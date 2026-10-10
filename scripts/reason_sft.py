"""Reasoning cold-start SFT, adapted from chat_sft without changing its baseline.

Run with uv run --no-sync python -m scripts.reason_sft --model-tag d24
--model-step 486 --output-tag d24-reason-smoke --num-iterations 50.
"""
import argparse
import gc
import json
import os
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import (
    CHECKPOINT_SOURCES,
    find_largest_model,
    find_last_step,
    load_model,
    save_checkpoint,
)
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
)
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from nanochat.loss_eval import evaluate_bpb
from nanochat.reasoning_runtime import prepare_output_directory, validate_output_tag
from nanochat.tokenizer import get_token_bytes
from scripts.chat_eval import run_chat_eval
from tasks.reasoning_data import build_sft_mixture


# -----------------------------------------------------------------------------
# CLI arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cold-start reasoning SFT with filtered traces and replay")
    # Logging
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
    # Runtime
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    # Model loading
    parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
    parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
    parser.add_argument("--source", choices=tuple(CHECKPOINT_SOURCES), default="sft")
    parser.add_argument("--output-tag", required=True, help="New checkpoint directory; must not already exist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    # Training horizon
    parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
    # Explicit conservative batches; only context defaults to the checkpoint.
    parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: checkpoint)")
    parser.add_argument("--device-batch-size", type=int, default=2, help="per-device microbatch (default: 2)")
    parser.add_argument("--total-batch-size", type=int, default=16384, help="tokens per optimizer step (default: 16384)")
    # Use checkpoint learning rates when recorded, otherwise original SFT defaults.
    parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: checkpoint)")
    parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: checkpoint)")
    parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: checkpoint)")
    parser.add_argument("--init-lr-frac", type=float, default=0.3, help="initial LR as fraction of base LR")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
    parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
    parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
    parser.add_argument("--eval-tokens", type=int, default=32768, help="number of tokens to evaluate val loss on")
    parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
    parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
    parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
    # Data mixture
    parser.add_argument("--metamath-rows", type=int, default=25000)
    parser.add_argument("--omi2-rows", type=int, default=25000)
    parser.add_argument("--procedural-rows", type=int, default=15000)
    parser.add_argument("--replay-frac", type=float, default=0.25)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    user_config = vars(args).copy()
    # -----------------------------------------------------------------------------

    # Compute init
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

    # wandb logging init
    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

    # Flash Attention status
    if not HAS_FA3:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

    # Resolve the parent identifiers so saved provenance never relies on auto-selection.
    base_dir = get_base_dir()
    input_root = os.path.join(base_dir, CHECKPOINT_SOURCES[args.source])
    args.model_tag = args.model_tag or find_largest_model(input_root)
    if args.model_step is None:
        args.model_step = find_last_step(os.path.join(input_root, args.model_tag))
    # Load the model and tokenizer
    model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    context_length = model.config.sequence_len
    if args.max_seq_len is None:
        args.max_seq_len = context_length
    if not 2 <= args.max_seq_len <= context_length:
        raise ValueError("max_seq_len must fit the checkpoint context")
    if args.num_iterations == 0 or args.num_iterations < -1:
        raise ValueError("num_iterations must be positive or -1 for a full epoch")
    if min(args.device_batch_size, args.total_batch_size, args.eval_tokens) < 1:
        raise ValueError("batch sizes and eval tokens must be positive")
    if not 0 <= args.warmup_ratio <= 1 or not 0 <= args.warmdown_ratio <= 1:
        raise ValueError("schedule ratios must be in [0, 1]")
    if args.warmup_ratio + args.warmdown_ratio > 1:
        raise ValueError("warmup and warmdown cannot overlap")
    if args.init_lr_frac <= 0 or args.final_lr_frac < 0:
        raise ValueError("learning-rate fractions must be valid")
    validate_output_tag(args.output_tag)
    checkpoint_dir = os.path.join(base_dir, CHECKPOINT_SOURCES["reason_sft"], args.output_tag)
    prepare_output_directory(Path(checkpoint_dir))

    # Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
    pretrain_user_config = meta.get("user_config", {})
    for name, fallback, source in [
        ("embedding_lr",      0.3,   pretrain_user_config),
        ("unembedding_lr",    0.004, pretrain_user_config),
        ("matrix_lr",         0.02,  pretrain_user_config),
    ]:
        arg_val = getattr(args, name)
        pretrain_val = source.get(name)
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(args, name, resolved)
            print0(f"Inherited {name}={resolved} from pretrained checkpoint")
        elif pretrain_val is not None and arg_val != pretrain_val:
            print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
        else:
            print0(f"Using {name}={arg_val}")

    orig_model = model
    model = torch.compile(model, dynamic=False) if args.compile else model
    num_flops_per_token = model.estimate_flops()
    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
    assert args.total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({args.total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
    grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
    token_bytes = get_token_bytes(device=device)

    # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
    # Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
    optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

    # Reasoning SFT deliberately starts a fresh optimizer.
    user_config = vars(args).copy()

    # GradScaler for fp16 training (bf16/fp32 don't need it)
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    # Override the initial learning rate as a fraction of the base learning rate
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

    # Filtering happens before row caps, replay and oversampling.
    train_dataset, val_dataset = build_sft_mixture(
        tokenizer, metamath_rows=args.metamath_rows, omi2_rows=args.omi2_rows,
        gsm8k_epochs=args.gsm8k_epochs, procedural_rows=args.procedural_rows,
        replay_fraction=args.replay_frac, seed=args.seed,
        validation_size=args.validation_size,
        context_limit=min(args.max_seq_len, 1536), assistant_limit=1024,
    )
    print0(f"Filtered mixture: {len(train_dataset):,} train / {len(val_dataset):,} validation rows")
    # DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
    # A big problem is that we don't know the final num_iterations in advance. So we create
    # shared state variables and update them from within the data generator.
    last_step = False # we will toggle this to True when we reach the end of the training dataset
    approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
    current_epoch = 1 # track epoch for logging
    def sft_data_generator_bos_bestfit(split: str, buffer_size: int = 100) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        BOS-aligned dataloader for SFT with bestfit-pad packing.

        Each row in the batch starts with BOS (beginning of a conversation).
        Conversations are packed using best-fit algorithm. When no conversation fits,
        the row is padded (instead of cropping) to ensure no tokens are ever discarded.
        Padding positions have targets masked with -1 (ignore_index for cross-entropy).
        """
        nonlocal approx_progress, current_epoch
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        dataset = train_dataset if split == "train" else val_dataset
        dataset_size = len(dataset)
        assert dataset_size > 0
        row_capacity = args.max_seq_len + 1  # +1 for target at last position
        bos_token = tokenizer.get_bos_token_id()

        # Conversation buffer: list of (token_ids, loss_mask) tuples
        conv_buffer = []
        cursor = ddp_rank % dataset_size  # Each rank processes different conversations (for fetching)
        consumed = ddp_rank  # Track actual consumption separately from buffering
        epoch = 1
        it = 0  # iteration counter

        def refill_buffer() -> None:
            nonlocal cursor, epoch
            while len(conv_buffer) < buffer_size:
                conversation = dataset[cursor]
                ids, mask = tokenizer.render_conversation(conversation, max_tokens=None)
                if len(ids) > row_capacity:
                    raise ValueError("Filtered conversation exceeds packing capacity")
                if not any(mask):
                    raise ValueError("Conversation has no supervised tokens")
                conv_buffer.append((ids, mask))
                cursor += ddp_world_size
                if cursor >= dataset_size:
                    cursor = cursor % dataset_size
                    epoch += 1
                    # Note: last_step is now triggered based on consumption, not fetching

        while True:
            rows = []
            mask_rows = []
            row_lengths = []  # Track actual content length (excluding padding) for each row
            for _ in range(args.device_batch_size):
                row = []
                mask_row = []
                padded = False
                while len(row) < row_capacity:
                    # Ensure buffer has conversations
                    while len(conv_buffer) < buffer_size:
                        refill_buffer()

                    remaining = row_capacity - len(row)

                    # Find largest conversation that fits entirely
                    best_idx = -1
                    best_len = 0
                    for i, (conv, _) in enumerate(conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        # Found a conversation that fits - use it entirely
                        conv, conv_mask = conv_buffer.pop(best_idx)
                        row.extend(conv)
                        mask_row.extend(conv_mask)
                        consumed += ddp_world_size  # Track actual consumption
                    else:
                        # No conversation fits - pad the remainder instead of cropping
                        # This ensures we never discard any tokens
                        content_len = len(row)
                        row.extend([bos_token] * remaining)  # Pad with BOS tokens
                        mask_row.extend([0] * remaining)
                        padded = True
                        break  # Row is now full (with padding)

                # Track content length: full row if no padding, otherwise the length before padding
                if padded:
                    row_lengths.append(content_len)
                else:
                    row_lengths.append(row_capacity)
                rows.append(row[:row_capacity])
                mask_rows.append(mask_row[:row_capacity])

            # Stopping condition to respect num_iterations, if given
            it += 1
            # Explicit horizons are checked against completed optimizer steps below.

            # Update progress tracking (based on consumed, not cursor, to account for buffering)
            if split == "train":
                current_epoch = epoch
                if args.num_iterations > 0:
                    approx_progress = it / (args.num_iterations * grad_accum_steps)
                else:
                    approx_progress = consumed / dataset_size

            # Build tensors
            use_cuda = device_type == "cuda"
            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
            targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

            # Apply the loss mask from render_conversation (mask=1 for assistant completions,
            # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
            # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
            mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
            mask_targets = mask_tensor[:, 1:].to(device=device)
            targets[mask_targets == 0] = -1

            # Mask out padding positions in targets (set to -1 = ignore_index)
            # For each row, positions >= (content_length - 1) in targets should be masked
            for i, content_len in enumerate(row_lengths):
                if content_len < row_capacity:
                    targets[i, content_len-1:] = -1

            yield inputs, targets

    train_loader = sft_data_generator_bos_bestfit("train")
    build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
    progress = 0 # will go from 0 to 1 over the course of the epoch

    # Learning rate schedule (linear warmup, constant, linear warmdown)
    # Same shape as base_train but uses progress (0→1) instead of absolute step counts,
    # because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
    def get_lr_multiplier(progress: float) -> float:
        if progress < args.warmup_ratio:
            return (progress + 1e-8) / args.warmup_ratio
        elif progress <= 1.0 - args.warmdown_ratio:
            return 1.0
        else:
            decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
            return (1 - decay) * 1.0 + decay * args.final_lr_frac

    # Momentum scheduler for Muon optimizer
    def get_muon_momentum(it: int) -> float:
        frac = min(it / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        return momentum

    # -----------------------------------------------------------------------------
    # Training loop
    x, y = next(train_loader) # prefetch the very first batch of data
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    ema_beta = 0.9 # EMA decay factor
    total_training_time = 0 # total wall-clock time of training
    step = 0
    latest_chatcore = {}
    run_started = time.monotonic()
    metrics_path = os.path.join(checkpoint_dir, "training.jsonl")
    while True:
        if args.num_iterations > 0:
            last_step = step >= args.num_iterations
        flops_so_far = num_flops_per_token * args.total_batch_size * step

        # Synchronize last_step across all ranks to avoid hangs in the distributed setting
        if ddp:
            last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
            dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
            last_step = bool(last_step_tensor.item())

        # once in a while: evaluate the val bpb (all ranks participate)
        if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            min_val_bpb = min(min_val_bpb, val_bpb)
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            })
            model.train()

        # once in a while: estimate the ChatCORE metric (all ranks participate)
        # use the original uncompiled model because the inputs keep changing shape
        if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
            model.eval()
            engine = Engine(orig_model, tokenizer)
            all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
            categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
            baseline_accuracies = {
                'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
                'GSM8K': 0.0, 'HumanEval': 0.0,
            }
            task_results = {}
            for task_name in all_tasks:
                limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
                max_problems = None if limit < 0 else limit  # -1 means no limit
                acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                    batch_size=args.device_batch_size, max_problems=max_problems, max_new_tokens=1024)
                task_results[task_name] = acc
                print0(f"  {task_name}: {100*acc:.2f}%")
            # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
            latest_chatcore = task_results
            centered = {task: (task_results[task] - baseline_accuracies[task]) / (1.0 - baseline_accuracies[task]) for task in all_tasks}
            chatcore = sum(centered.values()) / len(all_tasks)
            chatcore_cat = sum(centered[task] for task in categorical_tasks) / len(categorical_tasks)
            print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "chatcore_metric": chatcore,
                "chatcore_cat": chatcore_cat,
                **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
            })
            model.train()

        # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
        if last_step:
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "val_bpb": val_bpb, # loss at last step
                    "model_config": vars(orig_model.config),
                    "parent": {"source": args.source, "model_tag": args.model_tag,
                               "step": meta.get("step", args.model_step)},
                    "training_seconds": total_training_time,
                    "wall_seconds": time.monotonic() - run_started,
                    "peak_memory_bytes": get_max_memory(),
                    "train_rows": len(train_dataset),
                    "validation_rows": len(val_dataset),
                    "chatcore_tasks": latest_chatcore,
                    "user_config": user_config, # inputs to the training script
                },
                rank=ddp_rank,
            )

        if last_step:
            break

        # -------------------------------------------------------------------------
        # single training step
        # evaluate the gradient
        synchronize()
        t0 = time.time()
        step_progress = step / max(1, args.num_iterations - 1) if args.num_iterations > 0 else min(1.0, progress)
        loss_sum = 0.0
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SFT loss")
            loss_sum += loss.detach().item()
            loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # Account for the batch just trained, before prefetch advances the loader.
            progress = min(1.0, max(progress, approx_progress))
            if args.num_iterations < 0 and approx_progress >= 1:
                last_step = True
            x, y = next(train_loader)
        # step the optimizer
        lrm = get_lr_multiplier(step_progress)
        muon_momentum = get_muon_momentum(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
        if scaler is not None:
            scaler.unscale_(optimizer)
            if is_ddp_initialized():
                for v in scaler._found_inf_per_device(optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        t1 = time.time()
        dt = t1 - t0
        # -------------------------------------------------------------------------

        # State
        step += 1

        # logging
        train_loss = loss_sum / grad_accum_steps
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**step) # debias the EMA
        pct_done = 100 * progress
        tok_per_sec = int(args.total_batch_size / dt)
        flops_per_sec = num_flops_per_token * args.total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        total_training_time += dt
        if master_process:
            with open(metrics_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "loss": train_loss, "lr_multiplier": lrm, "seconds": dt, "tokens_per_second": tok_per_sec, "peak_memory_bytes": get_max_memory()}) + "\n")
        print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
        if step % 10 == 0:
            wandb_run.log({
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": current_epoch,
            })

        # The garbage collector spends ~500ms scanning for cycles quite frequently.
        # We manually manage it to avoid these pauses during training.
        if step == 1:
            gc.collect() # manually collect a lot of garbage from setup
            gc.freeze() # freeze all currently surviving objects and exclude them from GC
            gc.disable() # disable GC entirely except:
        elif step % 5000 == 0: # every 5000 steps...
            gc.collect() # manually collect, just to be safe for very long runs

    # print a few more stats
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time/60:.2f}m")
    print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

    # cleanup
    wandb_run.finish() # wandb run finish
    compute_cleanup()


if __name__ == "__main__":
    main()
