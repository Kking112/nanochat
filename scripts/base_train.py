"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import re
import csv
import sys
import json
import time
import math
import random
import hashlib
import argparse
import subprocess
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear, parse_layout
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Looped (weight-shared depth) study, see looped_nanochat_proposal.md section 9.5. All off by default.
parser.add_argument("--layout", type=str, default="", help="'P,KxR,C' = P prelude, K core blocks applied R times, C coda blocks (e.g. '2,4x2,2'), or 'N' = plain N-layer model. Replaces --depth.")
parser.add_argument("--width-depth", type=int, default=12, help="with --layout: model_dim = width_depth * aspect_ratio (the role --depth plays for the width)")
parser.add_argument("--ref-layout", type=str, default="", help="with --layout: derive horizon, batch size, LR batch scaling and weight decay scaling from a model of this layout instead of from this model, so they are identical across arms")
parser.add_argument("--horizon-frac", type=float, default=1.0, help="with --layout: train for this fraction of the horizon. Only shortens num_iterations: batch size, LR scale and weight decay stay those of the full horizon")
parser.add_argument("--matrix-lr-mult", type=float, default=1.0, help="multiplier on --matrix-lr")
parser.add_argument("--seed", type=int, default=-1, help="seed for parameter init only, data order is not affected (-1 = leave the global default seed alone)")
parser.add_argument("--train-loops", type=str, default="", help="with --layout: comma-separated loop counts, e.g. '1,2,3,4'. One is sampled uniformly per optimizer step. Evaluation uses the layout's own R.")
parser.add_argument("--arm", type=str, default="", help="name of the study arm. If given, the run is logged to <results-dir>/logs/*.jsonl and appended as one row to <results-dir>/looped_results.csv")
parser.add_argument("--results-dir", type=str, default="results", help="where --arm writes its results")
parser.add_argument("--no-save-optimizer", action="store_true", help="do not save optimizer state in the final checkpoint (saves disk). Intermediate checkpoints (--num-checkpoints) always include it")
parser.add_argument("--num-checkpoints", type=int, default=0, help="save this many evenly spaced intermediate checkpoints (with optimizer state) during the run, each replacing the previous one, so that an interrupted run can be resumed with --auto-resume")
parser.add_argument("--auto-resume", action="store_true", help="if the checkpoint directory holds an intermediate checkpoint of this run, resume from it. The data stream is replayed exactly (not the approximate --resume-from-step): the resumed run trains on the same tokens in the same order as an uninterrupted one")
parser.add_argument("--exit-after-step", type=int, default=-1, help="testing only: exit right after the checkpoint of this step was saved, to simulate an interruption")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
if args.layout:
    # the default tag is d{depth}: every arm would overwrite the same directory, and evals that
    # guess the model tag prefer d<number> directories
    assert args.model_tag, "--layout requires an explicit --model-tag"
    # Without it an arm derives horizon, batch size and weight decay from its own size and still runs
    # to completion, e.g. an 8-layer arm gets 1872 steps at weight decay 0.377 instead of 2520 at 0.28
    assert args.ref_layout, "--layout requires --ref-layout (use the arm's own layout for a standalone model)"
else:
    assert not (args.ref_layout or args.train_loops or args.horizon_frac != 1.0), "--ref-layout, --train-loops and --horizon-frac require --layout"
if args.auto_resume:
    assert args.num_checkpoints > 0 and args.resume_from_step == -1, "--auto-resume goes with --num-checkpoints and replaces --resume-from-step"
train_loops = sorted({int(r) for r in args.train_loops.split(",")}) if args.train_loops else []
script_t0 = time.time()
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth, layout=None):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data).
    With a layout, depth only sets the width and the layout sets the (unique) layers."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    layer_kwargs = parse_layout(layout) if layout else dict(n_layer=depth)
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern, **layer_kwargs,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
if args.layout:
    model = build_model_meta(args.width_depth, args.layout)
else:
    model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
if args.seed >= 0:
    # init_weights is the only consumer of the global RNG (dataloader, optimizer and evals use
    # none or their own), so the seed changes the parameter init and nothing else
    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(args.seed)
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
if args.auto_resume and os.path.isdir(checkpoint_dir):
    # The latest complete intermediate checkpoint: all three files present (the meta file is written
    # last and every file is renamed into place, so a checkpoint cut short by a crash is never complete)
    saved_steps = sorted(int(f[len("model_"):-len(".pt")]) for f in os.listdir(checkpoint_dir) if re.match(r"model_\d+\.pt$", f))
    complete = [s for s in saved_steps
                if os.path.exists(os.path.join(checkpoint_dir, f"meta_{s:06d}.json")) and os.path.exists(os.path.join(checkpoint_dir, f"optim_{s:06d}_rank{ddp_rank:d}.pt"))]
    if complete:
        args.resume_from_step = complete[-1]
        print0(f"Auto-resume: found intermediate checkpoint at step {args.resume_from_step} in {checkpoint_dir}")
resuming = args.resume_from_step != -1
exact_resume = resuming and args.auto_resume # replay the data stream instead of the approximate dataloader resume
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    if exact_resume:
        # the checkpoint must be from this very run configuration, or the resumed run is a chimera
        same = {k: meta_data["user_config"].get(k) for k in ("layout", "width_depth", "ref_layout", "seed", "matrix_lr_mult", "horizon_frac", "train_loops", "arm", "depth", "total_batch_size", "max_seq_len")}
        mine = {k: user_config.get(k) for k in same}
        assert same == mine, f"checkpoint is from a different run configuration:\n{same}\n{mine}"
        assert not meta_data["loop_state"].get("final", False), "this run already finished"
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if args.layout:
    # The loop count is a Python int attribute of the model, so every distinct R compiles its own
    # graph (on top of the train/eval graphs). Make room, and make running out an error instead
    # of a silent fall back to eager mode.
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.fail_on_recompile_limit_hit = True
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
# The horizon, and with it the batch size, LR scaling and weight decay scaling below, normally derive
# from the model's own size. With --ref-layout they derive from a reference model instead, so that
# they come out identical for every arm of a study no matter how many parameters the arm has.
horizon_scaling_params = num_scaling_params
if args.ref_layout:
    horizon_scaling_params = get_scaling_params(build_model_meta(args.width_depth, args.ref_layout))
    print0(f"Horizon and derived hyperparameters use reference layout {args.ref_layout}: {horizon_scaling_params:,} scaling params (this model: {num_scaling_params:,})")
target_tokens = int(args.target_param_data_ratio * horizon_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for {'layout ' + args.layout if args.layout else f'depth {args.depth}'}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * args.matrix_lr_mult * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
# With an exact resume the loader starts from the beginning and is fast-forwarded below (the stream
# is deterministic and does not depend on the seed), instead of the approximate row-group resume.
dataloader_resume_state_dict = None if (not resuming or exact_resume) else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
if args.horizon_frac != 1.0:
    # Shorten the run only. Everything derived from the horizon above (batch size, LR scale, weight
    # decay) deliberately stays at its full-horizon value, the schedules below span the short run.
    assert args.num_iterations <= 0 and args.target_flops <= 0, "--horizon-frac applies to the data:param ratio horizon only"
    num_iterations = round(num_iterations * args.horizon_frac)
    print0(f"Horizon fraction {args.horizon_frac}: training for {num_iterations:,} iterations")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Looped study: per-step loop counts, FLOPs accounting, results logging, diagnostics

# Random loop count training: the R of every optimizer step is fixed up front, from an RNG that is
# independent of --seed, so every seed (and a resumed run) sees the same sequence of R.
LOOP_SCHEDULE_SEED = 1337
loop_rng = random.Random(LOOP_SCHEDULE_SEED)
loop_schedule = [loop_rng.choice(train_loops) for _ in range(num_iterations)] if train_loops else None
# FLOPs per token depend on R, so training FLOPs are accumulated with the R actually used per step
flops_per_token_at = {r: orig_model.estimate_flops(num_loops=r) for r in train_loops}
def get_flops_per_token(it):
    return flops_per_token_at[loop_schedule[it]] if loop_schedule else num_flops_per_token
def get_flops_so_far(it):
    if loop_schedule:
        return total_batch_size * sum(flops_per_token_at[r] for r in loop_schedule[:it])
    return num_flops_per_token * total_batch_size * it
def set_eval_loops():
    # evaluation always uses the layout's own R, whatever the last training step used
    if loop_schedule:
        orig_model.set_num_loops(model_config.n_loop)

# Data order check: hash the first rows of the training stream. Counted in rows and not in
# micro-batches, so that the hash does not depend on --device-batch-size.
# Capped at the rows the run trains on: the loader prefetches one micro-batch past the last step,
# and those extra rows do depend on the micro-batch size.
DATA_HASH_ROWS = min(320, num_iterations * total_batch_size // args.max_seq_len)
data_hasher, data_hash_rows = hashlib.sha256(), 0
def hash_data(inputs, targets):
    # called on receipt of every batch: the dataloader reuses its buffers
    global data_hash_rows
    n = min(len(inputs), DATA_HASH_ROWS - data_hash_rows)
    if n > 0 and args.arm and (not resuming or exact_resume): # the .cpu() is a sync point: keep it off the stock path
        rows = torch.cat([inputs[:n], targets[:n]], dim=1).cpu().numpy() # row by row: (input, target) pairs
        data_hasher.update(rows.tobytes())
        data_hash_rows += n
hash_data(x, y)
if exact_resume:
    # Replay the stream up to the batch the interrupted run would have used next. Before the
    # interruption the loop had consumed 1 (the first prefetch) + step * grad_accum batches; x holds
    # batch 0 now, so after step * grad_accum more fetches it holds the right one. The hash of the
    # leading rows is rebuilt on the way, so the data check covers a resumed run too.
    skip = args.resume_from_step * (total_batch_size // (args.device_batch_size * args.max_seq_len * ddp_world_size))
    print0(f"Exact resume: replaying {skip:,} micro-batches of the data stream ...")
    t_replay = time.time()
    for i in range(skip):
        x, y, dataloader_state_dict = next(train_loader)
        hash_data(x, y)
        if (i + 1) % 2000 == 0:
            print0(f"  replayed {i + 1:,}/{skip:,} micro-batches ({time.time() - t_replay:.0f}s)")
    print0(f"Exact resume: data stream at step {args.resume_from_step} after {time.time() - t_replay:.0f}s")

# Core blocks are the ones that are visited more than once
core_layers = range(model_config.n_prelude, model_config.n_prelude + model_config.n_core) if model_config.n_prelude >= 0 else range(0)
def get_grad_norms():
    """Gradient norm of the matrices of the core blocks vs those of all other blocks."""
    sq = {True: [], False: []}
    for u, block in enumerate(orig_model.transformer.h):
        sq[u in core_layers].extend(p.grad.float().square().sum() for p in block.parameters() if p.grad is not None)
    norm_of = lambda terms: torch.stack(terms).sum().sqrt().item() if terms else 0.0
    return {"grad_norm_core": norm_of(sq[True]), "grad_norm_noncore": norm_of(sq[False])}

study = bool(args.arm) and master_process # log this run as an arm of the study
if study:
    from nanochat.flash_attention import USE_FA3 as attn_is_fa3
    git = lambda *cmd: subprocess.run(["git", *cmd], capture_output=True, text=True).stdout.strip()
    # Dirty = modified or untracked files in the code and its environment. Not the whole repo: the
    # results of earlier runs are untracked files too, and would mark every later run as dirty.
    CODE_PATHS = ["nanochat", "scripts", "runs", "tasks", "tests", "pyproject.toml", "uv.lock"]
    git_hash = git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", *CODE_PATHS) else "")
    if exact_resume and meta_data["loop_state"].get("run_id"):
        # the same run continues: keep its id and append to its log
        run_id = meta_data["loop_state"]["run_id"]
        jsonl_path = os.path.join(args.results_dir, "logs", f"{run_id}.jsonl")
    else:
        run_id = f"{output_dirname}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)
        jsonl_path = os.path.join(args.results_dir, "logs", f"{run_id}.jsonl")
        assert not os.path.exists(jsonl_path), f"refusing to overwrite the log of a past run: {jsonl_path}"
    # a fixed batch of validation rows for the residual stream diagnostics
    diag_x = next(build_val_loader())[0][:4].clone()
def log_event(kind, **data):
    if study:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, **data}) + "\n")

param_counts_table = {
    "unique block params": param_counts["transformer_matrices"], "lm_head": param_counts["lm_head"], "wte": param_counts["wte"],
    "value embed params (lookups, no FLOPs)": param_counts["value_embeds"], "scalars": param_counts["scalars"],
    "unique layers U": param_counts["unique_layers"], "effective layers E": param_counts["effective_layers"],
    "FLOPs per token": num_flops_per_token,
}
print0("Parameter table:")
for key, value in param_counts_table.items():
    print0(f"  {key:40s}: {value:,}")
log_event("config",
    run_id=run_id if study else None, arm=args.arm, layout=args.layout, ref_layout=args.ref_layout, seed=args.seed,
    git_hash=git_hash if study else None, user_config=user_config, model_config=model_config_kwargs, param_counts=param_counts,
    num_flops_per_token=num_flops_per_token, flops_per_token_at_loops=flops_per_token_at,
    attention="fa3" if study and attn_is_fa3 else "sdpa", compute_dtype=str(COMPUTE_DTYPE),
    gpu=torch.cuda.get_device_name(0) if device_type == "cuda" else device_type, torch_version=torch.__version__,
    num_iterations=num_iterations, total_batch_size=total_batch_size, target_tokens=target_tokens,
    batch_lr_scale=batch_lr_scale, weight_decay_scaled=weight_decay_scaled, matrix_lr=args.matrix_lr * args.matrix_lr_mult * batch_lr_scale,
    train_loops=train_loops, loop_schedule_seed=LOOP_SCHEDULE_SEED,
    loop_schedule_sha256=hashlib.sha256(bytes(loop_schedule)).hexdigest() if loop_schedule else None,
)

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

if loop_schedule:
    # Compile the training graph of every R now. Otherwise the first use of an R would recompile
    # in the middle of the run and distort the timings. No optimizer step, no data consumed.
    for r in train_loops:
        orig_model.set_num_loops(r)
        model(x, y).backward()
        print0(f"Compiled training graph for R={r}")
    model.zero_grad(set_to_none=True)

val_bpb_history = {} # step -> val bpb
step_times = [] # seconds per step, after the first 10 steps
final_core_metric = None
diverged = False
wall_clock_before = 0.0 # wall-clock of earlier segments of this run, if resumed
ckpt_every = num_iterations // args.num_checkpoints if args.num_checkpoints > 0 else 0
last_intermediate_step = None # the intermediate checkpoint to delete after the next one is saved
if exact_resume:
    saved = meta_data["loop_state"]
    val_bpb_history = {int(k): v for k, v in saved["val_bpb_history"].items()}
    step_times = saved["step_times"]
    final_core_metric = saved["final_core_metric"]
    diverged = saved["diverged"]
    wall_clock_before = saved["wall_clock_s"]
    last_intermediate_step = args.resume_from_step
    log_event("resume", step=step, git_hash=git_hash if study else None, wall_clock_before=wall_clock_before)

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = get_flops_so_far(step)
    set_eval_loops()

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        val_bpb_history[step] = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        log_event("eval", step=step, val_bpb=val_bpb, total_training_flops=flops_so_far, total_training_time=total_training_time)
        if study:
            # Residual stream RMS after every effective layer (mean over batch and positions), for
            # each loop count of interest. The end of core iteration i of a P,KxR,C layout is
            # entry P + K*(i+1) - 1. Uncompiled model: this must not touch the compiled graphs.
            residual_rms = {}
            with torch.no_grad(), disable_fp8(orig_model):
                for r in (train_loops or [model_config.n_loop]):
                    orig_model.set_num_loops(r)
                    residual_rms[r] = []
                    orig_model(diag_x, diag=residual_rms[r])
            orig_model.set_num_loops(model_config.n_loop)
            log_event("diag", step=step, residual_rms=residual_rms)
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        final_core_metric = results["core_metric"]
        log_event("core", step=step, core_metric=results["core_metric"], centered_results=results["centered_results"])
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    intermediate = not last_step and step > 0 and step != args.resume_from_step and ckpt_every > 0 and step % ckpt_every == 0
    if last_step or intermediate or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            None if (args.no_save_optimizer and not intermediate) else optimizer.state_dict(), # optimizer state: an intermediate checkpoint exists to be resumed from
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    # the rest is what --auto-resume needs to continue the study's bookkeeping
                    "val_bpb_history": val_bpb_history,
                    "step_times": step_times,
                    "final_core_metric": final_core_metric,
                    "diverged": diverged,
                    "wall_clock_s": wall_clock_before + time.time() - script_t0,
                    "run_id": run_id if study else None,
                    "final": last_step,
                },
            },
            rank=ddp_rank,
        )
        if intermediate or last_step:
            # the checkpoint before this one has served its purpose
            if last_intermediate_step is not None and last_intermediate_step != step:
                for name in ([f"model_{last_intermediate_step:06d}.pt", f"meta_{last_intermediate_step:06d}.json"] if ddp_rank == 0 else []) + [f"optim_{last_intermediate_step:06d}_rank{ddp_rank:d}.pt"]:
                    path = os.path.join(checkpoint_dir, name)
                    if os.path.exists(path):
                        os.remove(path)
            last_intermediate_step = step
        if intermediate and step == args.exit_after_step:
            print0(f"--exit-after-step {step}: exiting right after the checkpoint, as if interrupted")
            wandb_run.finish()
            compute_cleanup()
            sys.exit(0)

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    if loop_schedule:
        orig_model.set_num_loops(loop_schedule[step]) # the same R for all micro-batches of this step
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        hash_data(x, y)
    grad_norms = get_grad_norms() if study and step % 100 == 0 else {} # before the optimizer touches the grads
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = get_flops_per_token(step) * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
        step_times.append(dt)
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)
        log_event("train", **log_data, **grad_norms, num_loops=orig_model.num_loops)

    # A diverged run is a result of the study: record it and stop, do not try to rescue it.
    # Study runs only: without --arm the loop trains on through a non-finite loss, as upstream does.
    loss_is_finite = math.isfinite(train_loss_f)
    if args.arm and is_ddp_initialized():
        # the loss is rank-local: all ranks have to leave the loop together
        any_nonfinite = torch.tensor(float(not loss_is_finite), device=device)
        dist.all_reduce(any_nonfinite, op=dist.ReduceOp.MAX)
        loss_is_finite = any_nonfinite.item() == 0
    if args.arm and not loss_is_finite:
        print0(f"Step {step:05d} | training loss is {train_loss_f}: run diverged, stopping")
        diverged = True
        break

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Looped study: one row per run, appended. Rows of past runs are never edited.
if study:
    final_val_bpb = float("nan") if diverged else val_bpb_history.get(step, "not evaluated") # e.g. --eval-every -1
    # the second divergence criterion of the study: the final bpb is above the step-500 value
    steps_from_500 = [it for it in sorted(val_bpb_history) if it >= 500]
    val_bpb_at_500 = val_bpb_history[steps_from_500[0]] if steps_from_500 and steps_from_500[0] < step else None
    if not diverged and step in val_bpb_history and val_bpb_at_500 is not None:
        diverged = final_val_bpb > val_bpb_at_500
    timed_steps = len(step_times) # total_training_time skips the first 10 steps
    row = {
        "arm": args.arm, "layout": args.layout or str(args.depth), "seed": args.seed, "lr_mult": args.matrix_lr_mult,
        "horizon_frac": args.horizon_frac, "steps": step, "tokens": total_batch_size * step,
        "train_flops": get_flops_so_far(step), "final_val_bpb": final_val_bpb, "core": final_core_metric,
        "tokens_per_sec": round(timed_steps * total_batch_size / total_training_time) if total_training_time > 0 else None,
        # the mean above suffers from anything else that uses the GPU during the run, the median step does not
        "tokens_per_sec_median": round(total_batch_size / sorted(step_times)[len(step_times) // 2]) if step_times else None,
        "peak_vram_mib": round(get_max_memory() / 1024 / 1024), "wall_clock_s": round(wall_clock_before + time.time() - script_t0),
        "diverged": int(diverged), "git_hash": git_hash,
        "train_loops": args.train_loops, "model_tag": output_dirname, "run_id": run_id,
        "data_sha256": data_hasher.hexdigest() if data_hash_rows > 0 else None, "data_hash_rows": data_hash_rows,
    }
    with torch.no_grad():
        learned_scalars = {"resid_lambdas": orig_model.resid_lambdas.tolist(), "x0_lambdas": orig_model.x0_lambdas.tolist(), "backout_lambda": orig_model.backout_lambda.item()}
    log_event("final", **row, val_bpb_at_500=val_bpb_at_500, min_val_bpb=min_val_bpb, **learned_scalars)
    csv_path = os.path.join(args.results_dir, "looped_results.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print0(f"Appended results row to {csv_path} (diverged={int(diverged)}), log: {jsonl_path}")

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
