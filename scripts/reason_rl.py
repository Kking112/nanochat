"""Reasoning RL with correctness-separated shaped rewards or a binary ablation.

Example: uv run --no-sync python -m scripts.reason_rl --model-tag d24-reason \
    --output-tag d24-shaped-42 --num-steps 5 --examples-per-step 4 --num-samples 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nanochat.reasoning_runtime import (
    Curriculum,
    backward_group,
    optimizer_step_if_active,
    prepare_output_directory,
    render_completion_prompt,
    sample_completions,
    stable_seed,
    synchronize_curriculum,
    training_batch,
    validate_output_tag,
)


def build_parser() -> argparse.ArgumentParser:
    from nanochat.rewards import RewardConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="dummy", help="wandb run name; dummy disables network logging")
    parser.add_argument("--device-type", default="")
    parser.add_argument("--source", choices=("base", "sft", "rl", "reason_sft", "reason_rl"), default="reason_sft")
    parser.add_argument("--model-tag")
    parser.add_argument("--model-step", type=int)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--output-dir", type=Path, help="Fresh directory for logs; defaults outside Git")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--examples-per-step", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--procedural-size", type=int, default=100_000)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--reward-mode", choices=("binary", "shaped"), default="shaped")
    defaults = RewardConfig()
    for item in fields(defaults):
        default = getattr(defaults, item.name)
        parser.add_argument("--" + item.name.replace("_", "-"), type=type(default), default=default)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    validate_output_tag(args.output_tag)
    for name in ("num_steps", "device_batch_size", "examples_per_step", "num_samples", "max_new_tokens", "procedural_size", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("temperature", "top_k", "embedding_lr", "unembedding_lr", "matrix_lr", "weight_decay", "init_lr_frac"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")


def prompt_route(seed: int, prompt_number: int, sizes: dict[str, int]) -> tuple[str, int]:
    rng = random.Random(stable_seed(seed, "prompt", prompt_number))
    name = rng.choices(("gsm8k", "chain", "sorting", "countdown"), weights=(3, 1, 1, 1), k=1)[0]
    return name, rng.randrange(sizes[name])


def main(argv: list[str] | None = None) -> None:
    from nanochat.checkpoint_manager import (
        CHECKPOINT_SOURCES,
        find_largest_model,
        find_last_step,
        load_model,
        save_checkpoint,
    )
    from nanochat.common import (
        DummyWandb,
        autodetect_device_type,
        compute_cleanup,
        compute_init,
        get_base_dir,
        print0,
    )
    from nanochat.engine import Engine, use_calculator
    from nanochat.rewards import RewardConfig, score_response
    from tasks.gsm8k_reasoning import GSM8KReasoning
    from tasks.procedural import make_procedural_tasks

    args = build_parser().parse_args(argv)
    validate_args(args)
    reward_config = RewardConfig(**{item.name: getattr(args, item.name) for item in fields(RewardConfig)})
    ddp, rank, _, world_size, device = compute_init(args.device_type or autodetect_device_type())
    wandb_run: Any = DummyWandb()
    try:
        if args.examples_per_step % world_size:
            raise ValueError("Examples per step must be divisible by the distributed world size")
        torch.manual_seed(stable_seed(args.seed, rank))
        base_dir = Path(get_base_dir())
        source_dir = base_dir / CHECKPOINT_SOURCES[args.source]
        args.model_tag = args.model_tag or find_largest_model(str(source_dir))
        args.model_step = args.model_step if args.model_step is not None else find_last_step(str(source_dir / args.model_tag))
        checkpoint_dir = base_dir / CHECKPOINT_SOURCES["reason_rl"] / args.output_tag
        if checkpoint_dir.resolve() == (source_dir / args.model_tag).resolve():
            raise ValueError("Output checkpoint tag must differ from its input checkpoint directory")
        output_dir = args.output_dir or base_dir / "reasoning_runs" / args.output_tag
        prepare_output_directory(checkpoint_dir)
        prepare_output_directory(output_dir)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        if rank == 0:
            (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if args.run != "dummy" and rank == 0:
            import wandb
            wandb_run = wandb.init(project="nanochat-reason-rl", name=args.run, config=config)
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
        engine = Engine(model, tokenizer)
        context_length = int(model.config.sequence_len)
        tasks: dict[str, Any] = {"gsm8k": GSM8KReasoning(split="train", seed=args.seed)}
        tasks.update(make_procedural_tasks(split="train", seed=args.seed, size=args.procedural_size, difficulty=1))
        sizes = {name: len(task) for name, task in tasks.items()}
        if any(size == 0 for size in sizes.values()):
            raise ValueError("Every RL task must contain prompts")
        curriculum = Curriculum.from_state_dict(meta["reasoning_curriculum"]) if args.source == "reason_rl" and "reasoning_curriculum" in meta else Curriculum()
        optimizer = model.setup_optimizer(
            unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr,
            matrix_lr=args.matrix_lr, weight_decay=args.weight_decay,
        )
        for group in optimizer.param_groups:
            group["lr"] *= args.init_lr_frac
            group["initial_lr"] = group["lr"]
        examples_per_rank = args.examples_per_step // world_size
        optimizer_updates = 0
        total_tokens = 0
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with (output_dir / f"rollouts_rank{rank}.jsonl").open("w") as rollout_file, (output_dir / f"metrics_rank{rank}.jsonl").open("w") as metrics_file:
            for step in range(args.num_steps):
                step_start = time.perf_counter()
                observations: list[tuple[int, str, float]] = []
                sample_metrics: list[dict[str, Any]] = []
                active_groups = 0
                zero_variance_groups = 0
                total_loss = 0.0
                step_tokens = 0
                model.zero_grad(set_to_none=True)
                for name, difficulty in curriculum.difficulties.items():
                    tasks[name].difficulty = difficulty
                for local_index in range(examples_per_rank):
                    prompt_number = step * args.examples_per_step + local_index * world_size + rank
                    name, index = prompt_route(args.seed, prompt_number, sizes)
                    task = tasks[name]
                    conversation = task[index]
                    prompt = render_completion_prompt(tokenizer, conversation)
                    model.eval()
                    with torch.no_grad():
                        completions = sample_completions(
                            engine, tokenizer, prompt, context_length=context_length,
                            num_samples=args.num_samples, device_batch_size=args.device_batch_size,
                            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_k=args.top_k, seed=stable_seed(args.seed, "rollout", prompt_number),
                        )
                    breakdowns = [score_response(
                        task, conversation, completion.text, use_calculator,
                        config=reward_config, progress=step / max(1, args.num_steps - 1),
                        mode=args.reward_mode, terminated=completion.terminated,
                        completion_tokens=completion.completion_tokens,
                        generation_budget=completion.effective_budget,
                    ) for completion in completions]
                    correctness = [int(task.evaluate(conversation, completion.text) == 1) for completion in completions]
                    observations.append((prompt_number, name, sum(correctness) / len(correctness)))
                    for sample_index, (completion, breakdown, correct) in enumerate(zip(completions, breakdowns, correctness)):
                        components = breakdown.to_dict()
                        sample_metrics.append(components)
                        rollout_file.write(json.dumps({
                            "step": step, "prompt_number": prompt_number, "task": name, "index": index,
                            "difficulty": curriculum.difficulties.get(name), "sample": sample_index,
                            "prompt": conversation["messages"][0]["content"], "response": completion.text,
                            "correct": correct, "stop_reason": completion.stop_reason,
                            "completion_tokens": completion.completion_tokens, "effective_budget": completion.effective_budget,
                            "reward_diagnostics": components,
                        }) + "\n")
                    step_tokens += sum(completion.completion_tokens for completion in completions)
                    inputs, targets = training_batch(completions, tokenizer.encode_special("<|assistant_end|>"), device)
                    rewards = torch.tensor([breakdown.total for breakdown in breakdowns], dtype=torch.float32, device=device)
                    zero_variance_groups += int(torch.equal(rewards, rewards[:1].expand_as(rewards)))
                    model.train()
                    active, loss = backward_group(model, inputs, targets, rewards, device_batch_size=args.device_batch_size, examples_per_rank=examples_per_rank)
                    active_groups += int(active)
                    total_loss += loss
                lrm = 1.0 - step / args.num_steps
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * lrm
                updated, gradient_norm = optimizer_step_if_active(model, optimizer, active_groups, device)
                optimizer_updates += int(updated)
                synchronize_curriculum(curriculum, observations)
                # Counts and scalar diagnostic sums are reduced over all sampled responses.
                local_summary = {
                    "tokens": step_tokens, "sample_count": len(sample_metrics), "active_groups": active_groups,
                    "zero_variance_groups": zero_variance_groups,
                    "loss": total_loss, "correct_sum": sum(item[2] for item in observations),
                    "components": {key: sum(float(row[key]) for row in sample_metrics) for key, value in sample_metrics[0].items() if isinstance(value, (int, float, bool))},
                }
                summaries: list[Any] = [local_summary]
                if ddp:
                    summaries = [None] * world_size
                    dist.all_gather_object(summaries, local_summary)
                count = sum(item["sample_count"] for item in summaries)
                global_tokens = sum(item["tokens"] for item in summaries)
                total_tokens += global_tokens
                duration = time.perf_counter() - step_start
                metrics = {
                    "attempted_rollout_steps": step + 1, "optimizer_updates": optimizer_updates,
                    "active_groups": sum(item["active_groups"] for item in summaries),
                    "zero_variance_fraction": sum(item["zero_variance_groups"] for item in summaries) / args.examples_per_step,
                    "binary_correctness": sum(item["correct_sum"] for item in summaries) / args.examples_per_step,
                    "gradient_norm_local": gradient_norm, "loss": sum(item["loss"] for item in summaries) / world_size,
                    "response_tokens": global_tokens, "response_tokens_mean": global_tokens / count,
                    "elapsed_seconds": duration, "tokens_per_second": global_tokens / duration,
                    "peak_vram_bytes_local": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    "lrm": lrm, "curriculum": curriculum.state_dict(),
                    "training_diagnostics": {key: sum(item["components"][key] for item in summaries) / count for key in local_summary["components"]},
                }
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                rollout_file.flush()
                print0(json.dumps(metrics))
                wandb_run.log({"step": step, **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}, **{f"reward/{key}": value for key, value in metrics["training_diagnostics"].items()}})
                if (step + 1) % args.save_every == 0 or step == args.num_steps - 1:
                    save_checkpoint(str(checkpoint_dir), step + 1, model.state_dict(), optimizer.state_dict(), {
                        "model_config": vars(model.config), "user_config": config,
                        "parent_checkpoint": {"source": args.source, "model_tag": args.model_tag, "step": args.model_step},
                        "reasoning_curriculum": curriculum.state_dict(), "attempted_rollout_steps": step + 1,
                        "optimizer_updates": optimizer_updates, "fresh_optimizer": True,
                        "elapsed_seconds": time.perf_counter() - start_time,
                    }, rank=rank)
            if rank == 0:
                elapsed = time.perf_counter() - start_time
                (output_dir / "summary.json").write_text(json.dumps({
                    "config": config, "attempted_rollout_steps": args.num_steps,
                    "optimizer_updates": optimizer_updates, "elapsed_seconds": elapsed,
                    "response_tokens": total_tokens, "tokens_per_second": total_tokens / elapsed,
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    "curriculum": curriculum.state_dict(), "checkpoint_dir": str(checkpoint_dir),
                }, indent=2) + "\n")
    finally:
        wandb_run.finish()
        compute_cleanup()


if __name__ == "__main__":
    main()
