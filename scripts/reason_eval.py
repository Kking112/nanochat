"""Reproducible binary reasoning evaluation; reward values are not evaluation scores."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.distributed as dist

from nanochat.reasoning_runtime import (
    prepare_output_directory,
    render_completion_prompt,
    sample_completions,
    stable_seed,
)


class Outcome(TypedDict):
    correct: int
    response: str
    completion_tokens: int
    stop_reason: str
    effective_budget: int


class EvaluationRecord(TypedDict):
    task: str
    index: int
    question: str
    greedy: Outcome
    sampled: list[Outcome]


def wilson_interval(successes: int, count: int) -> list[float]:
    """95% Wilson score interval over independent held-out prompts."""
    if count <= 0:
        raise ValueError("At least one evaluation prompt is required")
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_records(records: list[EvaluationRecord], num_samples: int) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    correct1 = sum(record["greedy"]["correct"] for record in records)
    correctk = sum(any(outcome["correct"] for outcome in record["sampled"]) for record in records)
    greedy = [record["greedy"] for record in records]
    sampled = [outcome for record in records for outcome in record["sampled"]]
    result: dict[str, Any] = {
        "examples": len(records), "greedy_pass@1": correct1 / len(records),
        "greedy_pass@1_ci95": wilson_interval(correct1, len(records)),
        f"sampled_pass@{num_samples}": correctk / len(records),
        f"sampled_pass@{num_samples}_ci95": wilson_interval(correctk, len(records)),
    }
    for name, outcomes in (("greedy", greedy), ("sampled", sampled)):
        result[f"{name}_response_tokens_mean"] = sum(row["completion_tokens"] for row in outcomes) / len(outcomes)
        result[f"{name}_response_tokens_total"] = sum(row["completion_tokens"] for row in outcomes)
        result[f"{name}_truncation_rate"] = sum(row["stop_reason"] == "length" for row in outcomes) / len(outcomes)
        result[f"{name}_normal_termination_rate"] = sum(row["stop_reason"] == "assistant_end" for row in outcomes) / len(outcomes)
    return result


def evaluate_prompt(
    task: Any, conversation: dict[str, Any], engine: Any, tokenizer: Any, *,
    task_name: str, index: int, context_length: int, max_new_tokens: int,
    num_samples: int, device_batch_size: int, temperature: float, top_k: int,
    seed: int,
) -> EvaluationRecord:
    prompt = render_completion_prompt(tokenizer, conversation)
    modes: dict[str, list[Outcome]] = {}
    for mode, samples, sampling_temperature in (("greedy", 1, 0.0), ("sampled", num_samples, temperature)):
        completions = sample_completions(
            engine, tokenizer, prompt, context_length=context_length, num_samples=samples,
            device_batch_size=device_batch_size, max_new_tokens=max_new_tokens,
            temperature=sampling_temperature, top_k=top_k,
            seed=stable_seed(seed, task_name, index, mode),
        )
        modes[mode] = [{
            "correct": int(task.evaluate(conversation, completion.text) == 1),
            "response": completion.text, "completion_tokens": completion.completion_tokens,
            "stop_reason": completion.stop_reason, "effective_budget": completion.effective_budget,
        } for completion in completions]
    return {
        "task": task_name, "index": index, "question": conversation["messages"][0]["content"],
        "greedy": modes["greedy"][0], "sampled": modes["sampled"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("base", "sft", "rl", "reason_sft", "reason_rl"), default="reason_rl")
    parser.add_argument("--model-tag")
    parser.add_argument("--model-step", type=int)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh result directory outside Git")
    parser.add_argument("--tasks", default="gsm8k,chain,sorting,countdown")
    parser.add_argument("--difficulties", default="1,3")
    parser.add_argument("--max-examples", type=int, help="Optional cap per task; omitted evaluates all GSM8K test questions")
    parser.add_argument("--procedural-examples", type=int, default=256)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=8, help="Number of independently sampled responses for sampled pass@k")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=1729, help="Held-out procedural dataset seed; match across experimental arms")
    return parser


def main(argv: list[str] | None = None) -> None:
    from nanochat.checkpoint_manager import (
        CHECKPOINT_SOURCES,
        find_largest_model,
        find_last_step,
        load_model,
    )
    from nanochat.common import (
        autodetect_device_type,
        compute_cleanup,
        compute_init,
        get_base_dir,
        print0,
    )
    from nanochat.engine import Engine
    from tasks.gsm8k_reasoning import GSM8KReasoning
    from tasks.procedural import make_procedural_tasks

    args = build_parser().parse_args(argv)
    names = list(dict.fromkeys(item.strip() for item in args.tasks.split(",")))
    difficulties = list(dict.fromkeys(int(item) for item in args.difficulties.split(",")))
    if not set(names) <= {"gsm8k", "chain", "sorting", "countdown"} or any(not 1 <= item <= 5 for item in difficulties):
        raise ValueError("Unknown task or difficulty outside 1..5")
    if min(args.procedural_examples, args.device_batch_size, args.num_samples, args.max_new_tokens) <= 0 or (args.max_examples is not None and args.max_examples <= 0):
        raise ValueError("Evaluation counts and token budgets must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0 or args.top_k < 0:
        raise ValueError("Sampled evaluation requires positive temperature and nonnegative top-k")
    ddp, rank, _, world_size, device = compute_init(args.device_type or autodetect_device_type())
    try:
        prepare_output_directory(args.output_dir)
        base_dir = Path(get_base_dir())
        source_dir = base_dir / CHECKPOINT_SOURCES[args.source]
        args.model_tag = args.model_tag or find_largest_model(str(source_dir))
        args.model_step = args.model_step if args.model_step is not None else find_last_step(str(source_dir / args.model_tag))
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        if rank == 0:
            (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
        engine = Engine(model, tokenizer)
        tasks: dict[str, Any] = {}
        if "gsm8k" in names:
            tasks["gsm8k"] = GSM8KReasoning(split="test", seed=args.data_seed)
        for difficulty in difficulties:
            procedural = make_procedural_tasks(split="test", seed=args.data_seed, size=args.procedural_examples, difficulty=difficulty)
            for name in names:
                if name != "gsm8k":
                    tasks[f"{name}_d{difficulty}"] = procedural[name]
        all_records: list[EvaluationRecord] = []
        task_timings: dict[str, float] = {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with (args.output_dir / f"completions_rank{rank}.jsonl").open("w") as output:
            for name, task in tasks.items():
                task_started = time.perf_counter()
                size = len(task) if args.max_examples is None else min(len(task), args.max_examples)
                for index in range(rank, size, world_size):
                    with torch.no_grad():
                        record = evaluate_prompt(
                            task, task[index], engine, tokenizer, task_name=name, index=index,
                            context_length=model.config.sequence_len, max_new_tokens=args.max_new_tokens,
                            num_samples=args.num_samples, device_batch_size=args.device_batch_size,
                            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
                        )
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    all_records.append(record)
                    print0(f"{name}: {index + 1}/{size} greedy={record['greedy']['correct']} sampled={sum(row['correct'] for row in record['sampled'])}/{args.num_samples}")
                if ddp:
                    dist.barrier()
                task_timings[name] = time.perf_counter() - task_started
        if ddp:
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, all_records)
            all_records = [record for rank_records in gathered for record in rank_records]
        if rank == 0:
            elapsed = time.perf_counter() - started
            per_task = {
                name: {**summarize_records([record for record in all_records if record["task"] == name], args.num_samples), "elapsed_seconds": task_timings[name]}
                for name in tasks
            }
            total_tokens = sum(record["greedy"]["completion_tokens"] + sum(row["completion_tokens"] for row in record["sampled"]) for record in all_records)
            result = {
                "config": config, "model_config": meta["model_config"], "tasks": per_task,
                "elapsed_seconds": elapsed, "response_tokens": total_tokens,
                "tokens_per_second": total_tokens / elapsed,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                "metric_notes": "Binary task correctness. pass@k is the fraction of prompts with at least one success among k independent draws. CI95 is Wilson over prompts. Token counts include tool output and genuine stop tokens. Exact question filtering does not establish absence of semantic contamination.",
            }
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print0(json.dumps(result, indent=2))
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
