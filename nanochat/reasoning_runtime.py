"""Generation and policy-gradient helpers for the optional reasoning stages.

The adapter retains real stop tokens and masks without changing Engine. Training
uses one token denominator per prompt group, regardless of microbatch boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
import torch.distributed as dist


class GenerationTokenizer(Protocol):
    def encode_special(self, text: str) -> int: ...
    def get_bos_token_id(self) -> int: ...
    def decode(self, tokens: list[int]) -> str: ...
    def render_conversation(self, conversation: dict[str, Any], max_tokens: int | None = None) -> tuple[list[int], list[int]]: ...


class GenerationEngine(Protocol):
    def generate(self, tokens: list[int], **kwargs: Any) -> Iterator[tuple[list[int], list[int]]]: ...


@dataclass
class Completion:
    tokens: list[int]
    masks: list[int]
    prompt_length: int
    stop_reason: Literal["assistant_end", "bos", "length"]
    effective_budget: int
    text: str

    @property
    def completion_tokens(self) -> int:
        """Generated length, including a genuine terminal token and tool output."""
        return len(self.tokens) - self.prompt_length

    @property
    def terminated(self) -> bool:
        return self.stop_reason == "assistant_end"


def stable_seed(*parts: object) -> int:
    """A cross-process seed unaffected by Python's randomized string hash."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def render_completion_prompt(tokenizer: GenerationTokenizer, conversation: dict[str, Any]) -> list[int]:
    """Remove the gold response and render without the legacy 2048-token crop."""
    messages = conversation["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Reasoning tasks must end with a gold assistant response")
    prompt_conversation = {**conversation, "messages": messages[:-1]}
    ids, _ = tokenizer.render_conversation(prompt_conversation, max_tokens=None)
    return ids + [tokenizer.encode_special("<|assistant_start|>")]


def validate_output_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) or tag in (".", ".."):
        raise ValueError("Output tag must be a simple directory name")
    return tag


def prepare_output_directory(path: Path) -> None:
    """Reserve a fresh artifact directory, with collective error propagation."""
    repository = Path(__file__).resolve().parents[1]
    if path.resolve().is_relative_to(repository):
        raise ValueError("Datasets, checkpoints, and raw results must be stored outside the repository")
    distributed = dist.is_available() and dist.is_initialized()
    error: list[Any] = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = str(exc)
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise FileExistsError(f"Cannot reserve fresh output directory {path}: {error[0]}")


def sample_completions(
    engine: GenerationEngine,
    tokenizer: GenerationTokenizer,
    prompt: list[int],
    *,
    context_length: int,
    num_samples: int,
    device_batch_size: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 42,
) -> list[Completion]:
    """Consume Engine.generate, dropping emissions after each row's first stop.

    BOS is a stop, but is not normal assistant termination. Its real sampled
    target is retained; padding and forced calculator tokens receive zero masks.
    No artificial assistant-end target is added to a truncated completion.
    """
    if not prompt or len(prompt) >= context_length:
        raise ValueError("Prompt must be nonempty and leave room in the model context")
    if min(num_samples, device_batch_size, max_new_tokens) <= 0:
        raise ValueError("Sample count, device batch size, and completion budget must be positive")
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Temperature and top-k must be nonnegative")
    budget = min(max_new_tokens, context_length - len(prompt))
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    completions: list[Completion] = []
    for offset in range(0, num_samples, device_batch_size):
        count = min(device_batch_size, num_samples - offset)
        sequences = [prompt.copy() for _ in range(count)]
        masks = [[0] * len(prompt) for _ in range(count)]
        stops: list[Literal["assistant_end", "bos", "length"]] = ["length"] * count
        finished = [False] * count
        stream = engine.generate(
            prompt, num_samples=count, max_tokens=budget, temperature=temperature,
            top_k=top_k or None, seed=stable_seed(seed, offset),
        )
        try:
            for column_index, (column, column_masks) in enumerate(stream):
                if len(column) != count or len(column_masks) != count:
                    raise ValueError("Engine emitted an unexpected sample count")
                if column_index >= budget:
                    break
                for row, (token, mask) in enumerate(zip(column, column_masks)):
                    if finished[row]:
                        continue
                    sequences[row].append(token)
                    masks[row].append(int(mask))
                    if token in (assistant_end, bos):
                        stops[row] = "assistant_end" if token == assistant_end else "bos"
                        finished[row] = True
                if all(finished):
                    break
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        for sequence, mask, stop in zip(sequences, masks, stops):
            text_tokens = sequence[len(prompt):]
            if stop != "length":
                text_tokens = text_tokens[:-1]
            completions.append(Completion(sequence, mask, len(prompt), stop, budget, tokenizer.decode(text_tokens)))
    return completions


def training_batch(
    completions: Sequence[Completion], pad_token: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not completions:
        raise ValueError("At least one completion is required")
    max_length = max(len(item.tokens) for item in completions)
    ids = torch.tensor(
        [item.tokens + [pad_token] * (max_length - len(item.tokens)) for item in completions],
        dtype=torch.long, device=device,
    )
    masks = torch.tensor(
        [item.masks + [0] * (max_length - len(item.masks)) for item in completions],
        dtype=torch.bool, device=device,
    )
    targets = ids[:, 1:].clone()
    targets[~masks[:, 1:]] = -1
    return ids[:, :-1], targets


def backward_group(
    model: Any, inputs: torch.Tensor, targets: torch.Tensor, rewards: torch.Tensor,
    *, device_batch_size: int, examples_per_rank: int,
) -> tuple[bool, float]:
    """Mean-only advantages; exact token normalization for uneven microbatches."""
    if device_batch_size <= 0 or examples_per_rank <= 0:
        raise ValueError("Batch sizes must be positive")
    if not len(rewards) or len(rewards) != len(inputs) or not torch.isfinite(rewards).all():
        raise ValueError("One finite reward is required per sample")
    if inputs.shape != targets.shape:
        raise ValueError("Inputs and targets must have matching shapes")
    advantages = rewards - rewards.mean()
    valid_tokens = (targets >= 0).sum()
    if torch.equal(rewards, rewards[:1].expand_as(rewards)) or not valid_tokens.item():
        return False, 0.0
    denominator = valid_tokens * examples_per_rank
    total_loss = 0.0
    for offset in range(0, len(inputs), device_batch_size):
        stop = offset + device_batch_size
        batch_inputs, batch_targets = inputs[offset:stop], targets[offset:stop]
        nll = model(batch_inputs, batch_targets, loss_reduction="none").view_as(batch_inputs)
        loss = (nll * advantages[offset:stop, None]).sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite policy-gradient loss")
        loss.backward()
        total_loss += float(loss.detach())
    return True, total_loss


def optimizer_step_if_active(
    model: Any, optimizer: Any, active_groups: int, device: torch.device,
) -> tuple[bool, float]:
    """All ranks either enter the optimizer collectives or all skip the step."""
    active = torch.tensor(active_groups, dtype=torch.long, device=device)
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
    if active.item() == 0:
        model.zero_grad(set_to_none=True)
        return False, 0.0
    squared_norm = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        squared_norm += parameter.grad.float().square().sum()
    finite = torch.isfinite(squared_norm).to(torch.int32)
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError("Nonfinite gradient; optimizer update aborted")
    optimizer.step()
    model.zero_grad(set_to_none=True)
    return True, float(squared_norm.sqrt())


@dataclass
class Curriculum:
    """A prompt contributes its fraction of binary-correct sampled completions."""
    difficulties: dict[str, int] = field(default_factory=lambda: {name: 1 for name in ("chain", "sorting", "countdown")})
    windows: dict[str, list[float]] = field(default_factory=lambda: {name: [] for name in ("chain", "sorting", "countdown")})
    window_size: int = 100

    def observe(self, task: str, correctness: float) -> None:
        if task not in self.difficulties:
            return
        if not 0 <= correctness <= 1:
            raise ValueError("Prompt correctness must be between zero and one")
        window = self.windows[task]
        window.append(correctness)
        if len(window) >= self.window_size:
            rate = sum(window) / len(window)
            change = 1 if rate > 0.7 else -1 if rate < 0.2 else 0
            self.difficulties[task] = max(1, min(5, self.difficulties[task] + change))
            window.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"difficulties": self.difficulties.copy(), "windows": {key: list(value) for key, value in self.windows.items()}, "window_size": self.window_size}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Curriculum:
        instance = cls()
        expected = set(instance.difficulties)
        if set(state["difficulties"]) != expected or set(state["windows"]) != expected:
            raise ValueError("Curriculum task names do not match")
        instance.window_size = int(state.get("window_size", 100))
        if instance.window_size <= 0:
            raise ValueError("Curriculum window must be positive")
        for name in expected:
            difficulty = int(state["difficulties"][name])
            window = [float(item) for item in state["windows"][name]]
            if not 1 <= difficulty <= 5 or len(window) >= instance.window_size or any(not 0 <= item <= 1 for item in window):
                raise ValueError("Invalid curriculum checkpoint")
            instance.difficulties[name], instance.windows[name] = difficulty, window
        return instance


def synchronize_curriculum(curriculum: Curriculum, observations: list[tuple[int, str, float]]) -> None:
    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, observations)
        observations = [item for rank_items in gathered for item in rank_items]
    for _, task, correctness in sorted(observations):
        curriculum.observe(task, correctness)
