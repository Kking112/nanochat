"""CPU fixtures for termination, policy gradients, curriculum and evaluation."""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nanochat.reasoning_runtime import (
    Completion,
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
from scripts.reason_eval import evaluate_prompt, summarize_records, wilson_interval


class TinyTokenizer:
    def __init__(self) -> None:
        self.limits: list[int | None] = []

    def encode_special(self, text: str) -> int:
        return {"<|assistant_end|>": 9, "<|assistant_start|>": 8}[text]

    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, tokens: list[int]) -> str:
        return "".join({10: "#### 4", 11: "x", 12: "tool"}.get(token, str(token)) for token in tokens)

    def render_conversation(self, conversation: dict[str, Any], max_tokens: int | None = 2048) -> tuple[list[int], list[int]]:
        self.limits.append(max_tokens)
        ids = [1] * len(str(conversation["messages"][0]["content"]))
        return ids[:max_tokens], [0] * len(ids[:max_tokens])


class ColumnEngine:
    def __init__(self, columns: list[tuple[list[int], list[int]]] | None = None) -> None:
        self.columns = columns
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def generate(self, tokens: list[int], **kwargs: Any) -> Iterator[tuple[list[int], list[int]]]:
        self.calls.append({"tokens": tokens, **kwargs})
        count = kwargs["num_samples"]
        columns = self.columns or [([10] * count, [1] * count), ([9] * count, [1] * count)]
        try:
            yield from columns[:kwargs["max_tokens"]]
        finally:
            self.closed += 1


def complete(engine: ColumnEngine, **kwargs: Any) -> list[Completion]:
    settings = {"context_length": 20, "num_samples": 1, "device_batch_size": 2, "max_new_tokens": 8}
    settings.update(kwargs)
    return sample_completions(engine, TinyTokenizer(), [1, 8], **settings)


def test_generation_stops_per_row_and_retains_real_targets() -> None:
    engine = ColumnEngine([
        ([10, 11, 11], [1, 1, 1]),
        ([9, 12, 11], [1, 0, 1]),
        ([11, 0, 11], [1, 1, 1]),
        ([12, 11, 11], [0, 1, 1]),
    ])
    rows = complete(engine, num_samples=3, device_batch_size=3, max_new_tokens=4)
    assert rows[0].tokens == [1, 8, 10, 9]
    assert rows[0].text == "#### 4"
    assert rows[0].terminated and rows[0].completion_tokens == 2
    assert rows[1].tokens == [1, 8, 11, 12, 0]
    assert rows[1].masks == [0, 0, 1, 0, 1]
    assert rows[1].stop_reason == "bos" and not rows[1].terminated
    assert rows[2].stop_reason == "length" and rows[2].completion_tokens == 4
    assert engine.closed == 1
    inputs, targets = training_batch(rows, pad_token=9, device=torch.device("cpu"))
    assert inputs.shape == targets.shape == (3, 5)
    assert targets[0].tolist() == [-1, 10, 9, -1, -1]
    assert targets[1].tolist() == [-1, 11, -1, 0, -1]
    # Truncation never fabricates a stop token as a supervised target.
    assert targets[2].tolist() == [-1, 11, 11, 11, 11]


@pytest.mark.parametrize("samples,batch,expected", [(5, 2, [2, 2, 1]), (1, 8, [1]), (8, 3, [3, 3, 2])])
def test_sampling_partial_microbatches(samples: int, batch: int, expected: list[int]) -> None:
    engine = ColumnEngine()
    rows = complete(engine, num_samples=samples, device_batch_size=batch)
    assert len(rows) == samples
    assert [call["num_samples"] for call in engine.calls] == expected
    assert len({call["seed"] for call in engine.calls}) == len(expected)
    assert all(row.terminated for row in rows)


def test_context_budget_and_no_silent_prompt_cropping() -> None:
    tokenizer = TinyTokenizer()
    conversation = {"messages": [{"role": "user", "content": "x" * 3000}, {"role": "assistant", "content": "gold"}]}
    before = copy.deepcopy(conversation)
    prompt = render_completion_prompt(tokenizer, conversation)
    assert len(prompt) == 3001 and tokenizer.limits == [None]
    assert conversation == before
    with pytest.raises(ValueError, match="leave room"):
        sample_completions(ColumnEngine(), tokenizer, prompt, context_length=2048, num_samples=1, device_batch_size=1, max_new_tokens=20)
    engine = ColumnEngine()
    row = complete(engine, context_length=3, max_new_tokens=20)[0]
    assert engine.calls[0]["max_tokens"] == row.effective_budget == 1
    assert row.tokens == [1, 8, 10] and row.stop_reason == "length"


@pytest.mark.parametrize("kwargs", [{"num_samples": 0}, {"device_batch_size": 0}, {"max_new_tokens": 0}, {"temperature": -1}, {"top_k": -1}])
def test_invalid_sampling_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        complete(ColumnEngine(), **kwargs)


class TinyLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.unused = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[int] = []

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, loss_reduction: str) -> torch.Tensor:
        assert loss_reduction == "none"
        self.calls.append(len(inputs))
        return self.weight * inputs.float() * (targets >= 0)


def gradient_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11]])
    targets = inputs.clone()
    targets[0, 1:] = -1
    targets[3, :2] = -1
    rewards = torch.tensor([0.0, 0.25, 1.0, 0.5, 1.0])
    return inputs, targets, rewards


def test_microbatch_gradients_use_one_group_token_denominator() -> None:
    inputs, targets, rewards = gradient_batch()
    whole, chunked = TinyLossModel(), TinyLossModel()
    whole_active, whole_loss = backward_group(whole, inputs, targets, rewards, device_batch_size=8, examples_per_rank=2)
    chunk_active, chunk_loss = backward_group(chunked, inputs, targets, rewards, device_batch_size=2, examples_per_rank=2)
    assert whole_active and chunk_active
    assert chunked.calls == [2, 2, 1]
    assert whole_loss == pytest.approx(chunk_loss)
    assert torch.allclose(whole.weight.grad, chunked.weight.grad)
    expected = (inputs * (targets >= 0) * (rewards - rewards.mean())[:, None]).sum() / ((targets >= 0).sum() * 2)
    assert whole.weight.grad.item() == pytest.approx(expected.item())


def test_constant_reward_skips_backward_and_optimizer_momentum() -> None:
    model = TinyLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    inputs, targets, rewards = gradient_batch()
    active, _ = backward_group(model, inputs, targets, rewards, device_batch_size=2, examples_per_rank=1)
    updated, gradient_norm = optimizer_step_if_active(model, optimizer, int(active), torch.device("cpu"))
    assert updated and gradient_norm > 0
    state = copy.deepcopy(optimizer.state_dict())
    parameter_before = model.weight.detach().clone()
    model.calls.clear()
    active, loss = backward_group(model, inputs, targets, torch.ones_like(rewards), device_batch_size=2, examples_per_rank=1)
    assert not active and loss == 0 and model.calls == []
    updated, gradient_norm = optimizer_step_if_active(model, optimizer, 0, torch.device("cpu"))
    assert not updated and gradient_norm == 0
    assert torch.equal(model.weight, parameter_before)
    assert optimizer.state_dict()["state"][0]["step"] == state["state"][0]["step"]


def test_inactive_rank_supplies_zero_gradients_when_other_rank_active(monkeypatch: pytest.MonkeyPatch) -> None:
    from nanochat import reasoning_runtime as runtime

    model = TinyLossModel()
    observed: list[torch.Tensor] = []

    class RecordingOptimizer:
        def step(self) -> None:
            observed.extend(parameter.grad.clone() for parameter in model.parameters())

    def all_reduce(tensor: torch.Tensor, op: Any) -> None:
        if tensor.dtype == torch.long:
            tensor.add_(1)  # Another rank had an active group.

    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "all_reduce", all_reduce)
    updated, norm = optimizer_step_if_active(model, RecordingOptimizer(), 0, torch.device("cpu"))
    assert updated and norm == 0
    assert len(observed) == 2 and all(torch.count_nonzero(item) == 0 for item in observed)


def test_nonfinite_rewards_or_gradients_abort_before_update() -> None:
    model = TinyLossModel()
    inputs, targets, rewards = gradient_batch()
    rewards[0] = float("nan")
    with pytest.raises(ValueError, match="finite reward"):
        backward_group(model, inputs, targets, rewards, device_batch_size=2, examples_per_rank=1)
    model.weight.grad = torch.tensor(float("inf"))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    with pytest.raises(FloatingPointError, match="gradient"):
        optimizer_step_if_active(model, optimizer, 1, torch.device("cpu"))
    assert model.weight.item() == 0.25


def test_curriculum_thresholds_bounds_and_prompt_windows() -> None:
    curriculum = Curriculum()
    for _ in range(99):
        curriculum.observe("chain", 1.0)
    assert curriculum.difficulties["chain"] == 1
    curriculum.observe("chain", 1.0)
    assert curriculum.difficulties["chain"] == 2 and curriculum.windows["chain"] == []
    for _ in range(100):
        curriculum.observe("chain", 0.0)
        curriculum.observe("sorting", 0.0)
    assert curriculum.difficulties["chain"] == curriculum.difficulties["sorting"] == 1
    curriculum.difficulties["countdown"] = 5
    for _ in range(100):
        curriculum.observe("countdown", 1.0)
    assert curriculum.difficulties["countdown"] == 5
    curriculum.observe("gsm8k", 1.0)
    assert "gsm8k" not in curriculum.windows
    synchronize_curriculum(curriculum, [(2, "chain", 0.5), (1, "chain", 1.0)])
    assert curriculum.windows["chain"] == [1.0, 0.5]


def test_checkpoint_roundtrip_includes_curriculum_and_update_counts(tmp_path: Path) -> None:
    from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint

    curriculum = Curriculum()
    curriculum.observe("sorting", 0.75)
    metadata = {"reasoning_curriculum": curriculum.state_dict(), "attempted_rollout_steps": 5, "optimizer_updates": 3}
    save_checkpoint(str(tmp_path), 5, {"weight": torch.tensor([1.0, 2.0])}, {"state": {}}, metadata)
    model_data, optimizer_data, loaded = load_checkpoint(str(tmp_path), 5, torch.device("cpu"), load_optimizer=True)
    assert torch.equal(model_data["weight"], torch.tensor([1.0, 2.0]))
    assert optimizer_data == {"state": {}} and loaded == metadata
    restored = Curriculum.from_state_dict(loaded["reasoning_curriculum"])
    assert restored.state_dict() == curriculum.state_dict()
    restored.observe("sorting", 0.0)
    assert curriculum.windows["sorting"] == [0.75]


def test_evaluation_reports_binary_metrics_and_independent_seeded_draws() -> None:
    class Task:
        @staticmethod
        def evaluate(conversation: dict[str, Any], response: str) -> int:
            return int(response == "#### 4")

    engine = ColumnEngine()
    tokenizer = TinyTokenizer()
    record = evaluate_prompt(
        Task(), {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "gold"}]}, engine, tokenizer,
        task_name="gsm8k", index=2, context_length=20, max_new_tokens=8,
        num_samples=8, device_batch_size=3, temperature=1.0, top_k=50, seed=42,
    )
    summary = summarize_records([record], 8)
    assert summary["greedy_pass@1"] == summary["sampled_pass@8"] == 1
    assert summary["sampled_truncation_rate"] == 0
    assert summary["sampled_response_tokens_total"] == 16
    assert [call["temperature"] for call in engine.calls] == [0.0, 1.0, 1.0, 1.0]
    assert len({call["seed"] for call in engine.calls}) == 4
    assert tokenizer.limits == [None]
    assert 0 < summary["greedy_pass@1_ci95"][0] < 1
    assert wilson_interval(0, 10)[0] == 0
    assert not any("reward" in key for key in summary)


def test_output_reservation_and_tag_validation(tmp_path: Path) -> None:
    output = tmp_path / "reason-run"
    prepare_output_directory(output)
    with pytest.raises(FileExistsError, match="fresh output"):
        prepare_output_directory(output)
    for tag in ("../parent", "/tmp/path", "x/y", "..", ""):
        with pytest.raises(ValueError, match="simple directory"):
            validate_output_tag(tag)
    assert validate_output_tag("d24-shaped-42") == "d24-shaped-42"
    assert stable_seed(42, "chain", 1) == stable_seed(42, "chain", 1)
    assert stable_seed(42, "chain", 1) != stable_seed(43, "chain", 1)


def test_scripts_are_import_safe_and_reward_cli_covers_configuration() -> None:
    from dataclasses import fields

    from nanochat.rewards import RewardConfig

    rl = importlib.import_module("scripts.reason_rl")
    evaluation = importlib.import_module("scripts.reason_eval")
    args = rl.build_parser().parse_args(["--output-tag", "smoke"])
    rl.validate_args(args)
    assert args.source == "reason_sft" and args.max_new_tokens == 768
    for item in fields(RewardConfig):
        assert getattr(args, item.name) == getattr(RewardConfig(), item.name)
    assert evaluation.build_parser().parse_args(["--output-dir", "/tmp/eval"]).max_new_tokens == 1024
    sizes = dict.fromkeys(("gsm8k", "chain", "sorting", "countdown"), 100)
    assert rl.prompt_route(42, 3, sizes) == rl.prompt_route(42, 3, sizes)


def test_rl_entrypoint_writes_component_logs_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny CPU run checks wiring, gradients, isolated outputs, and JSON schema."""
    from nanochat import checkpoint_manager, common, engine
    from scripts import reason_rl
    from tasks import gsm8k_reasoning, procedural

    class FixtureTask:
        difficulty = 1

        def __len__(self) -> int:
            return 10

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "#### 4"}], "reasoning": {"gold_answer": "4", "intermediates": [], "process_available": False}}

        def evaluate(self, conversation: dict[str, Any], response: str) -> int:
            return int(response == "#### 4")

        def parse_answer(self, answer: str) -> str | None:
            return answer if answer.isdigit() else None

        def answer_partial(self, conversation: dict[str, Any], response: str) -> float:
            return float(self.evaluate(conversation, response))

    class MixedEngine(ColumnEngine):
        def generate(self, tokens: list[int], **kwargs: Any) -> Iterator[tuple[list[int], list[int]]]:
            count = kwargs["num_samples"]
            yield [10 if i % 2 == 0 else 11 for i in range(count)], [1] * count
            yield [9] * count, [1] * count

    model = TinyLossModel()
    model.config = SimpleNamespace(sequence_len=32, n_layer=1)
    model.setup_optimizer = lambda **kwargs: torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(common, "compute_init", lambda device_type: (False, 0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr(common, "compute_cleanup", lambda: None)
    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(checkpoint_manager, "load_model", lambda *args, **kwargs: (model, TinyTokenizer(), {"model_config": vars(model.config)}))
    monkeypatch.setattr(engine, "Engine", lambda *args: MixedEngine())
    monkeypatch.setattr(gsm8k_reasoning, "GSM8KReasoning", lambda **kwargs: FixtureTask())
    monkeypatch.setattr(procedural, "make_procedural_tasks", lambda **kwargs: {name: FixtureTask() for name in ("chain", "sorting", "countdown")})
    reason_rl.main([
        "--source", "sft", "--model-tag", "d24", "--model-step", "486", "--output-tag", "tiny-rl",
        "--device-type", "cpu", "--num-steps", "2", "--examples-per-step", "4", "--num-samples", "3",
        "--device-batch-size", "2", "--max-new-tokens", "8",
    ])
    output = tmp_path / "reasoning_runs" / "tiny-rl"
    metrics = [json.loads(line) for line in (output / "metrics_rank0.jsonl").read_text().splitlines()]
    records = [json.loads(line) for line in (output / "rollouts_rank0.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    assert len(metrics) == 2 and len(records) == 24
    assert summary["optimizer_updates"] == summary["attempted_rollout_steps"] == 2
    assert metrics[0]["zero_variance_fraction"] == 0
    assert all(record["reward_diagnostics"]["band"] in {"correct", "incorrect"} for record in records)
    checkpoint = tmp_path / checkpoint_manager.CHECKPOINT_SOURCES["reason_rl"] / "tiny-rl"
    _, optimizer_state, meta = checkpoint_manager.load_checkpoint(str(checkpoint), 2, torch.device("cpu"), load_optimizer=True)
    assert optimizer_state is not None and meta["parent_checkpoint"]["step"] == 486
    assert meta["reasoning_curriculum"] == summary["curriculum"]
