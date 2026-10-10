"""Offline acceptance of reasoning datasets, seeded tasks, and uncropped mixtures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from nanochat.reasoning import parse_reasoning
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer
from tasks.common import Task
from tasks.gsm8k_reasoning import GSM8KReasoning
from tasks.metamath import MetaMathQA
from tasks.openmathinstruct2 import OpenMathInstruct2
from tasks.procedural import ArithmeticChain, Countdown, Sorting, legal_countdown_value
from tasks.reasoning_data import (
    IndexedTask,
    LengthFiltered,
    build_sft_mixture,
    numeric_solution_answer,
    question_keys,
    render_untruncated,
    without_overlap,
)
from tasks.routed import RoutedTaskMixture


@pytest.fixture(scope="module")
def tokenizer() -> RustBPETokenizer:
    corpus = ["The answer is 42. <think> Sort 1, 2, 3. Numbers and calculations.\n"] * 12
    return RustBPETokenizer.train_from_iterator(iter(corpus), 256 + len(SPECIAL_TOKENS) + 24)


class RowsTask(Task):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__()
        self.rows = rows

    def num_examples(self) -> int:
        return len(self.rows)

    def get_example(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def conversation(question: str = "Question", answer: str = "Answer") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


@pytest.mark.parametrize("task_type", [ArithmeticChain, Sorting, Countdown])
@pytest.mark.parametrize("difficulty", [1, 3, 5])
def test_procedural_gold_determinism_and_split_spaces(task_type: type, difficulty: int) -> None:
    train = task_type(size=20, difficulty=difficulty, seed=8)
    same = task_type(size=20, difficulty=difficulty, seed=8)
    validation = task_type(split="validation", size=20, difficulty=difficulty, seed=8)
    test = task_type(split="test", size=20, difficulty=difficulty, seed=8)
    for index in range(20):
        example = train[index]
        assert example == same[index]
        assert example != validation[index] != test[index]
        output = example["messages"][-1]["content"]
        assert train.evaluate(example, output) == 1
        assert train.answer_partial(example, output) == 1
        assert parse_reasoning(output, train.parse_answer).format_score == 1
        for bad in ("", "####", "#### 1\n#### 2", "#### not an answer"):
            assert train.evaluate(example, bad) == 0
        assert example["reasoning"]["difficulty"] == difficulty
    assert not ({next(iter(question_keys(train[i]))) for i in range(20)} &
                {next(iter(question_keys(test[i]))) for i in range(20)})


def test_procedural_difficulty_and_bounds() -> None:
    task = ArithmeticChain(size=3)
    original = task[0]
    task.difficulty = 3
    assert task[0] != original
    with pytest.raises(ValueError):
        task.difficulty = 6
    with pytest.raises(ValueError):
        ArithmeticChain(split="random")
    with pytest.raises(IndexError):
        task[3]


def test_sorting_position_credit() -> None:
    task = Sorting()
    example = {"reasoning": {"gold_answer": "1, 2, 3"}}
    assert task.answer_partial(example, "#### 1, 3, 2") == pytest.approx(1 / 3)
    assert task.answer_partial(example, "#### 1, 2, 3, 4") == 0
    assert task.parse_answer("[1, 2, 3]") == (1, 2, 3)
    assert task.parse_answer("[1, 2, 3") is None


@pytest.mark.parametrize("expression", ["6", "1 + 2 + 3 + 0", "1 + 2 + 3 + 1", "6 + 0", "2 ** 3", "__import__('os')", "1 / (2 - 2)", "1.0 + 2 + 3"])
def test_countdown_rejects_illegal_expressions(expression: str) -> None:
    assert legal_countdown_value(expression, [1, 2, 3]) is None


def test_countdown_legal_closeness_only() -> None:
    task = Countdown()
    example = {"reasoning": {"numbers": [1, 2, 3], "target": 7}}
    assert task.answer_partial(example, "#### 1+2+3") == 0.5
    assert task.answer_partial(example, "#### 7") == 0
    assert task.evaluate(example, "#### 1+2*3") == 1
    assert legal_countdown_value("(2+2)*3", [2, 2, 3]) == 12
    assert legal_countdown_value("(2+2)*3", [2, 3]) is None


def test_gsm_conversion_preserves_tools_and_raw_intermediates(tokenizer: RustBPETokenizer) -> None:
    task = GSM8KReasoning(rows=[{"question": "What is half of 1 plus 1?", "answer": "Half is <<1/2=0.5>>0.5. Then <<0.5+1=1.5>>1.5.\n#### 1.5"}])
    example = task[0]
    parts = example["messages"][-1]["content"]
    assert [part["text"] for part in parts if part["type"] == "python"] == ["1/2", "0.5+1"]
    assert example["reasoning"]["intermediates"] == ["0.5", "1.5"]
    assert "reasoning" not in example["messages"][-1]
    assert task.evaluate(example, "#### 3/2") == 1
    assert task.answer_partial(example, "#### 1.4999") == 0
    ids, mask = render_untruncated(tokenizer, example)
    assert len(ids) == len(mask) and any(mask)
    output_start = tokenizer.encode_special("<|output_start|>")
    output_end = tokenizer.encode_special("<|output_end|>")
    in_output = False
    for token, supervised in zip(ids, mask):
        if token == output_start:
            in_output = True
        if in_output:
            assert supervised == 0
        if token == output_end:
            in_output = False


@pytest.mark.parametrize("response, expected", [("The answer is: 3", "3"),
    ("#### 3\nThe answer is: 3", "3"), (r"We get \boxed{\frac{1}{2}}.", "1/2"),
    ("#### 3\nThe answer is: 4", None), ("The answer is: x", None),
    (r"The answer is: \sqrt{5}", None), ("There are 3 of them", None),
    ("#### 3 or 4", None)])
def test_numeric_source_answer_signals(response: str, expected: str | None) -> None:
    assert numeric_solution_answer(response) == expected


def test_metamath_numeric_only_original_questions_and_single_marker() -> None:
    rows = [{"query": "Rephrase", "original_question": "Original", "response": "3+2=5.\n#### 5\nThe answer is: 5"},
            {"query": "Other", "response": "The answer is: (1, 2)"},
            {"query": "Bad", "response": "#### 5\nThe answer is: 6"}]
    task = MetaMathQA(rows=rows)
    assert len(task) == 1
    example = task[0]
    output = example["messages"][-1]["content"]
    assert output.count("####") == 1
    assert parse_reasoning(output).format_score == 1
    assert not example["reasoning"]["process_available"]
    assert "original" in question_keys(example)
    assert len(without_overlap(task, {"original"})) == 0


def test_openmath_numeric_expected_source_family_and_conflict() -> None:
    row = {"problem": "How many?", "generated_solution": r"Compute \boxed{2}.", "expected_answer": "2", "problem_source": "augmented_math"}
    rows = [row, {**row, "problem_source": "unrelated"}, {**row, "expected_answer": "3"},
            {**row, "expected_answer": "(1,2)"}]
    task = OpenMathInstruct2(rows=rows)
    assert len(task) == 1
    assert task.evaluate(task[0], "#### 2.0") == 1


def test_adapters_use_existing_parquet_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    def fake_loader(*args: Any, **kwargs: Any) -> list:
        calls.append((*args, kwargs))
        return []
    for module, cls in (("tasks.gsm8k_reasoning", GSM8KReasoning), ("tasks.metamath", MetaMathQA),
                        ("tasks.openmathinstruct2", OpenMathInstruct2)):
        monkeypatch.setattr(module + ".load_hub_dataset", fake_loader)
        assert len(cls()) == 0
    assert calls == [("openai/gsm8k", "main", {"split": "train"}),
                     ("meta-math/MetaMathQA", "default", {"split": "train"}),
                     ("nvidia/OpenMathInstruct-2", "default", {"split": "train_1M"})]


def test_routed_mixture_evaluation_and_no_source_mutation() -> None:
    tasks = {"chain": ArithmeticChain(size=3), "sorting": Sorting(size=4)}
    original = tasks["chain"][0]
    mix = RoutedTaskMixture(tasks, seed=7)
    assert len(mix) == 7
    assert mix.index_map == RoutedTaskMixture(tasks, seed=7).index_map
    assert Counter(mix[index]["task_route"] for index in range(7)) == {"chain": 3, "sorting": 4}
    for index in range(7):
        example = mix[index]
        assert example["task_name"] == example["task_route"]
        assert mix.evaluate(example, example["messages"][-1]["content"]) == 1
    assert tasks["chain"][0] == original and "task_route" not in original


def test_routed_reward_forwards_source_implementation_and_kwargs() -> None:
    class RewardOwner(ArithmeticChain):
        def reward(self, conversation: dict[str, Any], response: str, bonus: int = 0) -> float:
            return 123.0 + bonus
    task = IndexedTask(RewardOwner(size=1), [0])
    mix = RoutedTaskMixture({"owner": task})
    assert mix.reward(mix[0], "ignored", bonus=5) == 128


def test_new_tasks_return_shaped_breakdowns_without_loading_engine() -> None:
    task = Sorting(size=1)
    example = task[0]
    result = task.reward(example, example["messages"][-1]["content"], calculator=lambda value: None)
    assert result.total == pytest.approx(0.925)
    assert not result.process_available
    assert task.reward(example, example["messages"][-1]["content"], calculator=lambda value: None, mode="binary").total == 1


def test_length_exact_boundaries_cache_and_no_cropping(tokenizer: RustBPETokenizer, tmp_path: Path) -> None:
    row = conversation(answer="x" * 2400)
    task = RowsTask([row])
    ids, _ = render_untruncated(tokenizer, row)
    assert len(ids) > 2048
    start = ids.index(tokenizer.encode_special("<|assistant_start|>"))
    assistant_length = len(ids) - start
    exact = LengthFiltered(task, tokenizer, len(ids), assistant_length, tmp_path)
    assert len(exact) == 1 and not exact.cache_hit
    assert exact[0] == row
    assert LengthFiltered(task, tokenizer, len(ids), assistant_length, tmp_path).cache_hit
    assert len(LengthFiltered(task, tokenizer, len(ids) - 1, assistant_length, tmp_path)) == 0
    assert len(LengthFiltered(task, tokenizer, len(ids), assistant_length - 1, tmp_path)) == 0
    changed = LengthFiltered(task, tokenizer, len(ids), assistant_length, tmp_path, conversion_version="changed")
    assert not changed.cache_hit
    new_data = LengthFiltered(RowsTask([conversation(answer="z" * 2400)]), tokenizer, len(ids), assistant_length, tmp_path)
    assert not new_data.cache_hit


def test_length_counts_every_assistant_turn_and_tool_output(tokenizer: RustBPETokenizer, tmp_path: Path) -> None:
    row = conversation()
    row["messages"][-1]["content"] = [{"type": "text", "text": "Tool"}, {"type": "python", "text": "1+1"},
                                      {"type": "python_output", "text": "x" * 1500}, {"type": "text", "text": "2"}]
    ids, mask = render_untruncated(tokenizer, row)
    assert sum(mask) < 100 and len(ids) > 1500
    assert len(LengthFiltered(RowsTask([row]), tokenizer, 3000, 1024, tmp_path)) == 0
    multi = conversation(answer="x" * 100)
    multi["messages"] += conversation(answer="yes")["messages"]
    assert len(LengthFiltered(RowsTask([multi]), tokenizer, 3000, 50, tmp_path)) == 0


def test_length_cap_after_filter_and_corrupt_cache(tokenizer: RustBPETokenizer, tmp_path: Path) -> None:
    task = RowsTask([conversation(answer="x" * 2000), conversation(answer="yes"), conversation(answer="okay")])
    filtered = LengthFiltered(task, tokenizer, 100, 50, tmp_path, max_rows=1)
    assert filtered.indices == [1]
    cache = tmp_path / f"length-{filtered.fingerprint}.json"
    data = json.loads(cache.read_text())
    data["indices"] = [99]
    cache.write_text(json.dumps(data))
    refreshed = LengthFiltered(task, tokenizer, 100, 50, tmp_path, max_rows=1)
    assert refreshed.indices == [1] and not refreshed.cache_hit


def test_sft_caps_replay_validation_and_overlap(tokenizer: RustBPETokenizer, tmp_path: Path) -> None:
    gsm = GSM8KReasoning(rows=[{"question": f"GSM question {i}", "answer": f"1+{i}=<<1+{i}={i+1}>>{i+1}.\n#### {i+1}"} for i in range(7)])
    meta = MetaMathQA(rows=[{"query": f"Meta question {i}", "original_question": "Eval held out" if i == 0 else f"Original {i}",
                             "response": "The answer is: 1"} for i in range(8)])
    smol = RowsTask([conversation(f"Replay smol {i}", "Unchanged response") for i in range(10)])
    mmlu = RowsTask([conversation(f"Replay mmlu {i}", "A") for i in range(10)])
    train, val = build_sft_mixture(tokenizer, metamath_rows=3, omi2_rows=0, gsm8k_epochs=2, procedural_rows=0,
                                  replay_fraction=0.25, validation_size=1, cache_dir=tmp_path,
                                  evaluation_questions=["eval held out"], sources={"gsm8k": gsm, "metamath": meta, "smoltalk": smol, "mmlu": mmlu})
    counts = Counter(train[i]["task_route"] for i in range(len(train)))
    assert counts == {"gsm8k": 12, "metamath": 3, "smoltalk": 1, "mmlu": 4}
    train_questions = set().union(*(question_keys(train[i]) for i in range(len(train))))
    val_questions = set().union(*(question_keys(val[i]) for i in range(len(val))))
    assert not train_questions & val_questions
    assert "eval held out" not in train_questions
    assert len(val) == 2
    for index in range(len(train)):
        example = train[index]
        if example["task_route"] == "smoltalk":
            assert example["messages"][-1]["content"] == "Unchanged response"
        if example["task_route"] == "mmlu":
            assert example["messages"][-1]["content"] == "A"


def test_sft_zero_caps_skip_external_training_sources(tokenizer: RustBPETokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("disabled external source loaded")
    for module in ("tasks.gsm8k_reasoning", "tasks.metamath", "tasks.openmathinstruct2", "tasks.smoltalk", "tasks.mmlu"):
        monkeypatch.setattr(module + ".load_hub_dataset", forbidden)
    train, val = build_sft_mixture(tokenizer, metamath_rows=0, omi2_rows=0, gsm8k_epochs=0, procedural_rows=12,
                                  replay_fraction=0, validation_size=1, cache_dir=tmp_path, evaluation_questions=[])
    assert len(train) == 12 and len(val) == 3
    assert all(train[index]["reasoning"]["split"] == "train" for index in range(len(train)))
    assert all(val[index]["reasoning"]["split"] == "validation" for index in range(len(val)))
