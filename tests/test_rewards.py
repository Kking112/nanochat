"""Band dominance, process exploits and penalties tested without Torch or datasets."""

import random
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from nanochat.reasoning import extract_answer_marker, normalize_numeric
from nanochat.rewards import RewardConfig, combine_reward, score_response


class NumericTask:
    def parse_answer(self, answer: str) -> str | None:
        return normalize_numeric(answer)

    def evaluate(self, conversation: Mapping[str, Any], response: str) -> int:
        return int(self.parse_answer(extract_answer_marker(response) or "") == conversation["reasoning"]["gold_answer"])

    def answer_partial(self, conversation: Mapping[str, Any], response: str) -> float:
        return float(self.evaluate(conversation, response))


CONVERSATION = {"reasoning": {"gold_answer": "10", "intermediates": ["5", "10"]}}


def calculator(expression: str) -> object:
    return {"2+3": 5, "5*2": 10, "1+1": 2}.get(expression.replace(" ", ""))


def score(trace: str = "2+3=5; 5*2=10", answer: str = "10", **kwargs: Any):
    return score_response(NumericTask(), CONVERSATION, f"<think>{trace}</think>\n#### {answer}", calculator, **kwargs)


def test_exact_default_reward_and_full_verified_gold() -> None:
    result = score()
    assert result.total == 1
    assert result.correctness and result.band == "correct"
    assert result.r_format == result.r_process == result.r_answer_partial == 1
    assert result.s == result.q == 1
    wrong = score(answer="11")
    assert wrong.r_answer_partial == 0
    assert wrong.r_process == 0.9
    assert wrong.total == pytest.approx(0.5 * (0.15 + 0.4 * 0.9))


def test_randomized_dominance_including_malformed_correct_responses() -> None:
    rng = random.Random(31)
    for _ in range(2000):
        components = [rng.uniform(-1, 2) for _ in range(3)]
        penalty = rng.uniform(0, 3)
        progress = rng.random()
        wrong = combine_reward(False, *components, penalty, progress=progress)
        correct = combine_reward(True, *components, penalty, progress=progress)
        assert 0 <= wrong.total <= 0.5
        assert 0.7 <= correct.total <= 1
        assert wrong.total < correct.total
    malformed = score_response(NumericTask(), CONVERSATION, "#### 10\nextra", calculator, terminated=False, completion_tokens=768)
    assert malformed.r_format == 0
    assert malformed.total == 0.7


def test_binary_correctness_exclusively_selects_band_even_with_partial_credit() -> None:
    class PartialTask(NumericTask):
        def answer_partial(self, conversation: Mapping[str, Any], response: str) -> float:
            return 1.0

    result = score_response(PartialTask(), CONVERSATION, "<think>2+3=5;5*2=10</think>\n#### 11", calculator)
    assert result.r_answer_partial == 1
    assert not result.correctness
    assert result.total <= 0.5


@pytest.mark.parametrize("answer,expected", [("10", 1.0), ("11", 0.0)])
def test_binary_mode_ignores_shaping(answer: str, expected: float) -> None:
    result = score(trace="false", answer=answer, mode="binary", terminated=False, completion_tokens=768)
    assert result.total == expected
    assert result.mode == "binary"


def test_unrelated_equation_spam_never_adds_process_reward() -> None:
    empty = score(trace="No work yet", answer="2")
    spam = score(trace="1+1=2; " * 100, answer="2")
    assert spam.r_process == empty.r_process == 0
    assert spam.total <= empty.total
    assert spam.repetition_penalty > 0


def test_repeated_reference_values_count_only_once() -> None:
    single = score(trace="2+3=5", answer="0")
    duplicate = score(trace="2+3=5; " * 3, answer="0")
    assert single.intermediate_coverage == duplicate.intermediate_coverage == 0.5
    assert single.r_process == duplicate.r_process == 0.45
    duplicate_gold = {"reasoning": {"gold_answer": "10", "intermediates": ["5", "5", "10"]}}
    result = score_response(NumericTask(), duplicate_gold, "<think>2+3=5</think>\n#### 0", calculator)
    assert result.intermediate_coverage == 0.5


def test_false_statements_reduce_quality_and_cannot_claim_reference_results() -> None:
    clean = score()
    false = score(trace="2+3=5; 5*2=10; 1+1=10")
    assert false.intermediate_coverage == clean.intermediate_coverage == 1
    assert false.invalid_arithmetic_penalty == pytest.approx(0.2 / 3)
    assert false.q < clean.q
    assert 0.7 <= false.total < clean.total
    invented = score(trace="1+1=10")
    assert invented.intermediate_coverage == 0


def test_missing_reference_data_is_explicit_and_neutral() -> None:
    conversation = {"reasoning": {"gold_answer": "10", "intermediates": [], "process_available": False}}
    result = score_response(NumericTask(), conversation, "<think>2+3=5</think>\n#### 10", calculator)
    assert not result.process_available
    assert result.r_process == 0.5
    assert result.intermediate_coverage == 0
    assert not result.consistency_available


def test_nonnumeric_final_answer_renormalizes_applicable_process_terms() -> None:
    class StructuredTask(NumericTask):
        def parse_answer(self, answer: str) -> tuple[int, ...]:
            return tuple(int(n) for n in answer.split(","))

    result = score_response(StructuredTask(), CONVERSATION, "<think>2+3=5;5*2=10</think>\n#### 5,10", calculator)
    assert result.r_process == 1
    assert not result.consistency_available


@pytest.mark.parametrize("budget", [64, 128, 768])
def test_truncation_penalty_is_monotonic_and_only_for_missing_normal_stop(budget: int) -> None:
    penalties = [score(terminated=False, completion_tokens=count, generation_budget=budget).truncation_penalty for count in range(budget + 1)]
    assert penalties == sorted(penalties)
    assert penalties[0] == 0
    assert penalties[-1] == pytest.approx(0.1)
    assert score(terminated=True, completion_tokens=budget, generation_budget=budget).truncation_penalty == 0


def test_truncation_ramp_starts_in_final_128_tokens() -> None:
    assert score(terminated=False, completion_tokens=640).truncation_penalty == 0
    assert score(terminated=False, completion_tokens=704).truncation_penalty == pytest.approx(0.05)


def test_repetition_penalty_threshold_and_trailing_text() -> None:
    at_threshold = score(trace="one two three four one two three four")
    assert at_threshold.repetition_penalty == 0
    response = "<think>2+3=5;5*2=10</think>\n#### 10\n "
    task = NumericTask()
    clean = score_response(task, CONVERSATION, response, calculator)
    extra = score_response(task, CONVERSATION, response + "more", calculator)
    assert clean.trailing_text_penalty == 0
    assert extra.trailing_text_penalty == 0.1
    assert extra.total < clean.total


def test_annealing_transfers_weights_with_normalized_sums() -> None:
    config = RewardConfig()
    start, end = config.effective_weights(0), config.effective_weights(1)
    assert start.incorrect_process == 0.4 and start.correct_process == 0.5
    assert end.incorrect_process == pytest.approx(0.1)
    assert end.correct_process == pytest.approx(0.1)
    assert end.incorrect_answer == pytest.approx(0.75)
    assert end.correct_format == pytest.approx(0.9)
    for progress in (0, 0.25, 0.5, 1):
        weights = config.effective_weights(progress)
        assert weights.incorrect_format + weights.incorrect_answer + weights.incorrect_process == pytest.approx(1)
        assert weights.correct_format + weights.correct_process == pytest.approx(1)


@pytest.mark.parametrize("fields", [
    {"incorrect_max": 0.7}, {"correct_min": 0.4}, {"correct_span": 0.4},
    {"incorrect_answer_weight": -0.1}, {"correct_process_weight": 0.9},
    {"incorrect_process_weight": 0.5}, {"process_weight_final": 0.6},
    {"truncation_window": 0}, {"repetition_threshold": 1}, {"truncation_max": float("nan")},
])
def test_invalid_configuration_is_rejected(fields: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        replace(RewardConfig(), **fields)


def test_custom_band_boundaries_and_component_logging() -> None:
    config = replace(RewardConfig(), incorrect_max=0.4, correct_min=0.6, correct_span=0.2)
    assert score(config=config).total == pytest.approx(0.8)
    assert score(answer="0", config=config).total <= 0.4
    fields = score(progress=0.5).to_dict()
    assert fields["band"] == "correct"
    assert fields["correctness"] is True
    for name in ("s", "q", "total", "r_format", "r_answer_partial", "r_process", "invalid_arithmetic_penalty", "truncation_penalty", "repetition_penalty", "trailing_text_penalty", "incorrect_process_weight", "correct_process_weight"):
        assert name in fields


@pytest.mark.parametrize("progress", [-0.1, 1.1, float("nan")])
def test_invalid_annealing_progress_is_rejected(progress: float) -> None:
    with pytest.raises(ValueError):
        score(progress=progress)
