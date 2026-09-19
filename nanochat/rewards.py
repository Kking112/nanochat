"""Separated correctness-band rewards; training diagnostics, never evaluation metrics."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Mapping, Protocol

from nanochat.reasoning import Calculator, normalize_numeric, parse_reasoning, repeated_ngram_fraction, verify_arithmetic


class RewardTask(Protocol):
    def evaluate(self, conversation: Mapping[str, Any], response: str) -> int: ...
    def parse_answer(self, answer: str) -> object | None: ...
    def answer_partial(self, conversation: Mapping[str, Any], response: str) -> float: ...


@dataclass(frozen=True)
class EffectiveWeights:
    incorrect_format: float
    incorrect_answer: float
    incorrect_process: float
    correct_format: float
    correct_process: float


@dataclass(frozen=True)
class RewardConfig:
    incorrect_max: float = 0.5
    correct_min: float = 0.7
    correct_span: float = 0.3
    incorrect_format_weight: float = 0.15
    incorrect_answer_weight: float = 0.45
    incorrect_process_weight: float = 0.40
    correct_format_weight: float = 0.50
    correct_process_weight: float = 0.50
    process_weight_final: float = 0.1
    invalid_arithmetic_max: float = 0.2
    truncation_max: float = 0.1
    repetition_max: float = 0.1
    repetition_threshold: float = 0.2
    trailing_text_penalty: float = 0.1
    truncation_window: int = 128

    def __post_init__(self) -> None:
        numbers = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in numbers.values()):
            raise ValueError("Reward configuration values must be finite and nonnegative")
        if not (self.incorrect_max < self.correct_min and self.correct_min + self.correct_span <= 1.0):
            raise ValueError("Correct and incorrect bands must be separated and lie within [0, 1]")
        if not math.isclose(self.incorrect_format_weight + self.incorrect_answer_weight + self.incorrect_process_weight, 1.0, abs_tol=1e-12):
            raise ValueError("Incorrect-band weights must sum to one")
        if not math.isclose(self.correct_format_weight + self.correct_process_weight, 1.0, abs_tol=1e-12):
            raise ValueError("Correct-band weights must sum to one")
        if self.process_weight_final > min(self.incorrect_process_weight, self.correct_process_weight):
            raise ValueError("Final process weight must not exceed either starting process weight")
        if self.repetition_threshold >= 1 or self.truncation_window < 1:
            raise ValueError("Repetition threshold must be below one and truncation window positive")

    def effective_weights(self, progress: float) -> EffectiveWeights:
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("Annealing progress must be within [0, 1]")
        incorrect_transfer = progress * (self.incorrect_process_weight - self.process_weight_final)
        correct_transfer = progress * (self.correct_process_weight - self.process_weight_final)
        return EffectiveWeights(
            self.incorrect_format_weight,
            self.incorrect_answer_weight + incorrect_transfer,
            self.incorrect_process_weight - incorrect_transfer,
            self.correct_format_weight + correct_transfer,
            self.correct_process_weight - correct_transfer,
        )


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class BandScores:
    s: float
    q: float
    total: float


def combine_reward(
    correct: bool,
    r_format: float,
    r_answer_partial: float,
    r_process: float,
    penalties: float,
    config: RewardConfig = RewardConfig(),
    progress: float = 0.0,
) -> BandScores:
    """Clip quality scores only; binary correctness exclusively selects the band."""
    if any(not math.isfinite(value) for value in (r_format, r_answer_partial, r_process, penalties)):
        raise ValueError("Reward inputs must be finite")
    weights = config.effective_weights(progress)
    s = clip01(weights.incorrect_format * r_format + weights.incorrect_answer * r_answer_partial + weights.incorrect_process * r_process - penalties)
    q = clip01(weights.correct_format * r_format + weights.correct_process * r_process - penalties)
    total = config.correct_min + config.correct_span * q if correct else config.incorrect_max * s
    return BandScores(s, q, total)


@dataclass(frozen=True)
class RewardBreakdown:
    correctness: bool
    band: Literal["correct", "incorrect"]
    mode: Literal["binary", "shaped"]
    s: float
    q: float
    r_format: float
    r_answer_partial: float
    r_process: float
    process_available: bool
    intermediate_coverage: float
    final_consistency: float
    consistency_available: bool
    checked_statements: int
    invalid_statements: int
    invalid_fraction: float
    repeated_fraction: float
    invalid_arithmetic_penalty: float
    truncation_penalty: float
    repetition_penalty: float
    trailing_text_penalty: float
    penalties: float
    incorrect_format_weight: float
    incorrect_answer_weight: float
    incorrect_process_weight: float
    correct_format_weight: float
    correct_process_weight: float
    total: float

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)


def score_response(
    task: RewardTask,
    conversation: Mapping[str, Any],
    response: str,
    calculator: Calculator,
    config: RewardConfig = RewardConfig(),
    progress: float = 0.0,
    mode: Literal["binary", "shaped"] = "shaped",
    terminated: bool = True,
    completion_tokens: int = 0,
    generation_budget: int = 768,
) -> RewardBreakdown:
    """Score a response using task truth and independently verified trace quality.

    Missing reference intermediates give neutral process quality (0.5), explicitly
    marked unavailable. Correct but unrelated equations never earn coverage.
    """
    if mode not in ("binary", "shaped"):
        raise ValueError("Reward mode must be binary or shaped")
    if generation_budget <= 0 or completion_tokens < 0:
        raise ValueError("Generation budget must be positive and completion length nonnegative")
    weights = config.effective_weights(progress)
    correct = task.evaluate(conversation, response) == 1
    parsed = parse_reasoning(response, task.parse_answer)
    partial = float(task.answer_partial(conversation, response))
    if not math.isfinite(partial) or not 0 <= partial <= 1:
        raise ValueError("Task partial-answer score must lie in [0, 1]")
    verification = verify_arithmetic(parsed.think or "", calculator)
    metadata = conversation.get("reasoning", {})
    gold_values = {number for value in metadata.get("intermediates", []) if (number := normalize_numeric(value)) is not None}
    process_available = bool(gold_values) and metadata.get("process_available", True)
    coverage = len(gold_values.intersection(verification.verified_values)) / len(gold_values) if process_available else 0.0
    final_number = normalize_numeric(parsed.answer)
    consistency_available = bool(process_available and final_number is not None)
    # Consistency is only applicable to relevant verified work: an unrelated
    # equation matching a wrong final answer must not farm process reward.
    last_value = verification.verified_values[-1] if verification.verified_values else None
    consistency = float(bool(consistency_available and last_value in gold_values and last_value == final_number))
    process = (0.9 * coverage + 0.1 * consistency) / (1.0 if consistency_available else 0.9) if process_available else 0.5
    repeated_fraction = repeated_ngram_fraction(response)
    invalid_penalty = config.invalid_arithmetic_max * verification.invalid_fraction
    ramp_window = min(config.truncation_window, generation_budget)
    truncation_penalty = 0.0 if terminated else config.truncation_max * clip01((completion_tokens - generation_budget + ramp_window) / ramp_window)
    repetition_penalty = config.repetition_max * max(0.0, (repeated_fraction - config.repetition_threshold) / (1.0 - config.repetition_threshold))
    trailing_penalty = config.trailing_text_penalty if parsed.trailing_text else 0.0
    penalties = invalid_penalty + truncation_penalty + repetition_penalty + trailing_penalty
    scores = combine_reward(correct, parsed.format_score, partial, process, penalties, config, progress)
    return RewardBreakdown(
        correctness=correct, band="correct" if correct else "incorrect", mode=mode,
        s=scores.s, q=scores.q, r_format=parsed.format_score, r_answer_partial=partial,
        r_process=process, process_available=bool(process_available),
        intermediate_coverage=coverage, final_consistency=consistency,
        consistency_available=consistency_available, checked_statements=len(verification.statements),
        invalid_statements=verification.invalid_count, invalid_fraction=verification.invalid_fraction,
        repeated_fraction=repeated_fraction, invalid_arithmetic_penalty=invalid_penalty,
        truncation_penalty=truncation_penalty, repetition_penalty=repetition_penalty,
        trailing_text_penalty=trailing_penalty, penalties=penalties,
        incorrect_format_weight=weights.incorrect_format, incorrect_answer_weight=weights.incorrect_answer,
        incorrect_process_weight=weights.incorrect_process, correct_format_weight=weights.correct_format,
        correct_process_weight=weights.correct_process, total=float(correct) if mode == "binary" else scores.total,
    )
