"""Pure CPU format, arithmetic boundary and exact normalization acceptance tests."""

from fractions import Fraction

import pytest

from nanochat.reasoning import (
    arithmetic_statements,
    extract_answer_marker,
    extract_intermediate_values,
    normalize_numeric,
    parse_reasoning,
    repeated_ngram_fraction,
    verify_arithmetic,
)


def calculator(expression: str) -> object:
    # Local, deterministic fixtures only; production injects the engine calculator.
    return {"2+3": 5, "5*2": 10, "1+1": 2, "0.1+0.2": 0.1 + 0.2, "4/2": 2}.get(expression.replace(" ", ""))


@pytest.mark.parametrize("value,expected", [
    ("$ 1,000.0", "1000"), ("-0.00", "0"), (" .5 ", "1/2"),
    ("2/4", "1/2"), ("1.25e2", "125"), ("+42", "42"),
    (r"\frac{25}{9}", "25/9"), (r"\boxed{91}", "91"),
    (r"\$3.50", "7/2"), ("$42$", "42"), (Fraction(3, 7), "3/7"),
])
def test_numeric_normalization(value: object, expected: str) -> None:
    assert normalize_numeric(value) == expected


@pytest.mark.parametrize("value", ["1,2", "nan", "inf", "1/0", "42 people", "1 and 2", "", True, "1e999999999"])
def test_ambiguous_or_non_numeric_answers_rejected(value: object) -> None:
    assert normalize_numeric(value) is None


def test_answer_comparison_has_no_float_tolerance() -> None:
    assert normalize_numeric("0.30000000000000004") != normalize_numeric("0.3")


def test_valid_format_and_trailing_text() -> None:
    parsed = parse_reasoning("<think>2+3=5</think>\nThe answer is five.\n#### 5\nextra")
    assert parsed.think == "2+3=5"
    assert parsed.answer == "5"
    assert parsed.format_score == 1
    assert parsed.trailing_text == "extra"


@pytest.mark.parametrize("text", [
    "<think>open\n#### 5", "<think><think>nested</think></think>\n#### 5",
    "<think>a</think><think>b</think>\n#### 5", "</think>x<think>\n#### 5",
    "<think >bad</think>\n#### 5", "<THINK>bad</think>\n#### 5",
    "<think>x</think><think\n#### 5", "<think>x</ think>\n#### 5",
])
def test_malformed_tags_receive_no_format_credit(text: str) -> None:
    parsed = parse_reasoning(text)
    assert not parsed.think_valid
    assert not parsed.answer_valid
    assert parsed.format_score == 0


def test_empty_block_only_loses_think_credit() -> None:
    parsed = parse_reasoning("<think> \n </think>\n#### 5")
    assert not parsed.think_valid
    assert parsed.answer_valid
    assert parsed.format_score == 0.5


@pytest.mark.parametrize("text", [
    "<think>work\n#### 5</think>",
    "#### 5\n<think>work</think>",
    "<think>work</think>\n#### 5\n#### 5",
    "<think>#### 5</think>\n#### 5",
    "<think>work</think>\nAnswer #### 5",
    "<think>work</think>\n#### five",
])
def test_answer_marker_must_be_unique_parsable_line_after_block(text: str) -> None:
    parsed = parse_reasoning(text)
    assert parsed.think_valid
    assert not parsed.answer_valid
    assert parsed.format_score == 0.5


def test_custom_task_answer_parser() -> None:
    parsed = parse_reasoning("<think>sort</think>\n#### 1, 2", lambda value: tuple(int(n) for n in value.split(",")))
    assert parsed.answer == (1, 2)
    assert parsed.format_score == 1
    assert extract_answer_marker("answer only\n#### 42") == "42"
    assert extract_answer_marker("#### 42\n#### 43") is None


def test_arithmetic_preserves_trace_order_and_normalizes_results() -> None:
    text = "First 2+3=5. Then <<5*2=10>>\n<|python_start|>4/2<|python_end|><|output_start|>2.0<|output_end|>"
    statements = arithmetic_statements(text)
    assert [statement.source for statement in statements] == ["equation", "annotation", "tool"]
    assert extract_intermediate_values(text, calculator) == ["5", "10", "2"]


def test_tool_results_require_matched_boundaries_and_verification() -> None:
    text = (
        "<|output_start|>5<|output_end|> "
        "<|python_start|>2+3<|python_end|><|output_start|>99<|output_end|> "
        "<|python_start|>5*2<|python_end|><|output_start|>10"
    )
    checked = verify_arithmetic(text, calculator)
    assert len(checked.statements) == 1
    assert checked.invalid_count == 1
    assert checked.verified_values == ()


def test_invalid_equations_do_not_cover_intermediates() -> None:
    checked = verify_arithmetic("2+3=10; 5*2=10; 1+1=2", calculator)
    assert checked.verified_values == ("10", "2")
    assert checked.invalid_count == 1
    assert checked.invalid_fraction == pytest.approx(1 / 3)


def test_only_verification_allows_calculator_roundoff() -> None:
    assert verify_arithmetic("0.1+0.2=0.3", calculator).invalid_count == 0
    assert verify_arithmetic("500000000000+500000000000=1000000000001", lambda _: 1000000000000).invalid_count == 1
    assert verify_arithmetic("500000000000+500000000000=1000000000001", lambda _: 1000000000000.0).invalid_count == 1


def test_ngram_repetition_is_case_insensitive_and_bounded() -> None:
    assert repeated_ngram_fraction("one two three") == 0
    assert repeated_ngram_fraction("one two three four five") == 0
    assert repeated_ngram_fraction("One two THREE four one TWO three FOUR") == pytest.approx(1 / 5)
    assert 0.9 < repeated_ngram_fraction("one two three four " * 100) < 1
    with pytest.raises(ValueError):
        repeated_ngram_fraction("x", n=0)
