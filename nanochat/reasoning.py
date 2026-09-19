"""Shared reasoning syntax and deterministic, Torch-independent parsing helpers."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
import re
from typing import Callable, TypedDict


THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_MARKER = "####"
CONVERSION_VERSION = "reasoning-v1"


class _RequiredReasoningMetadata(TypedDict):
    gold_answer: str
    intermediates: list[str]


class ReasoningMetadata(_RequiredReasoningMetadata, total=False):
    process_available: bool
    task: str


Calculator = Callable[[str], object]
AnswerParser = Callable[[str], object | None]

# Commas must be thousands separators: accepting arbitrary commas changes values.
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(rf"{_NUMBER}(?:\s*/\s*{_NUMBER})?\Z")
_TAG_RE = re.compile(r"<\s*/?\s*think\b[^>]*(?:>|$)", re.IGNORECASE)
_MARKER_RE = re.compile(r"^[ \t]*####[ \t]+([^\r\n]+)", re.MULTILINE)


def normalize_numeric(value: str | int | float | Decimal | Fraction) -> str | None:
    """Return an exact canonical rational; currency/grouping are presentation only.

    Fractions avoid decimal rounding and make ``0.5`` and ``1/2`` identical.
    No numerical-distance tolerance is used for answers.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal, Fraction)):
        return None
    if isinstance(value, Fraction):
        number = value
    else:
        text = str(value).strip()
        boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}", text)
        if boxed:
            text = boxed.group(1).strip()
        text = text.replace(r"\$", "$")
        if text.startswith("$") and text.endswith("$") and len(text) > 1:
            text = text[1:-1].strip()
        if text.startswith("$"):
            text = text[1:].strip()
        fraction = re.fullmatch(r"([+-]?)\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", text)
        if fraction:
            text = f"{fraction.group(1)}{fraction.group(2)}/{fraction.group(3)}"
        if len(text) > 256 or not _NUMBER_RE.fullmatch(text):
            return None
        if any(abs(int(exponent)) > 256 for exponent in re.findall(r"[eE]([+-]?\d+)", text)):
            return None
        try:
            parts = [Fraction(part.strip().replace(",", "")) for part in text.split("/")]
            number = parts[0] if len(parts) == 1 else parts[0] / parts[1]
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    return str(number)


def extract_answer_marker(text: str) -> str | None:
    """Extract the unique answer line, independently of think-format correctness."""
    if text.count(ANSWER_MARKER) != 1:
        return None
    match = _MARKER_RE.search(text)
    return match.group(1).strip() if match else None


@dataclass(frozen=True)
class ParsedReasoning:
    think: str | None
    answer_text: str | None
    answer: object | None
    think_valid: bool
    answer_valid: bool
    trailing_text: str

    @property
    def format_score(self) -> float:
        return 0.5 * (int(self.think_valid) + int(self.answer_valid))


def parse_reasoning(text: str, answer_parser: AnswerParser = normalize_numeric) -> ParsedReasoning:
    """Require a single exact tag pair and one parsable answer line after it.

    An empty, otherwise structural block loses its block credit only. Nested,
    repeated, malformed or unclosed tags cannot establish an answer's location.
    """
    tags = list(_TAG_RE.finditer(text))
    structural = len(tags) == 2 and [tag.group() for tag in tags] == [THINK_START, THINK_END]
    think = text[tags[0].end():tags[1].start()] if structural else None
    marker = _MARKER_RE.search(text) if text.count(ANSWER_MARKER) == 1 else None
    answer_text = marker.group(1).strip() if marker else None
    answer: object | None = None
    if answer_text:
        try:
            answer = answer_parser(answer_text)
        except (ValueError, TypeError, ArithmeticError):
            pass
    answer_valid = bool(structural and marker and marker.start() >= tags[1].end() and answer is not None)
    trailing_text = text[marker.end():].strip() if marker else ""
    return ParsedReasoning(think, answer_text, answer, bool(think and think.strip()), answer_valid, trailing_text)


def split_think_answer(text: str, answer_parser: AnswerParser = normalize_numeric) -> ParsedReasoning:
    """Compatibility name for consumers that split the shared reasoning format."""
    return parse_reasoning(text, answer_parser)


@dataclass(frozen=True)
class ArithmeticStatement:
    expression: str
    result: str
    source: str


@dataclass(frozen=True)
class ArithmeticVerification:
    statements: tuple[ArithmeticStatement, ...]
    verified_values: tuple[str, ...]
    invalid_count: int

    @property
    def invalid_fraction(self) -> float:
        return self.invalid_count / len(self.statements) if self.statements else 0.0


_TOOL_RE = re.compile(
    r"<\|python_start\|>(.*?)<\|python_end\|>\s*"
    r"<\|output_start\|>(.*?)<\|output_end\|>", re.DOTALL,
)
_GOLD_RE = re.compile(r"<<([^<>]+)=([^<>]+)>>")
# Restrict free-text expressions to arithmetic. Tool calls are independently
# checked by the injected runtime calculator, including its supported count().
_EQUATION_RE = re.compile(
    rf"(?<![\w.])([+\-]?(?:\d[\d,.]*|\.\d+|\()[\d \t,.()+*/\-]*?)"
    rf"[ \t]*=[ \t]*({_NUMBER}(?:[ \t]*/[ \t]*{_NUMBER})?)(?!\w|\.\d)"
)


def arithmetic_statements(text: str) -> tuple[ArithmeticStatement, ...]:
    """Extract equations and paired tool results without crossing tool boundaries."""
    found: list[tuple[int, ArithmeticStatement]] = []
    masked = list(text)
    for pattern, source in ((_TOOL_RE, "tool"), (_GOLD_RE, "annotation")):
        for match in pattern.finditer(text):
            result = normalize_numeric(match.group(2).strip())
            if result is not None:
                found.append((match.start(), ArithmeticStatement(match.group(1).strip(), result, source)))
            masked[match.start():match.end()] = " " * (match.end() - match.start())
    # Unpaired tool blocks never turn into prose equations or trusted outputs.
    prose = "".join(masked)
    prose = re.sub(r"<\|(?:python_start|output_start)\|>.*?(?:<\|(?:python_end|output_end)\|>|$)", lambda m: " " * len(m.group()), prose, flags=re.DOTALL)
    for match in _EQUATION_RE.finditer(prose):
        expression = match.group(1).strip()
        if not any(op in expression.lstrip("+-") for op in "+-*/"):
            continue
        result = normalize_numeric(match.group(2))
        if result is not None:
            found.append((match.start(), ArithmeticStatement(expression, result, "equation")))
    return tuple(statement for _, statement in sorted(found, key=lambda item: item[0]))


def verify_arithmetic(text: str, calculator: Calculator) -> ArithmeticVerification:
    """Only checked, correct results can cover reference intermediate values."""
    statements = arithmetic_statements(text)
    values: list[str] = []
    invalid_count = 0
    for statement in statements:
        try:
            raw_computed = calculator(statement.expression)
            computed = normalize_numeric(raw_computed)
        except (ValueError, TypeError, ArithmeticError):
            raw_computed = None
            computed = None
        valid = computed == statement.result
        if isinstance(raw_computed, float) and computed is not None and not valid:
            # Calculator float roundoff is allowed for equation verification,
            # never for task answer matching or reference-value membership.
            try:
                valid = math.isclose(raw_computed, float(Fraction(statement.result)), rel_tol=0.0, abs_tol=2 * math.ulp(raw_computed))
            except OverflowError:
                valid = False
        if valid:
            values.append(statement.result)
        else:
            invalid_count += 1
    return ArithmeticVerification(statements, tuple(values), invalid_count)


def extract_intermediate_values(text: str, calculator: Calculator | None = None) -> list[str]:
    """Extract explicit results; supply a calculator to require verification."""
    if calculator is not None:
        return list(verify_arithmetic(text, calculator).verified_values)
    return [statement.result for statement in arithmetic_statements(text)]


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    """Fraction of word n-gram occurrences beyond their first occurrence."""
    if n < 1:
        raise ValueError("n must be positive")
    words = re.findall(r"\b\w+\b", text.casefold())
    grams = [tuple(words[index:index + n]) for index in range(len(words) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0
