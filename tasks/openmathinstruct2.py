"""Numeric GSM8K/MATH-family OpenMathInstruct-2 train_1M (CC BY 4.0)."""
from __future__ import annotations

from typing import Any

from nanochat.reasoning import normalize_numeric
from tasks.common import load_hub_dataset
from tasks.reasoning_data import (
    ConvertedRows,
    numeric_conversation,
    numeric_solution_answer,
)


class OpenMathInstruct2(ConvertedRows):
    source_revision = "469216e3f46f4dacf476b382e192485ea51a143e"
    license = "CC-BY-4.0"

    def __init__(self, rows: Any = None, seed: int = 42, **kwargs: Any) -> None:
        if rows is None:
            rows = load_hub_dataset("nvidia/OpenMathInstruct-2", "default", split="train_1M")
        super().__init__(rows, self.convert_row, seed=seed, **kwargs)

    @staticmethod
    def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
        source = str(row.get("problem_source", "")).lower()
        if source not in {"gsm8k", "math", "augmented_gsm8k", "augmented_math"}:
            return None
        expected = normalize_numeric(str(row.get("expected_answer", "")))
        if expected is None:
            return None
        question, solution = row.get("problem", ""), row.get("generated_solution", "")
        answer = numeric_solution_answer(solution, expected)
        if answer is None:
            return None
        original = [value for value in (question, row.get("original_question"), row.get("original_problem"))
                    if isinstance(value, str) and value]
        try:
            conversation = numeric_conversation(question, solution, answer, "openmathinstruct2", original)
        except ValueError:
            return None
        conversation["reasoning"]["problem_source"] = source
        return conversation
