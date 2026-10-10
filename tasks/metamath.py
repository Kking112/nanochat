"""Conservative numeric-only MetaMathQA conversion (dataset card: MIT)."""
from __future__ import annotations

from typing import Any

from tasks.common import load_hub_dataset
from tasks.reasoning_data import (
    ConvertedRows,
    numeric_conversation,
    numeric_solution_answer,
)


class MetaMathQA(ConvertedRows):
    source_revision = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
    license = "MIT"

    def __init__(self, rows: Any = None, seed: int = 42, **kwargs: Any) -> None:
        if rows is None:
            rows = load_hub_dataset("meta-math/MetaMathQA", "default", split="train")
        super().__init__(rows, self.convert_row, seed=seed, **kwargs)

    @staticmethod
    def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
        question, solution = row.get("query", ""), row.get("response", "")
        answer = numeric_solution_answer(solution)
        if answer is None:
            return None
        original = [value for value in (question, row.get("original_question")) if isinstance(value, str) and value]
        try:
            return numeric_conversation(question, solution, answer, "metamath", original)
        except ValueError:
            return None
