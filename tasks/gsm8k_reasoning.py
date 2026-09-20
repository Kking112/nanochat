"""GSM8K reasoning supervision with original calculator calls and gold traces."""
from __future__ import annotations

import re
from typing import Any

from nanochat.reasoning import normalize_numeric
from tasks.common import load_hub_dataset
from tasks.procedural import FORMAT_INSTRUCTION
from tasks.reasoning_data import ConvertedRows


class GSM8KReasoning(ConvertedRows):
    source_revision = "740312add88f781978c0658806c59bc2815b9866"
    license = "MIT"

    def __init__(self, split: str = "train", rows: Any = None, seed: int = 42, **kwargs: Any) -> None:
        if split not in {"train", "test"}:
            raise ValueError("GSM8K split must be train or test")
        self.split = split
        if rows is None:
            rows = load_hub_dataset("openai/gsm8k", "main", split=split)
        super().__init__(rows, self.convert_row, seed=seed, **kwargs)

    @staticmethod
    def convert_row(row: dict[str, Any]) -> dict[str, Any] | None:
        question, solution = row.get("question", ""), row.get("answer", "")
        if solution.count("####") != 1 or not question.strip():
            return None
        body, raw_answer = solution.split("####")
        answer = normalize_numeric(raw_answer.strip())
        if answer is None or not body.strip() or "<think>" in body or "</think>" in body:
            return None
        parts = [{"type": "text", "text": "<think>\n"}]
        intermediates: list[str] = []
        for part in re.split(r"(<<[^>]+>>)", body.rstrip()):
            if part.startswith("<<") and part.endswith(">>"):
                expression, separator, result = part[2:-2].rpartition("=")
                if not separator or normalize_numeric(result) is None:
                    return None
                parts.extend([{"type": "python", "text": expression}, {"type": "python_output", "text": result}])
                intermediates.append(result.strip())
            elif part:
                parts.append({"type": "text", "text": part})
        parts.append({"type": "text", "text": f"\n</think>\n#### {answer}"})
        return {"messages": [{"role": "user", "content": question + "\n" + FORMAT_INSTRUCTION},
                             {"role": "assistant", "content": parts}],
                "reasoning": {"task": "gsm8k", "gold_answer": answer,
                              "intermediates": intermediates, "process_available": bool(intermediates),
                              "original_questions": [question]}}
