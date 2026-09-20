"""Reasoning conversion helpers, uncropped length filtering, and SFT assembly."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from nanochat.common import get_base_dir
from nanochat.reasoning import CONVERSION_VERSION, Calculator, normalize_numeric
from tasks.common import Task
from tasks.procedural import FORMAT_INSTRUCTION, answer_payload, reasoning_reward

if TYPE_CHECKING:
    from nanochat.rewards import RewardBreakdown

Conversation = dict[str, Any]


def normalized_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def question_keys(conversation: Conversation) -> set[str]:
    values = [message["content"] for message in conversation["messages"]
              if message["role"] == "user" and isinstance(message["content"], str)]
    metadata = conversation.get("reasoning", {})
    values.extend(metadata.get("original_questions", []))
    return {normalized_question(value.removesuffix("\n" + FORMAT_INSTRUCTION)) for value in values if value}


def _boxed_values(text: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\\boxed\{", text):
        start, depth = match.end(), 1
        for pos in range(start, len(text)):
            depth += (text[pos] == "{") - (text[pos] == "}")
            if depth == 0:
                values.append(text[start:pos])
                break
        else:
            values.append("")
    return values


def numeric_solution_answer(solution: str, expected: str | None = None) -> str | None:
    """Accept only mutually consistent, explicit numeric final-answer signals."""
    values = _boxed_values(solution)
    values.extend(match.group(1).strip() for match in re.finditer(r"####\s*([^\n]+)", solution))
    values.extend(match.group(1).strip().removesuffix(".") for match in re.finditer(
        r"(?:the (?:final )?answer is|final answer)\s*:?\s*([^\n]+)", solution, re.IGNORECASE))
    if expected is not None:
        values.append(expected)
    if not values:
        return None
    normalized = [normalize_numeric(value) for value in values]
    if any(value is None for value in normalized) or len(set(normalized)) != 1:
        return None
    return normalized[0]


def numeric_conversation(question: str, solution: str, answer: str, task: str,
                         original_questions: list[str] | None = None) -> Conversation:
    # Existing answer markers cannot appear inside the reasoning block, where they
    # would make the single final-answer marker ambiguous at train and test time.
    solution = re.sub(r"####", "Answer:", solution).strip()
    if not question.strip() or not solution or "<think>" in solution or "</think>" in solution:
        raise ValueError("Empty question/solution or preexisting reasoning delimiters")
    return {
        "messages": [{"role": "user", "content": question + "\n" + FORMAT_INSTRUCTION},
                     {"role": "assistant", "content": f"<think>\n{solution}\n</think>\n#### {answer}"}],
        "reasoning": {"task": task, "gold_answer": answer, "intermediates": [],
                      "process_available": False, "original_questions": original_questions or [question]},
    }


class NumericReasoningTask(Task):
    @property
    def eval_type(self) -> str:
        return "generative"

    def parse_answer(self, payload: str) -> str | None:
        return normalize_numeric(payload)

    def evaluate(self, conversation: Conversation, response: str) -> int:
        payload = answer_payload(response)
        answer = normalize_numeric(payload) if payload is not None else None
        return int(answer is not None and answer == normalize_numeric(conversation["reasoning"]["gold_answer"]))

    def answer_partial(self, conversation: Conversation, response: str) -> float:
        return float(self.evaluate(conversation, response))

    def reward(self, conversation: Conversation, response: str,
               calculator: Calculator | None = None, **kwargs: Any) -> RewardBreakdown:
        return reasoning_reward(self, conversation, response, calculator, **kwargs)


class ConvertedRows(NumericReasoningTask):
    """Convert lazily, retaining only valid row indices instead of duplicating text."""
    def __init__(self, rows: Any, convert: Callable[[dict[str, Any]], Conversation | None],
                 seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rows, self.convert, self.seed = rows, convert, seed
        self.indices = [index for index in range(len(rows)) if convert(rows[index]) is not None]
        random.Random(seed).shuffle(self.indices)

    def num_examples(self) -> int:
        return len(self.indices)

    def get_example(self, index: int) -> Conversation:
        if not 0 <= index < len(self.indices):
            raise IndexError(index)
        example = self.convert(self.rows[self.indices[index]])
        assert example is not None
        return example


class IndexedTask(Task):
    def __init__(self, task: Task, indices: Sequence[int]) -> None:
        super().__init__()
        self.task, self.indices = task, list(indices)

    @property
    def eval_type(self) -> str:
        return self.task.eval_type

    def num_examples(self) -> int:
        return len(self.indices)

    def get_example(self, index: int) -> Conversation:
        return self.task[self.indices[index]]

    def evaluate(self, conversation: Conversation, response: str) -> int:
        return self.task.evaluate(conversation, response)

    def parse_answer(self, payload: str) -> Any:
        return self.task.parse_answer(payload)

    def answer_partial(self, conversation: Conversation, response: str) -> float:
        return self.task.answer_partial(conversation, response)

    def reward(self, conversation: Conversation, response: str, **kwargs: Any) -> Any:
        return self.task.reward(conversation, response, **kwargs)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def dataset_fingerprint(task: Task) -> str:
    """Hash actual underlying data as well as view, seed, and conversion identity."""
    if isinstance(task, IndexedTask):
        return _digest_json([dataset_fingerprint(task.task), task.indices])
    rows = getattr(task, "rows", getattr(task, "ds", None))
    digest = hashlib.sha256()
    if rows is not None and hasattr(rows, "table"):
        digest.update(str(rows.table.schema).encode())
        for column in rows.table.columns:
            for chunk in column.chunks:
                for buffer in chunk.buffers():
                    if buffer is not None:
                        digest.update(buffer)
        permutation = getattr(rows, "permutation", None)
        if permutation is not None:
            digest.update(permutation.tobytes())
        digest.update(_digest_json(getattr(task, "indices", [])).encode())
    else:
        for index in range(len(task)):
            digest.update(_digest_json(task[index]).encode())
    digest.update(_digest_json([type(task).__module__, type(task).__name__, len(task), task.start,
                               task.stop, task.step, getattr(task, "seed", None),
                               getattr(task, "source_revision", None), CONVERSION_VERSION]).encode())
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> str:
    explicit = getattr(tokenizer, "fingerprint", None)
    if explicit is not None:
        return str(explicit)
    return hashlib.sha256(pickle.dumps(getattr(tokenizer, "enc", tokenizer))).hexdigest()


def render_untruncated(tokenizer: Any, conversation: Conversation) -> tuple[list[int], list[int]]:
    # The existing tokenizer uses ids[:max_tokens]; [:None] deliberately disables it.
    ids, masks = tokenizer.render_conversation(conversation, max_tokens=None)
    return ids, masks


class LengthFiltered(IndexedTask):
    """Keep complete examples under both limits; include tools in turn lengths."""
    def __init__(self, task: Task, tokenizer: Any, context_limit: int = 1536,
                 assistant_limit: int = 1024, cache_dir: str | Path | None = None,
                 conversion_version: str = CONVERSION_VERSION, max_rows: int | None = None) -> None:
        if context_limit <= 0 or assistant_limit <= 0 or (max_rows is not None and max_rows < 0):
            raise ValueError("Length limits must be positive and max_rows nonnegative")
        self.context_limit, self.assistant_limit = context_limit, assistant_limit
        self.cache_hit = False
        identity = {"dataset": dataset_fingerprint(task), "tokenizer": tokenizer_fingerprint(tokenizer),
                    "conversion": conversion_version, "context_limit": context_limit,
                    "assistant_limit": assistant_limit, "max_rows": max_rows}
        self.fingerprint = _digest_json(identity)
        folder = Path(cache_dir) if cache_dir is not None else Path(get_base_dir()) / "reasoning_cache"
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / f"length-{self.fingerprint}.json"
        with FileLock(str(cache) + ".lock"):
            if cache.exists():
                try:
                    cached = json.loads(cache.read_text())
                    indices = cached["indices"]
                    valid = cached["identity"] == identity and isinstance(indices, list)
                    valid = valid and all(type(i) is int and 0 <= i < len(task) for i in indices)
                    valid = valid and indices == sorted(set(indices))
                    valid = valid and (max_rows is None or len(indices) <= max_rows)
                    if not valid:
                        raise ValueError("Invalid retained-index cache")
                    self.cache_hit = True
                except (OSError, ValueError, KeyError, TypeError):
                    indices = self._scan(task, tokenizer, max_rows)
            else:
                indices = self._scan(task, tokenizer, max_rows)
            if not self.cache_hit:
                descriptor, temporary = tempfile.mkstemp(prefix=cache.name, suffix=".tmp", dir=folder)
                try:
                    with os.fdopen(descriptor, "w") as stream:
                        json.dump({"identity": identity, "indices": indices}, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, cache)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        super().__init__(task, indices)

    def _scan(self, task: Task, tokenizer: Any, max_rows: int | None) -> list[int]:
        retained: list[int] = []
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        for index in range(len(task)):
            if max_rows is not None and len(retained) >= max_rows:
                break
            ids, masks = render_untruncated(tokenizer, task[index])
            if len(ids) > self.context_limit or not any(masks):
                continue
            start: int | None = None
            valid = True
            for position, token in enumerate(ids):
                if token == assistant_start:
                    start = position
                elif token == assistant_end and start is not None:
                    # Include both assistant delimiters and every intervening tool token.
                    if position - start + 1 > self.assistant_limit:
                        valid = False
                        break
                    start = None
            if valid and start is None:
                retained.append(index)
        return retained


def without_overlap(task: Task, excluded: set[str]) -> IndexedTask:
    return IndexedTask(task, [index for index in range(len(task)) if not (question_keys(task[index]) & excluded)])


def _evaluation_questions() -> set[str]:
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K
    from tasks.humaneval import HumanEval
    from tasks.mmlu import MMLU
    tasks = [GSM8K(subset="main", split="test"), MMLU(subset="all", split="test"),
             ARC(subset="ARC-Easy", split="test"), ARC(subset="ARC-Challenge", split="test"),
             HumanEval()]
    questions: set[str] = set()
    for task in tasks:
        for index in range(len(task)):
            example = task[index]
            questions.update(question_keys(example))
            # MMLU and ARC render multiple-choice wrappers; compare original questions too.
            ds = getattr(task, "ds", None)
            if ds is not None:
                raw = ds[index]
                for key in ("question", "prompt", "original_question"):
                    if isinstance(raw.get(key), str):
                        questions.add(normalized_question(raw[key]))
    return questions


def build_sft_mixture(tokenizer: Any, metamath_rows: int = 25000, omi2_rows: int = 25000,
                      gsm8k_epochs: int = 4, procedural_rows: int = 15000,
                      replay_fraction: float = 0.25, seed: int = 42, validation_size: int = 128,
                      context_limit: int = 1536, assistant_limit: int = 1024,
                      cache_dir: str | Path | None = None,
                      evaluation_questions: Iterable[str] | None = None,
                      sources: dict[str, Task] | None = None) -> tuple[Task, Task]:
    """Filter, reserve validation, cap, oversample, then add unchanged replay rows.

    Replay is sized from final reasoning rows and follows the original *source*
    SmolTalk:MMLU×3 row ratio. Exact normalized question exclusion cannot establish
    absence of semantic contamination. Sources injection supports offline fixtures.
    """
    if min(metamath_rows, omi2_rows, gsm8k_epochs, procedural_rows, validation_size) < 0:
        raise ValueError("Row caps, passes, and validation size must be nonnegative")
    if not 0 <= replay_fraction < 1:
        raise ValueError("replay_fraction must be in [0, 1)")
    from tasks.gsm8k_reasoning import GSM8KReasoning
    from tasks.metamath import MetaMathQA
    from tasks.mmlu import MMLU
    from tasks.openmathinstruct2 import OpenMathInstruct2
    from tasks.procedural import make_procedural_tasks
    from tasks.routed import RoutedTaskMixture
    from tasks.smoltalk import SmolTalk

    supplied = dict(sources or {})
    excluded = (_evaluation_questions() if evaluation_questions is None
                else {normalized_question(q) for q in evaluation_questions})
    requested: dict[str, tuple[Task, int | None, int]] = {}
    if gsm8k_epochs:
        requested["gsm8k"] = (supplied["gsm8k"] if "gsm8k" in supplied else GSM8KReasoning(seed=seed), None, gsm8k_epochs)
    if metamath_rows:
        requested["metamath"] = (supplied["metamath"] if "metamath" in supplied else MetaMathQA(seed=seed), metamath_rows, 1)
    if omi2_rows:
        requested["openmathinstruct2"] = (supplied["openmathinstruct2"] if "openmathinstruct2" in supplied else OpenMathInstruct2(seed=seed), omi2_rows, 1)
    procedural_validation: dict[str, Task] = {}
    if procedural_rows:
        count = (procedural_rows + 2) // 3
        procedural_validation = make_procedural_tasks(split="validation", seed=seed, size=validation_size)
        for position, (name, task) in enumerate(make_procedural_tasks(seed=seed, size=count).items()):
            cap = procedural_rows // 3 + int(position < procedural_rows % 3)
            if cap:
                requested[name] = (supplied.get(name, task), cap, 1)
    train_sources, val_sources, heldout = {}, {}, set()
    prepared: dict[str, tuple[Task, int | None, int]] = {}
    for name, (task, cap, epochs) in requested.items():
        clean = without_overlap(task, excluded)
        filtered = LengthFiltered(clean, tokenizer, context_limit, assistant_limit, cache_dir=cache_dir,
                                  max_rows=None if cap is None else cap + validation_size)
        if name in procedural_validation:
            reserve = 0
            validation_task = supplied.get(f"{name}_validation", procedural_validation[name])
            validation_task = without_overlap(validation_task, excluded)
            val_sources[name] = LengthFiltered(validation_task, tokenizer, context_limit, assistant_limit,
                                               cache_dir=cache_dir, max_rows=validation_size)
        else:
            reserve = min(validation_size, max(0, len(filtered) - 1))
            val_sources[name] = IndexedTask(filtered, range(reserve))
        for index in range(len(val_sources[name])):
            heldout.update(question_keys(val_sources[name][index]))
        prepared[name] = (IndexedTask(filtered, range(reserve, len(filtered))), cap, epochs)
    for name, (task, cap, epochs) in prepared.items():
        clean = without_overlap(task, heldout)
        indices = list(range(len(clean) if cap is None else min(cap, len(clean))))
        train_sources[name] = IndexedTask(clean, indices * epochs)
    reasoning_count = sum(len(task) for task in train_sources.values())
    replay_count = round(reasoning_count * replay_fraction / (1 - replay_fraction))
    if replay_count:
        replay = {"smoltalk": supplied["smoltalk"] if "smoltalk" in supplied else SmolTalk(split="train"),
                  "mmlu": supplied["mmlu"] if "mmlu" in supplied else MMLU(subset="all", split="auxiliary_train")}
        denominator = len(replay["smoltalk"]) + 3 * len(replay["mmlu"])
        if not denominator:
            raise ValueError("Replay sources are empty")
        smol_count = round(replay_count * len(replay["smoltalk"]) / denominator)
        for name, count in (("smoltalk", smol_count), ("mmlu", replay_count - smol_count)):
            if not count:
                continue
            task = without_overlap(replay[name], excluded | heldout)
            indices = list(range(len(task)))
            random.Random(seed).shuffle(indices)
            task = IndexedTask(task, indices)
            task = LengthFiltered(task, tokenizer, context_limit, assistant_limit,
                                  cache_dir=cache_dir, max_rows=count)
            if not len(task):
                raise ValueError(f"No usable {name} replay rows")
            train_sources[name] = IndexedTask(task, [index % len(task) for index in range(count)])
    train = RoutedTaskMixture(train_sources, seed=seed)
    validation = RoutedTaskMixture(val_sources, seed=seed + 1)
    if not len(train):
        raise ValueError("Reasoning SFT mixture is empty after filtering")
    return train, validation
