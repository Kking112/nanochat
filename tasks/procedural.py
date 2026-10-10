"""Deterministic arithmetic, sorting, and legal-expression reasoning tasks."""
from __future__ import annotations

import ast
import hashlib
import random
import re
from collections import Counter
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from nanochat.reasoning import Calculator, extract_answer_marker, normalize_numeric
from tasks.common import Task

if TYPE_CHECKING:
    from nanochat.rewards import RewardBreakdown, RewardTask

Conversation = dict[str, Any]
FORMAT_INSTRUCTION = "Show your reasoning inside <think>...</think>, then end with one line '#### answer'."


answer_payload = extract_answer_marker


def reasoning_reward(task: RewardTask, conversation: Conversation, response: str,
                     calculator: Calculator | None = None, **kwargs: Any) -> RewardBreakdown:
    """Task-owned reward integration keeps the scoring library Torch independent."""
    from nanochat.rewards import score_response
    if calculator is None:
        from nanochat.engine import use_calculator
        calculator = use_calculator
    return score_response(task, conversation, response, calculator=calculator, **kwargs)


class ProceduralTask(Task):
    name = "procedural"

    def __init__(self, split: str = "train", seed: int = 42, size: int = 5000,
                 difficulty: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if size < 0:
            raise ValueError("size must be nonnegative")
        self.split, self.seed, self.size = split, seed, size
        self.difficulty = difficulty

    @property
    def difficulty(self) -> int:
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: int) -> None:
        if not 1 <= value <= 5:
            raise ValueError("difficulty must be between 1 and 5")
        self._difficulty = value

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return self.size

    def rng(self, index: int, attempt: int = 0) -> random.Random:
        if not 0 <= index < self.size:
            raise IndexError(index)
        identity = f"reasoning-v1:{self.name}:{self.split}:{self.seed}:{index}:{self.difficulty}:{attempt}"
        return random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))

    def get_example(self, index: int) -> Conversation:
        # Separate RNG streams alone still allow accidental identical easy tasks.
        # Partition actual questions so exact held-out overlap is impossible even
        # when the seed, index, or difficulty changes between training and testing.
        bucket = {"train": 0, "validation": 1, "test": 2}[self.split]
        for attempt in range(1000):
            example = self._generate(index, attempt)
            question = " ".join(example["messages"][0]["content"].casefold().split())
            if int.from_bytes(hashlib.sha256(question.encode()).digest(), "big") % 3 == bucket:
                return example
        raise RuntimeError("Could not generate a question in the requested split")

    def _generate(self, index: int, attempt: int) -> Conversation:
        raise NotImplementedError

    def conversation(self, index: int, question: str, solution: str, gold: str,
                     intermediates: list[str], **metadata: Any) -> Conversation:
        return {
            "messages": [{"role": "user", "content": question + "\n" + FORMAT_INSTRUCTION},
                         {"role": "assistant", "content": f"<think>\n{solution}\n</think>\n#### {gold}"}],
            "reasoning": {"task": self.name, "gold_answer": gold,
                          "intermediates": intermediates, "process_available": bool(intermediates),
                          "split": self.split, "seed": self.seed, "index": index,
                          "difficulty": self.difficulty, **metadata},
        }

    def parse_answer(self, payload: str) -> Any:
        return normalize_numeric(payload)

    def evaluate(self, conversation: Conversation, response: str) -> int:
        payload = answer_payload(response)
        predicted = self.parse_answer(payload) if payload is not None else None
        gold = self.parse_answer(conversation["reasoning"]["gold_answer"])
        return int(predicted is not None and predicted == gold)

    def answer_partial(self, conversation: Conversation, response: str) -> float:
        return float(self.evaluate(conversation, response))

    def reward(self, conversation: Conversation, response: str,
               calculator: Calculator | None = None, **kwargs: Any) -> RewardBreakdown:
        return reasoning_reward(self, conversation, response, calculator, **kwargs)


class ArithmeticChain(ProceduralTask):
    name = "chain"

    def _generate(self, index: int, attempt: int) -> Conversation:
        rng = self.rng(index, attempt)
        value = rng.randint(1, 10 * self.difficulty)
        instructions, equations, intermediates = [f"Start with {value}."], [], []
        for _ in range(self.difficulty + 2):
            operand = rng.randint(1, 5 * self.difficulty)
            operator = rng.choice(["+", "-", "*"] if self.difficulty > 1 else ["+", "-"])
            result = {"+": value + operand, "-": value - operand, "*": value * operand}[operator]
            instructions.append(f"Then {'add' if operator == '+' else 'subtract' if operator == '-' else 'multiply by'} {operand}.")
            equations.append(f"{value} {operator} {operand} = {result}")
            intermediates.append(str(result))
            value = result
        instructions.append("What is the final integer?")
        return self.conversation(index, " ".join(instructions), "\n".join(equations), str(value), intermediates)


class Sorting(ProceduralTask):
    name = "sorting"

    def _generate(self, index: int, attempt: int) -> Conversation:
        rng = self.rng(index, attempt)
        numbers = [rng.randint(-10 * self.difficulty, 10 * self.difficulty) for _ in range(3 + 2 * self.difficulty)]
        result = ", ".join(map(str, sorted(numbers)))
        return self.conversation(index, f"Sort these integers in ascending order: {', '.join(map(str, numbers))}.",
                                 f"Ordering from smallest to largest gives {result}.", result, [], numbers=numbers)

    def parse_answer(self, payload: str) -> tuple[int, ...] | None:
        payload = payload.strip()
        if payload.startswith("[") != payload.endswith("]"):
            return None
        payload = payload.removeprefix("[").removesuffix("]").strip()
        if not re.fullmatch(r"-?\d+(?:\s*,\s*-?\d+)*", payload):
            return None
        return tuple(int(value.strip()) for value in payload.split(","))

    def answer_partial(self, conversation: Conversation, response: str) -> float:
        payload = answer_payload(response)
        predicted = self.parse_answer(payload) if payload is not None else None
        gold = self.parse_answer(conversation["reasoning"]["gold_answer"])
        if predicted is None or gold is None or len(predicted) != len(gold):
            return 0.0
        return sum(a == b for a, b in zip(predicted, gold)) / len(gold)


def legal_countdown_value(expression: str, numbers: list[int]) -> Fraction | None:
    """Evaluate only +,-,*,/ trees using every supplied positive integer once."""
    if len(expression) > 512 or not re.fullmatch(r"[\d\s()+*/.-]+", expression):
        return None
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError):
        return None
    used: list[int] = []

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value > 0:
            used.append(node.value)
            return Fraction(node.value)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise TypeError("Unsupported expression")
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right

    try:
        value = visit(tree.body)
    except (ValueError, TypeError, ZeroDivisionError, RecursionError):
        return None
    return value if Counter(used) == Counter(numbers) else None


class Countdown(ProceduralTask):
    name = "countdown"

    def _generate(self, index: int, attempt: int) -> Conversation:
        rng = self.rng(index, attempt)
        numbers = [rng.randint(1, 5 + 2 * self.difficulty) for _ in range(min(6, self.difficulty + 2))]
        expression, value = str(numbers[0]), numbers[0]
        steps, intermediates = [], []
        for number in numbers[1:]:
            operator = rng.choice(["+", "*"] if self.difficulty <= 2 else ["+", "-", "*"])
            result = {"+": value + number, "-": value - number, "*": value * number}[operator]
            steps.append(f"{value} {operator} {number} = {result}")
            intermediates.append(str(result))
            expression = f"({expression} {operator} {number})"
            value = result
        shuffled = numbers[:]
        rng.shuffle(shuffled)
        question = (f"Use each of {', '.join(map(str, shuffled))} exactly once to make {value}. "
                    "Use only +, -, *, / and parentheses. Put the expression after ####.")
        return self.conversation(index, question, "\n".join(steps), expression, intermediates,
                                 numbers=shuffled, target=value)

    def parse_answer(self, payload: str) -> str | None:
        payload = payload.strip()
        if not payload or len(payload) > 512 or not re.fullmatch(r"[\d\s()+*/-]+", payload):
            return None
        try:
            tree = ast.parse(payload, mode="eval")
        except (SyntaxError, RecursionError):
            return None
        allowed = (ast.Expression, ast.BinOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div)
        if any(not isinstance(node, allowed) for node in ast.walk(tree)):
            return None
        return payload

    def evaluate(self, conversation: Conversation, response: str) -> int:
        payload = answer_payload(response)
        if payload is None:
            return 0
        metadata = conversation["reasoning"]
        value = legal_countdown_value(payload, metadata["numbers"])
        return int(value is not None and value == metadata["target"])

    def answer_partial(self, conversation: Conversation, response: str) -> float:
        payload = answer_payload(response)
        if payload is None:
            return 0.0
        metadata = conversation["reasoning"]
        value = legal_countdown_value(payload, metadata["numbers"])
        if value is None:
            return 0.0
        return 1.0 / (1.0 + float(abs(value - metadata["target"])))


def make_procedural_tasks(split: str = "train", seed: int = 42, size: int = 5000,
                          difficulty: int = 1) -> dict[str, ProceduralTask]:
    return {cls.name: cls(split=split, seed=seed, size=size, difficulty=difficulty)
            for cls in (ArithmeticChain, Sorting, Countdown)}
