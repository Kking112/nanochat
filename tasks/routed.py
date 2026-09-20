"""Shuffled mixtures retaining the source task needed to evaluate a response."""
from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from tasks.common import Task


class RoutedTaskMixture(Task):
    def __init__(self, tasks: Mapping[str, Task], seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tasks = dict(tasks)
        self.index_map = [(name, index) for name, task in self.tasks.items() for index in range(len(task))]
        random.Random(seed).shuffle(self.index_map)

    @property
    def eval_type(self) -> str:
        return "generative"

    def num_examples(self) -> int:
        return len(self.index_map)

    def get_example(self, index: int) -> dict[str, Any]:
        name, local = self.index_map[index]
        # Metadata only: source replay messages are not edited or wrapped.
        return {**self.tasks[name][local], "task_route": name, "task_name": name}

    def task_for(self, conversation: dict[str, Any]) -> Task:
        return self.tasks[conversation["task_route"]]

    def evaluate(self, conversation: dict[str, Any], response: str) -> int:
        return self.task_for(conversation).evaluate(conversation, response)

    def answer_partial(self, conversation: dict[str, Any], response: str) -> float:
        return self.task_for(conversation).answer_partial(conversation, response)

    def reward(self, conversation: dict[str, Any], response: str, **kwargs: Any) -> Any:
        return self.task_for(conversation).reward(conversation, response, **kwargs)
