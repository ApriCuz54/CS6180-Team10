from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Sample:
    id: str
    question: str
    answer: str


class TaskAdapter(Protocol):
    name: str
    task_description: str
    answer_format: str

    def load_samples(self, data_path: str, sample_limit: int | None = None) -> list[dict]:
        ...

    def extract_answer(self, raw_text: str) -> str:
        ...

    def evaluate(self, prediction: str, gold: str) -> bool:
        ...
