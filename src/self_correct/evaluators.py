from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FunctionEvaluator:
    """Wraps existing adapter.evaluate methods."""

    name: str
    evaluate_fn: Any

    def evaluate(self, prediction: str, gold: str, sample: dict[str, Any]) -> tuple[bool, str | None]:
        result = self.evaluate_fn(prediction, gold, sample)
        if isinstance(result, tuple) and len(result) == 2:
            return bool(result[0]), str(result[1]) if result[1] is not None else None
        return bool(result), None


def build_evaluator(task_cfg, adapter) -> Any:
    eval_type = (task_cfg.evaluator_type or "default").lower()

    if eval_type == "default":
        return FunctionEvaluator(name=f"{adapter.name}_default", evaluate_fn=adapter.evaluate)

    raise ValueError(f"Unsupported evaluator_type: {task_cfg.evaluator_type}")
