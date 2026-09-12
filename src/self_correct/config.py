from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class ModelConfig:
    generator: str
    critic: str
    classifier: str


@dataclass(frozen=True)
class TaskConfig:
    name: str
    data_path: str
    sample_limit: int | None = None
    evaluator_type: str = "default"
    evaluator_model: str | None = None
    evaluator_max_turns: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str
    provider: str
    api_base_url: str
    seeds: list[int]
    conditions: list[str]
    critique_styles: list[str]
    model: ModelConfig
    tasks: list[TaskConfig]
    max_iterations: int = 1

    @staticmethod
    def from_json_file(path: str) -> "ExperimentConfig":
        cfg_path = Path(path)
        raw: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Support both "seeds": [42, 123, 456] and legacy "seed": 42
        seeds = raw.get("seeds")
        if seeds is None:
            seeds = [int(raw.get("seed", 42))]
        return ExperimentConfig(
            output_dir=raw["output_dir"],
            provider=raw.get("provider", "openrouter"),
            api_base_url=raw.get("api_base_url", "https://openrouter.ai/api/v1"),
            seeds=[int(s) for s in seeds],
            conditions=raw.get("conditions", ["no_correction", "same_model", "cross_model"]),
            critique_styles=raw.get("critique_styles", ["targeted"]),
            model=ModelConfig(**raw["model"]),
            tasks=[TaskConfig(**task) for task in raw["tasks"]],
            max_iterations=int(raw.get("max_iterations", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
