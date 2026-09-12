from __future__ import annotations

from datetime import datetime
import random
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .config import ExperimentConfig
from .evaluators import build_evaluator
from .io_utils import ensure_dir, write_json, write_jsonl
from .metrics import aggregate_run_summaries, summarize_records
from .models import OpenAICompatClient
from .pipeline_core import run_condition
from .tasks.registry import get_task_adapter


def run_experiment(config: ExperimentConfig, api_key: str | None = None) -> dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = ensure_dir(Path(config.output_dir) / ts)
    runs_out = ensure_dir(base_out / "runs")

    if not api_key:
        raise ValueError("api_key is required for OpenAI-compatible providers")
    client = OpenAICompatClient(api_key=api_key, base_url=config.api_base_url)

    experiment_summary: dict[str, Any] = {
        "timestamp": ts,
        "output_dir": str(base_out),
        "seeds": config.seeds,
        "num_runs": len(config.seeds),
        "conditions": config.conditions,
        "critique_styles": config.critique_styles,
        "tasks": {},
    }

    experiment_start = time.perf_counter()

    for task_cfg in tqdm(config.tasks, desc="Tasks", dynamic_ncols=True):
        task_start = time.perf_counter()
        adapter = get_task_adapter(task_cfg.name)
        evaluator = build_evaluator(task_cfg, adapter)
        all_samples = adapter.load_samples(task_cfg.data_path)

        # Collect per-condition summaries across seeds
        per_key_runs: dict[str, list[dict[str, Any]]] = {}

        for seed in tqdm(config.seeds, desc=f"{task_cfg.name} | seeds", leave=False, dynamic_ncols=True):
            seed_start = time.perf_counter()
            random.seed(seed)
            samples = list(all_samples)
            random.shuffle(samples)
            if task_cfg.sample_limit:
                samples = samples[: task_cfg.sample_limit]

            for condition in tqdm(config.conditions, desc=f"{task_cfg.name} | seed {seed} | conditions", leave=False, dynamic_ncols=True):
                condition_start = time.perf_counter()
                styles = config.critique_styles if condition != "no_correction" else ["targeted"]
                for style in tqdm(styles, desc=f"{task_cfg.name} | seed {seed} | {condition} | styles", leave=False, dynamic_ncols=True):
                    style_start = time.perf_counter()
                    records = run_condition(
                        samples=samples,
                        task_description=adapter.task_description,
                        answer_format=adapter.answer_format,
                        extract_answer_fn=adapter.extract_answer,
                        evaluate_fn=evaluator.evaluate,
                        evaluator_name=evaluator.name,
                        condition=condition,
                        critique_style=style,
                        client=client,
                        generator_model=config.model.generator,
                        critic_model=config.model.critic,
                        classifier_model=config.model.classifier,
                        max_iterations=config.max_iterations,
                    )

                    key = f"{condition}__{style}"
                    run_dir = ensure_dir(runs_out / task_cfg.name / f"seed_{seed}")
                    write_jsonl(run_dir / f"{key}.jsonl", records)

                    summary = summarize_records(records)
                    summary["seed"] = seed
                    per_key_runs.setdefault(key, []).append(summary)
                    tqdm.write(
                        f"{task_cfg.name} | seed {seed} | {condition} | {style} finished in {time.perf_counter() - style_start:.1f}s"
                    )

                tqdm.write(
                    f"{task_cfg.name} | seed {seed} | {condition} finished in {time.perf_counter() - condition_start:.1f}s"
                )

            tqdm.write(f"{task_cfg.name} | seed {seed} finished in {time.perf_counter() - seed_start:.1f}s")

        # Aggregate across seeds
        task_results: dict[str, Any] = {}
        for key, summaries in per_key_runs.items():
            task_results[key] = aggregate_run_summaries(summaries)

        experiment_summary["tasks"][task_cfg.name] = task_results
        tqdm.write(f"{task_cfg.name} task finished in {time.perf_counter() - task_start:.1f}s")

    write_json(base_out / "summary.json", experiment_summary)
    write_json(base_out / "config_snapshot.json", config.to_dict())
    tqdm.write(f"Experiment finished in {time.perf_counter() - experiment_start:.1f}s")
    return experiment_summary
