"""Run an experiment from a config file and generate all plots.

One command to go from config to results + figures:

    python scripts/run_and_plot.py --config configs/experiment.example.json

Override settings without editing the config file:

    python scripts/run_and_plot.py --config configs/experiment.example.json \
        --max-iterations 3 \
        --conditions same_model cross_model \
        --critique-styles targeted generic

Plots are saved next to summary.json in the timestamped output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# add project root to path so src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.self_correct.config import ExperimentConfig
from src.self_correct.runner import run_experiment
from scripts.plot_results import generate_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run experiment + generate plots in one step.",
    )
    p.add_argument("--config", required=True, help="Path to experiment JSON config.")
    p.add_argument(
        "--api-key-env", default="OPENROUTER_API_KEY",
        help="Env var with the API key (default: OPENROUTER_API_KEY).",
    )

    # optional overrides
    p.add_argument("--max-iterations", type=int, default=None, help="Override max_iterations.")
    p.add_argument("--conditions", nargs="+", default=None, help="Override conditions list.")
    p.add_argument("--critique-styles", nargs="+", default=None, help="Override critique_styles list.")
    p.add_argument("--plots-only", default=None, help="Skip running; just plot from an existing summary.json path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # plots-only mode: just generate charts from existing results
    if args.plots_only:
        summary_path = Path(args.plots_only)
        if not summary_path.exists():
            print(f"File not found: {summary_path}")
            return 1
        generate_plots(summary_path, summary_path.parent / "figures")
        return 0

    # load config
    config = ExperimentConfig.from_json_file(args.config)

    # apply CLI overrides by rebuilding the frozen dataclass
    overrides: dict = {}
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations
    if args.conditions is not None:
        overrides["conditions"] = args.conditions
    if args.critique_styles is not None:
        overrides["critique_styles"] = args.critique_styles

    if overrides:
        from dataclasses import asdict
        d = asdict(config)
        d.update(overrides)
        from src.self_correct.config import ModelConfig, TaskConfig
        d["model"] = ModelConfig(**d["model"])
        d["tasks"] = [TaskConfig(**t) for t in d["tasks"]]
        config = ExperimentConfig(**d)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Missing API key. Set env var: {args.api_key_env}")
        return 1

    print(f"Running experiment (max_iterations={config.max_iterations})...")
    print(f"  conditions: {config.conditions}")
    print(f"  critique_styles: {config.critique_styles}")
    print()

    summary = run_experiment(config, api_key=api_key)

    # find the summary.json that was just written
    output_dir = Path(summary["output_dir"])
    summary_path = output_dir / "summary.json"

    print()
    print("Generating plots...")
    figures_dir = output_dir / "figures"
    generate_plots(summary_path, figures_dir)

    print()
    print(f"Results: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
