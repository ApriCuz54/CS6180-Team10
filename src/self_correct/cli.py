from __future__ import annotations

import argparse
import json
import os

from .config import ExperimentConfig
from .runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM self-correction experiments.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing API key.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_json_file(args.config)
    provider = (config.provider or "openrouter").lower()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set env var: {args.api_key_env}")

    summary = run_experiment(config, api_key=api_key)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
