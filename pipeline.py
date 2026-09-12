"""Notebook-friendly wrapper for generate -> critique -> revise experiments.

Keeps the original API shape used in benchmark notebooks while
calling into src/self_correct under the hood.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.self_correct.metrics import summarize_records
from src.self_correct.models import OpenAICompatClient
from src.self_correct.pipeline_core import run_condition
from src.self_correct.prompts import make_generate_prompt  # re-export for notebooks

GENERATOR_MODEL = "openai/gpt-4o-mini"
CRITIC_MODEL = "openai/gpt-4o"
CLASSIFIER_MODEL = "openai/gpt-4o-mini"

SEED = 42
NUM_SAMPLES = 250

client: OpenAICompatClient | None = None


def init_client(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> OpenAICompatClient:
    """Initialize an OpenAI-compatible client for notebook experiments."""
    global client
    client = OpenAICompatClient(api_key=api_key, base_url=base_url)
    return client


def call_model(prompt: str, model: str, max_tokens: int = 768) -> str:
    """Send a single prompt to the model. Used by notebooks for ad-hoc calls."""
    if client is None:
        raise RuntimeError("Client is not initialized. Call init_client(api_key) first.")
    return client.complete(prompt, model, max_tokens=max_tokens)


def run_pipeline(
    samples: list[dict[str, Any]],
    task_description: str,
    answer_format: str,
    extract_answer_fn,
    evaluate_fn,
    condition: str,
    critique_style: str = "targeted",
    max_iterations: int = 1,
) -> pd.DataFrame:
    """Run one condition on pre-loaded benchmark samples."""
    if client is None:
        raise RuntimeError("Client is not initialized. Call init_client(api_key) first.")

    records = run_condition(
        samples=samples,
        task_description=task_description,
        answer_format=answer_format,
        extract_answer_fn=extract_answer_fn,
        evaluate_fn=evaluate_fn,
        evaluator_name="notebook_default",
        condition=condition,
        critique_style=critique_style,
        client=client,
        generator_model=GENERATOR_MODEL,
        critic_model=CRITIC_MODEL,
        classifier_model=CLASSIFIER_MODEL,
        max_iterations=max_iterations,
    )
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate metrics for a DataFrame produced by run_pipeline."""
    return summarize_records(df.to_dict(orient="records"))
