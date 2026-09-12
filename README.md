# LLM Agent Self-Correction: When Does Critique-and-Revise Actually Help?

A systematic empirical study testing when LLM self-correction through critique-and-revise loops actually improves task performance, when it degrades it, and what drives the difference. We evaluate a modular **Generate -> Critique -> Revise** pipeline across three benchmarks under four correction conditions.

**Authors:** Aditya Chatterjee, Shrey Babulal Patel, Manav Kamleshbhai Dhamani

## Code Structure Overview

```
CS6180-Team10-main/
|
|-- src/self_correct/               # Core experiment package
|   |-- __init__.py
|   |-- cli.py                      # CLI entry point (--config flag)
|   |-- config.py                   # Dataclasses: ExperimentConfig, ModelConfig, TaskConfig
|   |-- evaluators.py               # Evaluator factory wrapping task adapters
|   |-- io_utils.py                 # JSON/JSONL read/write helpers
|   |-- metrics.py                  # NCV, fix/break rates, transition counts, critique correlation
|   |-- models.py                   # OpenAI-compatible API client with retry logic
|   |-- pipeline_core.py            # Core generate -> critique -> revise loop
|   |-- prompts.py                  # All prompt templates (generate, critique, revise, classify)
|   |-- runner.py                   # Multi-seed, multi-task experiment orchestrator
|   |-- tasks/                      # Benchmark adapters
|       |-- __init__.py
|       |-- base.py                 # TaskAdapter protocol definition
|       |-- registry.py             # Task name -> adapter lookup
|       |-- math500.py              # MATH500: load, extract, evaluate (exact match + numeric)
|       |-- hotpotqa.py             # HotpotQA: load, extract, evaluate (EM + F1)
|       |-- humaneval.py            # HumanEval: load, extract, evaluate (execution-based)
|
|-- scripts/
|   |-- run_and_plot.py             # End-to-end: run experiment + generate plots
|   |-- plot_results.py             # Generate plots from existing summary.json
|   |-- download_benchmarks.py      # Download MATH500, HotpotQA, HumanEval datasets
|
|-- configs/
|   |-- experiment.example.json     # Example experiment configuration
|
|-- data/                           # Benchmark data files (downloaded via script)
|   |-- math500_test.jsonl
|   |-- hotpot_dev_distractor_v1.json
|   |-- HumanEval.jsonl
|
|-- results/                        # Raw result CSVs (3 runs x 4 conditions)
|   |-- math500.csv
|   |-- hotpotqa.csv
|   |-- humaneval.csv
|
|-- notebooks/
|   |-- MathBenchmark_GenAI.ipynb    # Math benchmark notebook experiments
|   |-- Shared_pipeline_test.ipynb   # Shared pipeline testing notebook
|
|-- pipeline.py                     # Notebook-friendly wrapper (init_client, run_pipeline, summarize)
|-- main.py                         # Simple entry point
|-- LLM_correction_paper.tex        # Paper source (NeurIPS format)
|-- requirements.txt                # Python dependencies
|-- README.md                       # This file
```

### Key Modules

- **`pipeline_core.py`** implements the three-stage pipeline: for each sample, it generates an answer, optionally sends it through a critique -> revise loop (supporting multiple iterations), and records error transitions (correct->correct, wrong->correct, correct->wrong, wrong->wrong) at each step.

- **`prompts.py`** contains all four prompt templates: generate, targeted critique (asks for specific error identification), generic critique (open-ended review), and revise (instructs model to fix errors or keep answer).

- **`metrics.py`** computes Fix Rate, Break Rate, Net Correction Value (NCV), critique quality correlation (revision outcomes grouped by critique class), and per-iteration breakdowns for multi-iteration runs.

- **`runner.py`** orchestrates the full experiment: iterates over tasks, seeds, conditions (no_correction / same_model / cross_model), and critique styles (targeted / generic), writing per-sample JSONL logs and an aggregate summary.json.

- **Task adapters** (`tasks/math500.py`, `tasks/hotpotqa.py`, `tasks/humaneval.py`) handle dataset loading, answer extraction from LLM output, and task-specific evaluation (exact match for MATH500, EM + F1 for HotpotQA, test execution for HumanEval).

## Libraries and Versions

| Library | Version | Purpose |
|---|---|---|
| `openai` | >= 1.40.0 | OpenAI-compatible API client for LLM calls (used via OpenRouter) |
| `pandas` | >= 2.2.0 | DataFrame operations in notebook wrapper (`pipeline.py`) |
| `tqdm` | >= 4.66.0 | Progress bars for experiment runs |
| `matplotlib` | >= 3.8.0 | Plotting accuracy charts, transition diagrams, critique correlations |
| `numpy` | >= 1.26.0 | Numerical operations in plot generation |

**Python version:** 3.10+ (uses `X | Y` union type syntax and `from __future__ import annotations`)

**External API:** OpenRouter (default) or any OpenAI-compatible endpoint. Models used: GPT-4o-mini (generator/reviser/classifier), GPT-4o (cross-model critic).

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your API key:
```bash
# Linux/Mac
export OPENROUTER_API_KEY="your_key_here"

# Windows PowerShell
$env:OPENROUTER_API_KEY="your_key_here"
```

4. Download benchmark datasets:
```bash
python scripts/download_benchmarks.py
```

## Running Experiments

### Full experiment (run + plot)
```bash
python scripts/run_and_plot.py --config configs/experiment.example.json
```

### CLI only (no plots)
```bash
python -m src.self_correct.cli --config configs/experiment.example.json
```

### Plot from existing results
```bash
python scripts/plot_results.py outputs/<timestamp>/summary.json
```

### Configuration

Key fields in `configs/experiment.example.json`:

| Field | Default | Description |
|---|---|---|
| `conditions` | `["no_correction", "same_model", "cross_model"]` | Correction modes to run |
| `critique_styles` | `["targeted", "generic"]` | Targeted (specific errors) vs generic (open-ended) |
| `max_iterations` | `1` | Number of critique-revise loops per sample |
| `seeds` | `[42, 123, 456]` | Random seeds for reproducibility (3 runs) |
| `model.generator` | `openai/gpt-4o-mini` | Model for generation and revision |
| `model.critic` | `openai/gpt-4o` | Model for cross-model critique |
| `model.classifier` | `openai/gpt-4o-mini` | Model for critique quality classification |

### Output Structure

```
outputs/<timestamp>/
  summary.json                          # Aggregate metrics across all runs
  config_snapshot.json                   # Frozen configuration
  runs/<task>/seed_<N>/
    no_correction__targeted.jsonl        # Per-sample logs
    same_model__targeted.jsonl
    cross_model__targeted.jsonl
    same_model__generic.jsonl
  figures/                               # Generated plots
```
