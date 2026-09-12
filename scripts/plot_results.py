"""Generate plots from a summary.json produced by the experiment runner.

Usage:
    python scripts/plot_results.py outputs/20260414_120000/summary.json
    python scripts/plot_results.py outputs/20260414_120000/summary.json --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument("summary", help="Path to summary.json.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write charts. Defaults to summary file's parent.",
    )
    return parser.parse_args()


# -- helpers ------------------------------------------------------------------

def _flatten_task_results(summary: dict) -> list[dict]:
    rows: list[dict] = []
    tasks = summary.get("tasks", {})
    for task_name, runs in tasks.items():
        for run_name, metrics in runs.items():
            condition = run_name.split("__", 1)[0]
            row = {"task": task_name, "run": run_name, "condition": condition}
            row.update(metrics)
            rows.append(row)
    return rows


def _label_for_run(run_name: str) -> str:
    run_name = run_name.replace("__", "|")
    mapping = {
        "no_correction|targeted": "No\nCorrection",
        "same_model|targeted": "Same-Model\nTargeted",
        "cross_model|targeted": "Cross-Model\nTargeted",
        "same_model|generic": "Same-Model\nGeneric",
    }
    return mapping.get(run_name, run_name.replace("|", "\n"))


_SORT_ORDER = {
    "no_correction__targeted": 0,
    "same_model__targeted": 1,
    "cross_model__targeted": 2,
    "same_model__generic": 3,
}


def _sort_key(run_name: str) -> tuple[int, str]:
    return _SORT_ORDER.get(run_name, 99), run_name


# -- plot: accuracy bars ------------------------------------------------------

def plot_accuracy(rows: list[dict], out_path: Path, task_name: str) -> None:
    ordered = sorted(rows, key=lambda r: _sort_key(r["run"]))
    labels = [_label_for_run(r["run"]) for r in ordered]
    accuracies = [r.get("rev_accuracy", r.get("gen_accuracy", 0.0)) for r in ordered]
    baseline = next(
        (r.get("gen_accuracy", 0.0) for r in ordered if r["run"].startswith("no_correction")),
        accuracies[0],
    )
    n_value = max((int(r.get("n", 0)) for r in ordered), default=0)

    colors = ["#95A5A6", "#2ECC71", "#3498DB", "#E67E22", "#9B59B6", "#E74C3C"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(ordered)), accuracies,
        color=colors[: len(ordered)], edgecolor="black", linewidth=0.6, width=0.5,
    )
    ax.axhline(baseline, color="red", linestyle="--", linewidth=1.2, label="Baseline")
    ax.set_ylim(0, max(0.5, max(accuracies + [baseline]) + 0.08))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy ({task_name}, n={n_value})")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bar, value in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.008,
            f"{value:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -- plot: transition counts --------------------------------------------------

def plot_transitions(rows: list[dict], out_path: Path) -> None:
    transition_order = ["correct->correct", "wrong->correct", "correct->wrong", "wrong->wrong"]
    counts = {k: 0 for k in transition_order}
    for row in rows:
        for key, value in (row.get("transitions") or {}).items():
            counts[key] = counts.get(key, 0) + int(value)

    bar_colors = ["#2ECC71", "#3498DB", "#E74C3C", "#95A5A6"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.keys(), counts.values(), color=bar_colors)
    ax.set_title("Error Transition Counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -- plot: critique quality correlation ---------------------------------------

def plot_critique_correlation(rows: list[dict], out_path: Path) -> None:
    """Bar chart: revision accuracy and wrong->correct rate per critique class."""
    # merge correlation tables across all runs that have one
    merged: dict[str, dict] = {}
    for row in rows:
        corr = row.get("critique_quality_correlation")
        if not corr:
            continue
        for cls, stats in corr.items():
            if cls not in merged:
                merged[cls] = {"rev_correct": 0, "w2c": 0, "c2w": 0, "count": 0}
            merged[cls]["rev_correct"] += stats["revision_accuracy"] * stats["count"]
            merged[cls]["w2c"] += stats["wrong_to_correct"] * stats["count"]
            merged[cls]["c2w"] += stats["correct_to_wrong"] * stats["count"]
            merged[cls]["count"] += stats["count"]

    if not merged:
        return

    classes = sorted(merged.keys())
    rev_acc = [merged[c]["rev_correct"] / merged[c]["count"] for c in classes]
    w2c = [merged[c]["w2c"] / merged[c]["count"] for c in classes]
    c2w = [merged[c]["c2w"] / merged[c]["count"] for c in classes]
    counts = [merged[c]["count"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, rev_acc, width, label="Revision Accuracy", color="#2ECC71")
    ax.bar(x, w2c, width, label="Wrong->Correct", color="#3498DB")
    ax.bar(x + width, c2w, width, label="Correct->Wrong", color="#E74C3C")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(classes, counts)])
    ax.set_ylabel("Rate")
    ax.set_title("Revision Outcome by Critique Quality")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -- plot: per-iteration accuracy line ----------------------------------------

def plot_iteration_accuracy(rows: list[dict], out_path: Path) -> None:
    """Line chart: accuracy at each iteration depth (multi-iteration runs only)."""
    # find per_iteration data from any row that has it
    all_iters: list[dict] | None = None
    gen_acc = None
    for row in rows:
        pi = row.get("per_iteration")
        if pi:
            all_iters = pi
            gen_acc = row.get("gen_accuracy")
            break

    if not all_iters:
        return

    iters = [step["iteration"] for step in all_iters]
    accs = [step["accuracy"] for step in all_iters]

    fig, ax = plt.subplots(figsize=(7, 4))

    if gen_acc is not None:
        ax.axhline(gen_acc, color="red", linestyle="--", linewidth=1.2, label=f"Gen baseline ({gen_acc:.3f})")

    ax.plot(iters, accs, marker="o", linewidth=2, color="#3498DB", label="Revision accuracy")
    for i, acc in zip(iters, accs):
        ax.annotate(f"{acc:.3f}", (i, acc), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Iteration")
    ax.set_xticks(iters)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -- main ---------------------------------------------------------------------

def generate_plots(summary_path: Path, out_dir: Path) -> None:
    """Generate all available plots from a summary.json file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if isinstance(summary, dict) and "rows" in summary:
        rows = summary["rows"]
    else:
        rows = _flatten_task_results(summary)

    if not rows:
        print("No data found in summary.json")
        return

    task_name = summary.get("task", summary_path.stem or "experiment")

    plot_accuracy(rows, out_dir / "accuracy_by_run.png", task_name)
    print(f"  wrote accuracy_by_run.png")

    # only plot transitions for runs that have them
    rev_rows = [r for r in rows if r.get("transitions")]
    if rev_rows:
        plot_transitions(rev_rows, out_dir / "transitions.png")
        print(f"  wrote transitions.png")

    # critique quality correlation
    corr_rows = [r for r in rows if r.get("critique_quality_correlation")]
    if corr_rows:
        plot_critique_correlation(corr_rows, out_dir / "critique_correlation.png")
        print(f"  wrote critique_correlation.png")

    # per-iteration accuracy (multi-iteration runs)
    iter_rows = [r for r in rows if r.get("per_iteration")]
    if iter_rows:
        plot_iteration_accuracy(iter_rows, out_dir / "iteration_accuracy.png")
        print(f"  wrote iteration_accuracy.png")

    print(f"All plots saved to {out_dir}")


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    generate_plots(summary_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
