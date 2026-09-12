from __future__ import annotations

from collections import Counter
from typing import Any


def compute_transition(before_correct: bool, after_correct: bool) -> str:
    left = "correct" if before_correct else "wrong"
    right = "correct" if after_correct else "wrong"
    return f"{left}->{right}"


def compute_ncv(transitions: dict[str, int]) -> dict[str, float | None]:
    """Compute Fix Rate, Break Rate, and Net Correction Value from transition counts.

    NCV = Fix Rate - Break Rate * (p_correct / p_wrong)

    The weighting factor adjusts for base-rate asymmetry: when baseline accuracy
    is high, even a low Break Rate affects many problems in absolute terms.
    """
    cc = transitions.get("correct->correct", 0)
    cw = transitions.get("correct->wrong", 0)
    wc = transitions.get("wrong->correct", 0)
    ww = transitions.get("wrong->wrong", 0)

    baseline_correct = cc + cw
    baseline_wrong = wc + ww

    fix_rate: float | None = None
    break_rate: float | None = None
    ncv: float | None = None

    if baseline_wrong > 0:
        fix_rate = wc / baseline_wrong
    if baseline_correct > 0:
        break_rate = cw / baseline_correct

    if fix_rate is not None and break_rate is not None and baseline_wrong > 0:
        p_correct = baseline_correct / (baseline_correct + baseline_wrong)
        p_wrong = baseline_wrong / (baseline_correct + baseline_wrong)
        ncv = fix_rate - break_rate * (p_correct / p_wrong)

    return {
        "fix_rate": fix_rate,
        "break_rate": break_rate,
        "ncv": ncv,
    }


def critique_quality_correlation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Break down revision outcomes by critique class (accurate/vague/misleading).

    For each class, returns count, revision_accuracy, wrong_to_correct rate,
    and correct_to_wrong rate.
    """
    by_class: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        cc = r.get("critique_class")
        if cc is None:
            continue
        by_class.setdefault(cc, []).append(r)

    if not by_class:
        return {}

    table: dict[str, Any] = {}
    for cls, group in sorted(by_class.items()):
        n = len(group)
        rev_correct = sum(1 for r in group if r.get("correct_rev"))
        w2c = sum(1 for r in group if r.get("transition") == "wrong->correct")
        c2w = sum(1 for r in group if r.get("transition") == "correct->wrong")
        table[cls] = {
            "count": n,
            "revision_accuracy": rev_correct / n,
            "wrong_to_correct": w2c / n,
            "correct_to_wrong": c2w / n,
        }

    return table


def per_iteration_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accuracy and transition counts at each iteration depth.

    Only useful when max_iterations > 1. Returns a list ordered by iteration
    number with accuracy, transition counts, and critique class counts.
    """
    max_depth = 0
    for r in records:
        iters = r.get("iterations")
        if iters:
            max_depth = max(max_depth, len(iters))

    if max_depth == 0:
        return []

    result: list[dict[str, Any]] = []
    for depth in range(max_depth):
        correct = 0
        total = 0
        transitions: Counter[str] = Counter()
        critiques: Counter[str] = Counter()

        for r in records:
            iters = r.get("iterations")
            if not iters or depth >= len(iters):
                continue
            step = iters[depth]
            total += 1
            if step.get("correct_rev"):
                correct += 1
            if step.get("transition"):
                transitions[step["transition"]] += 1
            if step.get("critique_class"):
                critiques[step["critique_class"]] += 1

        result.append({
            "iteration": depth + 1,
            "accuracy": correct / max(total, 1),
            "n": total,
            "transitions": dict(transitions),
            "critique_quality": dict(critiques),
        })

    return result


def iteration_correlation(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Critique-quality correlation at each iteration depth.

    Returns one correlation table per iteration so you can see if later
    rounds have weaker signal.
    """
    max_depth = 0
    for r in records:
        iters = r.get("iterations")
        if iters:
            max_depth = max(max_depth, len(iters))

    if max_depth == 0:
        return []

    result: list[dict[str, Any]] = []
    for depth in range(max_depth):
        pseudo: list[dict[str, Any]] = []
        for r in records:
            iters = r.get("iterations")
            if not iters or depth >= len(iters):
                continue
            pseudo.append(iters[depth])

        table = critique_quality_correlation(pseudo)
        result.append({"iteration": depth + 1, "correlation": table})

    return result


def aggregate_run_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean and std across multiple per-seed run summaries."""
    if not summaries:
        return {}

    if len(summaries) == 1:
        result = dict(summaries[0])
        result["num_runs"] = 1
        return result

    result: dict[str, Any] = {
        "num_runs": len(summaries),
        "n": summaries[0].get("n", 0),
    }

    # Aggregate scalar numeric fields
    scalar_keys = [
        "gen_accuracy", "rev_accuracy", "accuracy_delta",
        "gen_mean_f1", "rev_mean_f1",
    ]
    for key in scalar_keys:
        values = [s[key] for s in summaries if key in s and s[key] is not None]
        if values:
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            result[key] = round(mean, 6)
            result[f"{key}_std"] = round(std, 6)

    # Aggregate ncv_metrics
    ncv_keys = ["fix_rate", "break_rate", "ncv"]
    ncv_dicts = [s.get("ncv_metrics", {}) for s in summaries]
    if any(ncv_dicts):
        agg_ncv: dict[str, Any] = {}
        for key in ncv_keys:
            values = [d[key] for d in ncv_dicts if d.get(key) is not None]
            if values:
                mean = sum(values) / len(values)
                std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
                agg_ncv[key] = round(mean, 6)
                agg_ncv[f"{key}_std"] = round(std, 6)
        if agg_ncv:
            result["ncv_metrics"] = agg_ncv

    # Sum transition counts across runs
    total_transitions: Counter[str] = Counter()
    for s in summaries:
        for k, v in (s.get("transitions") or {}).items():
            total_transitions[k] += v
    if total_transitions:
        result["transitions"] = dict(total_transitions)

    # Sum critique quality counts across runs
    total_critiques: Counter[str] = Counter()
    for s in summaries:
        for k, v in (s.get("critique_quality") or {}).items():
            total_critiques[k] += v
    if total_critiques:
        result["critique_quality"] = dict(total_critiques)

    # Keep per-run details for traceability
    result["per_run"] = summaries

    return result


def _extract_f1_scores(records: list[dict[str, Any]], note_field: str) -> list[float]:
    """Extract F1 values from eval notes like 'f1=0.5342'."""
    scores: list[float] = []
    for r in records:
        note = r.get(note_field)
        if note and isinstance(note, str) and note.startswith("f1="):
            try:
                scores.append(float(note[3:]))
            except ValueError:
                pass
    return scores


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0, "gen_accuracy": 0.0}

    gen_acc = sum(1 for r in records if bool(r.get("correct_gen"))) / n
    summary: dict[str, Any] = {"n": n, "gen_accuracy": gen_acc}

    gen_f1s = _extract_f1_scores(records, "eval_note_gen")
    if gen_f1s:
        summary["gen_mean_f1"] = sum(gen_f1s) / len(gen_f1s)

    has_revision = any(r.get("correct_rev") is not None for r in records)
    if has_revision:
        rev_vals = [bool(r.get("correct_rev")) for r in records if r.get("correct_rev") is not None]
        rev_acc = sum(1 for x in rev_vals if x) / max(len(rev_vals), 1)
        summary["rev_accuracy"] = rev_acc
        summary["accuracy_delta"] = rev_acc - gen_acc

        rev_f1s = _extract_f1_scores(records, "eval_note_rev")
        if rev_f1s:
            summary["rev_mean_f1"] = sum(rev_f1s) / len(rev_f1s)

        transitions = Counter(r["transition"] for r in records if r.get("transition"))
        critiques = Counter(r["critique_class"] for r in records if r.get("critique_class"))
        summary["transitions"] = dict(transitions)
        summary["ncv_metrics"] = compute_ncv(dict(transitions))
        summary["critique_quality"] = dict(critiques)

        corr = critique_quality_correlation(records)
        if corr:
            summary["critique_quality_correlation"] = corr

    # per-iteration breakdown (only when max_iterations > 1)
    has_iterations = any(r.get("iterations") for r in records)
    if has_iterations:
        summary["per_iteration"] = per_iteration_metrics(records)
        iter_corr = iteration_correlation(records)
        if iter_corr:
            summary["per_iteration_correlation"] = iter_corr

    return summary
