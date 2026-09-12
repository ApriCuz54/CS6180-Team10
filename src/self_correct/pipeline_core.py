from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from .metrics import compute_transition
from .prompts import (
    make_classify_critique_prompt,
    make_critique_prompt,
    make_generate_prompt,
    make_revise_prompt,
)


VALID_CONDITIONS = ("no_correction", "same_model", "cross_model")


def run_condition(
    samples: list[dict[str, Any]],
    task_description: str,
    answer_format: str,
    extract_answer_fn,
    evaluate_fn,
    evaluator_name: str,
    condition: str,
    critique_style: str,
    client,
    generator_model: str,
    critic_model: str,
    classifier_model: str,
    max_iterations: int = 1,
) -> list[dict[str, Any]]:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition: {condition}")

    records: list[dict[str, Any]] = []
    sample_bar = tqdm(samples, desc=f"{condition} | {task_description}", leave=False, dynamic_ncols=True)
    for sample_index, s in enumerate(sample_bar, start=1):
        sample_start = time.perf_counter()
        qid = s["id"]
        question = s["question"]
        gold = s["answer"]

        # generate
        tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)}: generating answer")
        gen_start = time.perf_counter()
        gen_prompt = make_generate_prompt(task_description, question, answer_format)
        raw_gen = client.complete(gen_prompt, generator_model, max_tokens=768)
        pred_gen = extract_answer_fn(raw_gen)
        correct_gen, eval_note_gen = _run_eval(evaluate_fn, pred_gen, gold, s)
        tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)}: generation done in {time.perf_counter() - gen_start:.1f}s")

        # Default values for no_correction
        iterations: list[dict[str, Any]] = []
        critique = None
        critique_class = None
        raw_rev = None
        pred_rev = None
        correct_rev = None
        eval_note_rev = None
        transition = None

        # critique -> revise loop
        if condition in ("same_model", "cross_model"):
            chosen_critic = generator_model if condition == "same_model" else critic_model
            current_answer = raw_gen
            prev_correct = correct_gen

            for iteration in range(1, max_iterations + 1):
                iter_label = f"iter {iteration}/{max_iterations}"

                # critique
                tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} {iter_label}: critiquing")
                crit_start = time.perf_counter()
                crit_prompt = make_critique_prompt(
                    task_description=task_description,
                    question=question,
                    original_answer=current_answer,
                    critique_style=critique_style,
                )
                iter_critique = client.complete(crit_prompt, chosen_critic, max_tokens=384)
                tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} {iter_label}: critique done in {time.perf_counter() - crit_start:.1f}s")

                # revise
                tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} {iter_label}: revising")
                rev_start = time.perf_counter()
                rev_prompt = make_revise_prompt(
                    task_description=task_description,
                    question=question,
                    original_answer=current_answer,
                    critique=iter_critique,
                    answer_format=answer_format,
                )
                iter_raw_rev = client.complete(rev_prompt, generator_model, max_tokens=768)
                iter_pred_rev = extract_answer_fn(iter_raw_rev)
                iter_correct_rev, iter_eval_note = _run_eval(evaluate_fn, iter_pred_rev, gold, s)
                iter_transition = compute_transition(prev_correct, iter_correct_rev)
                tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} {iter_label}: revision done in {time.perf_counter() - rev_start:.1f}s")

                # classify critique
                tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} {iter_label}: classifying critique")
                cls_start = time.perf_counter()
                cls_prompt = make_classify_critique_prompt(iter_critique)
                iter_critique_class = client.complete(cls_prompt, classifier_model, max_tokens=8).lower().strip()
                tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} {iter_label}: classification done in {time.perf_counter() - cls_start:.1f}s")

                iterations.append({
                    "iteration": iteration,
                    "critique": iter_critique,
                    "critique_class": iter_critique_class,
                    "raw_rev": iter_raw_rev,
                    "pred_rev": iter_pred_rev,
                    "correct_rev": iter_correct_rev,
                    "eval_note_rev": iter_eval_note,
                    "transition": iter_transition,
                })

                # feed this revision into the next round
                current_answer = iter_raw_rev
                prev_correct = iter_correct_rev

            # top-level fields = last iteration's values
            final = iterations[-1]
            critique = final["critique"]
            critique_class = final["critique_class"]
            raw_rev = final["raw_rev"]
            pred_rev = final["pred_rev"]
            correct_rev = final["correct_rev"]
            eval_note_rev = final["eval_note_rev"]
            transition = compute_transition(correct_gen, correct_rev)

        record: dict[str, Any] = {
            "id": qid,
            "question": question,
            "gold": gold,
            "evaluator": evaluator_name,
            "condition": condition,
            "critique_style": critique_style,
            "raw_gen": raw_gen,
            "pred_gen": pred_gen,
            "correct_gen": correct_gen,
            "eval_note_gen": eval_note_gen,
            "critique": critique,
            "critique_class": critique_class,
            "raw_rev": raw_rev,
            "pred_rev": pred_rev,
            "correct_rev": correct_rev,
            "eval_note_rev": eval_note_rev,
            "transition": transition,
        }

        if iterations:
            record["iterations"] = iterations

        records.append(record)
        tqdm.write(f"[{condition}] sample {sample_index}/{len(samples)} complete in {time.perf_counter() - sample_start:.1f}s")

    return records


def _run_eval(evaluate_fn, prediction: str, gold: str, sample: dict[str, Any]) -> tuple[bool, str | None]:
    result = evaluate_fn(prediction, gold, sample)
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), (str(result[1]) if result[1] is not None else None)
    return bool(result), None
