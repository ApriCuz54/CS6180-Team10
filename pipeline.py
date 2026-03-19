# ============================================================
# CS6180 Final Project — Shared Pipeline
# GENERATE → CRITIQUE → REVISE
#
# Import this module in each benchmark notebook.
# Do NOT run directly — use the benchmark notebooks instead.
# ============================================================

import os, time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Model config ─────────────────────────────────────────────
GENERATOR_MODEL  = "openai/gpt-4o-mini"
CRITIC_MODEL     = "openai/gpt-4o"
CLASSIFIER_MODEL = "openai/gpt-4o-mini"

# ── Reproducibility ──────────────────────────────────────────
SEED        = 42
NUM_SAMPLES = 250


# ── Client — initialized with key passed in from notebook ────
def init_client(api_key: str) -> OpenAI:
    """Call this from your benchmark notebook after loading your API key."""
    global client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


# ── Prompt Templates ─────────────────────────────────────────
def make_generate_prompt(task_description: str, question: str, answer_format: str) -> str:
    return f"""You are solving a {task_description}.

Question:
{question}

Instructions:
- Think step by step.
- {answer_format}
- Do not add any text after your final answer."""


def make_critique_prompt(task_description: str, question: str,
                          original_answer: str, critique_style: str = "targeted") -> str:
    if critique_style == "targeted":
        instruction = (
            "Identify specific errors in the answer above. "
            "For each error, explain what is wrong and why. "
            "If the answer is fully correct, say 'No errors found.'"
        )
    else:  # generic — used for ablation
        instruction = (
            "Review the answer above. "
            "Is it correct? If not, what could be improved?"
        )

    return f"""You are evaluating a solution to a {task_description}.

Question:
{question}

Answer to critique:
{original_answer}

Task:
{instruction}

Your critique:"""


def make_revise_prompt(task_description: str, question: str,
                        original_answer: str, critique: str, answer_format: str) -> str:
    return f"""You are solving a {task_description}.

Question:
{question}

Your previous answer:
{original_answer}

Critique of your answer:
{critique}

Instructions:
- Use the critique to fix any errors.
- If the critique says there are no errors, you may keep your answer.
- {answer_format}
- Do not add any text after your final answer.

Revised answer:"""


def make_classify_critique_prompt(critique: str) -> str:
    return f"""Classify the following critique into exactly one of these categories:
- accurate: the critique correctly identifies a real error
- vague: the critique is non-specific or unhelpful
- misleading: the critique incorrectly flags a correct answer or introduces wrong information

Critique:
{critique}

Respond with exactly one word: accurate, vague, or misleading."""


# ── Core API Call ─────────────────────────────────────────────
def call_model(prompt: str, model: str, max_tokens: int = 512,
               retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[WARN] Attempt {attempt+1} failed ({e}). Retrying in {wait}s...")
            time.sleep(wait)
    print(f"[ERROR] Model call failed after {retries} attempts.")
    return "[ERROR]"


# ── Pipeline Runner ───────────────────────────────────────────
def run_pipeline(
    samples: list[dict],
    task_description: str,
    answer_format: str,
    extract_answer_fn,
    evaluate_fn,
    condition: str,
    critique_style: str = "targeted",
) -> pd.DataFrame:
    assert condition in ("no_correction", "same_model", "cross_model")
    records = []

    for s in tqdm(samples, desc=f"{condition} | {task_description}"):
        qid      = s["id"]
        question = s["question"]
        gold     = s["answer"]

        gen_prompt  = make_generate_prompt(task_description, question, answer_format)
        raw_gen     = call_model(gen_prompt, GENERATOR_MODEL)
        pred_gen    = extract_answer_fn(raw_gen)
        correct_gen = evaluate_fn(pred_gen, gold)

        critique = critique_class = raw_rev = pred_rev = correct_rev = None

        if condition in ("same_model", "cross_model"):
            critic_model = GENERATOR_MODEL if condition == "same_model" else CRITIC_MODEL

            crit_prompt = make_critique_prompt(
                task_description, question, raw_gen, critique_style
            )
            critique  = call_model(crit_prompt, critic_model, max_tokens=256)

            rev_prompt = make_revise_prompt(
                task_description, question, raw_gen, critique, answer_format
            )
            raw_rev     = call_model(rev_prompt, GENERATOR_MODEL)
            pred_rev    = extract_answer_fn(raw_rev)
            correct_rev = evaluate_fn(pred_rev, gold)

            cls_prompt     = make_classify_critique_prompt(critique)
            critique_class = call_model(cls_prompt, CLASSIFIER_MODEL, max_tokens=10)

        transition = None
        if condition != "no_correction":
            transition = f"{'correct' if correct_gen else 'wrong'}→{'correct' if correct_rev else 'wrong'}"

        records.append({
            "id": qid, "question": question, "gold": gold,
            "condition": condition, "critique_style": critique_style,
            "raw_gen": raw_gen, "pred_gen": pred_gen, "correct_gen": correct_gen,
            "critique": critique, "critique_class": critique_class,
            "raw_rev": raw_rev, "pred_rev": pred_rev, "correct_rev": correct_rev,
            "transition": transition,
        })

    return pd.DataFrame(records)


# ── Summary Statistics ────────────────────────────────────────
def summarize(df: pd.DataFrame) -> dict:
    out = {"n": len(df), "gen_accuracy": df["correct_gen"].mean()}
    if df["condition"].iloc[0] != "no_correction":
        out["rev_accuracy"]    = df["correct_rev"].mean()
        out["accuracy_delta"]  = out["rev_accuracy"] - out["gen_accuracy"]
        out["transitions"]     = df["transition"].value_counts().to_dict()
        out["critique_quality"] = df["critique_class"].value_counts().to_dict()
    return out