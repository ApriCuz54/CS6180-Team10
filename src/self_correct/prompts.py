from __future__ import annotations


def make_generate_prompt(task_description: str, question: str, answer_format: str) -> str:
    return f"""You are solving a {task_description}.

Question:
{question}

Instructions:
- Think step by step.
- {answer_format}
- Do not add any text after your final answer."""


def make_critique_prompt(
    task_description: str,
    question: str,
    original_answer: str,
    critique_style: str = "targeted",
) -> str:
    if critique_style == "targeted":
        instruction = (
            "Identify specific errors in the answer above. "
            "For each error, explain what is wrong and why. "
            "If the answer is fully correct, say 'No errors found.'"
        )
    else:
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


def make_revise_prompt(
    task_description: str,
    question: str,
    original_answer: str,
    critique: str,
    answer_format: str,
) -> str:
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
