from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


class HumanEvalAdapter:
    name = "humaneval"
    task_description = "Python coding task"
    answer_format = "Return only valid Python code for the required function."

    def load_samples(self, data_path: str, sample_limit: int | None = None) -> list[dict]:
        path = Path(data_path)
        rows: list[dict] = []

        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                prompt = row.get("prompt", "")
                entry_point = row.get("entry_point", "")
                question = (
                    "Write a Python function that satisfies the prompt below.\n\n"
                    f"Prompt:\n{prompt}\n\n"
                    f"The function name must be: {entry_point}"
                )
                rows.append(
                    {
                        "id": str(row.get("task_id", idx)),
                        "question": question,
                        "answer": entry_point,
                        "test": row.get("test", ""),
                        "prompt": prompt,
                    }
                )
                if sample_limit and len(rows) >= sample_limit:
                    break

        return rows

    def extract_answer(self, raw_text: str) -> str:
        fenced = re.search(r"```(?:python)?\n(.*?)```", raw_text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return raw_text.strip()

    def evaluate(self, prediction: str, gold: str, sample: dict | None = None) -> bool:
        # gold is the entry_point name
        if f"def {gold}(" not in prediction:
            return False

        # If no test code available, fall back to structural check
        if sample is None or not sample.get("test"):
            return True

        # Build executable program: generated code + test harness
        code = prediction
        # If the prediction lacks imports from the original prompt, prepend them
        prompt = sample.get("prompt", "")
        if prompt:
            # Extract import lines from the prompt
            import_lines = [
                line for line in prompt.splitlines()
                if line.strip().startswith(("import ", "from "))
            ]
            if import_lines:
                imports = "\n".join(import_lines) + "\n\n"
                # Only prepend if the prediction doesn't already have them
                if import_lines[0] not in prediction:
                    code = imports + code

        full_code = code + "\n\n" + sample["test"]

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
