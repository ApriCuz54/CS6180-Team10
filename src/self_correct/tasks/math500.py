from __future__ import annotations

import json
import re
from pathlib import Path


class MATH500Adapter:
    name = "math500"
    task_description = "competition-level mathematical reasoning task"
    answer_format = "End with exactly one line: FINAL_ANSWER: <your answer>"

    def load_samples(self, data_path: str, sample_limit: int | None = None) -> list[dict]:
        path = Path(data_path)
        samples: list[dict] = []

        if path.suffix == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            for idx, row in enumerate(rows):
                samples.append(self._parse_row(row, idx))
                if sample_limit and len(samples) >= sample_limit:
                    break
        else:
            with path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    row = json.loads(line)
                    samples.append(self._parse_row(row, idx))
                    if sample_limit and len(samples) >= sample_limit:
                        break

        return samples

    def _parse_row(self, row: dict, idx: int) -> dict:
        question = row.get("problem", row.get("question", ""))
        # Gold answer: prefer pre-extracted "answer" field, else parse \boxed{} from "solution"
        gold = row.get("answer", "")
        if not gold and "solution" in row:
            gold = self._extract_boxed(row["solution"])
        if not gold:
            gold = row.get("answer_text", "")

        sample: dict = {
            "id": str(row.get("id", idx)),
            "question": question,
            "answer": self._normalize_answer(gold),
        }
        # Preserve difficulty metadata when available
        if "level" in row:
            sample["level"] = row["level"]
        if "type" in row:
            sample["subject"] = row["type"]
        elif "subject" in row:
            sample["subject"] = row["subject"]

        return sample

    def extract_answer(self, raw_text: str) -> str:
        # Try FINAL_ANSWER: pattern first
        match = re.search(r"FINAL_ANSWER:\s*(.+)", raw_text)
        if match:
            return self._normalize_answer(match.group(1).strip())

        # Try \boxed{} pattern
        boxed = self._extract_boxed(raw_text)
        if boxed:
            return self._normalize_answer(boxed)

        # Fallback: last number in the text
        nums = re.findall(r"[-+]?\d*\.?\d+", raw_text)
        if nums:
            return self._normalize_answer(nums[-1].strip())

        return self._normalize_answer(raw_text.strip())

    def evaluate(self, prediction: str, gold: str, sample: dict | None = None) -> bool:
        pred_norm = self._normalize_answer(prediction)
        gold_norm = self._normalize_answer(gold)

        if pred_norm == gold_norm:
            return True

        # Try numeric comparison as fallback
        try:
            return float(pred_norm) == float(gold_norm)
        except (ValueError, OverflowError):
            return False

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Extract the contents of the last \\boxed{...} in *text*, handling nested braces."""
        # Find all \boxed{ positions, take the last one
        idx = text.rfind("\\boxed{")
        if idx == -1:
            idx = text.rfind("\\boxed ")
            if idx == -1:
                return ""
            # Simple case: \boxed followed by a single token
            rest = text[idx + len("\\boxed "):]
            return rest.split()[0] if rest.split() else ""

        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            return text[start : pos - 1].strip()
        return ""

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize a math answer string for comparison."""
        text = text.strip()
        # Remove surrounding $...$
        if text.startswith("$") and text.endswith("$"):
            text = text[1:-1].strip()
        # Remove \text{...} wrappers
        text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
        # Remove \mathrm{...} wrappers
        text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
        # Remove \left and \right
        text = text.replace("\\left", "").replace("\\right", "")
        # Normalize \frac{a}{b} to a/b
        text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
        # Normalize \dfrac similarly
        text = re.sub(r"\\dfrac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
        # Remove common LaTeX spacing commands
        text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "")
        text = text.replace("\\quad", " ").replace("\\qquad", " ")
        # Remove trailing period
        text = text.rstrip(".")
        # Collapse whitespace
        text = " ".join(text.split())
        return text
