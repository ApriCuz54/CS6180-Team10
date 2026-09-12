from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path


class HotpotQAAdapter:
    name = "hotpotqa"
    task_description = "multi-hop question answering task"
    answer_format = "End with exactly one line: FINAL_ANSWER: <short answer phrase>"

    def load_samples(self, data_path: str, sample_limit: int | None = None) -> list[dict]:
        path = Path(data_path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        samples = []
        for idx, row in enumerate(rows):
            samples.append(
                {
                    "id": str(row.get("_id", idx)),
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )
            if sample_limit and len(samples) >= sample_limit:
                break
        return samples

    def extract_answer(self, raw_text: str) -> str:
        match = re.search(r"FINAL_ANSWER:\s*(.+)", raw_text)
        if match:
            return match.group(1).strip()
        return raw_text.strip().splitlines()[-1].strip() if raw_text.strip() else ""

    def evaluate(self, prediction: str, gold: str, sample: dict | None = None) -> tuple[bool, str]:
        em = self.exact_match(prediction, gold)
        f1 = self.f1_score(prediction, gold)
        return em, f"f1={f1:.4f}"

    def metrics(self, prediction: str, gold: str) -> dict[str, float]:
        return {
            "em": 1.0 if self.exact_match(prediction, gold) else 0.0,
            "f1": self.f1_score(prediction, gold),
        }

    def exact_match(self, prediction: str, gold: str) -> bool:
        return self._normalize(prediction) == self._normalize(gold)

    def f1_score(self, prediction: str, gold: str) -> float:
        pred_tokens = self._normalize(prediction).split()
        gold_tokens = self._normalize(gold).split()
        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())
