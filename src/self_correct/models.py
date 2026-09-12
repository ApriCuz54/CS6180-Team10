from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class ChatClient(Protocol):
    def complete(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        ...


@dataclass
class OpenAICompatClient:
    api_key: str
    base_url: str
    temperature: float = 0.0
    retries: int = 3

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        del max_tokens
        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                time.sleep(2 ** attempt)
        return "[ERROR]"
