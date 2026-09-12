from __future__ import annotations

from .math500 import MATH500Adapter
from .hotpotqa import HotpotQAAdapter
from .humaneval import HumanEvalAdapter


def get_task_adapter(name: str):
    adapters = {
        "math500": MATH500Adapter(),
        "hotpotqa": HotpotQAAdapter(),
        "humaneval": HumanEvalAdapter(),
    }
    key = name.lower().strip()
    if key not in adapters:
        raise ValueError(f"Unsupported task adapter: {name}")
    return adapters[key]
