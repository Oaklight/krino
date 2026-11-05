"""Top-m recall of the gold option, torch-free.

Shared by ``model/scripts/topm_recall.py`` and its tests.  These functions
define every number the diagnostic reports, so they live where the CI leg
without torch can import them.
"""

from __future__ import annotations

import math

MAX_M = 5


def gold_rank(probs: tuple[float, ...] | list[float], gold: int) -> int:
    """1-based rank of ``gold``; ties count against the gold."""
    return 1 + sum(1 for i, p in enumerate(probs) if i != gold and p >= probs[gold])


def k_bucket(k: int) -> str:
    return "3-5" if k <= 5 else "6+"


def summarize(rows: list[dict], max_m: int = MAX_M) -> dict:
    """Recall@1..max_m with binomial SE; ``None`` for an empty group."""
    out: dict = {"n": len(rows)}
    for m in range(1, max_m + 1):
        if not rows:
            out[f"recall@{m}"] = None
            out[f"se@{m}"] = None
            continue
        p = sum(r["rank"] <= m for r in rows) / len(rows)
        out[f"recall@{m}"] = round(p, 4)
        out[f"se@{m}"] = round(math.sqrt(p * (1 - p) / len(rows)), 4)
    return out
