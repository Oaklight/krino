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


def _rate(rows: list[dict], m: int) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    p = sum(r["rank"] <= m for r in rows) / len(rows)
    return round(p, 4), round(math.sqrt(p * (1 - p) / len(rows)), 4)


def summarize(rows: list[dict], max_m: int = MAX_M) -> dict:
    """Recall@1..max_m with binomial SE; ``None`` for an empty group.

    ``recall@m`` counts every row, so a row with K <= m is a free hit.
    ``informative@m`` keeps only rows with K > m, where the scorer can miss;
    ``n_informative@m`` is its denominator.  Rows without ``k`` count as
    informative.
    """
    out: dict = {"n": len(rows)}
    for m in range(1, max_m + 1):
        out[f"recall@{m}"], out[f"se@{m}"] = _rate(rows, m)
        informative = [r for r in rows if r.get("k", m + 1) > m]
        out[f"n_informative@{m}"] = len(informative)
        out[f"informative@{m}"], _ = _rate(informative, m)
    return out
