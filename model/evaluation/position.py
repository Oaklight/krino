"""The position probes' design, without torch (#357).

The probes bucket accuracy by the gold's natural rank and by its display rank,
permute the display order per item, and check the K=52 reproduction gate before
anything else is read. Those three pieces decide every number the probes
report, so they live here for the scripts and the tests to share.

The gate is the load-bearing one: if a draw drifts, the natural K=52 cell stops
matching the recorded cell and the run must stop loudly rather than be read as
a finding.
"""

from __future__ import annotations

import random

BUCKETS = 5
SEED = 342
ITEMS = 200
CELL = "natural_k52"
GATES = {"banking77": 72, "clinc150": 146}


def bucket(position: int, k: int) -> int:
    """One of five equal buckets, clipped to the last.

    Every position lands in exactly one bucket, including the last bucket when
    ``k`` is not a multiple of five.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 0 <= position < k:
        raise ValueError(f"position {position} outside [0, {k})")
    return min(BUCKETS - 1, position * BUCKETS // k)


def permuted(
    texts: list[str], gold: int, item_id: str, index: int, seed: int = SEED
) -> tuple[list[str], int]:
    """One seeded per-item permutation of the display order.

    The seed is keyed on ``(seed, item_id, index)``, so a permutation is
    reproducible from the item id alone, and different items get different
    permutations. The returned gold is the display position of the same text.
    """
    order = list(range(len(texts)))
    random.Random(f"{seed}:{item_id}:perm{index}").shuffle(order)
    return [texts[position] for position in order], order.index(gold)


def check_gate(
    rows: list[dict], dataset: str, expected: int | None = None
) -> dict:
    """The K=52 reproduction gate: the natural cell must match the record.

    A mismatch raises, so a drifted draw cannot be read as a result.
    """
    wanted = GATES[dataset] if expected is None else expected
    cell = [row for row in rows if row["cell"] == CELL]
    correct = sum(1 for row in cell if row["correct"])
    if len(cell) != ITEMS or correct != wanted:
        raise SystemExit(
            f"K=52 reproduction gate failed for {dataset}: "
            f"{correct}/{len(cell)} against the recorded {wanted}/{ITEMS}"
        )
    return {"items": len(cell), "correct": correct}
