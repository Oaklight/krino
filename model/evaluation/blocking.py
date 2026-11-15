"""The blocking readout's design, without torch (#356).

Split the K options into ``ceil(K / b)`` blocks of ``b`` options and combine
the block winners: scheme A compares the winners' raw code logits, B runs one
panel-2 prompt over the winners, C scores each winner by per-option judgement.

Everything that decides which options a block holds and whether a scheme is
correct lives here, so the scripts that run the model and the tests share one
definition. The module imports no torch, so the tests run without a GPU.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any

K = 150
SEED = 342
BLOCK_SIZES = (52, 32, 26)


def blocks_of(k: int, size: int) -> list[list[int]]:
    """Consecutive slices of ``range(k)``; the last block may be short.

    Every index appears exactly once and no index appears twice, which is the
    property the whole readout rests on: an option dropped here is
    unrecoverable and an option duplicated here can win twice.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if size <= 0:
        raise ValueError(f"block size must be positive, got {size}")
    return [list(range(start, min(start + size, k))) for start in range(0, k, size)]


def shuffled_order(item_id: str, k: int = K, seed: int = SEED) -> list[int]:
    """A per-item permutation of ``range(k)`` keyed on the item id.

    The repository's item-seed convention, ``random.Random(f"{seed}:{item_id}")``,
    so a permutation is reproducible from the id alone and independent of how
    many items the caller keeps.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    order = list(range(k))
    random.Random(f"{seed}:{item_id}").shuffle(order)
    return order


def shuffled_blocks(
    item_id: str, size: int, k: int = K, seed: int = SEED
) -> list[list[int]]:
    """Consecutive slices of the per-item permutation."""
    if size <= 0:
        raise ValueError(f"block size must be positive, got {size}")
    order = shuffled_order(item_id, k, seed)
    return [order[start : start + size] for start in range(0, k, size)]


def gold_block(blocks: Sequence[Sequence[int]], gold: int) -> int:
    """Index of the block holding the gold; raises if the gold is absent."""
    for index, block in enumerate(blocks):
        if gold in block:
            return index
    raise ValueError(f"gold {gold} is in no block")


def position_bin(local: int, width: int) -> str:
    """The gold's position in its block: first 20%, last 20%, or middle."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if not 0 <= local < width:
        raise ValueError(f"local {local} outside [0, {width})")
    if local < 0.2 * width:
        return "first"
    if local >= 0.8 * width:
        return "last"
    return "middle"


def scheme_a_correct(gold: int, winners: Sequence[int], logits: Sequence[float]) -> bool:
    """Scheme A: the winner with the highest raw logit, ties to the earlier block."""
    if len(winners) != len(logits):
        raise ValueError(f"{len(winners)} winners for {len(logits)} logits")
    pick = winners[max(range(len(logits)), key=lambda i: (logits[i], -i))]
    return pick == gold


def scheme_b_correct(
    gold: int, winners: Sequence[int], final_pick: int
) -> bool:
    """Scheme B: the gold must have won its block, and the final pass must pick it.

    The first clause is the structural claim of blocking: an option that loses
    its block is not in the final list, so no final pass can recover it. A
    final pass that returned the gold anyway would still be wrong.
    """
    return gold in winners and final_pick == gold


def decompose(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The two factors: block-win rate and final accuracy given a block win.

    Each row carries ``won_block`` and ``correct``. The product of the factors
    reproduces the end-to-end count by construction, which is what makes it a
    decomposition rather than a description.
    """
    rows = list(rows)
    n = len(rows)
    won = sum(1 for row in rows if row["won_block"])
    final = sum(1 for row in rows if row["won_block"] and row["correct"])
    return {
        "items": n,
        "won_block": won,
        "won_block_rate": won / n if n else 0.0,
        "final_given_won": final / won if won else 0.0,
        "product": final / n if n else 0.0,
    }
