"""LSH-based near-duplicate detection for synthetic data families.

Uses the standalone lsh module for MinHash/LSH, applies it to family
state text to find and remove near-duplicates.
"""

from __future__ import annotations

import logging
from typing import Any

from .lsh import LSHIndex

logger = logging.getLogger(__name__)


def dedup_families(
    families: list[dict[str, Any]],
    threshold: float = 0.7,
    num_perm: int = 128,
    bands: int = 16,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Remove near-duplicate families based on state text similarity.

    Returns (deduplicated_families, dropped_indices).
    """
    if not families:
        return [], []

    lsh = LSHIndex(num_perm=num_perm, bands=bands, seed=seed)
    kept: list[dict[str, Any]] = []
    dropped: list[int] = []
    signatures: list = []

    for idx, family in enumerate(families):
        state = family.get("state", "")
        mh = lsh.make_minhash(state)

        candidates = lsh.query(mh.signature)
        is_dup = False
        for cand_idx in candidates:
            sim = mh.jaccard(signatures[cand_idx])
            if sim >= threshold:
                is_dup = True
                logger.debug(
                    "Family %d is near-duplicate of %d (sim=%.3f): %.80s",
                    idx, cand_idx, sim, state,
                )
                break

        if is_dup:
            dropped.append(idx)
        else:
            lsh.insert(len(signatures), mh.signature)
            signatures.append(mh)
            kept.append(family)

    if dropped:
        logger.info(
            "Dedup: %d/%d families kept, %d dropped (threshold=%.2f)",
            len(kept), len(families), len(dropped), threshold,
        )
    else:
        logger.info("Dedup: all %d families are unique (threshold=%.2f)", len(families), threshold)

    return kept, dropped


def dedup_items(
    items: list[dict[str, Any]],
    threshold: float = 0.7,
    num_perm: int = 128,
    bands: int = 16,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], int]:
    """Remove near-duplicate items based on state text similarity.

    Groups items by state, deduplicates states, returns items whose state survived.
    Returns (deduplicated_items, num_dropped_items).
    """
    if not items:
        return [], 0

    state_to_items: dict[str, list[dict]] = {}
    for item in items:
        state = item.get("state", "")
        state_to_items.setdefault(state, []).append(item)

    unique_states = list(state_to_items.keys())
    pseudo_families = [{"state": s} for s in unique_states]

    kept_families, dropped_indices = dedup_families(
        pseudo_families, threshold, num_perm, bands, seed,
    )

    kept_states = {f["state"] for f in kept_families}
    kept_items = [item for item in items if item.get("state", "") in kept_states]
    num_dropped = len(items) - len(kept_items)

    return kept_items, num_dropped
