"""LSH-based near-duplicate detection for synthetic data families.

Uses MinHash on character n-grams to find families with overly similar
states, then drops duplicates keeping the first occurrence.

Pure Python, no external dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Any

logger = logging.getLogger(__name__)

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _hash_ngram(ngram: str) -> int:
    """Hash an n-gram to a 32-bit integer."""
    return struct.unpack("<I", hashlib.md5(ngram.encode()).digest()[:4])[0]


class MinHash:
    """MinHash signature for approximate Jaccard similarity."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = num_perm
        import random
        rng = random.Random(seed)
        self._a = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self._hashvalues = [_MERSENNE_PRIME] * num_perm

    def update(self, ngrams: set[str]) -> None:
        for ngram in ngrams:
            h = _hash_ngram(ngram)
            for i in range(self.num_perm):
                val = (self._a[i] * h + self._b[i]) % _MERSENNE_PRIME
                if val < self._hashvalues[i]:
                    self._hashvalues[i] = val

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(self._hashvalues)

    def jaccard(self, other: MinHash) -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("MinHash num_perm mismatch")
        return sum(
            a == b for a, b in zip(self._hashvalues, other._hashvalues)
        ) / self.num_perm


class LSHIndex:
    """Locality-Sensitive Hashing index for fast near-duplicate lookup."""

    def __init__(self, num_perm: int = 128, bands: int = 16, seed: int = 42):
        if num_perm % bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.seed = seed
        self._buckets: list[dict[tuple[int, ...], list[int]]] = [
            {} for _ in range(bands)
        ]
        self._signatures: list[tuple[int, ...]] = []

    def _band_hashes(self, sig: tuple[int, ...]) -> list[tuple[int, ...]]:
        return [
            sig[i * self.rows:(i + 1) * self.rows]
            for i in range(self.bands)
        ]

    def insert(self, idx: int, sig: tuple[int, ...]) -> None:
        self._signatures.append(sig)
        for band_idx, band_hash in enumerate(self._band_hashes(sig)):
            self._buckets[band_idx].setdefault(band_hash, []).append(idx)

    def query(self, sig: tuple[int, ...]) -> set[int]:
        """Return indices of candidate near-duplicates."""
        candidates: set[int] = set()
        for band_idx, band_hash in enumerate(self._band_hashes(sig)):
            bucket = self._buckets[band_idx].get(band_hash, [])
            candidates.update(bucket)
        return candidates

    def make_minhash(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self.num_perm, seed=self.seed)
        mh.update(_char_ngrams(text))
        return mh


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
    signatures: list[MinHash] = []

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

    # Group items by state text
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
