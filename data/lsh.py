"""Locality-Sensitive Hashing for near-duplicate detection.

MinHash on character n-grams with LSH banding for fast approximate
nearest-neighbor lookup. Pure Python, no external dependencies.

Usage::

    lsh = LSHIndex(num_perm=128, bands=16)

    # Build index
    for i, text in enumerate(documents):
        mh = lsh.make_minhash(text)
        lsh.insert(i, mh.signature)

    # Query
    query_mh = lsh.make_minhash("some new document")
    candidates = lsh.query(query_mh.signature)
    for c in candidates:
        sim = query_mh.jaccard(index_signatures[c])
        if sim >= threshold:
            print(f"Near-duplicate: {c} (sim={sim:.3f})")
"""

from __future__ import annotations

import hashlib
import random
import struct

_MERSENNE_PRIME = (1 << 61) - 1


def char_ngrams(text: str, n: int = 5) -> set[str]:
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
    """Locality-Sensitive Hashing index for fast near-duplicate lookup.

    Uses banding technique: the signature is split into `bands` bands of
    `rows` rows each. Two signatures that share at least one identical band
    are candidate duplicates. Increasing bands raises recall (more candidates)
    at the cost of precision; increasing rows does the opposite.

    For threshold ~0.5: use bands=32, rows=4 (128 perms)
    For threshold ~0.7: use bands=16, rows=8 (128 perms)
    For threshold ~0.9: use bands=8, rows=16 (128 perms)
    """

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
        """Insert a signature into the index."""
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

    def make_minhash(self, text: str, n: int = 5) -> MinHash:
        """Create a MinHash from text using character n-grams."""
        mh = MinHash(num_perm=self.num_perm, seed=self.seed)
        mh.update(char_ngrams(text, n))
        return mh

    def __len__(self) -> int:
        return len(self._signatures)
