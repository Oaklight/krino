"""Content-addressed option order (canonical order) for panel 2.

Sorting the options by ``sha256(text.strip())`` before rendering makes the
prompt a function of the option multiset, not of the order the caller supplied.
A natural and a reversed input then produce the same canonical sequence, so the
two prompts are byte-identical and the model cannot see which order was asked
for.

This removes variance to input order, not the model's position preference: the
model still prefers a position, but that position is now the same arbitrary
canonical slot on every run.  It is only valid where the option set is
unordered.  A score item's ``Level 0..k`` lines carry meaning in their order,
so canonical ordering must not be applied to them in production; the JevBench
run applies it uniformly only to measure whether accuracy drops.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def option_digest(text: str) -> bytes:
    """The content key an option is ordered by, whitespace-stripped."""
    return hashlib.sha256(text.strip().encode("utf-8")).digest()


def canonical_order(texts: Sequence[str]) -> list[int]:
    """The natural indices of the options in canonical display order.

    Sorting is stable on digest, so two options with identical stripped text
    keep their natural order; identical text also renders identically, so the
    prompt does not depend on which of the two a caller lists first.
    """
    return sorted(range(len(texts)), key=lambda index: option_digest(texts[index]))


def canonical_texts(texts: Sequence[str]) -> list[str]:
    """The option texts in canonical display order."""
    return [texts[index] for index in canonical_order(texts)]


def permutations_are_reversal_invariant(texts: Sequence[str]) -> bool:
    """Whether the canonical sequence of ``texts`` equals that of its reverse."""
    return canonical_texts(texts) == canonical_texts(list(reversed(texts)))
