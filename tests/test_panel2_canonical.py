"""Tests for the content-addressed option order (#341 follow-up)."""

from __future__ import annotations

import hashlib

from model.evaluation.panel2 import codes_for, render_panel2
from model.evaluation.panel2_canonical import (
    canonical_order,
    canonical_texts,
    permutations_are_reversal_invariant,
)


def digest(text: str) -> bytes:
    return hashlib.sha256(text.strip().encode("utf-8")).digest()


def test_order_is_digest_ascending_and_a_permutation():
    texts = ["alpha", "beta", "gamma", "delta"]
    order = canonical_order(texts)
    assert sorted(order) == list(range(len(texts)))
    assert [digest(texts[index]) for index in order] == sorted(
        digest(text) for text in texts
    )


def test_canonical_texts_are_the_same_from_natural_and_reversed_lists():
    texts = ["alpha", "beta", "gamma", "delta"]
    assert canonical_texts(texts) == canonical_texts(list(reversed(texts)))
    assert permutations_are_reversal_invariant(texts)


def test_canonical_prompts_are_byte_identical():
    texts = ["First option", "Second option", "Third option"]
    codes = codes_for(len(texts))
    forward = render_panel2(
        "The state.", "Which one?", canonical_texts(texts), codes
    )
    backward = render_panel2(
        "The state.", "Which one?", canonical_texts(list(reversed(texts))), codes
    )
    assert forward == backward


def test_duplicate_texts_still_render_the_same_prompt():
    texts = ["same", "other", "same"]
    assert canonical_texts(texts) == canonical_texts(list(reversed(texts)))
