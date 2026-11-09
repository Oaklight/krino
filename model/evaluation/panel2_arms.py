"""Option-to-code assignments for the panel-2 order arms (#339).

Panel 2's shipped scheme assigns codes by position: the first option is always
``A``.  Reversing the option list therefore changes which code names which
option, and the measured consequence is a 1-3 pp accuracy swing with 16-28%
of choices changing.  This module holds the two alternative assignments and
the reading conventions, with no torch import, so the script and the tests
share one definition and the assignment is a declared object rather than an
argument to a function call.

Three arms, and only three:

``positional``
    The shipped scheme.  Code ``CODES[i]`` sits next to the option at display
    position ``i``, so the option-to-code mapping depends on the display
    order.

``hashed``
    Content addressing.  Each option text is stripped, hashed with SHA-256,
    the options are sorted by the 32-byte digest in ascending order, and the
    i-th option in that order takes ``CODES[i]``.  The mapping is a function
    of the option set, so a reversed display gives every option the same code
    it had before: the position effect is removed by construction, not
    statistically.

``mean3``
    The mean of three declared random assignments.  For each seed ``s`` in
    ``SEEDS``, ``numpy.random.default_rng(s).permutation(K)`` maps the option
    at natural index ``i`` to ``CODES[permutation[i]]``.  The three per-seed
    distributions are averaged as probabilities per option and the mixture is
    argmaxed; averaging logits is not equivalent when the codes carry
    different biases.

Duplicate option texts inside one item have identical digests and no
content-determined order among themselves.  That case is refused rather than
tie-broken by input order, because the tie-break would reintroduce exactly
the positional dependence the arm exists to remove.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

from model.evaluation.panel2 import CODES

POSITIONAL = "positional"
HASHED = "hashed"
MEAN3 = "mean3"
ARMS = (POSITIONAL, HASHED, MEAN3)

SEEDS = (0, 1, 2)


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.strip().encode("utf-8")).digest()


def content_codes(texts: Sequence[str]) -> list[str]:
    """The code for each option, assigned by the option's own text.

    Args:
        texts: Option texts in the order they are listed for the item.

    Returns:
        One code per input text, at the same index as the input.

    Raises:
        ValueError: If two stripped option texts are identical.  Their digests
            are too, so the sorted order among them would fall back to the
            input order and the assignment would no longer be reversal
            invariant.  The run stops and reports rather than tie-breaking.
    """
    keys = [_digest(text) for text in texts]
    if len(set(keys)) != len(keys):
        duplicates = [
            index for index, key in enumerate(keys) if keys.count(key) > 1
        ]
        raise ValueError(
            "duplicate option texts have identical digests and no "
            f"content-determined order; refusing to assign codes: indices {duplicates}"
        )
    order = sorted(range(len(texts)), key=lambda index: keys[index])
    codes = [""] * len(texts)
    for position, index in enumerate(order):
        codes[index] = CODES[position]
    return codes


def seeded_codes(n_options: int, seed: int) -> list[str]:
    """The code for each option under one declared random assignment.

    The option at natural index ``i`` takes ``CODES[permutation[i]]``, with the
    permutation drawn from ``default_rng(seed)``.  Re-seeding per item makes
    the assignment reproducible within an item and independent across items.
    """
    permutation = np.random.default_rng(seed).permutation(n_options)
    return [CODES[int(index)] for index in permutation]


def natural_codes(arm: str, texts: Sequence[str]) -> list[list[str]]:
    """The option-to-code mapping(s) for one arm, indexed by natural option.

    Returns one mapping per forward pass: one for ``positional`` and
    ``hashed``, three for ``mean3``.
    """
    if arm == POSITIONAL:
        return [list(CODES[: len(texts)])]
    if arm == HASHED:
        return [content_codes(texts)]
    if arm == MEAN3:
        return [seeded_codes(len(texts), seed) for seed in SEEDS]
    raise ValueError(f"unknown arm {arm!r}")


def display_codes(
    arm: str, texts: Sequence[str], reversed_order: bool
) -> list[list[str]]:
    """The code shown next to each displayed option, per forward pass.

    The returned lists are in display order.  For ``hashed`` and ``mean3`` an
    option keeps its code when the display is reversed, which is the whole
    point of the arm; for ``positional`` the code follows the display position
    and therefore stays ``A, B, C, ...`` in both orders.
    """
    mappings = natural_codes(arm, texts)
    width = len(texts)
    if arm == POSITIONAL:
        # The code follows the display position, so the displayed list is
        # A, B, C, ... in both orders; only the option behind each code moves.
        return [list(CODES[:width])]
    if not reversed_order:
        return mappings
    reversed_mappings = []
    for mapping in mappings:
        reversed_mappings.append([mapping[width - 1 - position] for position in range(width)])
    return reversed_mappings


def option_index_for_display(width: int, reversed_order: bool) -> list[int]:
    """The natural option index at each display position."""
    if reversed_order:
        return [width - 1 - position for position in range(width)]
    return list(range(width))


def mixture_prediction(per_seed: Sequence[Sequence[float]]) -> int:
    """Argmax of the probability mean across the declared assignments.

    Each row is a distribution over options from one forward pass.  The mean
    is taken per option in probability space; averaging logits first would
    give a different answer whenever the code rows carry different biases.
    """
    if not per_seed:
        raise ValueError("mixture_prediction needs at least one distribution")
    width = len(per_seed[0])
    totals = [0.0] * width
    for distribution in per_seed:
        if len(distribution) != width:
            raise ValueError("distributions must have the same width")
        for index, value in enumerate(distribution):
            totals[index] += value
    return max(range(width), key=totals.__getitem__)


def probabilities_by_option(
    code_logits: Sequence[float], display_to_option: Sequence[int]
) -> list[float]:
    """Softmax over the K code columns, mapped back to natural option order.

    Args:
        code_logits: Logit of the code shown at each display position.
        display_to_option: The natural option index at each display position,
            from :func:`option_index_for_display`.

    Returns:
        A distribution over natural option indices.
    """
    if len(code_logits) != len(display_to_option):
        raise ValueError("code_logits and display_to_option must agree")
    probabilities = _softmax(list(code_logits))
    by_option = [0.0] * len(display_to_option)
    for position, option in enumerate(display_to_option):
        by_option[option] = probabilities[position]
    return by_option


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    exp = [math.exp(value - peak) for value in values]
    total = sum(exp)
    return [value / total for value in exp]

