"""Blend two per-option scorers, and the bucket and sampling rules around it.

``research/13`` found the backbone's native per-option score and the trained
choice head trade places at about twenty options: neither dominates, and a hard
switch at five options throws away whatever the losing scorer knew about each
item. The cheapest test of whether that matters is a weighted blend --

    blended_i = head_i / spread_head + lam * native_i / spread_native

-- with one free parameter, fitted on in-distribution items and read out on a
held-out benchmark.

The two scorers are on different scales (a trained head's logits against a
log-odds of independent per-option probabilities), and the imbalance grows with
the option count, so each is divided by a single number: the median per-item
margin on the fitting set. One number per scorer, not per item, so the weight
means the same thing on a two-option item as on an eighty-option one and a
single ``lam`` can be read across buckets.

Kept free of torch so the arithmetic and the sampling rule can be tested without
a model.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Sequence

# Option-count buckets that matter here: the crossover from research/13 sits
# between them, and every choice item in the corpus is either <= 5 or >= 20 --
# nothing sits in between.
SHORT_MAX = 5
LONG_MIN = 20


def margin(values: Sequence[float]) -> float:
    """Top score minus the runner-up, which is what decides a choice."""
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1]


def spread(margins: Sequence[float]) -> float:
    """One scale per scorer: the median of the positive per-item margins.

    The median rather than the mean because a handful of very confident items
    should not set the scale for all of them. Non-positive margins are dropped:
    a zero margin means the scorer cannot tell the top two apart, and including
    those would drag the scale toward zero and let tiny differences dominate.
    """
    positive = [value for value in margins if value > 0]
    return statistics.median(positive) if positive else 1.0


def blended_argmax(
    head: Sequence[float],
    native: Sequence[float],
    head_scale: float,
    native_scale: float,
    lam: float,
) -> int:
    """Argmax of the rescaled sum, ties broken toward the lower index.

    ``lam = 0`` recovers the head's own argmax and a large ``lam`` approaches
    the native one, so the two endpoints of a sweep are the two scorers rather
    than approximations of them.

    Raises:
        ValueError: If the two score vectors have different lengths.
    """
    if len(head) != len(native):
        raise ValueError("scorers must cover the same options")
    if not head:
        raise ValueError("at least one option is required")
    scores = [
        head_value / head_scale + lam * native_value / native_scale
        for head_value, native_value in zip(head, native)
    ]
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def bucket_of(n_options: int) -> str:
    """The option-count bucket an item is reported in."""
    if n_options <= SHORT_MAX:
        return f"<={SHORT_MAX}"
    if n_options >= LONG_MIN:
        return f">={LONG_MIN}"
    return f"{SHORT_MAX + 1}-{LONG_MIN - 1}"


def sample_choice_items(
    items: Sequence[Any],
    per_source: int,
    long_items: int,
    rng: random.Random,
) -> list[Any]:
    """Take a deterministic sample of one source's choice items.

    Short-list items are capped at ``per_source``; items with at least
    ``LONG_MIN`` options are capped at ``long_items`` instead, because scoring an
    item natively costs one forward pass per option and the long bucket would
    otherwise dominate the runtime. Sampling is per bucket so a source that is
    nearly all long items still contributes its short ones.

    Args:
        items: Choice items from one source.
        per_source: Cap for items below ``LONG_MIN`` options.
        long_items: Cap for items at or above ``LONG_MIN`` options.
        rng: Seeded generator, so a re-run reproduces the same sample.

    Returns:
        The long sample followed by the short sample, each in shuffled order.
    """
    short: list[Any] = []
    long: list[Any] = []
    for item in items:
        criteria = item.question.get("criteria") or []
        (long if len(criteria) >= LONG_MIN else short).append(item)
    rng.shuffle(short)
    rng.shuffle(long)
    return long[:long_items] + short[:per_source]
