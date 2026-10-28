"""Pure helpers for option-order sensitivity probes.

A decision model should not depend on the order in which its options are
presented.  Given the same option set, a model that scores each option
independently and then normalises should return the same label ->
probability map whatever the order.  Models that read position, attend across
the option list, or let one option's text leak into another's score can flip
their answer when the list is reordered.  JevBench's open rebuild is the
canonical example: reversing a yes/no question drops it from 72% to 21%.

These helpers do the bookkeeping for the probe and import nothing outside the
standard library, so they run in CI where torch is not installed.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence


def option_count_bucket(n_options: int) -> str:
    """Return the reporting bucket for an option count.

    The boundaries are the ones in the evaluation report: a binary question,
    a short multiple-choice question, a long list, and the many-option regime
    where decision models are known to collapse.

    Args:
        n_options: Number of options (or score levels) in the question.

    Returns:
        One of ``"2"``, ``"3-5"``, ``"6-20"`` or ``">20"``.
    """
    if n_options <= 2:
        return "2"
    if n_options <= 5:
        return "3-5"
    if n_options <= 20:
        return "6-20"
    return ">20"


def total_variation(left: Mapping[Any, float], right: Mapping[Any, float]) -> float:
    """Total-variation distance between two label -> probability maps.

    The distance is ``0.5 * sum_l |p_left(l) - p_right(l)|`` over the union of
    labels.  It is in ``[0, 1]`` and equals the maximum probability mass that
    has to move to turn one distribution into the other.

    Args:
        left: Probability keyed by option label.
        right: Probability keyed by option label.

    Returns:
        The distance in ``[0, 1]``.
    """
    labels = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(label, 0.0) - right.get(label, 0.0)) for label in labels
    )


def remap_probe_probs(
    probs: Sequence[float], source_labels: Sequence[Any]
) -> dict[Any, float]:
    """Map probe-position probabilities back onto the original option labels.

    A probe reorders the options before the forward pass, so the output index
    no longer names the same option.  ``source_labels[i]`` is the original
    label of whatever option was placed at probe position ``i``; this function
    moves each probability back to that label so the result can be compared
    with the unperturbed distribution.

    Args:
        probs: Probability at each probe position.
        source_labels: Original label of the option at each probe position.

    Returns:
        Probability keyed by original option label.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(probs) != len(source_labels):
        raise ValueError(
            f"probs and source_labels must have the same length, "
            f"got {len(probs)} and {len(source_labels)}"
        )
    return {label: float(p) for label, p in zip(source_labels, probs)}


def make_permutation(n: int, seed: int, item_id: str, tag: str) -> list[int]:
    """Return a deterministic permutation of ``range(n)`` for one probe.

    Seeding from the item id rather than a shared stream keeps the probe
    reproducible when a run is capped at a different number of items: items
    keep the permutation they had in the full run.

    Args:
        n: Number of positions to permute.
        seed: Run-level seed.
        item_id: Stable question identifier.
        tag: Probe name, so two probes on one item do not share a permutation.

    Returns:
        A list of ``n`` distinct indices in ``range(n)``.
    """
    rng = random.Random(f"{seed}:{item_id}:{tag}")
    permutation = list(range(n))
    rng.shuffle(permutation)
    return permutation
