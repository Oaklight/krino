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


def option_count(
    question: dict[str, Any], teacher_probs: dict[str, Any] | None = None
) -> int:
    """Number of options (or score levels) a forward pass will score.

    Mirrors the branch ``_eval_item`` takes in ``model/training/supervised.py``:
    a noul item uses its augmented option list only when the item carries more
    than two teacher probabilities.

    Args:
        question: The ``TypedQuestion.question`` mapping.
        teacher_probs: Optional teacher distribution attached to the item.

    Returns:
        The option or level count.
    """
    q_type = question.get("type", "noul")
    if q_type == "choice":
        return len(question.get("criteria") or {})
    if q_type == "score":
        return len(question.get("criteria") or [])
    augmented = question.get("augmented_options")
    if augmented and teacher_probs and len(teacher_probs) > 2:
        return len(augmented)
    return 2


def select_items_by_option_limit(
    items: Sequence[Any], max_options: int
) -> tuple[list[Any], int]:
    """Split items into those within the option guard and those above it.

    Args:
        items: Objects with a ``question`` attribute (and optionally
            ``teacher_probs``), such as ``TypedQuestion``.
        max_options: Maximum option count; ``0`` or a negative value disables
            the guard.

    Returns:
        ``(kept, skipped)`` where ``skipped`` is the number of items dropped.
    """
    if max_options <= 0:
        return list(items), 0
    kept: list[Any] = []
    skipped = 0
    for item in items:
        if (
            option_count(item.question, getattr(item, "teacher_probs", None))
            > max_options
        ):
            skipped += 1
        else:
            kept.append(item)
    return kept, skipped


def tie_indices(probabilities: Sequence[float]) -> list[int]:
    """Indices that share the exact maximum probability."""
    if not probabilities:
        return []
    best = max(probabilities)
    return [index for index, p in enumerate(probabilities) if p == best]


def tie_stable_argmax(probabilities: Sequence[float], order_keys: Sequence[Any]) -> int:
    """Argmax that breaks exact ties by the smallest ``order_keys`` value.

    Python's ``max`` returns the first maximum, so on an exact tie the label
    depends on where the option happened to sit in the list.  That makes a
    reordering probe count tie-breaking as sensitivity and makes the headline
    accuracy itself depend on presentation order.  Breaking ties by a key that
    travels with the label (its original index) removes both effects.

    Args:
        probabilities: Probability per position.
        order_keys: Comparable tie-break key per position; lower wins.

    Returns:
        Index of the winning position.

    Raises:
        ValueError: If the inputs are empty or have different lengths.
    """
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if len(probabilities) != len(order_keys):
        raise ValueError(
            f"probabilities and order_keys must have the same length, "
            f"got {len(probabilities)} and {len(order_keys)}"
        )
    candidates = tie_indices(probabilities)
    return min(candidates, key=lambda index: order_keys[index])


def duplicate_option_texts(texts: Sequence[str]) -> dict[str, int]:
    """Effective option texts that occur more than once, with their counts.

    A duplicated option makes the gold label ambiguous: two positions carry
    the same evidence, so the argmax can only be decided by position.  The
    evaluation reports these items separately rather than letting them decide
    accuracy silently.
    """
    counts: dict[str, int] = {}
    for text in texts:
        counts[text] = counts.get(text, 0) + 1
    return {text: count for text, count in counts.items() if count > 1}


def chance_accuracy(n_options: int) -> float:
    """Accuracy of a uniform guess over ``n_options`` options."""
    return 1.0 / n_options if n_options > 0 else 0.0


def normalized_accuracy(accuracy: float, mean_chance: float) -> float:
    """Rescale accuracy to the distance from chance to perfect.

    Raw accuracy falls with option count even when discrimination is constant,
    so a many-option cell looks worse than a binary one.  ``(acc - chance) /
    (1 - chance)`` puts both on a comparable scale.  A value of 0 is chance
    and 1 is perfect.
    """
    if mean_chance >= 1.0:
        return 0.0
    return (accuracy - mean_chance) / (1.0 - mean_chance)


def nested_option_subsets(
    n_options: int,
    gold_index: int,
    sizes: Sequence[int],
    seed: int,
    item_id: str,
) -> dict[int, list[int]]:
    """Nested option subsets that always contain the gold option.

    Removing options from the same questions isolates option count from source
    difficulty: accuracy by subset size is a controlled curve, whereas
    bucketing whole sources confounds the two.

    Args:
        n_options: Total number of options on the item.
        gold_index: Index of the gold option.
        sizes: Requested subset sizes; values outside ``[2, n_options]`` are
            skipped.
        seed: Run-level seed.
        item_id: Stable question identifier.

    Returns:
        Mapping ``size -> option indices`` with the gold option included.
    """
    subsets: dict[int, list[int]] = {}
    for size in sizes:
        if size < 2 or size > n_options:
            continue
        if size == n_options:
            subset = list(range(n_options))
        else:
            distractors = [index for index in range(n_options) if index != gold_index]
            rng = random.Random(f"{seed}:{item_id}:nested:{size}")
            rng.shuffle(distractors)
            subset = [gold_index] + distractors[: size - 1]
        rng = random.Random(f"{seed}:{item_id}:nested-order:{size}")
        rng.shuffle(subset)
        subsets[size] = subset
    return subsets
