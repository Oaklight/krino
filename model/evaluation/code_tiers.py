"""The code-tier experiment's design, without torch (#350 follow-up).

Panel 2 names each option with a one-token code. The shipped alphabet is 52
letters plus 32 symbols, and the K=80 dev items show the symbol half is dead.
The experiment keeps the items, the option order and the gold's position fixed
and varies only which tier labels the gold option, so the comparison is paired
within item.

This module holds everything that decides the prompts and the reading, so the
script that runs the model and the tests that check the design share one
definition:

* ``assign_codes`` — 52 letters on the non-slot positions, six tier codes on the
  slots, with the gold's slot always among them;
* ``swap_position`` — the smallest position that does not carry a tier code,
  never a literal position 0 (which is the gold's own position for one item);
* ``reading`` — rejected / adoptable / undecided, fixed before the run.
"""

from __future__ import annotations

import random
from math import comb
from typing import Any

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
TIER_SLOTS = 6
K = len(ALPHABET) + TIER_SLOTS

# Pre-registered bounds, in percentage points.
REJECTION_BOUND_PP = -10.0
ADOPTION_POINT_PP = -5.0
ADOPTION_CI_LOW_PP = -10.0
ALPHA = 0.05
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 342


def item_seed(item_id: str, seed: int) -> random.Random:
    """A generator keyed on the item id, so a subset is item-independent."""
    return random.Random(f"{seed}:{item_id}")


def tier_positions(gold_pos: int, n_options: int = K) -> list[int]:
    """The six slot positions: the gold's, plus the last five that are not it."""
    if not 0 <= gold_pos < n_options:
        raise ValueError(f"gold_pos {gold_pos} outside [0, {n_options})")
    tail = [p for p in range(n_options - 1, -1, -1) if p != gold_pos][: TIER_SLOTS - 1]
    return sorted([gold_pos, *tail])


def letter_positions(gold_pos: int, n_options: int = K) -> list[int]:
    """The positions that carry letters: everything that is not a slot."""
    slots = set(tier_positions(gold_pos, n_options))
    return [p for p in range(n_options) if p not in slots]


def swap_position(gold_pos: int, n_options: int = K) -> int:
    """The position exchanged with the gold when the gold should carry a letter.

    It is the smallest position that does not carry a tier code, not position 0:
    when the gold sits at position 0 the literal rule would swap it with itself
    and the two conditions would be identical, silently.
    """
    return letter_positions(gold_pos, n_options)[0]


def assign_codes(
    gold_pos: int, tier_codes: list[str], gold_in_tier: bool, n_options: int = K
) -> list[str]:
    """Codes by position for one condition.

    ``gold_in_tier`` puts one of ``tier_codes`` on the gold's position; otherwise
    the codes at the gold and at :func:`swap_position` are exchanged, so the gold
    carries a letter and the multiset of codes is unchanged.
    """
    if len(tier_codes) != TIER_SLOTS:
        raise ValueError(f"need {TIER_SLOTS} tier codes, got {len(tier_codes)}")
    if len(set(tier_codes)) != TIER_SLOTS:
        raise ValueError(f"tier codes must be distinct, got {tier_codes}")
    slots = tier_positions(gold_pos, n_options)
    letters = letter_positions(gold_pos, n_options)
    if len(letters) != len(ALPHABET):
        raise ValueError(f"{len(letters)} letter positions for {len(ALPHABET)} letters")
    codes: list[str | None] = [None] * n_options
    for code, position in zip(tier_codes, slots):
        codes[position] = code
    for code, position in zip(ALPHABET, letters):
        codes[position] = code
    if not gold_in_tier:
        swap = swap_position(gold_pos, n_options)
        codes[gold_pos], codes[swap] = codes[swap], codes[gold_pos]
    return [str(code) for code in codes]


def paired_codes(
    gold_pos: int, tier_codes: list[str], n_options: int = K
) -> dict[bool, list[str]]:
    """Both conditions for one item, with the gold's code checked to differ.

    A pair whose two conditions give the gold the same code carries no
    information; it is returned as an empty dict so the caller can skip it and
    count it, rather than comparing an item with itself.
    """
    pair = {
        True: assign_codes(gold_pos, tier_codes, True, n_options),
        False: assign_codes(gold_pos, tier_codes, False, n_options),
    }
    if pair[True][gold_pos] == pair[False][gold_pos]:
        return {}
    return pair


def subsample(
    criteria: dict, gold_key: str, keep: int, rng: random.Random
) -> tuple[list[str], int]:
    """Keep the gold option and ``keep - 1`` distractors, in original order."""
    keys = list(criteria)
    gold_index = keys.index(gold_key)
    others = [i for i in range(len(keys)) if i != gold_index]
    if len(others) < keep - 1:
        raise ValueError(f"only {len(others)} distractors for {keep} options")
    kept = sorted([gold_index, *rng.sample(others, keep - 1)])
    texts = [str(criteria[keys[i]]) for i in kept]
    return texts, kept.index(gold_index)


def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the two discordant counts."""
    total = b + c
    if total == 0:
        return 1.0
    low = min(b, c)
    return min(1.0, 2 * sum(comb(total, i) for i in range(low + 1)) * (0.5**total))


def bootstrap_ci(
    pairs: list[tuple[bool, bool]],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile 95% CI of the paired difference, in percentage points.

    ``pairs`` are ``(gold_in_tier_correct, letter_gold_correct)`` per item. The
    resampling seed is fixed, so the interval is reproducible and could be
    written into a pre-registration.
    """
    if not pairs:
        return {"low": 0.0, "high": 0.0, "reps": reps, "seed": seed}
    n = len(pairs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        total = 0
        for _ in range(n):
            in_tier, letter = pairs[rng.randrange(n)]
            total += int(in_tier) - int(letter)
        diffs.append(100.0 * total / n)
    diffs.sort()
    return {
        "low": diffs[int(0.025 * reps)],
        "high": diffs[int(0.975 * reps)],
        "reps": reps,
        "seed": seed,
    }


def reading(difference_pp: float, exact_p: float, ci_low: float) -> str:
    """The pre-registered three-way reading of one tier's paired difference."""
    if difference_pp <= REJECTION_BOUND_PP and exact_p < ALPHA:
        return "rejected"
    if difference_pp > ADOPTION_POINT_PP and ci_low > ADOPTION_CI_LOW_PP:
        return "adoptable"
    return "undecided"


def evaluate_pairs(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """Accuracy, paired difference, exact p, CI and reading for one tier."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no pairs")
    tier_correct = sum(1 for in_tier, _ in pairs if in_tier)
    letter_correct = sum(1 for _, letter in pairs if letter)
    b = sum(1 for in_tier, letter in pairs if letter and not in_tier)
    c = sum(1 for in_tier, letter in pairs if in_tier and not letter)
    difference = 100.0 * (c - b) / n
    p = mcnemar_p(b, c)
    ci = bootstrap_ci(pairs)
    return {
        "items": n,
        "gold_in_tier_correct": tier_correct,
        "gold_in_tier_accuracy": tier_correct / n,
        "letter_gold_correct": letter_correct,
        "letter_gold_accuracy": letter_correct / n,
        "difference_pp": difference,
        "discordant": {"tier_correct_letter_wrong": c, "tier_wrong_letter_correct": b},
        "exact_p": p,
        "bootstrap_ci_pp": ci,
        "reading": reading(difference, p, ci["low"]),
    }
