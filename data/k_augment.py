"""Option-count augmentation for choice items.

The choice head encodes each option independently, pools, and takes a softmax
over K logits.  That is well defined for any K, but the training data only ever
supplies K in 2..80, with 92.7% of evaluation items at K <= 5.  Accuracy
therefore collapses as options are added: on the synthetic padded pool the
pooled-attention baseline falls from 0.677 at K=4 to 0.217 at K=256, where the
goal asks for a 5 pp difference between the <=5-option and >20-option regimes.

This module widens the option count at training time.  A fraction of choice
items get their option list padded up to a target K drawn log-uniformly from a
configured range, keeping the gold option and filling the rest with distractor
option texts borrowed from other choice items.  Nothing is written to disk and
no new labels are needed: the gold key is preserved, so supervision is unchanged.

Distractors are taken from the same source first and from other choice sources
only when a source cannot supply enough distinct texts (banking77 has 80 option
texts, so a K=256 target must borrow).  Mixing domains is deliberate: an item
that must reject 200 options from a foreign domain is harder than one that must
reject 200 near-synonyms, and the evaluation regime adds generic distractors.

Padded items drop ``teacher_probs``.  A teacher distribution assigns exactly zero
mass to options it never saw, so keeping it would train the model to treat
plausible distractors as impossible rather than merely unlikely.

The module is torch-free so the augmentation can be unit-tested on any
interpreter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .format import TypedQuestion

# Option texts shorter than this add no signal (empty strings, stray marks).
_MIN_DISTRACTOR_CHARS = 1


@dataclass(frozen=True)
class KAugmentConfig:
    """How many options to add, and to which items.

    Args:
        fraction: share of choice items to pad, in [0, 1].  The rest keep their
            natural option count.  This is the main compute knob: the cost of a
            choice item grows linearly in K, so a fraction of 0.2 with a mean
            target of 45 roughly triples an epoch's option encodes.
        min_k: smallest target K.  Padding never shrinks an item, so an item
            already at or above ``min_k`` may still be padded if the drawn target
            exceeds its natural count.
        max_k: largest target K.  Must be at least ``min_k``.
        seed: base seed; each epoch mixes it with the epoch index so an item sees
            a different width and a different distractor set every epoch.
    """

    fraction: float = 0.2
    min_k: int = 8
    max_k: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")
        if self.min_k < 2:
            raise ValueError(f"min_k must be at least 2 (a choice needs 2 options), got {self.min_k}")
        if self.max_k < self.min_k:
            raise ValueError(f"max_k ({self.max_k}) must be >= min_k ({self.min_k})")


def _option_text(value: object, key: str) -> str:
    """Text of an option: the value when usable, else the key."""
    text = value if isinstance(value, str) and value.strip() else key
    return text


def build_distractor_pool(items: list[TypedQuestion]) -> dict[str, list[str]]:
    """Collect option texts per source, in first-seen order and de-duplicated.

    Only choice items contribute.  Keeping the source split lets
    :func:`pad_choice_item` prefer same-domain distractors.
    """
    pools: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for item in items:
        if item.question.get("type") != "choice":
            continue
        criteria = item.question.get("criteria")
        if not isinstance(criteria, dict):
            continue
        source = item.source
        bucket = pools.setdefault(source, [])
        bucket_seen = seen.setdefault(source, set())
        for key, value in criteria.items():
            text = _option_text(value, key)
            if len(text) < _MIN_DISTRACTOR_CHARS or text in bucket_seen:
                continue
            bucket_seen.add(text)
            bucket.append(text)
    return pools


def _flatten_pool(pool: dict[str, list[str]]) -> list[str]:
    flat: list[str] = []
    for texts in pool.values():
        flat.extend(texts)
    return flat


def draw_target_k(rng: random.Random, cfg: KAugmentConfig) -> int:
    """Draw a log-uniform integer in ``[min_k, max_k]``.

    Log-uniform rather than uniform: the interesting regime is wide option
    lists, but a uniform draw over 8..256 would spend most of its mass far above
    the widths the evaluation actually uses.  Log spacing keeps the small widths
    (which dominate the real eval) sampled as often as the large ones.
    """
    low = math.log(cfg.min_k)
    high = math.log(cfg.max_k)
    value = math.exp(rng.uniform(low, high))
    return max(cfg.min_k, min(cfg.max_k, int(round(value))))


def pad_choice_item(
    item: TypedQuestion,
    target_k: int,
    pool: dict[str, list[str]],
    rng: random.Random,
    flat_pool: list[str] | None = None,
) -> TypedQuestion:
    """Return a copy of ``item`` with at least ``target_k`` options.

    The gold key and every original option are preserved verbatim, so the label
    stays valid and the item's supervision is unchanged.  Distractor keys are
    namespaced (``__distractor_00``) to guarantee they cannot collide with a
    gold key, and ``teacher_probs`` is cleared because a teacher distribution
    has no mass for options it never saw.

    Returns the original item when it cannot be padded safely: not a choice
    item, a list-valued criteria (some older loaders), a label missing from the
    criteria, or no usable distractor texts.
    """
    question = item.question
    if question.get("type") != "choice":
        return item
    criteria = question.get("criteria")
    if not isinstance(criteria, dict):
        return item
    if item.label not in criteria:
        return item

    natural = len(criteria)
    if target_k <= natural:
        return item

    own = pool.get(item.source, [])
    flat = flat_pool if flat_pool is not None else _flatten_pool(pool)
    if not own and not flat:
        return item

    original_texts = {_option_text(v, k) for k, v in criteria.items()}
    wanted = target_k - natural

    # Same-source distractors first, then the rest of the corpus.  Shuffle both
    # so the extra options are not always the same lexicographic prefix of the
    # pool, which would correlate option position with source of origin.
    own_shuffled = list(own)
    rng.shuffle(own_shuffled)
    flat_shuffled = list(flat)
    rng.shuffle(flat_shuffled)

    added: list[str] = []
    taken: set[str] = set()
    for text in own_shuffled + flat_shuffled:
        if len(added) >= wanted:
            break
        if text in original_texts or text in taken:
            continue
        taken.add(text)
        added.append(text)

    if not added:
        return item

    new_criteria = dict(criteria)
    for offset, text in enumerate(added):
        new_criteria[f"__distractor_{offset:03d}"] = text

    new_question = dict(question)
    new_question["criteria"] = new_criteria
    return TypedQuestion(
        id=item.id,
        state=item.state,
        question=new_question,
        label=item.label,
        source=item.source,
        split=item.split,
        group=item.group,
        teacher_probs=None,
    )


def apply_k_augment(
    items: list[TypedQuestion],
    cfg: KAugmentConfig,
    epoch: int = 0,
) -> list[TypedQuestion]:
    """Pad a random ``cfg.fraction`` of the choice items in ``items``.

    Deterministic for a given ``(cfg, epoch, item order)``: the item's index
    seeds its own RNG, so the same epoch reproduces the same widths after a
    restart, while a different epoch sees different ones.

    Non-choice items and items that cannot be padded pass through unchanged, so
    the returned list has the same length and order as the input.
    """
    if cfg.fraction <= 0.0:
        return items

    pool = build_distractor_pool(items)
    if not pool:
        return items
    flat_pool = _flatten_pool(pool)

    out: list[TypedQuestion] = []
    for index, item in enumerate(items):
        if item.question.get("type") != "choice":
            out.append(item)
            continue
        # A stable integer mix rather than hash(): Python's string hashing is
        # salted per process, and reproducibility across restarts is the point.
        rng = random.Random(cfg.seed * 1_000_003 + epoch * 7_919 + index)
        # Draw the coin from a per-item RNG so changing the fraction does not
        # reshuffle every item that was already selected.
        if rng.random() >= cfg.fraction:
            out.append(item)
            continue
        target_k = draw_target_k(rng, cfg)
        out.append(pad_choice_item(item, target_k, pool, rng, flat_pool=flat_pool))
    return out
