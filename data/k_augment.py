"""Option-count augmentation for choice items.

The choice head encodes each option independently, pools, and takes a softmax
over K logits.  That is well defined for any K, but the training data only ever
supplies K in 2..80, with 92.7% of evaluation *choice* items at K <= 5.  Accuracy
degrades as options are added, and the size of that degradation depends on the
checkpoint: on a fixed 300-item banking77 sweep the 0.6B 5-epoch
pooled-attention baseline falls from 0.677 at K=4 to 0.217 at K=256, while the
4B 10-epoch model falls only from 0.977 to 0.850 between K=4 and K=80 — 12.67 pp,
against a goal that asks for 5 pp between the <=5-option and >20-option regimes.
All three measurements are committed: ``model/experiments/optcount_pooled.json``
(0.6B), ``model/experiments/optcount_4b_ep10.json`` (4B) and
``model/experiments/k_augment_cost.json``.  The 0.6B curve is not a description of
the 4B model, so this module is aimed at the smaller residual gap, not the
collapse.

This module widens the option count at training time.  A fraction of choice
items get their option list padded up to a target K drawn log-uniformly from a
configured range, keeping the gold option and filling the rest with distractor
option texts borrowed from other choice items.  Nothing is written to disk and
no new labels are needed: the gold key is preserved, so supervision is unchanged.

Provenance: the distractor pool is built only from the ``items`` handed to
:func:`apply_k_augment`.  Training passes the sampled training epoch, so an
evaluation item can never enter the pool or be padded.  Texts are compared after
``strip().casefold()``, so a case or whitespace variant of the gold option cannot
slip through the exact-match guards and create an item with two defensible
answers, one of them labelled wrong.  Semantic near-duplicates are not
detectable by string comparison and remain a residual rate.

Distractors are taken from the same source first and from other choice sources
only when a source cannot supply enough distinct texts (banking77 has 80 option
texts, so a K=256 target must borrow).  Mixing domains is deliberate: an item
that must reject 200 options from a foreign domain is harder than one that must
reject 200 near-synonyms, and the evaluation regime adds generic distractors.

Padded items drop ``teacher_probs``.  A teacher distribution assigns exactly zero
mass to options it never saw, so keeping it would train the model to treat
plausible distractors as impossible rather than merely unlikely.  That switch is
a second experimental variable: a padded item is trained with hard cross-entropy
while the same item in an unpadded arm may be trained against the KL objective.
To keep an A/B attributable to K alone, every selected item drops its teacher
distribution whether or not it can actually be widened, and
``KAugmentConfig(control=True)`` runs the same selection and the same drop with
no padding, for the baseline arm.

Cost: on the real 182,262-item pool at fraction 0.2, the default K range [8, 80]
takes an epoch's option encodes from 44,558 to 69,467 (1.56x); the optional K
range [8, 256] takes them to 107,053 (2.40x), and about 60% of that extra work
sits above the widest evaluation item (77 options).  Both are recorded in
``model/experiments/k_augment_cost.json``.  Those are encode-count ratios, not
memory ratios: the option encode is one batched forward, so peak activations
scale with batch x K.  A 0.6B forward at batch 1 rose from 1.20 GB at 5 options
to 2.36 GB at 256 options (#160), and the 4B run has no measured profile at
K=256.  Batch size 32 at K=256 is the profile to check before using the wide
range.

The module is torch-free so the augmentation can be unit-tested on any
interpreter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from .format import TypedQuestion

# Option texts shorter than this add no signal (stray marks, single characters).
# A blank value is skipped rather than replaced by its option key: a bare key is
# not an option text, and admitting one puts the literal "B" in the option list.
_MIN_DISTRACTOR_CHARS = 3


def _norm(text: str) -> str:
    """Comparison key for duplicate detection.

    Option texts are borrowed across sources with different casing and spacing
    conventions, so exact string equality lets ``"True"`` and ``"true"`` both
    land next to the gold ``"true"``.
    """
    return text.strip().casefold()


@dataclass(frozen=True)
class KAugmentConfig:
    """How many options to add, and to which items.

    Args:
        fraction: share of choice items to pad, in [0, 1].  The rest keep their
            natural option count.  This is the main compute knob: cost grows with
            the sum of option counts.
        min_k: smallest target K.  Padding never shrinks an item, so an item
            already at or above ``min_k`` may still be padded if the drawn target
            exceeds its natural count.
        max_k: largest target K.  Must be at least ``min_k``.  Defaults to 80,
            the widest evaluation item: padding beyond that buys widths nothing
            is measured at and, on the real pool, about 60% of the extra encode
            cost.  Raise it deliberately if training past evaluation width is
            wanted for robustness.
        seed: base seed; each epoch mixes it with the epoch index so an item sees
            a different width and a different distractor set every epoch.
        control: run the selection and clear ``teacher_probs`` on the selected
            items but never pad.  Use this for the baseline arm of an A/B so the
            only difference between arms is the option list, not the loss type.
    """

    fraction: float = 0.2
    min_k: int = 8
    max_k: int = 80
    seed: int = 42
    control: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction}")
        if self.min_k < 2:
            raise ValueError(f"min_k must be at least 2 (a choice needs 2 options), got {self.min_k}")
        if self.max_k < self.min_k:
            raise ValueError(f"max_k ({self.max_k}) must be >= min_k ({self.min_k})")


@dataclass(frozen=True)
class KAugmentStats:
    """What one epoch's augmentation did.

    Reported at the start of an epoch so a thin distractor pool cannot quietly
    weaken the intervention: ``under_filled`` counts selected items whose
    realised width stayed below the drawn target, and ``realised_mean`` against
    ``target_mean`` shows the size of that truncation.
    """

    choice_items: int = 0
    selected: int = 0
    padded: int = 0
    under_filled: int = 0
    teacher_probs_dropped: int = 0
    encodes_before: int = 0
    encodes_after: int = 0
    target_mean: float = 0.0
    realised_mean: float = 0.0
    realised_max: int = 0


def _option_text(value: object, key: str) -> str:
    """Text the training path feeds for an option: the value when usable, else the key.

    This mirrors ``compute_loss``, which passes ``v or k``.  It is used only to
    compare distractors against the item's own options; pool collection rejects
    blank values outright instead of substituting the key.
    """
    text = value if isinstance(value, str) and value.strip() else key
    return text


def build_distractor_pool(items: list[TypedQuestion]) -> dict[str, list[str]]:
    """Collect option texts per source, in first-seen order and de-duplicated.

    Only choice items contribute, and only from ``items``: the pool is the
    provenance boundary that keeps evaluation data out of training distractors.
    Blank or very short values are skipped rather than replaced by their key.
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
        for value in criteria.values():
            text = value.strip() if isinstance(value, str) else ""
            if len(text) < _MIN_DISTRACTOR_CHARS:
                continue
            norm = _norm(text)
            if norm in bucket_seen:
                continue
            bucket_seen.add(norm)
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
    lists, but a uniform draw over 8..80 would spend most of its mass far above
    the widths the evaluation actually uses.  Log spacing keeps the small widths
    (which dominate the real eval) sampled as often as the large ones.
    """
    low = math.log(cfg.min_k)
    high = math.log(cfg.max_k)
    value = math.exp(rng.uniform(low, high))
    return max(cfg.min_k, min(cfg.max_k, int(round(value))))


def _draw_texts(
    rng: random.Random,
    texts: list[str],
    wanted: int,
    excluded: set[str],
    taken: set[str],
) -> list[str]:
    """Draw up to ``wanted`` unused texts without copying or shuffling ``texts``.

    Sampling indices costs O(wanted) in the common case rather than the O(len)
    copy-and-shuffle a per-item full-pool shuffle would pay.  A margin covers
    collisions with the item's own options; if the margin is exhausted, another
    draw continues from the indices not yet tried.  The whole list is only
    traversed when it genuinely cannot supply ``wanted`` distinct texts.
    """
    if wanted <= 0 or not texts:
        return []
    n = len(texts)
    draw = min(n, wanted + len(excluded) + 4)
    tried: set[int] = set()
    picked: list[str] = []
    while len(picked) < wanted and len(tried) < n:
        batch = rng.sample(range(n), min(draw, n - len(tried)))
        fresh = [i for i in batch if i not in tried]
        tried.update(fresh)
        for i in fresh:
            if len(picked) >= wanted:
                break
            text = texts[i]
            norm = _norm(text)
            if norm in excluded or norm in taken:
                continue
            taken.add(norm)
            picked.append(text)
    return picked


def _without_teacher_probs(item: TypedQuestion) -> TypedQuestion:
    """Return ``item`` with ``teacher_probs`` cleared, or ``item`` when already clear."""
    if getattr(item, "teacher_probs", None) is None:
        return item
    return replace(item, teacher_probs=None)


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

    # Compare normalised, so a case or whitespace variant of an existing option
    # cannot be added as a distractor that is really a second correct answer.
    original_texts = {_norm(_option_text(v, k)) for k, v in criteria.items()}
    wanted = target_k - natural
    taken: set[str] = set()

    # Same-source distractors first, then the rest of the corpus.
    added = _draw_texts(rng, own, wanted, original_texts, taken)
    if len(added) < wanted and flat:
        added += _draw_texts(rng, flat, wanted - len(added), original_texts, taken)

    if not added:
        return item

    new_criteria = dict(criteria)
    for offset, text in enumerate(added):
        new_criteria[f"__distractor_{offset:03d}"] = text

    new_question = dict(question)
    new_question["criteria"] = new_criteria
    return replace(item, question=new_question, teacher_probs=None)


def shrink_choice_item(
    item: TypedQuestion, target_k: int, rng: random.Random
) -> TypedQuestion:
    """Return a copy of ``item`` with at most ``target_k`` options, gold included.

    :func:`pad_choice_item` only widens, which is right for training — nothing is
    ever removed, so supervision is untouched — but it cannot vary K *downward*.
    A source whose items already carry its full option inventory is then
    unchanged by every target below that count: banking77 puts all 80 intents on
    every item, so a pad-only sweep scores the same unmodified item at K=4, 8,
    16, 32, 64 and 80 and reports perfect option-count robustness.

    This keeps the gold key and ``target_k - 1`` others drawn with ``rng``, in the
    source's own order so the item still reads as its author wrote it. Used by
    ``model/scripts/option_count_sweep.py`` to hold an item fixed while its option
    count changes; training does not call it.

    Args:
        item: A choice item whose ``criteria`` is a mapping.
        target_k: Option count to keep, including the gold.
        rng: Seeded generator deciding which other options survive.

    Returns:
        A copy with at most ``target_k`` options, or ``item`` itself when it is
        not a choice item, the gold is missing, or it is already small enough.
    """
    criteria = item.question.get("criteria")
    if not isinstance(criteria, dict) or item.label not in criteria:
        return item
    if len(criteria) <= target_k:
        return item

    others = [key for key in criteria if key != item.label]
    rng.shuffle(others)
    kept = {item.label, *others[: max(target_k - 1, 0)]}
    new_question = dict(item.question)
    new_question["criteria"] = {
        key: value for key, value in criteria.items() if key in kept
    }
    return replace(item, question=new_question)


def apply_k_augment_with_stats(
    items: list[TypedQuestion],
    cfg: KAugmentConfig,
    epoch: int = 0,
) -> tuple[list[TypedQuestion], KAugmentStats]:
    """Pad a random ``cfg.fraction`` of the choice items and report what happened.

    Deterministic for a given ``(cfg, epoch, item order)``: the item's index
    seeds its own RNG, so the same epoch reproduces the same widths after a
    restart, while a different epoch sees different ones.

    Every selected item loses ``teacher_probs``, padded or not, so a paired
    control run (``cfg.control``) differs from the augmented run only in the
    option list.  Non-choice items and items that cannot be padded pass through
    unchanged, so the returned list has the same length and order as the input.
    """
    stats = KAugmentStats()
    encodes_before = sum(
        len(it.question["criteria"])
        for it in items
        if it.question.get("type") == "choice"
        and isinstance(it.question.get("criteria"), dict)
    )
    if cfg.fraction <= 0.0:
        return items, replace(stats, encodes_before=encodes_before, encodes_after=encodes_before)

    pool = {} if cfg.control else build_distractor_pool(items)
    flat_pool = _flatten_pool(pool)

    out: list[TypedQuestion] = []
    choice_items = selected = padded = under_filled = dropped = 0
    encodes_after = 0
    targets: list[int] = []
    realised: list[int] = []
    for index, item in enumerate(items):
        criteria = item.question.get("criteria")
        if item.question.get("type") != "choice" or not isinstance(criteria, dict):
            out.append(item)
            continue
        choice_items += 1
        natural = len(criteria)
        # A stable integer mix rather than hash(): Python's string hashing is
        # salted per process, and reproducibility across restarts is the point.
        rng = random.Random(cfg.seed * 1_000_003 + epoch * 7_919 + index)
        # Draw the coin from a per-item RNG so changing the fraction does not
        # reshuffle every item that was already selected.  The coin comes before
        # the target K so the control arm selects exactly the same items.
        if rng.random() >= cfg.fraction:
            out.append(item)
            encodes_after += natural
            continue
        selected += 1
        had_teacher = getattr(item, "teacher_probs", None) is not None

        if cfg.control:
            new_item = _without_teacher_probs(item)
        else:
            target_k = draw_target_k(rng, cfg)
            padded_item = pad_choice_item(item, target_k, pool, rng, flat_pool=flat_pool)
            # Clear the teacher distribution on selected-but-unpaddable items too;
            # the matched control run clears it on the same selection.
            new_item = (
                _without_teacher_probs(item) if padded_item is item else padded_item
            )
            new_width = len(new_item.question["criteria"])
            if target_k > natural:
                # Count every attempted widening, including one that added
                # nothing at all: a thin pool must not pass unrecorded.
                targets.append(target_k)
                realised.append(new_width)
                if new_width < target_k:
                    under_filled += 1
                if new_width > natural:
                    padded += 1
        if had_teacher and new_item is not item and getattr(new_item, "teacher_probs", None) is None:
            dropped += 1
        out.append(new_item)
        encodes_after += len(new_item.question["criteria"])

    stats = KAugmentStats(
        choice_items=choice_items,
        selected=selected,
        padded=padded,
        under_filled=under_filled,
        teacher_probs_dropped=dropped,
        encodes_before=encodes_before,
        encodes_after=encodes_after,
        target_mean=(sum(targets) / len(targets)) if targets else 0.0,
        realised_mean=(sum(realised) / len(realised)) if realised else 0.0,
        realised_max=max(realised) if realised else 0,
    )
    return out, stats


def apply_k_augment(
    items: list[TypedQuestion],
    cfg: KAugmentConfig,
    epoch: int = 0,
) -> list[TypedQuestion]:
    """Pad a random ``cfg.fraction`` of the choice items in ``items``.

    Convenience wrapper over :func:`apply_k_augment_with_stats` for callers that
    do not need the run statistics.
    """
    return apply_k_augment_with_stats(items, cfg, epoch=epoch)[0]
