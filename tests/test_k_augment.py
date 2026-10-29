"""Option-count augmentation must widen choice items without touching their labels.

The effect it addresses is measured, not hypothetical, but it is smaller on the
current model than the original sweep suggested: the 0.6B 5-epoch baseline drops
from 0.677 at 4 options to 0.217 at 256 (model/experiments/optcount_pooled.json),
while the 4B 10-epoch model falls from 0.977 at K=4 to 0.850 at K=80
(model/experiments/optcount_4b_ep10.json) — 12.67 pp against a 5 pp goal.

The tests pin the invariants that make padding and shrinking safe: the gold key
survives, no option text (including a case or whitespace variant) is duplicated,
distractors come only from the supplied items, keys cannot collide, non-choice
items pass through untouched, the same (seed, epoch) reproduces the same widths,
shrinking keeps the source order and is reproducible per seed, and the
teacher-distribution switch is matched between the augmented and control arms.

The tests are torch-free on purpose: CI installs neither torch nor transformers,
so a guard that skips on a missing torch would make the whole file pass without
running anything.
"""

from __future__ import annotations

import random

import pytest

from data.format import TypedQuestion
from dataclasses import replace

from data.k_augment import (
    KAugmentConfig,
    apply_k_augment,
    apply_k_augment_with_stats,
    build_distractor_pool,
    draw_target_k,
    pad_choice_item,
    shrink_choice_item,
)


def _choice(
    item_id: str,
    label: str,
    criteria: dict[str, str],
    source: str = "src",
    teacher_probs: dict[str, float] | None = None,
) -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "choice", "instructions": "pick", "criteria": criteria},
        label=label,
        source=source,
        split="train",
        teacher_probs=teacher_probs,
    )


def _pool_items(
    n: int,
    source: str = "src",
    width: int = 4,
    teacher_probs: dict[str, float] | None = None,
) -> list[TypedQuestion]:
    items = []
    for i in range(n):
        criteria = {f"k{j}": f"{source} option {i}-{j}" for j in range(width)}
        items.append(
            _choice(f"{source}-{i}", f"k{i % width}", criteria, source=source, teacher_probs=teacher_probs)
        )
    return items


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        KAugmentConfig(fraction=1.5)
    with pytest.raises(ValueError):
        KAugmentConfig(min_k=1)
    with pytest.raises(ValueError):
        KAugmentConfig(min_k=32, max_k=16)


def test_default_max_k_stays_within_evaluation_width():
    # The widest evaluation item is banking77 at 77 options.  Padding past it
    # buys widths nothing is measured at and most of the extra encode cost.
    assert KAugmentConfig().max_k == 80


def test_draw_target_k_stays_in_range():
    cfg = KAugmentConfig(min_k=8, max_k=256)
    rng = random.Random(0)
    values = [draw_target_k(rng, cfg) for _ in range(2000)]
    assert min(values) >= 8
    assert max(values) <= 256
    # Log-uniform must not pile up at the top of the range.
    assert sum(v <= 32 for v in values) / len(values) > 0.3


def test_pad_preserves_gold_and_original_options():
    items = _pool_items(40)
    target = items[0]
    pool = build_distractor_pool(items)

    padded = pad_choice_item(target, target_k=32, pool=pool, rng=random.Random(1))

    assert padded is not target
    assert padded.label == target.label
    assert padded.label in padded.question["criteria"]
    assert len(padded.question["criteria"]) >= 32
    for key, value in target.question["criteria"].items():
        assert padded.question["criteria"][key] == value
    # State, instructions and identity are unchanged.
    assert padded.state == target.state
    assert padded.id == target.id
    assert padded.question["instructions"] == target.question["instructions"]


def test_pad_adds_distinct_texts():
    items = _pool_items(60)
    pool = build_distractor_pool(items)
    padded = pad_choice_item(items[0], target_k=64, pool=pool, rng=random.Random(2))
    texts = [_norm(v) for v in padded.question["criteria"].values()]
    assert len(texts) == len(set(texts)), "a distractor duplicated another option's text"


def _norm(text: str) -> str:
    return text.strip().casefold()


def test_pad_rejects_case_and_whitespace_variants_of_gold():
    item = _choice("t", "gold", {"gold": "True", "other": "Other option"})
    pool = {"src": ["true", "  TRUE  ", "valid distractor", "aDdItIoNaL text"]}
    padded = pad_choice_item(item, target_k=4, pool=pool, rng=random.Random(0))
    added = set(padded.question["criteria"].values()) - set(item.question["criteria"].values())
    assert {_norm(t) for t in added} == {"valid distractor", "additional text"}


def test_pad_drops_teacher_probs():
    item = _choice("t", "k0", {"k0": "a", "k1": "b"}, teacher_probs={"k0": 0.7, "k1": 0.3})
    pool = {"src": ["c", "d", "e"]}
    padded = pad_choice_item(item, target_k=4, pool=pool, rng=random.Random(0))
    assert padded.teacher_probs is None, "teacher mass on unseen options would teach impossibility"


def test_pad_is_a_noop_when_target_not_wider():
    items = _pool_items(10)
    pool = build_distractor_pool(items)
    item = items[0]
    assert pad_choice_item(item, target_k=2, pool=pool, rng=random.Random(0)) is item


def test_pad_prefers_same_source():
    a = _choice("a", "k0", {"k0": "alpha zero", "k1": "alpha one"}, source="alpha")
    b = _choice("b", "k0", {"k0": "beta zero", "b1": "beta one"}, source="beta")
    pool = build_distractor_pool([a, b])

    padded = pad_choice_item(a, target_k=4, pool=pool, rng=random.Random(0))
    added = set(padded.question["criteria"].values()) - set(a.question["criteria"].values())
    # Only one same-source text is available, so exactly one foreign text sneaks in.
    assert "beta zero" in added or "beta one" in added
    assert len(added) == 2


def test_pad_skips_unpaddable_items():
    pool = {"src": ["c", "d"]}
    rng = random.Random(0)

    noul = TypedQuestion(
        id="n", state="s", question={"type": "noul", "instructions": "i"},
        label=True, source="src", split="train",
    )
    assert pad_choice_item(noul, target_k=8, pool=pool, rng=rng) is noul

    list_criteria = TypedQuestion(
        id="l", state="s",
        question={"type": "choice", "instructions": "i", "criteria": ["a", "b"]},
        label="a", source="src", split="train",
    )
    assert pad_choice_item(list_criteria, target_k=8, pool=pool, rng=rng) is list_criteria

    missing_label = _choice("m", "absent", {"k0": "a", "k1": "b"})
    assert pad_choice_item(missing_label, target_k=8, pool=pool, rng=rng) is missing_label


def test_pad_borrows_only_texts_from_the_supplied_items():
    # The provenance boundary: distractors are borrowed from the items passed
    # to the augmenter, which in training is the sampled training epoch.  An
    # item that was not passed (e.g. evaluation data) cannot reach the pool.
    train = [_choice("t", "g", {"g": "train gold text", "o": "train other text"}, source="train")]
    held_out = [_choice("e", "g", {"g": "eval gold text", "x": "eval only text"}, source="eval")]

    pool = build_distractor_pool(train)
    padded = pad_choice_item(train[0], target_k=3, pool=pool, rng=random.Random(0))
    added = set(padded.question["criteria"].values()) - set(train[0].question["criteria"].values())

    supplied = {v for item in train for v in item.question["criteria"].values()}
    held_out_texts = {v for item in held_out for v in item.question["criteria"].values()}
    assert added <= supplied
    assert not (added & held_out_texts)


def test_pad_does_not_mutate_the_pool():
    items = _pool_items(30)
    pool = build_distractor_pool(items)
    snapshot = {source: list(texts) for source, texts in pool.items()}
    pad_choice_item(items[0], target_k=16, pool=pool, rng=random.Random(3))
    assert pool == snapshot


def test_apply_keeps_length_and_order_and_only_touches_choice():
    items = _pool_items(200)
    items.append(TypedQuestion(
        id="n", state="s", question={"type": "noul", "instructions": "i"},
        label=True, source="src", split="train",
    ))
    cfg = KAugmentConfig(fraction=1.0, min_k=16, max_k=64, seed=7)

    out = apply_k_augment(items, cfg, epoch=0)

    assert len(out) == len(items)
    assert [o.id for o in out] == [i.id for i in items]
    assert out[-1] is items[-1], "non-choice items must pass through untouched"
    widths = [len(o.question["criteria"]) for o in out[:-1]]
    assert min(widths) >= 16


def test_apply_is_deterministic_per_epoch_and_varies_across_epochs():
    items = _pool_items(100)
    cfg = KAugmentConfig(fraction=1.0, min_k=16, max_k=256, seed=11)

    a = [len(i.question["criteria"]) for i in apply_k_augment(items, cfg, epoch=0)]
    b = [len(i.question["criteria"]) for i in apply_k_augment(items, cfg, epoch=0)]
    c = [len(i.question["criteria"]) for i in apply_k_augment(items, cfg, epoch=1)]

    assert a == b, "same epoch must reproduce the same widths"
    assert a != c, "a new epoch should explore different widths"


def test_fraction_zero_is_a_passthrough():
    items = _pool_items(50)
    out = apply_k_augment(items, KAugmentConfig(fraction=0.0), epoch=0)
    assert out is items


def test_fraction_controls_the_share_selected():
    items = _pool_items(400)
    cfg = KAugmentConfig(fraction=0.25, min_k=16, max_k=64, seed=3)
    out = apply_k_augment(items, cfg, epoch=0)
    padded = sum(
        1 for o, i in zip(out, items) if len(o.question["criteria"]) > len(i.question["criteria"])
    )
    assert 0.15 * len(items) <= padded <= 0.35 * len(items)


def test_distractor_pool_dedupes_and_skips_short_or_blank_values():
    items = [
        _choice("a", "k0", {"k0": "shared", "k1": "shared"}),
        _choice("b", "k0", {"k0": "unique", "k1": ""}),
        _choice("c", "k0", {"k0": "ab", "k1": "  real text  "}),
    ]
    pool = build_distractor_pool(items)
    # Blank and sub-3-character values are skipped, not replaced by their keys:
    # a bare "k1" in the option list carries no signal.
    assert pool["src"] == ["shared", "unique", "real text"]


def test_selected_item_drops_teacher_probs_even_when_not_padded():
    # target_k == natural: nothing to add, but the selected item must still lose
    # its teacher distribution so the control arm can match it.
    item = _choice("t", "k0", {"k0": "a", "k1": "b"}, teacher_probs={"k0": 0.7, "k1": 0.3})
    cfg = KAugmentConfig(fraction=1.0, min_k=2, max_k=2, seed=1)
    out = apply_k_augment([item], cfg, epoch=0)
    assert len(out[0].question["criteria"]) == 2
    assert out[0].teacher_probs is None


def test_control_matches_the_augmented_selection_without_padding():
    teacher = {"k0": 0.5, "k1": 0.5}
    items = _pool_items(200, teacher_probs=teacher)
    aug_cfg = KAugmentConfig(fraction=0.3, min_k=16, max_k=64, seed=5)
    ctl_cfg = KAugmentConfig(fraction=0.3, min_k=16, max_k=64, seed=5, control=True)

    aug, aug_stats = apply_k_augment_with_stats(items, aug_cfg, epoch=2)
    ctl, ctl_stats = apply_k_augment_with_stats(items, ctl_cfg, epoch=2)

    # Same coin, same items: control copies an item exactly when the augmented
    # arm touched it, and never widens one.
    touched_aug = {i for i, (a, orig) in enumerate(zip(aug, items)) if a is not orig}
    touched_ctl = {i for i, (c, orig) in enumerate(zip(ctl, items)) if c is not orig}
    assert touched_aug == touched_ctl
    assert aug_stats.selected == ctl_stats.selected == len(touched_aug)
    assert ctl_stats.padded == 0
    assert ctl_stats.teacher_probs_dropped == aug_stats.teacher_probs_dropped == ctl_stats.selected
    for c, orig in zip(ctl, items):
        assert len(c.question["criteria"]) == len(orig.question["criteria"])
        if c is not orig:
            assert c.teacher_probs is None
        else:
            assert c.teacher_probs == teacher


def test_stats_report_under_fill_when_the_pool_is_too_thin():
    items = _pool_items(10, width=2)  # 20 distinct texts in the whole pool
    cfg = KAugmentConfig(fraction=1.0, min_k=100, max_k=100, seed=1)
    _, stats = apply_k_augment_with_stats(items, cfg, epoch=0)

    assert stats.selected == stats.padded == len(items)
    assert stats.under_filled == len(items)
    assert stats.target_mean == 100.0
    assert stats.realised_max < 100
    assert stats.encodes_after < stats.target_mean * len(items)


def test_stats_report_encodes_and_padding_counts():
    items = _pool_items(20, width=4)
    cfg = KAugmentConfig(fraction=1.0, min_k=8, max_k=8, seed=2)
    out, stats = apply_k_augment_with_stats(items, cfg, epoch=0)

    assert stats.choice_items == len(items)
    assert stats.selected == len(items)
    assert stats.padded == len(items)
    assert stats.under_filled == 0
    assert stats.encodes_before == 4 * len(items)
    assert stats.encodes_after == sum(len(i.question["criteria"]) for i in out)
    assert stats.encodes_after > stats.encodes_before


def _wide_item(n_options: int = 80) -> TypedQuestion:
    criteria = {f"intent_{i:02d}": f"text {i}" for i in range(n_options)}
    return TypedQuestion(
        id="wide-1",
        state="I want to order a coffee",
        question={"type": "choice", "instructions": "Pick", "criteria": criteria},
        label="intent_07",
        source="src",
        split="test",
    )


def test_shrink_keeps_the_gold_and_the_requested_width():
    shrink = shrink_choice_item
    item = _wide_item(80)

    shrunk = shrink(item, 4, random.Random(0))

    criteria = shrunk.question["criteria"]
    assert len(criteria) == 4
    assert item.label in criteria
    assert shrunk.label == item.label


def test_shrink_is_a_no_op_at_or_above_the_natural_width():
    shrink = shrink_choice_item
    item = _wide_item(80)

    # This is the bug the function exists to prevent: a pad-only sweep left the
    # item untouched at every K below its natural width and scored it six times.
    assert shrink(item, 80, random.Random(0)) is item
    assert shrink(item, 120, random.Random(0)) is item


def test_shrink_preserves_source_order_of_the_kept_options():
    shrink = shrink_choice_item
    item = _wide_item(80)

    shrunk = shrink(item, 5, random.Random(3))

    kept = list(shrunk.question["criteria"])
    original = list(item.question["criteria"])
    assert kept == [key for key in original if key in set(kept)]


def test_shrink_is_deterministic_for_a_seed_and_varies_across_seeds():
    shrink = shrink_choice_item
    item = _wide_item(80)

    first = list(shrink(item, 6, random.Random(11)).question["criteria"])
    again = list(shrink(item, 6, random.Random(11)).question["criteria"])
    other = list(shrink(item, 6, random.Random(12)).question["criteria"])

    assert first == again
    assert first != other


def test_shrink_leaves_non_choice_and_label_mismatched_items_alone():
    shrink = shrink_choice_item
    noul = TypedQuestion(
        id="n-1",
        state="s",
        question={"type": "noul", "instructions": "i", "criteria": {"yes": "", "no": ""}},
        label="true",
        source="src",
        split="test",
    )
    missing = _wide_item(80)
    missing = replace(missing, label="not_a_key")

    assert shrink(noul, 1, random.Random(0)) is noul
    assert shrink(missing, 4, random.Random(0)) is missing
