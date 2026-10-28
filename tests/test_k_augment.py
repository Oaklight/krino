"""Option-count augmentation must widen choice items without touching their labels.

The failure it addresses is measured, not hypothetical: the pooled-attention
baseline drops from 0.677 accuracy at 4 options to 0.217 at 256 on the same
items, because training never presents more than 80 options. These tests pin the
invariants that make padding safe — the gold key survives, no option text is
duplicated, keys cannot collide, non-choice items pass through untouched, and
the same (seed, epoch) reproduces the same widths.
"""

from __future__ import annotations

import pytest

from data.format import TypedQuestion
from data.k_augment import (
    KAugmentConfig,
    apply_k_augment,
    build_distractor_pool,
    draw_target_k,
    pad_choice_item,
)


def _choice(item_id: str, label: str, criteria: dict[str, str], source: str = "src") -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "choice", "instructions": "pick", "criteria": criteria},
        label=label,
        source=source,
        split="train",
    )


def _pool_items(n: int, source: str = "src", width: int = 4) -> list[TypedQuestion]:
    items = []
    for i in range(n):
        criteria = {f"k{j}": f"{source} option {i}-{j}" for j in range(width)}
        items.append(_choice(f"{source}-{i}", f"k{i % width}", criteria, source=source))
    return items


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        KAugmentConfig(fraction=1.5)
    with pytest.raises(ValueError):
        KAugmentConfig(min_k=1)
    with pytest.raises(ValueError):
        KAugmentConfig(min_k=32, max_k=16)


def test_draw_target_k_stays_in_range():
    cfg = KAugmentConfig(min_k=8, max_k=256)
    rng = __import__("random").Random(0)
    values = [draw_target_k(rng, cfg) for _ in range(2000)]
    assert min(values) >= 8
    assert max(values) <= 256
    # Log-uniform must not pile up at the top of the range.
    assert sum(v <= 32 for v in values) / len(values) > 0.3


def test_pad_preserves_gold_and_original_options():
    items = _pool_items(40)
    target = items[0]
    pool = build_distractor_pool(items)

    padded = pad_choice_item(target, target_k=32, pool=pool, rng=__import__("random").Random(1))

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
    padded = pad_choice_item(items[0], target_k=64, pool=pool, rng=__import__("random").Random(2))
    texts = list(padded.question["criteria"].values())
    assert len(texts) == len(set(texts)), "a distractor duplicated another option's text"


def test_pad_drops_teacher_probs():
    item = _choice("t", "k0", {"k0": "a", "k1": "b"})
    item.teacher_probs = {"k0": 0.7, "k1": 0.3}
    pool = {"src": ["c", "d", "e"]}
    padded = pad_choice_item(item, target_k=4, pool=pool, rng=__import__("random").Random(0))
    assert padded.teacher_probs is None, "teacher mass on unseen options would teach impossibility"


def test_pad_is_a_noop_when_target_not_wider():
    items = _pool_items(10)
    pool = build_distractor_pool(items)
    item = items[0]
    assert pad_choice_item(item, target_k=2, pool=pool, rng=__import__("random").Random(0)) is item


def test_pad_prefers_same_source():
    a = _choice("a", "k0", {"k0": "a0", "k1": "a1"}, source="alpha")
    b = _choice("b", "k0", {"k0": "b0", "b1": "b1"}, source="beta")
    pool = build_distractor_pool([a, b])

    padded = pad_choice_item(a, target_k=4, pool=pool, rng=__import__("random").Random(0))
    added = set(padded.question["criteria"].values()) - set(a.question["criteria"].values())
    # Only one same-source text is available, so exactly one foreign text sneaks in.
    assert "b0" in added or "b1" in added
    assert len(added) == 2


def test_pad_skips_unpaddable_items():
    pool = {"src": ["c", "d"]}
    rng = __import__("random").Random(0)

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


def test_distractor_pool_dedupes_and_falls_back_to_keys():
    items = [
        _choice("a", "k0", {"k0": "shared", "k1": "shared"}),
        _choice("b", "k0", {"k0": "unique", "k1": ""}),
    ]
    pool = build_distractor_pool(items)
    # A blank option text is not usable as a distractor, so the key stands in —
    # the same `value or key` rule the training path uses.
    assert pool["src"] == ["shared", "unique", "k1"]
