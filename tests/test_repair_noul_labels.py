"""Tests for noul label repair: agreement filter, gold mix, band filter."""

from __future__ import annotations

import json
from pathlib import Path

from data.format import TypedQuestion
from data.pipeline import load_jsonl, save_jsonl
from model.scripts.repair_noul_labels import (
    gold_bool,
    main as repair_main,
    mix_with_gold,
    p_gold_from_base,
    teacher_argmax_true,
    teacher_distribution,
)


def _noul(item_id: str = "n1", label: bool = True, probs=None, **changes) -> TypedQuestion:
    defaults = dict(
        id=item_id,
        state="the evidence supports the claim",
        question={"type": "noul", "instructions": "is it supported?"},
        label=label,
        source="fixture",
        split="train",
        teacher_probs={"jev": probs if probs is not None else {"true": 0.8, "false": 0.2}},
        teacher_name="jev",
    )
    defaults.update(changes)
    return TypedQuestion(**defaults)


def _choice(item_id: str = "c1") -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        label="a",
        source="fixture",
        split="train",
        teacher_probs={"jev": {"a": 0.9, "b": 0.1}},
    )


def test_teacher_distribution_reads_namespaced_and_flat():
    dist, ns = teacher_distribution({"jev": {"true": 0.7, "false": 0.3}})
    assert ns == "jev" and dist == {"true": 0.7, "false": 0.3}

    dist, ns = teacher_distribution({"true": 0.7, "false": 0.3})
    assert ns is None and dist == {"true": 0.7, "false": 0.3}

    assert teacher_distribution({"jev": {"borderline": 0.5}}) is None
    assert teacher_distribution(None) is None


def test_gold_bool_and_argmax():
    assert gold_bool(True) is True
    assert gold_bool("yes") is True
    assert gold_bool("FALSE") is False
    assert gold_bool(0.5) is None
    assert teacher_argmax_true({"true": 0.8, "false": 0.2}) is True
    assert teacher_argmax_true({"no": 0.9, "yes": 0.1}) is False


def test_mix_with_gold_flips_a_confident_wrong_teacher():
    mixed = mix_with_gold({"true": 0.8, "false": 0.2}, gold=False, weight=0.5)
    assert abs(mixed["true"] - 0.4) < 1e-9
    assert abs(mixed["false"] - 0.6) < 1e-9
    assert abs(sum(mixed.values()) - 1.0) < 1e-9

    hard = mix_with_gold({"true": 0.8, "false": 0.2}, gold=False, weight=1.0)
    assert hard == {"true": 0.0, "false": 1.0}

    yes_no = mix_with_gold({"yes": 0.2, "no": 0.8}, gold=True, weight=0.5)
    assert abs(yes_no["yes"] - 0.6) < 1e-9
    assert abs(yes_no["no"] - 0.4) < 1e-9


def test_p_gold_from_base_maps_true_false_and_yes_no():
    row = {"keys": ["false", "true"], "probs": [0.3, 0.7]}
    assert abs(p_gold_from_base(row, True) - 0.7) < 1e-9
    assert abs(p_gold_from_base(row, False) - 0.3) < 1e-9
    row = {"keys": ["no", "yes"], "probs": [0.2, 0.8]}
    assert abs(p_gold_from_base(row, True) - 0.8) < 1e-9
    assert p_gold_from_base({"keys": ["a", "b"], "probs": [0.5, 0.5]}, True) is None


def _run(tmp_path: Path, items, *extra: str):
    items_path = tmp_path / "items.jsonl"
    out_path = tmp_path / "out.jsonl"
    save_jsonl(items, items_path)
    import sys

    argv = sys.argv
    sys.argv = [
        "repair_noul_labels.py",
        "--items", str(items_path),
        "--out", str(out_path),
        *extra,
    ]
    try:
        assert repair_main() == 0
    finally:
        sys.argv = argv
    return load_jsonl(out_path), json.loads(out_path.with_suffix(".stats.json").read_text())


def test_agreement_keep_drops_wrong_noul_and_keeps_other_types(tmp_path: Path):
    items = [
        _noul("agree", True, {"true": 0.9, "false": 0.1}),
        _noul("disagree", True, {"true": 0.1, "false": 0.9}),
        _choice("choice"),
    ]
    out, stats = _run(tmp_path, items, "--noul-agreement", "keep")
    assert [item.id for item in out] == ["agree", "choice"]
    assert stats["counts"]["noul_agree"] == 1
    assert stats["counts"]["noul_disagree"] == 1
    assert stats["counts"]["dropped_agreement"] == 1


def test_agreement_exclude_keeps_only_wrong_noul(tmp_path: Path):
    items = [
        _noul("agree", True, {"true": 0.9, "false": 0.1}),
        _noul("disagree", True, {"true": 0.1, "false": 0.9}),
    ]
    out, _ = _run(tmp_path, items, "--noul-agreement", "exclude")
    assert [item.id for item in out] == ["disagree"]


def test_gold_mix_rewrites_namespaced_distribution(tmp_path: Path):
    items = [_noul("n1", label=False, probs={"true": 0.8, "false": 0.2})]
    out, stats = _run(tmp_path, items, "--noul-gold-mix", "0.5")
    dist = out[0].teacher_probs["jev"]
    assert abs(dist["true"] - 0.4) < 1e-9
    assert abs(dist["false"] - 0.6) < 1e-9
    assert out[0].teacher_name == "jev"
    assert stats["counts"]["noul_gold_mixed"] == 1


def test_band_upper_drops_easy_noul(tmp_path: Path):
    items = [
        _noul("easy", True),
        _noul("hard", True),
        _choice("choice"),
    ]
    base_path = tmp_path / "base.jsonl"
    rows = [
        {"id": "easy", "q_type": "noul", "keys": ["false", "true"], "probs": [0.1, 0.9]},
        {"id": "hard", "q_type": "noul", "keys": ["false", "true"], "probs": [0.7, 0.3]},
        {"id": "choice", "q_type": "choice", "keys": ["a", "b"], "probs": [0.6, 0.4]},
    ]
    base_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out, stats = _run(
        tmp_path, items,
        "--base-probs", str(base_path),
        "--noul-band-upper", "0.6",
    )
    # p(gold=True): easy 0.9 -> dropped, hard 0.3 -> kept, choice untouched.
    assert [item.id for item in out] == ["hard", "choice"]
    assert stats["counts"]["dropped_band"] == 1


def test_augmented_noul_and_unusable_targets_are_kept(tmp_path: Path):
    augmented = _noul(
        "aug",
        True,
        {"borderline": 0.6, "more_info_needed": 0.4},
        question={
            "type": "noul",
            "instructions": "q",
            "augmented_options": {"borderline": "b", "more_info_needed": "m"},
        },
    )
    missing = _noul("missing", True, None)
    missing = TypedQuestion(**{**missing.to_dict(), "teacher_probs": None})
    out, stats = _run(
        tmp_path, [augmented, missing],
        "--noul-agreement", "keep",
        "--noul-gold-mix", "0.5",
    )
    assert [item.id for item in out] == ["aug", "missing"]
    assert out[0].teacher_probs["jev"] == {"borderline": 0.6, "more_info_needed": 0.4}
    assert stats["counts"]["noul_unusable_target"] == 1


def test_mixed_flat_and_namespaced_prefers_the_teacher_namespace():
    dist, ns = teacher_distribution(
        {"true": 0.56, "false": 0.44, "jev": {"true": 0.52, "false": 0.48}}
    )
    assert ns == "jev"
    assert dist == {"true": 0.52, "false": 0.48}


def test_extra_noul_appends_new_noul_only(tmp_path: Path):
    base_items = [_noul("a", True), _choice("choice")]
    extra_items = [_noul("b", True), _noul("a", True), _choice("other_choice")]
    base_path = tmp_path / "base.jsonl"
    extra_path = tmp_path / "extra.jsonl"
    save_jsonl(base_items, base_path)
    save_jsonl(extra_items, extra_path)
    out_path = tmp_path / "out.jsonl"
    import sys

    argv = sys.argv
    sys.argv = [
        "repair_noul_labels.py",
        "--items", str(base_path),
        "--extra-noul", str(extra_path),
        "--out", str(out_path),
    ]
    try:
        assert repair_main() == 0
    finally:
        sys.argv = argv
    out = load_jsonl(out_path)
    assert {item.id for item in out} == {"a", "b", "choice"}
    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["counts"]["extra_noul_added"] == 1
    assert stats["counts"]["extra_noul_duplicate"] == 1
    assert stats["counts"]["extra_noul_skipped"] == 1
