"""Tests for the surface-cue baseline."""

from __future__ import annotations

import json
from pathlib import Path

from model.scripts.abstention_surface_baseline import (
    _auc,
    _evaluate,
    _feature_names,
    _features,
    _read_pairs,
)


def _row(
    item_id: str,
    state: str,
    *,
    answerable: bool | None = None,
    reason: str | None = None,
    source: str = "boolq",
    qtype: str = "noul",
    n_options: int = 0,
) -> dict:
    question: dict = {"type": qtype, "instructions": "q"}
    if n_options:
        question["criteria"] = {f"o{i}": "x" for i in range(n_options)}
    row = {
        "id": item_id,
        "state": state,
        "question": question,
        "label": True,
        "source": source,
        "split": "train",
        "group": "g",
    }
    if answerable is not None:
        row["answerable"] = answerable
    if reason is not None:
        row["reason"] = reason
    return row


def test_auc_perfect_and_reverse():
    scores = [0.9, 0.8, 0.2, 0.1]
    assert _auc(scores, [1, 1, 0, 0]) == 1.0
    assert _auc(scores, [0, 0, 1, 1]) == 0.0


def test_auc_with_all_ties_is_chance():
    assert _auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_read_pairs_derives_base_ids(tmp_path: Path):
    original = _row("b1", "Acme makes widgets.", answerable=True)
    variant = _row(
        "b1-unans-support_removed-1",
        "Acme ships widgets.",
        answerable=False,
        reason="support_removed",
    )
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(original) + "\n" + json.dumps(variant) + "\n", encoding="utf-8"
    )
    originals, variants = _read_pairs(path)
    assert set(originals) == {"b1"}
    assert variants[0][0] == "b1"
    assert variants[0][1] == "support_removed"


def test_features_carry_no_words():
    a = _row("a", "alpha beta gamma delta", source="boolq")
    b = _row("b", "omega zeta theta sigma", source="boolq")
    sources = ["boolq"]
    types = ["noul"]
    first = _features(a, sources, types)
    second = _features(b, sources, types)
    assert first == second  # same surface shape, different words


def test_feature_names_include_source_and_type():
    rows = [
        _row("a", "x", source="boolq", qtype="noul"),
        _row("b", "y", source="fever", qtype="choice", n_options=3),
    ]
    names = _feature_names(rows)
    assert "source=boolq" in names
    assert "source=fever" in names
    assert "type=choice" in names
    assert "log_chars" in names


def test_a_length_shortcut_is_detected():
    rows = []
    for index in range(60):
        base = f"b{index}"
        original = _row(base, "x" * 180)
        variant = dict(original)
        variant["id"] = f"{base}-unans-entity_corrupted-1"
        variant["state"] = "x" * 240
        variant["answerable"] = False
        variant["reason"] = "entity_corrupted"
        for row, label in ((original, 0), (variant, 1)):
            row = dict(row)
            row["_base"] = base
            rows.append((row, label))
    result = _evaluate(rows, seed=3, train_fraction=0.5, ridge=1.0, draws=20)
    assert result["auc"] > 0.9


def test_equal_length_variants_are_near_chance():
    rows = []
    for index in range(80):
        base = f"b{index}"
        original = _row(base, ("x" * 40 + " ") * 3)
        variant = dict(original)
        variant["id"] = f"{base}-unans-entity_corrupted-1"
        variant["answerable"] = False
        variant["reason"] = "entity_corrupted"
        for row, label in ((original, 0), (variant, 1)):
            row = dict(row)
            row["_base"] = base
            rows.append((row, label))
    result = _evaluate(rows, seed=4, train_fraction=0.5, ridge=1.0, draws=20)
    assert 0.3 < result["auc"] < 0.7
