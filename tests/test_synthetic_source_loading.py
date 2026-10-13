"""Tests for synthetic source loading in the multitask training script.

The `synthetic` source has no `path` in the R2 configs — its data is assembled
on the fly from families + variants + labels. These tests pin the special case
in `load_source_items()` that routes `synthetic` to `load_synthetic()`.
"""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from data.format import TypedQuestion  # noqa: E402


def _make_item(item_id: str, split: str = "train") -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": "noul", "instructions": "q?"},
        label=True,
        source="synthetic",
        split=split,
    )


def test_synthetic_without_path_uses_load_synthetic(monkeypatch):
    """A `synthetic` source with no path must go through load_synthetic()."""
    import model.scripts.train_multitask as tm

    sentinel = [_make_item("synthetic-a-0001-noul-x"), _make_item("synthetic-a-0002-noul-y")]
    called = {"n": 0}

    def fake_load_synthetic():
        called["n"] += 1
        return iter(sentinel)

    monkeypatch.setattr(tm, "load_synthetic", fake_load_synthetic)

    train_items, eval_items = tm.load_source_items("synthetic", {"weight": 1.0}, random.Random(0))

    assert called["n"] == 1
    assert [it.id for it in train_items] == [it.id for it in sentinel]
    assert eval_items == []


def test_synthetic_respects_max_train(monkeypatch):
    """max_train caps the synthetic train split."""
    import model.scripts.train_multitask as tm

    items = [_make_item(f"synthetic-a-{i:04d}-noul-x") for i in range(50)]
    monkeypatch.setattr(tm, "load_synthetic", lambda: iter(items))

    train_items, _ = tm.load_source_items("synthetic", {"weight": 1.0, "max_train": 10}, random.Random(0))
    assert len(train_items) == 10


def test_synthetic_explicit_path_still_read_from_file(monkeypatch, tmp_path):
    """An explicit path overrides the on-the-fly assembly."""
    import model.scripts.train_multitask as tm

    def boom():
        raise AssertionError("load_synthetic must not be called when path is set")

    monkeypatch.setattr(tm, "load_synthetic", boom)

    jsonl = tmp_path / "synthetic.jsonl"
    jsonl.write_text(json.dumps(_make_item("from-file").to_dict()) + "\n")

    train_items, _ = tm.load_source_items("synthetic", {"path": str(jsonl), "weight": 1.0}, random.Random(0))
    assert [it.id for it in train_items] == ["from-file"]


def test_non_synthetic_missing_path_warns_and_skips(monkeypatch, capsys):
    """Other sources keep the old behaviour: missing file means skip."""
    import model.scripts.train_multitask as tm

    train_items, eval_items = tm.load_source_items("does_not_exist", {"weight": 1.0}, random.Random(0))

    assert train_items == []
    assert eval_items == []
    assert "not found, skipping does_not_exist" in capsys.readouterr().err
