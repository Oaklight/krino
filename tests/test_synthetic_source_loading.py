"""Tests for source loading in the multitask training pipeline.

Covers the synthetic special case, the shared synthetic predicate, the default
path convention, and the warn-on-empty guard. Imports nothing from torch, so
it runs on a bare interpreter (and therefore in CI).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from data.format import TypedQuestion
from data.sources import (
    DEFAULT_BENCHMARK_PREFIX,
    collect_synthetic_sources,
    is_synthetic_source,
    load_source_items,
    resolve_path,
)


def _make_item(
    item_id: str, split: str = "train", qtype: str = "noul"
) -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state="state",
        question={"type": qtype, "instructions": "q?"},
        label=True,
        source="synthetic",
        split=split,
    )


# ---------------------------------------------------------------------------
# Synthetic special case
# ---------------------------------------------------------------------------


def test_synthetic_without_path_uses_load_synthetic(monkeypatch):
    """A pathless synthetic source must go through load_synthetic()."""
    import data.sources as src

    sentinel = [
        _make_item("synthetic-a-0001-noul-x"),
        _make_item("synthetic-a-0002-noul-y"),
    ]
    called = {"n": 0}

    def fake_load_synthetic():
        called["n"] += 1
        return iter(sentinel)

    monkeypatch.setattr(src, "load_synthetic", fake_load_synthetic)

    train_items, eval_items = load_source_items(
        "synthetic", {"weight": 1.0}, random.Random(0)
    )

    assert called["n"] == 1
    assert [it.id for it in train_items] == [it.id for it in sentinel]
    assert eval_items == []


def test_synthetic_respects_max_train(monkeypatch):
    """max_train caps the synthetic train split."""
    import data.sources as src

    items = [_make_item(f"synthetic-a-{i:04d}-noul-x") for i in range(50)]
    monkeypatch.setattr(src, "load_synthetic", lambda: iter(items))

    train_items, _ = load_source_items(
        "synthetic", {"weight": 1.0, "max_train": 10}, random.Random(0)
    )
    assert len(train_items) == 10


def test_synthetic_empty_assembly_warns_and_returns_empty(monkeypatch, capsys):
    """An empty synthetic assembly must warn rather than fail silently."""
    import data.sources as src

    monkeypatch.setattr(src, "load_synthetic", lambda: iter([]))

    train_items, eval_items = load_source_items(
        "synthetic", {"weight": 1.0}, random.Random(0)
    )

    assert train_items == []
    assert eval_items == []
    assert "assembled 0 items" in capsys.readouterr().err


def test_synthetic_explicit_path_still_read_from_file(monkeypatch, tmp_path):
    """An explicit path overrides the on-the-fly assembly."""
    import data.sources as src

    def boom():
        raise AssertionError("load_synthetic must not be called when path is set")

    monkeypatch.setattr(src, "load_synthetic", boom)

    jsonl = tmp_path / "synthetic.jsonl"
    jsonl.write_text(json.dumps(_make_item("from-file").to_dict()) + "\n")

    train_items, _ = load_source_items(
        "synthetic", {"path": str(jsonl), "weight": 1.0}, random.Random(0)
    )
    assert [it.id for it in train_items] == ["from-file"]


# ---------------------------------------------------------------------------
# Shared synthetic predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cfg,expected",
    [
        ("synthetic", {}, True),
        ("synthetic_aug", {}, True),
        ("banking77", {}, False),
        ("banking77", {"path": "data/benchmarks/synthetic/x.jsonl"}, True),
        ("banking77", {"path": "data/benchmarks/banking77.jsonl"}, False),
    ],
)
def test_is_synthetic_source(name, cfg, expected):
    assert is_synthetic_source(name, cfg) is expected


_SOURCE_MATRIX = [
    ("synthetic", {}),
    ("synthetic_aug", {}),
    ("plain_noul", {}),
    ("by_synthetic_path", {"path": "data/benchmarks/synthetic/x.jsonl"}),
    ("plain_bench", {"path": "data/benchmarks/plain_bench.jsonl"}),
]


def test_collect_synthetic_sources_agrees_with_predicate():
    """The sampler's classification is exactly the shared predicate's."""
    sources_cfg = dict(_SOURCE_MATRIX)
    expected = {name for name, cfg in _SOURCE_MATRIX if is_synthetic_source(name, cfg)}
    assert expected == {"synthetic", "synthetic_aug", "by_synthetic_path"}
    assert collect_synthetic_sources(sources_cfg) == expected


def test_collect_synthetic_sources_override_wins():
    """An explicit sampling.synthetic_sources list bypasses auto-detection."""
    sources_cfg = {"synthetic": {}, "banking77": {}}
    assert collect_synthetic_sources(sources_cfg, ["banking77"]) == {"banking77"}
    assert collect_synthetic_sources(sources_cfg, []) == set()


def test_synthetic_named_source_without_path_uses_assembly(monkeypatch):
    """A source named `synthetic_*` with no path is treated as synthetic."""
    import data.sources as src

    sentinel = [_make_item("synthetic-aug-0001-noul-x")]
    monkeypatch.setattr(src, "load_synthetic", lambda: iter(sentinel))

    train_items, _ = load_source_items(
        "synthetic_aug", {"weight": 1.0}, random.Random(0)
    )
    assert [it.id for it in train_items] == ["synthetic-aug-0001-noul-x"]


# ---------------------------------------------------------------------------
# Default path convention
# ---------------------------------------------------------------------------


def test_default_path_uses_data_benchmarks_prefix(capsys):
    """A pathless non-synthetic source resolves under data/benchmarks/."""
    train_items, eval_items = load_source_items(
        "no_such_source", {"weight": 1.0}, random.Random(0)
    )

    assert train_items == []
    assert eval_items == []
    err = capsys.readouterr().err
    assert f"{DEFAULT_BENCHMARK_PREFIX}/no_such_source.jsonl" in err
    assert "model/data/" not in err


def test_resolve_path_anchors_at_repo_root():
    """ROOT must point at the repo root, so the default prefix resolves there."""
    import data.sources as src

    # tests/ sits directly under the repo root, so parents[1] is the root.
    assert src.ROOT == Path(__file__).resolve().parents[1]


def test_resolve_path_joins_onto_repo_root():
    resolved = resolve_path(DEFAULT_BENCHMARK_PREFIX)
    assert resolved.is_absolute()
    assert resolved.name == "benchmarks"


def test_pathless_source_loads_from_default_prefix(monkeypatch, tmp_path):
    """Pin prefix + ROOT + load together: a fixture under the prefix is found."""
    import data.sources as src

    monkeypatch.setattr(src, "ROOT", tmp_path)
    fixture_dir = tmp_path / DEFAULT_BENCHMARK_PREFIX
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "fixture_src.jsonl").write_text(
        json.dumps(_make_item("from-default-path").to_dict()) + "\n"
    )

    train_items, _ = load_source_items("fixture_src", {"weight": 1.0}, random.Random(0))
    assert [it.id for it in train_items] == ["from-default-path"]
