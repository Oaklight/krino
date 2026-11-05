"""Tests for the paired McNemar loader and test. Torch-free."""

from __future__ import annotations

import json

from model.scripts.paired_mcnemar import load, mcnemar

ROWS = [{"id": "a", "correct": True}, {"id": "b", "correct": False}]


def test_load_jsonl(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    assert load(path) == {"a": True, "b": False}


def test_load_json_object(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"predictions": ROWS}))
    assert load(path) == {"a": True, "b": False}


def test_mcnemar_exact():
    first = {"a": True, "b": True, "c": False}
    second = {"a": False, "b": True, "c": False}
    assert mcnemar(first, second, ["a", "b", "c"]) == (1, 0, 1.0)
