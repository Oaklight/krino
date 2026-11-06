"""Tests for the ``answerable=false`` corpus census.

The census is evidence, not a metric: it must count the rows the trainer would
otherwise drive a hard label on, keep the corpus path out of the output, and
separate an unanswerable row that still looks like ordinary labelled data.
"""

from __future__ import annotations

import json

from model.scripts.census_answerable import main


def _row(**overrides):
    row = {
        "id": "row",
        "state": "the evidence",
        "question": {"type": "noul", "instructions": "is it true?"},
        "label": False,
        "source": "unit",
        "split": "train",
        "answerable": False,
        "reason": "insufficient",
        "provenance": "execution",
    }
    row.update(overrides)
    return row


def _run(tmp_path, rows):
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "census.json"
    assert main(["--items", str(path), "--out", str(out)]) == 0
    return json.loads(out.read_text())


def test_the_dangerous_shape_is_counted_separately(tmp_path):
    rows = [
        _row(id="a", reason="insufficient"),
        _row(id="b", reason="out_of_scope"),
        # A false row whose label is not one of the options is unanswerable but
        # not the dangerous shape: the trainer must not train on that label.
        _row(
            id="c",
            question={"type": "choice", "criteria": {"a": "alpha", "b": "beta"}},
            label="gone",
            reason="none_of_these",
        ),
        _row(id="d", answerable=None, reason=None),
        _row(id="e", answerable=None),  # reason without answerable=false
        _row(id="f", reason=None),  # false without a reason
    ]
    result = _run(tmp_path, rows)
    false = result["answerable_false"]
    assert result["total"] == 6
    assert result["answerable_values"] == {"false": 4, "unset": 2}
    assert false["count"] == 4
    assert false["dangerous_well_formed"] == 3
    assert false["dangerous_by_reason"] == {
        "insufficient": 1,
        "out_of_scope": 1,
        "": 1,
    }
    assert false["dangerous_by_source"] == {"unit": 3}
    assert result["reason_without_answerable_false"] == 1
    assert result["answerable_false_without_reason"] == 1


def test_well_formed_options_and_label_validity_are_reported(tmp_path):
    rows = [
        _row(id="a", answerable=False),
        _row(
            id="b",
            question={"type": "choice", "criteria": {"a": "alpha", "b": "beta"}},
            label="a",
            reason="none_of_these",
        ),
        _row(
            id="c",
            question={"type": "score", "criteria": ["low", "mid", "high"]},
            label=2,
            reason="out_of_scope",
        ),
    ]
    false = _run(tmp_path, rows)["answerable_false"]
    assert false["well_formed_options"] == 3
    assert false["label_valid"] == 3
    assert false["label_in_options"] == 3
    assert false["dangerous_well_formed"] == 3
    assert false["by_type"] == {"choice": 1, "noul": 1, "score": 1}
    assert false["option_counts"] == {"3": 1, "2": 2}


def test_the_output_keeps_only_the_basename(tmp_path):
    result = _run(tmp_path, [_row()])
    assert result["items"] == "sample.jsonl"
    assert "/" not in result["items"]


def test_examples_are_bounded(tmp_path):
    rows = [_row(id=str(index)) for index in range(5)]
    result = _run(tmp_path, rows)
    assert len(result["answerable_false"]["examples"]) == 5
