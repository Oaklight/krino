"""Tests for canonical label semantics (#234 / builder contract)."""

from __future__ import annotations

import pytest

from data.labels import (
    FRACTIONAL,
    LEVEL,
    LabelError,
    OPTION,
    WITHHELD_GOLD,
    annotate,
    label_kind,
)


def _choice(label, criteria=None, answerable=None):
    item = {
        "id": "c-1",
        "question": {"type": "choice", "criteria": criteria or {"a": "A", "b": "B"}},
        "label": label,
    }
    if answerable is not None:
        item["answerable"] = answerable
    return item


def _score(label, levels=5):
    return {
        "id": "s-1",
        "question": {"type": "score", "criteria": [f"L{i}" for i in range(levels)]},
        "label": label,
    }


def test_choice_label_in_criteria_is_option():
    assert label_kind(_choice("a")) == OPTION


def test_choice_withheld_gold_requires_answerable_false():
    assert label_kind(_choice("a", answerable=False)) == OPTION
    assert label_kind(_choice("withheld", answerable=False)) == WITHHELD_GOLD


def test_choice_label_outside_criteria_fails_with_address_and_keys():
    with pytest.raises(LabelError) as excinfo:
        label_kind(_choice("c"))
    message = str(excinfo.value)
    assert "c-1" in message
    assert "'c'" in message
    assert "['a', 'b']" in message


def test_score_integer_is_level():
    assert label_kind(_score(2)) == LEVEL
    assert label_kind(_score("3")) == LEVEL


def test_score_fraction_is_fractional():
    assert label_kind(_score(1.5)) == FRACTIONAL
    assert label_kind(_score(2.2)) == FRACTIONAL


def test_score_invalid_labels_fail_with_address():
    with pytest.raises(LabelError, match="s-1.*not numeric"):
        label_kind(_score("high"))
    with pytest.raises(LabelError, match="s-1.*outside"):
        label_kind(_score(9))


def test_noul_has_no_label_kind_and_annotate_leaves_it_alone():
    item = {
        "id": "n-1",
        "question": {"type": "noul", "criteria": None},
        "label": True,
    }
    assert label_kind(item) is None
    assert annotate(item) is item
    assert "label_kind" not in item


def test_annotate_writes_kind_on_choice_and_score_rows():
    choice = annotate(_choice("a"))
    score = annotate(_score(1.5))
    assert choice["label_kind"] == OPTION
    assert score["label_kind"] == FRACTIONAL
