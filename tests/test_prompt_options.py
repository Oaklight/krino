"""Tests for the shared noul option construction (#278)."""

from __future__ import annotations

from model.evaluation.prompt_options import noul_option_texts


def test_criteria_phrasing_is_used_when_present():
    criteria = {"true": "supported", "false": "not supported"}
    assert noul_option_texts(criteria) == [" supported", " not supported"]


def test_missing_criteria_falls_back_to_yes_no():
    assert noul_option_texts(None) == [" yes", " no"]
    assert noul_option_texts({}) == [" yes", " no"]


def test_partial_criteria_falls_back_per_side():
    assert noul_option_texts({"true": "correct"}) == [" correct", " no"]
    assert noul_option_texts({"false": "incorrect"}) == [" yes", " incorrect"]


def test_true_always_comes_first():
    texts = noul_option_texts({"true": "a", "false": "b"})
    assert texts[0] == " a"
    assert texts[1] == " b"
