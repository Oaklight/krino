"""The judgement readout's option construction (#277 follow-up).

The readout decides what text sits after ``Option:`` and which index is the gold
answer. Both are easy to get wrong in a way that still produces a number, so they
are pinned here without loading a model.
"""

from __future__ import annotations

from types import SimpleNamespace

from model.evaluation.judgement_prompt import (
    JUDGEMENT_TEMPLATE,
    NO,
    YES,
    candidates,
    render,
)


def _item(q_type: str, criteria, label, instructions="Do the thing."):
    return SimpleNamespace(
        id="easy-intent-00",
        state="A situation.",
        question={"type": q_type, "instructions": instructions, "criteria": criteria},
        label=label,
    )


def test_noul_keeps_the_questions_own_wording_true_first():
    texts, gold = candidates(_item("noul", {"true": "Sí", "false": "No"}, True))
    assert texts == ["Sí", "No"]
    assert gold == 0


def test_noul_false_label_points_at_the_second_option():
    texts, gold = candidates(_item("noul", None, False))
    assert texts == ["yes", "no"]
    assert gold == 1


def test_choice_presents_descriptions_and_indexes_the_gold_key():
    item = _item("choice", {"a": "First", "b": "Second", "c": "Third"}, "b")
    texts, gold = candidates(item)
    assert texts == ["First", "Second", "Third"]
    assert gold == 1


def test_choice_returns_none_when_the_label_is_not_an_option():
    assert candidates(_item("choice", {"a": "First"}, "zzz")) is None


def test_score_presents_named_levels_and_rounds_the_gold():
    texts, gold = candidates(_item("score", ["low", "mid", "high"], 2.4))
    assert texts == ["Level 0: low", "Level 1: mid", "Level 2: high"]
    assert gold == 2


def test_template_ends_with_the_answer_cue():
    assert render("S", "Q", "O") == (
        "State: S\nQuestion: Q\nOption: O\n"
        "Does this option correctly answer the question? Answer:"
    )
    assert JUDGEMENT_TEMPLATE.endswith("Answer:")


def test_the_scored_pair_is_yes_then_no():
    assert (YES, NO) == (" yes", " no")
