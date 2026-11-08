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
    POLARITY_TEMPLATE,
    YES,
    candidates,
    polarity_gold,
    render,
    render_polarity,
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


def test_noul_without_criteria_is_not_an_option_set():
    # No criteria means no option text to judge; the caller reads polarity at
    # Answer: instead of asking whether the word "yes" answers the question.
    assert candidates(_item("noul", None, False)) is None
    assert candidates(_item("noul", {}, True)) is None


def test_polarity_gold_reads_a_boolean_label():
    assert polarity_gold(_item("noul", None, True)) is True
    assert polarity_gold(_item("noul", None, False)) is False
    assert polarity_gold(_item("noul", None, "yes")) is True
    assert polarity_gold(_item("noul", None, "False")) is False


def test_polarity_branch_never_fires_on_a_criteria_bearing_noul():
    # Every JevBench noul item carries criteria, so the acceptance set cannot
    # reach this branch and the repair cannot be tuned against it.
    assert polarity_gold(_item("noul", {"true": "Sí", "false": "No"}, True)) is None


def test_polarity_branch_is_noul_only():
    assert polarity_gold(_item("choice", None, "a")) is None
    assert polarity_gold(_item("score", ["low", "high"], 1)) is None


def test_polarity_template_has_no_option_line():
    assert render_polarity("S", "Q") == "State: S\nQuestion: Q\nAnswer:"
    assert "Option:" not in POLARITY_TEMPLATE


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


def test_record_correct_honours_an_explicit_outcome():
    from model.evaluation.jevbench import record_correct

    # A one-proposition record states its own outcome; the argmax over a single
    # logit would otherwise be correct by construction.
    assert record_correct(
        {"logits": [1.0], "label_order": [0], "label_idx": 0, "correct": False}
    ) is False
    assert record_correct(
        {"logits": [-1.0], "label_order": [0], "label_idx": 0, "correct": True}
    ) is True
