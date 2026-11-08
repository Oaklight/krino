"""The judgement readout's option construction (#277 follow-up).

The readout decides what text sits after ``Option:`` and which index is the gold
answer. Both are easy to get wrong in a way that still produces a number, so they
are pinned here without loading a model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


class _FixedScorer:
    """Return the same ``(log P(yes), log P(no))`` pair for every option."""

    def __init__(self, yes: float, no: float):
        self.yes = yes
        self.no = no
        self.prompts: list[str] = []

    def score_options(self, prompt, options):
        assert options == [" yes", " no"]
        self.prompts.append(prompt)
        return [self.yes, self.no]


def test_option_branch_ranks_on_yes_and_records_the_no_side():
    # #310: the ranking statistic is log P(" yes") alone. log P(" no") is
    # recorded beside it, and must not enter ``logits``.
    pytest.importorskip("torch")
    from model.scripts.eval_jevbench_judgement import prediction_row, score_item

    scorer = _FixedScorer(yes=-0.75, no=-2.25)
    item = _item("choice", {"a": "First", "b": "Second"}, "a")
    record = score_item(scorer, item)

    assert record["mode"] == "option"
    assert record["logits"] == [-0.75, -0.75]
    assert record["logp_yes"] == [-0.75, -0.75]
    assert record["logp_no"] == [-2.25, -2.25]
    assert record["label_idx"] == 0
    assert len(scorer.prompts) == 2

    row = prediction_row(item, record)
    assert row["logp_yes"] == [-0.75, -0.75]
    assert row["logp_no"] == [-2.25, -2.25]
    assert row["predicted_idx"] == 0
    assert row["mode"] == "option"


def test_polarity_branch_records_both_sides_and_keeps_the_difference():
    # A criteria-less noul item is thresholded on log P(" yes") - log P(" no"),
    # so ``logits`` stays the signed difference while both sides are recorded.
    pytest.importorskip("torch")
    from model.scripts.eval_jevbench_judgement import prediction_row, score_item

    scorer = _FixedScorer(yes=-0.5, no=-1.5)
    item = _item("noul", None, True)
    record = score_item(scorer, item)

    assert record["mode"] == "polarity"
    assert record["logits"] == [-0.5 - -1.5]
    assert record["logp_yes"] == [-0.5]
    assert record["logp_no"] == [-1.5]
    assert record["correct"] is True
    assert len(scorer.prompts) == 1

    # The persisted row must not put the difference under a log-probability
    # name; that mapping is a separate step from the one ``score_item`` does.
    row = prediction_row(item, record)
    assert row["logp_yes"] == [-0.5]
    assert row["logp_no"] == [-1.5]
    assert row["predicted_idx"] == 0
    assert row["mode"] == "polarity"
