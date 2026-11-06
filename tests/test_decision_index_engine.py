"""Tests for the Decision Index engine adapter.

The adapter is deliberately torch-free at import time, so these tests run in the
normal environment even though the model that will use it needs torch.
"""

from __future__ import annotations

import pytest

from model.evaluation.gate import Prediction
from model.scripts.decision_index_engine import (
    KrinoEngine,
    Unsupported,
    capacity_violation,
)


class FakeScorer:
    """Minimal stand-in for ``RawReranker``: returns a fixed distribution."""

    def __init__(self, keys, probs):
        self.keys = tuple(keys)
        self.probs = tuple(probs)

    def predict_row(self, item):
        return Prediction(
            item_id=item.id,
            q_type=item.question["type"],
            label=None,
            probs=self.probs,
            keys=self.keys,
        )


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class CapacityScorer(FakeScorer):
    tokenizer = FakeTokenizer()
    prefix_tokens = ["<p>", "<p>"]
    suffix_tokens = ["</p>", "</p>"]
    shared_prefix = True
    max_length = 14


def choice_question():
    return {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Billing", "tech": "Technical", "account": "Account"},
    }


def noul_question():
    return {"type": "noul", "instructions": "Is this urgent?"}


def test_choice_keeps_every_option_and_returns_the_argmax():
    engine = KrinoEngine(
        FakeScorer(("billing", "tech", "account"), (0.2, 0.5, 0.3)),
        strict_length=False,
    )
    response, _ = engine("a state", {"q1": choice_question()})
    answer = response["answers"]["q1"]
    assert answer["type"] == "choice"
    assert answer["choice"] == "tech"
    assert answer["probabilities"] == {
        "billing": 0.2,
        "tech": 0.5,
        "account": 0.3,
    }


def test_probabilities_are_not_rounded():
    engine = KrinoEngine(
        FakeScorer(("billing", "tech", "account"), (0.123456789, 0.5, 0.376543211)),
        strict_length=False,
    )
    response, _ = engine("a state", {"q1": choice_question()})
    assert response["answers"]["q1"]["probabilities"]["billing"] == 0.123456789


def test_noul_returns_the_yes_probability():
    engine = KrinoEngine(FakeScorer(("no", "yes"), (0.3, 0.7)), strict_length=False)
    response, _ = engine("a state", {"q1": noul_question()})
    assert response["answers"]["q1"] == {"type": "noul", "noul": 0.7}


def test_every_question_in_a_request_is_answered():
    engine = KrinoEngine(FakeScorer(("no", "yes"), (0.3, 0.7)), strict_length=False)
    questions = {"a": noul_question(), "b": noul_question()}
    response, _ = engine("a state", questions)
    assert set(response["answers"]) == {"a", "b"}


def test_score_questions_are_refused():
    engine = KrinoEngine(FakeScorer(("0", "1"), (0.5, 0.5)), strict_length=False)
    question = {
        "type": "score",
        "instructions": "How bad?",
        "criteria": ["low", "high"],
    }
    with pytest.raises(Unsupported):
        engine("a state", {"q1": question})


def test_a_missing_option_is_an_error_not_a_silent_distribution():
    engine = KrinoEngine(
        FakeScorer(("billing", "tech"), (0.5, 0.5)), strict_length=False
    )
    with pytest.raises(ValueError, match="omitted options"):
        engine("a state", {"q1": choice_question()})


def test_strict_length_refuses_instead_of_truncating():
    engine = KrinoEngine(
        CapacityScorer(("a", "b"), (0.5, 0.5)),
        strict_length=True,
        instruction="Rate.",
    )
    question = {
        "type": "choice",
        "instructions": "Which one?",
        "criteria": {"a": "a long description that will not fit", "b": "b"},
    }
    with pytest.raises(Unsupported, match="budget is"):
        engine("a state", {"q1": question})


def test_capacity_check_is_off_when_not_requested():
    engine = KrinoEngine(
        CapacityScorer(("a", "b"), (0.5, 0.5)),
        strict_length=False,
    )
    question = {
        "type": "choice",
        "instructions": "Which one?",
        "criteria": {"a": "a", "b": "b"},
    }
    response, _ = engine("a state", {"q1": question})
    assert set(response["answers"]["q1"]["probabilities"]) == {"a", "b"}


def test_capacity_violation_reports_a_missing_budget():
    scorer = CapacityScorer(("a", "b"), (0.5, 0.5))
    scorer.max_length = 5
    reason = capacity_violation(
        scorer, "state", "question", "choice", [("a", "a"), ("b", "b")], "Rate."
    )
    assert reason is not None
    assert "no budget remains" in reason


def test_capacity_violation_is_none_when_the_scorer_cannot_be_checked():
    class Opaque:
        def predict_row(self, item):  # pragma: no cover - not called
            raise AssertionError

    assert (
        capacity_violation(Opaque(), "state", "q", "choice", [("a", "a")], "Rate.")
        is None
    )
