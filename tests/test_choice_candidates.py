"""Tests for answering choice questions with the score head, one candidate at a time."""

from __future__ import annotations

import math

import pytest

from data.format import TypedQuestion
from data.score_form import CHOICE_SCORE_LEVELS, to_score_candidates
from model.evaluation.choice_candidates import (
    candidate_questions,
    predict_from_candidates,
    score_candidates,
    top_level_log_odds,
)


def _choice(label: str = "refund") -> TypedQuestion:
    return TypedQuestion.choice(
        id="c-1",
        state="I was charged twice and want my money back.",
        instructions="Which intent does this message have?",
        criteria={
            "refund": "The customer wants a refund.",
            "cancel": "The customer wants to cancel an order.",
            "status": "The customer wants an order status update.",
        },
        label=label,
        source="test",
        split="test",
    )


class _FakeModel:
    """Scores the top level by how well the candidate matches a chosen phrase."""

    def __init__(self, phrase: str = "refund", strength: float = 6.0) -> None:
        self.phrase = phrase
        self.strength = strength
        self.calls: list[str] = []

    def forward_score(self, state, instructions, criteria):
        self.calls.append(instructions)
        assert list(criteria) == list(CHOICE_SCORE_LEVELS)
        top = self.strength if self.phrase in instructions else -self.strength
        # Every level but the top sits at zero, so P(top) is a clean sigmoid.
        return [[0.0] * (len(criteria) - 1) + [top]]


class TestTopLevelLogOdds:
    def test_uniform_levels_give_the_log_odds_of_one_over_n(self):
        assert top_level_log_odds([0.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(
            math.log(0.2) - math.log(0.8)
        )

    def test_it_increases_with_the_top_level_logit(self):
        low = top_level_log_odds([0.0, 0.0, 1.0])
        high = top_level_log_odds([0.0, 0.0, 3.0])
        assert high > low

    def test_a_saturated_prediction_stays_finite(self):
        value = top_level_log_odds([-1e6, -1e6, 1e6])
        assert math.isfinite(value)

    def test_empty_logits_are_an_error(self):
        with pytest.raises(ValueError):
            top_level_log_odds([])


class TestCandidateQuestions:
    def test_one_question_per_option_in_the_original_order(self):
        candidates = candidate_questions(_choice())
        assert [c.question["score_form"]["candidate"] for c in candidates] == [
            "refund",
            "cancel",
            "status",
        ]

    def test_candidates_carry_the_shared_ordinal_scale(self):
        for candidate in candidate_questions(_choice()):
            assert list(candidate.question["criteria"]) == list(CHOICE_SCORE_LEVELS)


class TestScoreCandidates:
    def test_it_scores_every_option(self):
        model = _FakeModel()
        scores = score_candidates(model, _choice())
        assert len(scores) == 3
        assert len(model.calls) == 3

    def test_the_matching_candidate_scores_highest(self):
        scores = score_candidates(_FakeModel(phrase="refund"), _choice())
        assert scores[0] > scores[1] and scores[0] > scores[2]

    def test_a_different_match_wins_for_a_different_phrase(self):
        scores = score_candidates(_FakeModel(phrase="order status"), _choice())
        assert max(range(3), key=lambda i: scores[i]) == 2


class TestPredictFromCandidates:
    def test_it_returns_the_key_the_scores_and_the_option_order(self):
        key, scores, keys = predict_from_candidates(_FakeModel(), _choice())
        assert key == "refund"
        assert keys == ["refund", "cancel", "status"]
        assert len(scores) == 3

    def test_ties_go_to_the_earlier_option(self):
        class _Flat:
            def forward_score(self, state, instructions, criteria):
                return [[0.0] * len(criteria)]

        key, _, _ = predict_from_candidates(_Flat(), _choice())
        assert key == "refund"

    def test_a_mismatched_option_count_is_an_error(self, monkeypatch):
        class _Any:
            def forward_score(self, state, instructions, criteria):
                return [[0.0] * len(criteria)]

        item = _choice()
        # Two candidates for three options: the bridge must not silently return
        # a prediction that cannot be mapped back to the option order.
        monkeypatch.setattr(
            "model.evaluation.choice_candidates.candidate_questions",
            lambda _item: candidate_questions(_item)[:2],
        )
        with pytest.raises(ValueError):
            predict_from_candidates(_Any(), item)
