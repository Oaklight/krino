"""Tests for score-first conversion of typed questions."""

from __future__ import annotations

from collections import Counter

import pytest

from data.format import TypedQuestion
from data.score_form import (
    CHOICE_SCORE_LEVELS,
    NOUL_SCORE_LEVELS,
    convert_training_items,
    from_score,
    round_trip,
    score_to_choice,
    score_to_noul,
    to_score,
    to_score_candidates,
)


def _noul(
    label: bool | str = True,
    teacher_probs: dict[str, float] | None = None,
) -> TypedQuestion:
    question = TypedQuestion.noul(
        id="noul-1",
        state="The customer says: please close my account.",
        instructions="Is the customer asking to close their account?",
        label=label,  # type: ignore[arg-type]
        source="test",
        split="train",
    )
    question.teacher_probs = teacher_probs
    return question


def _choice(label: str = "close_account") -> TypedQuestion:
    return TypedQuestion.choice(
        id="choice-1",
        state="The customer says: please close my account.",
        instructions="Which intent category does this message belong to?",
        criteria={
            "close_account": "The customer wants to close their account.",
            "cancel_transfer": "The customer wants to cancel a transfer.",
            "check_balance": "The customer wants to check their balance.",
        },
        label=label,
        source="test",
        split="train",
    )


def _score(label: float = 2.5) -> TypedQuestion:
    return TypedQuestion.score(
        id="score-1",
        state="The customer says: this is the third time I have called.",
        instructions="How frustrated is the customer?",
        criteria=["calm", "mild", "annoyed", "angry", "furious"],
        label=label,
        source="test",
        split="train",
    )


def _assert_probs(
    actual: dict[str, float] | None,
    expected: dict[str, float],
) -> None:
    assert actual is not None
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)


class TestNoulToScore:
    def test_true_label_becomes_one(self):
        score_question = to_score(_noul(True))
        assert score_question.question["type"] == "score"
        assert score_question.question["criteria"] == ["false", "true"]
        assert score_question.label == 1.0

    def test_false_label_becomes_zero(self):
        assert to_score(_noul(False)).label == 0.0

    def test_identity_fields_preserved(self):
        question = _noul(True)
        question.group = "group-1"
        score_question = to_score(question)
        assert score_question.id == question.id
        assert score_question.state == question.state
        assert score_question.source == question.source
        assert score_question.split == question.split
        assert score_question.group == "group-1"
        assert score_question.question["instructions"] == question.question["instructions"]

    @pytest.mark.parametrize("p_true", [0.0, 0.1, 0.5, 0.7, 0.9, 1.0])
    def test_expected_score_equals_p_true(self, p_true):
        """Pinning test: the score must be P(true), not 1 - P(true).

        The expected level index over ["false", "true"] is
        0 * P(false) + 1 * P(true) = P(true).  If the level order, the label
        encoding, or the teacher-probability remapping changes to anything
        that does not preserve this identity, this test fails.
        """
        question = _noul(True, teacher_probs={"true": p_true, "false": 1.0 - p_true})
        score_question = to_score(question)
        assert score_question.question["criteria"] == ["false", "true"]
        assert score_question.teacher_probs is not None
        expected = sum(
            level * score_question.teacher_probs[str(level)] for level in range(2)
        )
        assert expected == pytest.approx(p_true)
        assert from_score(expected, question)["noul"] == pytest.approx(p_true)

    @pytest.mark.parametrize("label,p_true", [(True, 1.0), (False, 0.0)])
    def test_hard_label_round_trip_preserves_p_true(self, label, p_true):
        question = _noul(label)
        score_question = to_score(question)
        assert from_score(score_question.label, question)["noul"] == pytest.approx(p_true)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", 1.0),
            ("yes", 1.0),
            ("1", 1.0),
            ("false", 0.0),
            ("no", 0.0),
            ("0", 0.0),
        ],
    )
    def test_string_labels(self, raw, expected):
        assert to_score(_noul(raw)).label == expected

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            to_score(_noul("maybe"))

    def test_teacher_probs_with_noul_key(self):
        score_question = to_score(_noul(True, teacher_probs={"noul": 0.8}))
        _assert_probs(score_question.teacher_probs, {"0": 0.2, "1": 0.8})

    def test_unknown_teacher_probs_dropped(self):
        assert to_score(_noul(True, teacher_probs={"foo": 0.5})).teacher_probs is None

    def test_augmented_teacher_probs_dropped(self):
        question = _noul(
            True,
            teacher_probs={"true": 0.6, "false": 0.1, "unsure": 0.3},
        )
        assert to_score(question).teacher_probs is None


class TestChoiceToScore:
    def test_criteria_are_option_texts_in_order(self):
        question = _choice()
        score_question = to_score(question)
        assert score_question.question["type"] == "score"
        assert score_question.question["criteria"] == list(
            question.question["criteria"].values()
        )
        assert score_question.question["score_form"]["option_keys"] == list(
            question.question["criteria"]
        )

    def test_label_is_gold_index(self):
        assert to_score(_choice("close_account")).label == 0.0
        assert to_score(_choice("cancel_transfer")).label == 1.0
        assert to_score(_choice("check_balance")).label == 2.0

    def test_from_score_recovers_hard_label(self):
        question = _choice("cancel_transfer")
        result = from_score(to_score(question).label, question)
        assert result["choice"] == "cancel_transfer"
        assert result["probabilities"] == {
            "close_account": 0.0,
            "cancel_transfer": 1.0,
            "check_balance": 0.0,
        }

    def test_round_trip_exact_but_lossy(self):
        result = round_trip(_choice())
        assert result["exact"] is True
        assert result["lossy"] is True
        assert result["note"]

    def test_spread_distribution_bucketizes_to_zero_mass_option(self):
        """Pin the lossiness of the single-question ordinal encoding.

        Options are unordered.  A categorical distribution with all its mass
        on options 0 and 2 has expected index 1.0, so the score form
        "recovers" option 1 even though its probability is zero.
        """
        question = _choice()
        keys = list(question.question["criteria"])
        probabilities = {keys[0]: 0.5, keys[1]: 0.0, keys[2]: 0.5}
        expected_index = sum(
            index * probabilities[key] for index, key in enumerate(keys)
        )
        assert expected_index == pytest.approx(1.0)
        assert from_score(expected_index, question)["choice"] == keys[1]
        assert max(probabilities, key=probabilities.get) != keys[1]

    def test_option_order_changes_the_score_value(self):
        question = _choice("close_account")
        criteria = dict(question.question["criteria"])
        reordered = dict(reversed(list(criteria.items())))
        reordered_question = TypedQuestion.choice(
            id=question.id,
            state=question.state,
            instructions=question.question["instructions"],
            criteria=reordered,
            label=question.label,
            source=question.source,
            split=question.split,
        )
        assert to_score(question).label == 0.0
        assert to_score(reordered_question).label == 2.0
        assert (
            from_score(to_score(question).label, question)["choice"]
            == "close_account"
        )
        assert (
            from_score(to_score(reordered_question).label, reordered_question)["choice"]
            == "close_account"
        )

    def test_label_not_in_criteria_raises(self):
        # Pin the message, not just the exception type. Without the explicit
        # guard, `keys.index` raises "'zzz' is not in list", which tells the
        # caller nothing about the options that were available.
        with pytest.raises(ValueError, match="is not one of the option keys"):
            to_score(_choice("not_an_option"))

    def test_empty_criteria_raises(self):
        question = TypedQuestion(
            id="empty",
            state="state",
            question={"type": "choice", "instructions": "instructions", "criteria": {}},
            label="a",
            source="test",
            split="train",
        )
        with pytest.raises(ValueError):
            to_score(question)

    def test_teacher_probs_remapped_to_level_indices(self):
        question = _choice()
        question.teacher_probs = {
            "close_account": 0.2,
            "cancel_transfer": 0.5,
            "check_balance": 0.3,
        }
        _assert_probs(
            to_score(question).teacher_probs,
            {"0": 0.2, "1": 0.5, "2": 0.3},
        )

    def test_teacher_probs_mismatch_dropped(self):
        question = _choice()
        question.teacher_probs = {"a": 0.5, "b": 0.5}
        assert to_score(question).teacher_probs is None


class TestChoiceCandidates:
    def test_one_question_per_option(self):
        question = _choice()
        candidates = to_score_candidates(question)
        keys = list(question.question["criteria"])
        assert len(candidates) == len(keys)
        assert [candidate.id for candidate in candidates] == [
            f"{question.id}#{key}" for key in keys
        ]
        assert len({candidate.id for candidate in candidates}) == len(candidates)
        assert all(
            candidate.state == question.state
            and candidate.source == question.source
            and candidate.split == question.split
            and candidate.group == question.group
            for candidate in candidates
        )

    def test_shared_ordinal_scale(self):
        candidates = to_score_candidates(_choice())
        assert all(
            candidate.question["criteria"] == list(CHOICE_SCORE_LEVELS)
            for candidate in candidates
        )

    def test_candidate_metadata_and_instructions(self):
        question = _choice()
        for candidate in to_score_candidates(question):
            metadata = candidate.question["score_form"]
            key = metadata["candidate"]
            assert metadata["candidate_text"] == question.question["criteria"][key]
            assert metadata["option_keys"] == list(question.question["criteria"])
            assert metadata["candidate_text"] in candidate.question["instructions"]

    def test_duplicate_option_texts_stay_distinct(self):
        """Two options can share text; their prompts must not collapse.

        hellaswag endings repeat, so ``{0: " a dog", 1: " a dog", ...}`` is a
        real criteria shape.  Identical prompts force a deterministic scorer
        to give identical scores, and the argmax tie-break then returns key 0
        even when the gold answer is key 1.
        """
        question = TypedQuestion.choice(
            id="dup",
            state="context",
            instructions="Which ending completes the context?",
            criteria={"0": " a dog", "1": " a dog", "2": " a cat"},
            label="1",
            source="test",
            split="train",
        )
        candidates = to_score_candidates(question)
        prompts = [candidate.question["instructions"] for candidate in candidates]
        assert len(set(prompts)) == len(prompts)

        first, second = candidates[0], candidates[1]
        assert first.question["instructions"] != second.question["instructions"]
        assert first.question["score_form"]["candidate"] == "0"
        assert second.question["score_form"]["candidate"] == "1"
        # The gold label still marks the second one, so a correct scorer can
        # now separate the two.
        assert second.label == float(len(CHOICE_SCORE_LEVELS) - 1)
        assert first.label == 0.0

    def test_unique_texts_are_not_decorated(self):
        """The disambiguator only appears where it is needed."""
        for candidate in to_score_candidates(_choice()):
            assert "Candidate answer (" not in candidate.question["instructions"]
            assert "Candidate answer: " in candidate.question["instructions"]

    def test_label_not_in_criteria_raises_with_message(self):
        with pytest.raises(ValueError, match="is not one of the option keys"):
            to_score_candidates(_choice("not_an_option"))

    def test_gold_top_others_bottom(self):
        question = _choice("cancel_transfer")
        top_level = float(len(CHOICE_SCORE_LEVELS) - 1)
        for candidate in to_score_candidates(question):
            expected = (
                top_level
                if candidate.question["score_form"]["candidate"] == question.label
                else 0.0
            )
            assert candidate.label == expected

    def test_argmax_recovers_choice(self):
        question = _choice("cancel_transfer")
        scores = {
            candidate.question["score_form"]["candidate"]: candidate.label
            for candidate in to_score_candidates(question)
        }
        result = from_score(scores, question)
        assert result["choice"] == "cancel_transfer"
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
        assert (
            max(result["probabilities"], key=result["probabilities"].get)
            == "cancel_transfer"
        )

    def test_ties_break_by_option_order(self):
        question = _choice()
        keys = list(question.question["criteria"])
        result = from_score({key: 3.0 for key in keys}, question)
        assert result["choice"] == keys[0]
        for probability in result["probabilities"].values():
            assert probability == pytest.approx(1.0 / len(keys))

    def test_sequence_scores(self):
        question = _choice("cancel_transfer")
        assert from_score([0.0, 4.0, 0.0], question)["choice"] == "cancel_transfer"

    def test_sequence_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            from_score([1.0, 2.0], _choice())

    def test_missing_candidate_score_raises(self):
        with pytest.raises(ValueError):
            from_score({"close_account": 4.0}, _choice())

    def test_extra_candidate_score_raises(self):
        question = _choice()
        scores = {key: 0.0 for key in question.question["criteria"]}
        scores["bogus"] = 1.0
        with pytest.raises(ValueError):
            from_score(scores, question)

    def test_noul_and_score_return_single_element(self):
        noul_question = _noul(True)
        score_question = _score()
        assert to_score_candidates(noul_question) == [to_score(noul_question)]
        assert to_score_candidates(score_question) == [score_question]


class TestScoreIdentity:
    def test_to_score_is_identity(self):
        question = _score()
        assert to_score(question) is question

    def test_from_score_returns_score(self):
        assert from_score(2.5, _score()) == {"score": 2.5}

    def test_from_score_does_not_clamp(self):
        assert from_score(7.0, _score()) == {"score": 7.0}

    def test_round_trip_exact_and_lossless(self):
        result = round_trip(_score(3.25))
        assert result["type"] == "score"
        assert result["recovered"] == {"score": 3.25}
        assert result["recovered_label"] == 3.25
        assert result["exact"] is True
        assert result["lossy"] is False
        assert result["note"] is None

    def test_score_to_noul_two_levels(self):
        assert score_to_noul(1.0) is True
        assert score_to_noul(0.0) is False
        # Strictly greater-than, matching model/evaluation/accuracy.py.
        assert score_to_noul(0.5) is False
        assert score_to_noul(0.500001) is True

    def test_score_to_noul_multilevel(self):
        assert score_to_noul(0.0, n_levels=5) is False
        assert score_to_noul(2.0, n_levels=5) is False
        assert score_to_noul(2.1, n_levels=5) is True
        assert score_to_noul(4.0, n_levels=5) is True

    def test_score_to_noul_rejects_degenerate_scale(self):
        with pytest.raises(ValueError):
            score_to_noul(0.0, n_levels=1)

    def test_score_to_noul_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            score_to_noul(0.5, threshold=1.5)


class TestScoreToChoice:
    def test_bucketize_nearest_level(self):
        question = _choice()
        keys = list(question.question["criteria"])
        assert score_to_choice(0.0, question) == keys[0]
        assert score_to_choice(0.49, question) == keys[0]
        assert score_to_choice(0.5, question) == keys[1]  # half-up
        assert score_to_choice(1.5, question) == keys[2]
        assert score_to_choice(2.4, question) == keys[2]

    def test_clamps_out_of_range(self):
        question = _choice()
        keys = list(question.question["criteria"])
        assert score_to_choice(-10.0, question) == keys[0]
        assert score_to_choice(10.0, question) == keys[-1]

    def test_rejects_non_choice_question(self):
        with pytest.raises(ValueError):
            score_to_choice(0.0, _noul(True))


class TestFromScoreValidation:
    def test_noul_rejects_mapping(self):
        with pytest.raises(TypeError):
            from_score({"true": 0.7}, _noul(True))

    def test_noul_rejects_sequence(self):
        with pytest.raises(TypeError):
            from_score([0.7], _noul(True))

    def test_noul_clamps_probability(self):
        assert from_score(1.5, _noul(True)) == {"noul": 1.0}
        assert from_score(-0.5, _noul(True)) == {"noul": 0.0}

    def test_score_rejects_mapping(self):
        with pytest.raises(TypeError):
            from_score({"0": 1.0}, _score())

    def test_nan_prediction_raises(self):
        with pytest.raises(ValueError):
            from_score(float("nan"), _noul(True))

    def test_string_prediction_raises(self):
        with pytest.raises(TypeError):
            from_score("0.7", _noul(True))

    def test_unknown_type_raises(self):
        question = TypedQuestion(
            id="unknown",
            state="state",
            question={"type": "ranking", "instructions": "instructions"},
            label=1,
            source="test",
            split="train",
        )
        with pytest.raises(ValueError):
            to_score(question)
        with pytest.raises(ValueError):
            to_score_candidates(question)
        with pytest.raises(ValueError):
            from_score(1.0, question)


class TestRoundTrip:
    def test_noul(self):
        result = round_trip(_noul(True))
        assert result["type"] == "noul"
        assert result["recovered"] == {"noul": 1.0}
        assert result["recovered_label"] is True
        assert result["exact"] is True
        assert result["lossy"] is False
        assert result["note"] is None

    def test_noul_false(self):
        result = round_trip(_noul(False))
        assert result["recovered_label"] is False
        assert result["exact"] is True

    def test_choice(self):
        result = round_trip(_choice("check_balance"))
        assert result["type"] == "choice"
        assert result["recovered_label"] == "check_balance"
        assert result["exact"] is True
        assert result["lossy"] is True
        assert "unordered" in result["note"]

    def test_score_question_is_returned_unchanged(self):
        question = _score()
        question.group = "group-1"
        result = round_trip(question)
        assert result["score_question"] is question
        assert result["score_question"].group == "group-1"


class TestConstants:
    def test_noul_levels_order(self):
        assert NOUL_SCORE_LEVELS == ("false", "true")

    def test_choice_scale_is_ordered_and_unique(self):
        assert len(CHOICE_SCORE_LEVELS) >= 2
        assert all(
            isinstance(level, str) and level for level in CHOICE_SCORE_LEVELS
        )
        assert len(set(CHOICE_SCORE_LEVELS)) == len(CHOICE_SCORE_LEVELS)


class TestConvertTrainingItems:
    def test_single_preserves_count_and_makes_everything_score(self):
        items = [_noul(), _choice(), _score()]
        converted = convert_training_items(items, "single")
        assert len(converted) == len(items)
        assert all(item.question["type"] == "score" for item in converted)

    def test_candidates_expands_choice_only(self):
        items = [_noul(), _choice(), _score()]
        converted = convert_training_items(items, "candidates")
        # noul and score stay single; choice becomes one question per option.
        expected = 1 + len(items[1].question["criteria"]) + 1
        assert len(converted) == expected
        assert all(item.question["type"] == "score" for item in converted)

    def test_candidates_gold_index_is_recoverable(self):
        question = _choice("cancel_transfer")
        converted = convert_training_items([question], "candidates")
        top = float(len(CHOICE_SCORE_LEVELS) - 1)
        gold = [
            item
            for item in converted
            if item.question["score_form"]["candidate"] == "cancel_transfer"
        ]
        assert len(gold) == 1
        assert gold[0].label == top
        assert sum(1 for item in converted if item.label == top) == 1

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown score-form mode"):
            convert_training_items([_noul()], "bogus")


class TestScoreFormKeepsSamplerMix:
    """Conversion must not change the sampler's type balance.

    Every converted item declares ``type: score``.  If the sampler pooled on
    that field, ``type_ratios`` would stop applying and the epoch mix would be
    decided by source weights alone -- which would confound a score-form run
    against an ordinary one, because the objective *and* the type mix would
    change together.
    """

    def _items(self):
        items = []
        for source_index in range(4):
            for item_index in range(500):
                items.append(
                    TypedQuestion.choice(
                        id=f"choice-{source_index}-{item_index}",
                        state="state",
                        instructions="pick one",
                        criteria={"a": "a", "b": "b", "c": "c"},
                        label="a",
                        source=f"choice_src_{source_index}",
                        split="train",
                    )
                )
        items.extend(_noul() for _ in range(500))
        items.extend(_score() for _ in range(500))
        return items

    def test_type_ratios_still_apply_after_conversion(self):
        from data.sampler import MultitaskSampler, SamplerConfig, sampling_type

        items = self._items()
        config = SamplerConfig(
            type_ratios={"noul": 1.0, "choice": 1.0, "score": 1.0},
            epoch_size=3000,
        )
        base = MultitaskSampler(items, config, seed=0).sample_epoch(0)
        base_counts = Counter(sampling_type(item) for item in base)
        assert set(base_counts) == {"noul", "choice", "score"}

        for mode in ("single", "candidates"):
            converted = convert_training_items(items, mode)
            sampled = MultitaskSampler(converted, config, seed=0).sample_epoch(0)
            counts = Counter(sampling_type(item) for item in sampled)
            assert counts == base_counts, f"{mode} changed the type mix"
