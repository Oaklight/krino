"""Tests for the eval-variant generator.

Pure Python, so it runs without torch.
"""

from __future__ import annotations

import random

import pytest

from scripts.make_eval_variants import (
    NEUTRAL_INSTRUCTION,
    reverse_choice,
    reverse_score,
    shuffle_choice,
    shuffle_score,
    transform_item,
)


def _choice_item():
    return {
        "id": "q1",
        "state": "text",
        "question": {
            "type": "choice",
            "instructions": "Which one?",
            "criteria": {"a": "A", "b": "B", "c": "C"},
        },
        "label": "b",
        "source": "test",
        "split": "test",
    }


def _score_item():
    return {
        "id": "q2",
        "state": "text",
        "question": {
            "type": "score",
            "instructions": "How good?",
            "criteria": ["bad", "ok", "good"],
        },
        "label": 0.0,
        "source": "test",
        "split": "test",
    }


class TestReverseChoice:
    def test_order_reversed_and_label_preserved(self):
        criteria, label = reverse_choice({"a": "A", "b": "B", "c": "C"}, "b")
        assert list(criteria) == ["c", "b", "a"]
        assert label == "b"
        assert criteria["b"] == "B"

    def test_text_stays_with_its_key(self):
        criteria, _ = reverse_choice({"x": "first", "y": "second"}, "x")
        assert criteria["x"] == "first"


class TestReverseScore:
    def test_label_mirrored(self):
        levels, label = reverse_score(["bad", "ok", "good"], 0.0)
        assert levels == ["good", "ok", "bad"]
        assert label == 2.0

    def test_middle_level_is_fixed_point(self):
        levels, label = reverse_score(["a", "b", "c"], 1.0)
        assert levels == ["c", "b", "a"]
        assert label == 1.0

    def test_five_levels(self):
        _, label = reverse_score(list("abcde"), 4.0)
        assert label == 0.0


class TestShuffle:
    def test_choice_permutation_is_a_permutation(self):
        criteria, label = shuffle_choice({"a": "A", "b": "B", "c": "C"}, "c", random.Random(0))
        assert sorted(criteria) == ["a", "b", "c"]
        assert label == "c"

    def test_score_label_follows_its_level(self):
        levels = ["L0", "L1", "L2", "L3"]
        new_levels, new_label = shuffle_score(levels, 2.0, random.Random(7))
        assert new_levels[int(new_label)] == "L2"
        assert sorted(new_levels) == sorted(levels)

    def test_shuffle_is_deterministic(self):
        a = shuffle_score(["L0", "L1", "L2"], 1.0, random.Random(3))
        b = shuffle_score(["L0", "L1", "L2"], 1.0, random.Random(3))
        assert a == b


class TestTransformItem:
    def test_neutral_replaces_instructions_only(self):
        item = _choice_item()
        out = transform_item(item, "neutral", random.Random(0))
        assert out["question"]["instructions"] == NEUTRAL_INSTRUCTION
        assert out["question"]["criteria"] == item["question"]["criteria"]
        assert out["label"] == item["label"]

    def test_id_is_tagged(self):
        out = transform_item(_choice_item(), "reversed", random.Random(0))
        assert out["id"].endswith("::reversed")

    def test_original_item_is_not_mutated(self):
        item = _choice_item()
        transform_item(item, "reversed", random.Random(0))
        assert list(item["question"]["criteria"]) == ["a", "b", "c"]

    def test_noul_untouched_by_reversal(self):
        item = {
            "id": "n1",
            "state": "s",
            "question": {"type": "noul", "instructions": "Is it?", "criteria": {"false": "no", "true": "yes"}},
            "label": True,
            "source": "t",
            "split": "test",
        }
        out = transform_item(item, "reversed", random.Random(0))
        assert out["label"] is True
        assert out["question"]["criteria"] == {"false": "no", "true": "yes"}

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            transform_item(_choice_item(), "sideways", random.Random(0))

    def test_shuffled_score_item_keeps_level_semantics(self):
        item = _score_item()
        out = transform_item(item, "shuffled", random.Random(5))
        levels = out["question"]["criteria"]
        assert levels[int(out["label"])] == "bad"


class TestNonsenseInstruction:
    def test_replaces_the_instruction_and_keeps_everything_else(self):
        from scripts.make_eval_variants import NONSENSE_INSTRUCTION, transform_item
        import random as _random

        item = {
            "id": "q1",
            "label": True,
            "question": {"type": "noul", "instructions": "Is X true?", "criteria": {}},
        }
        out = transform_item(item, "nonsense", _random.Random(42))
        assert out["question"]["instructions"] == NONSENSE_INSTRUCTION
        assert out["label"] is True
        assert out["id"] == "q1::nonsense"

    def test_differs_from_neutral(self):
        from scripts.make_eval_variants import NEUTRAL_INSTRUCTION, NONSENSE_INSTRUCTION

        assert NEUTRAL_INSTRUCTION != NONSENSE_INSTRUCTION


class TestShuffleScoreFractionalLabels:
    def test_fractional_label_stays_between_its_neighbours(self):
        from scripts.make_eval_variants import shuffle_score
        import random as _random

        levels = ["a", "b", "c", "d", "e"]
        for seed in range(20):
            new_levels, new_label = shuffle_score(levels, 2.5, _random.Random(seed))
            low = new_levels.index("c")   # old level 2, weight 0.5
            high = new_levels.index("d")  # old level 3, weight 0.5
            # The expected level is the midpoint of the two positions in the
            # new scale, in either direction -- not rounded onto one of them.
            assert new_label == (low + high) / 2
            assert new_label != float(int(new_label)) or abs(low - high) % 2 == 0

    def test_integer_label_moves_without_interpolation(self):
        from scripts.make_eval_variants import shuffle_score
        import random as _random

        levels = ["a", "b", "c", "d", "e"]
        for seed in range(20):
            new_levels, new_label = shuffle_score(levels, 2.0, _random.Random(seed))
            assert new_levels[int(new_label)] == "c"
