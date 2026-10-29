"""Tests for the shared JevBench scoring rule and aggregation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from model.evaluation.jevbench import prediction_rows, record_correct, summarize, tier_from_id


def record(item_id: str, logits: list[float], label_idx: int, q_type: str = "choice"):
    return {
        "type": q_type,
        "logits": logits,
        "label_order": list(range(len(logits))),
        "label_idx": label_idx,
        "_item": SimpleNamespace(id=item_id, question={"type": q_type}),
    }


def test_tier_comes_from_the_id_prefix():
    assert tier_from_id("easy-1") == "easy"
    assert tier_from_id("hard-111") == "hard"
    assert tier_from_id("original-72") == "original"


def test_an_unexpected_prefix_is_kept_rather_than_dropped():
    assert tier_from_id("renamed-7") == "renamed"


def test_correct_when_the_gold_option_has_the_highest_logit():
    assert record_correct(record("easy-1", [0.1, 2.0, 0.3], 1)) is True


def test_wrong_when_another_option_wins():
    assert record_correct(record("easy-1", [3.0, 2.0, 0.3], 1)) is False


def test_a_tie_is_decided_by_the_option_key_not_its_presentation_position():
    # Options 0 and 1 are tied. In the original order, key 0 wins.
    before = record("easy-1", [1.0, 1.0, 0.0], 0)
    assert record_correct(before) is True

    # Same items reordered so key 0 now sits at position 1, and the gold index
    # moves with it. Key 0 still wins, so the item is still correct.
    after = record("easy-1", [1.0, 1.0, 0.0], 1)
    after["label_order"] = [1, 0, 2]
    assert record_correct(after) is True

    # Same reordering, but the gold option is now key 1 at position 0: the tie
    # goes to key 0 at position 1, so this run is wrong.
    lost = record("easy-1", [1.0, 1.0, 0.0], 0)
    lost["label_order"] = [1, 0, 2]
    assert record_correct(lost) is False


def test_a_supplied_prediction_skips_the_softmax():
    odd = {"logits": [float("nan")], "label_order": [0], "label_idx": 0}
    assert record_correct(odd, prediction=0) is True


def test_temperature_does_not_change_the_argmax():
    rec = record("hard-1", [0.4, 2.0, 0.1], 1)
    assert record_correct(rec, temperature=0.5) is True
    assert record_correct(rec, temperature=8.0) is True


def test_summarize_counts_by_tier_type_and_overall():
    records = [
        record("easy-1", [2.0, 0.0], 0),
        record("easy-2", [0.0, 2.0], 0),
        record("hard-1", [2.0, 0.0], 0, q_type="noul"),
        record("hard-2", [0.0, 2.0], 1, q_type="noul"),
    ]
    summary = summarize(records)
    assert summary["overall"] == {"accuracy": 0.75, "correct": 3, "total": 4, "failed": 0}
    assert summary["by_tier"]["easy"] == {"accuracy": 0.5, "correct": 1, "total": 2}
    assert summary["by_tier"]["hard"] == {"accuracy": 1.0, "correct": 2, "total": 2}
    assert summary["by_type"]["choice"]["total"] == 2
    assert summary["by_tier_type"]["hard"]["noul"] == {
        "accuracy": 1.0,
        "correct": 2,
        "total": 2,
    }


def test_summarize_reports_unscored_items_rather_than_hiding_them():
    summary = summarize([record("easy-1", [2.0, 0.0], 0)], failed=5)
    assert summary["overall"]["total"] == 1
    assert summary["overall"]["failed"] == 5


def test_summarize_of_nothing_is_zero_and_not_a_division_error():
    summary = summarize([])
    assert summary["overall"] == {"accuracy": 0.0, "correct": 0, "total": 0, "failed": 0}
    assert summary["by_tier"] == {}


@pytest.mark.parametrize("tier", ["easy", "hard", "original"])
def test_each_tier_is_reported_under_its_own_name(tier):
    summary = summarize([record(f"{tier}-1", [2.0, 0.0], 0)])
    assert list(summary["by_tier"]) == [tier]


def test_prediction_rows_name_the_option_that_was_chosen():
    rows = prediction_rows([record("hard-a", [0.1, 2.0, 0.3], 1), record("hard-b", [2.0, 0.0], 1)])
    assert [row["predicted"] for row in rows] == [1, 0]
    assert rows[0]["gold"] == 1 and rows[0]["correct"] is True
    assert rows[1]["correct"] is False
    assert [row["id"] for row in rows] == ["hard-a", "hard-b"]


def test_prediction_rows_expose_a_shared_key_bias():
    # Two items whose gold answers differ, both predicted to the same option:
    # the count of identical predictions is what accuracy alone cannot show.
    rows = prediction_rows(
        [record("hard-a", [2.0, 0.0, 0.0], 1), record("hard-b", [2.0, 0.0, 0.0], 2)]
    )
    assert [row["predicted"] for row in rows] == [0, 0]
    assert [row["correct"] for row in rows] == [False, False]


def test_prediction_rows_fall_back_to_indices_without_labels():
    item = SimpleNamespace(id="hard-c", question={"type": "score"})
    rows = prediction_rows(
        [{"type": "score", "logits": [0.0, 5.0], "label_order": [0, 1], "label_idx": 0, "_item": item}]
    )
    assert rows[0]["predicted"] == 1
    assert rows[0]["gold"] == 0
    assert rows[0]["n_options"] == 0
