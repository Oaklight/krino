"""Tests for the MMLU row converter.

Covers the pure row-to-item conversion only; the surrounding load_mmlu()
needs network and pyarrow, so it is exercised by the materialize script
instead of in CI.
"""

from __future__ import annotations

from data.pipeline import MMLU_KEYS, _mmlu_row_to_item


def _row(**over):
    row = {
        "question": "Which is larger?",
        "choices": ["A", "B", "C", "D"],
        "answer": 1,
        "subject": "elementary_mathematics",
    }
    row.update(over)
    return row


class TestMmluRowToItem:
    def test_basic_conversion(self):
        item = _mmlu_row_to_item(_row(), "auxiliary_train", 7, "train")
        assert item is not None
        assert item.split == "train"
        assert item.label == MMLU_KEYS[1]
        assert set(item.question["criteria"]) == set(MMLU_KEYS)
        assert item.question["type"] == "choice"
        assert item.group == "mmlu-elementary_mathematics"

    def test_id_encodes_tag_and_index(self):
        item = _mmlu_row_to_item(_row(), "validation", 3, "test")
        assert item.id == "mmlu-validation-00003"

    def test_non_four_way_row_is_dropped(self):
        assert _mmlu_row_to_item(_row(choices=["A", "B"]), "t", 0, "train") is None

    def test_out_of_range_answer_is_dropped(self):
        assert _mmlu_row_to_item(_row(answer=9), "t", 0, "train") is None
        assert _mmlu_row_to_item(_row(answer=-1), "t", 0, "train") is None

    def test_non_integer_answer_is_dropped(self):
        assert _mmlu_row_to_item(_row(answer="B"), "t", 0, "train") is None

    def test_missing_subject_is_tolerated(self):
        row = _row()
        del row["subject"]
        item = _mmlu_row_to_item(row, "t", 0, "train")
        assert item is not None
        assert item.group == "mmlu-"
