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
        assert item.group == "mmlu-unknown"
        assert item.question["instructions"] == "Answer this multiple-choice question."

    def test_empty_subject_matches_missing_subject(self):
        """auxiliary_train carries the key with an empty value, not absent."""
        item = _mmlu_row_to_item(_row(subject=""), "t", 0, "train")
        assert item is not None
        assert item.group == "mmlu-unknown"


class TestLoadMmluSplits:
    """Pin which split each source of rows ends up in.

    load_mmlu() needs network and pyarrow, so the parquet fetch is stubbed.
    Without this, reinstating the original bug (mapping the auxiliary_train
    rows back to "test") would leave every other test green.
    """

    @staticmethod
    def _stub(monkeypatch, **overrides):
        import data.pipeline as pipeline

        row = {
            "question": "q",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
            "subject": "abstract_algebra",
        }

        def fake(dataset, config, split):
            assert dataset == "cais/mmlu"
            if config == "auxiliary_train":
                return [{"train": {**row, "subject": ""}}]
            return [{**row, **overrides.get(split, {})}]

        monkeypatch.setattr(pipeline, "_load_hf_parquet", fake)
        return pipeline

    def test_auxiliary_train_rows_are_train(self, monkeypatch):
        pipeline = self._stub(monkeypatch)
        by_split = {}
        for item in pipeline.load_mmlu():
            by_split.setdefault(item.split, []).append(item)

        assert len(by_split["train"]) == 1
        assert by_split["train"][0].id.startswith("mmlu-auxiliary_train-")

    def test_both_all_splits_are_eval(self, monkeypatch):
        pipeline = self._stub(monkeypatch)
        splits = [item.split for item in pipeline.load_mmlu()]
        assert splits.count("test") == 2
        assert "train" in splits

    def test_empty_subject_gets_a_usable_instruction(self, monkeypatch):
        pipeline = self._stub(monkeypatch)
        train = [i for i in pipeline.load_mmlu() if i.split == "train"][0]
        assert train.question["instructions"] == "Answer this multiple-choice question."
        assert train.group == "mmlu-unknown"

    def test_populated_subject_is_used(self, monkeypatch):
        pipeline = self._stub(monkeypatch)
        test = [i for i in pipeline.load_mmlu() if i.split == "test"][0]
        assert test.question["instructions"] == "Answer this abstract algebra question."
