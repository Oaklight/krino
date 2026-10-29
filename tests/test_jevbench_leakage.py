"""Tests for the JevBench held-out check.

The audit decides whether a number we report on a "held-out" set is actually
held out, so the logic that maps a pair of items to "identical", "label
conflict" or "state repeat" has to be right before the zeros it reports mean
anything. Pure Python, no corpus and no network: every fixture is written to a
temporary directory in the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_jevbench_leakage import (
    TrainingEntry,
    audit,
    canonical_question,
    classify_exact,
    load_probes,
    normalise,
    shingle_hashes,
)


def _row(state: str, question: dict, *, split: str = "train", label: str = "a", item_id: str = "i") -> dict:
    return {"id": item_id, "state": state, "question": question, "split": split, "label": label}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class TestNormalise:
    def test_collapses_whitespace_and_case(self):
        assert normalise("  The   QUICK\nbrown  fox ") == "the quick brown fox"

    def test_is_idempotent(self):
        once = normalise("A  b\tC")
        assert normalise(once) == once


class TestShingleHashes:
    def test_identical_text_shares_every_shingle(self):
        text = "one two three four five six seven eight nine ten"
        assert shingle_hashes(text) == shingle_hashes(text.upper())

    def test_whitespace_and_case_do_not_change_the_set(self):
        assert shingle_hashes("one two three four five six seven eight") == shingle_hashes(
            "one  two three four FIVE six seven eight"
        )

    def test_shifted_text_keeps_most_shingles(self):
        base = " ".join(f"word{index}" for index in range(40))
        shifted = "introduction " + " ".join(f"word{index}" for index in range(1, 40))
        overlap = len(shingle_hashes(base) & shingle_hashes(shifted)) / len(shingle_hashes(base))
        assert overlap > 0.9

    def test_short_text_falls_back_to_one_shingle(self):
        # A state shorter than the shingle size must still be comparable, or it
        # scores containment 0 against everything and reads as "not leaked".
        assert len(shingle_hashes("too short")) == 1

    def test_empty_text_has_no_shingles(self):
        assert shingle_hashes("   ") == set()


class TestCanonicalQuestion:
    def test_criteria_order_does_not_change_the_key(self):
        left = {"type": "choice", "instructions": "pick", "criteria": {"a": "x", "b": "y"}}
        right = {"type": "choice", "instructions": "pick", "criteria": {"b": "y", "a": "x"}}
        assert canonical_question(left) == canonical_question(right)

    def test_mapping_and_list_shapes_can_agree(self):
        mapping = {"type": "score", "instructions": "rate", "criteria": {"a": "x", "b": "y"}}
        as_list = {"type": "score", "instructions": "rate", "criteria": ["x", "y"]}
        # The two shapes differ in what they carry (keys vs not), so the keys
        # must not collide: a list and a mapping are different questions.
        assert canonical_question(mapping) != canonical_question(as_list)

    def test_instructions_are_part_of_the_question(self):
        left = {"type": "noul", "instructions": "is it true"}
        right = {"type": "noul", "instructions": "is it false"}
        assert canonical_question(left) != canonical_question(right)

    def test_missing_question_is_stable(self):
        assert canonical_question(None) == canonical_question({})


class TestClassifyExact:
    entry = TrainingEntry(source="sst2", item_id="1", question="Q", gold="true")

    def test_same_question_and_gold_is_identical(self):
        assert classify_exact(self.entry, "Q", "true") == "identical"

    def test_same_question_different_gold_is_a_conflict(self):
        assert classify_exact(self.entry, "Q", "false") == "label_conflict"

    def test_different_question_is_only_a_state_repeat(self):
        # PIQA reuses a goal sentence under different option pairs; that is not
        # leakage and must not be reported as such.
        assert classify_exact(self.entry, "Q2", "true") == "state_repeat"


class TestAudit:
    def _setup(self, tmp_path: Path, train_rows: list[dict]) -> Path:
        data_dir = tmp_path / "benchmarks"
        data_dir.mkdir()
        _write(data_dir / "sst2.jsonl", train_rows)
        # The probe file sits inside the same directory and must not be indexed
        # as training data even when it is named like a source.
        _write(
            data_dir / "jevbench.jsonl",
            [_row("a state that appears only in the probe file", {"type": "noul"}, split="test")],
        )
        return data_dir

    def _probes(self, tmp_path: Path, rows: list[dict]) -> list:
        path = tmp_path / "probes" / "easy.jsonl"
        path.parent.mkdir()
        _write(path, rows)
        return load_probes([path])

    def test_an_identical_item_is_counted_as_leakage(self, tmp_path):
        question = {"type": "noul", "instructions": "is it positive"}
        data_dir = self._setup(
            tmp_path, [_row("the movie was wonderful", question, label="true")]
        )
        probes = self._probes(
            tmp_path,
            [{**_row("The movie was WONDERFUL", question), "expected": "true"}],
        )

        report = audit(probes, data_dir, 0.5)

        tier = report["by_tier"]["easy"]
        assert tier["identical"] == 1
        assert tier["label_conflict"] == 0

    def test_a_conflicting_label_is_not_counted_as_leakage(self, tmp_path):
        question = {"type": "noul", "instructions": "is it positive"}
        data_dir = self._setup(
            tmp_path, [_row("the movie was wonderful", question, label="true")]
        )
        probes = self._probes(
            tmp_path,
            [{**_row("the movie was wonderful", question), "expected": "false"}],
        )

        report = audit(probes, data_dir, 0.5)

        tier = report["by_tier"]["easy"]
        assert tier["identical"] == 0
        assert tier["label_conflict"] == 1

    def test_a_near_duplicate_is_reported_though_not_exact(self, tmp_path):
        long_state = " ".join(f"word{index}" for index in range(40))
        noul = {"type": "noul", "instructions": "is it true"}
        data_dir = self._setup(tmp_path, [_row(long_state, noul)])
        probe_state = " ".join(["introduction"] + long_state.split()[1:])
        probes = self._probes(tmp_path, [{**_row(probe_state, noul), "expected": "a"}])

        report = audit(probes, data_dir, 0.5)

        tier = report["by_tier"]["easy"]
        assert tier["identical"] == 0
        assert len(tier["near"]) == 1
        assert tier["near"][0]["containment"] > 0.9

    def test_unrelated_items_report_zero_and_probe_file_is_not_training_data(self, tmp_path):
        noul = {"type": "noul", "instructions": "is it true"}
        data_dir = self._setup(
            tmp_path, [_row("completely unrelated training text about weather", noul)]
        )
        probes = self._probes(
            tmp_path,
            [{**_row("a state that appears only in the probe file", noul), "expected": "a"}],
        )

        report = audit(probes, data_dir, 0.5)

        assert report["training_items"] == 1
        tier = report["by_tier"]["easy"]
        assert tier["identical"] == 0
        assert tier["near"] == []
        # The probe file inside data/benchmarks was not indexed as training data.
        assert report["probes"] == 1

    def test_tiers_are_reported_separately(self, tmp_path):
        noul = {"type": "noul", "instructions": "is it true"}
        data_dir = self._setup(tmp_path, [_row("shared state text for both tiers here", noul)])
        first = tmp_path / "probes" / "easy.jsonl"
        second = tmp_path / "probes" / "hard.jsonl"
        first.parent.mkdir()
        _write(first, [{**_row("shared state text for both tiers here", noul), "expected": "a"}])
        _write(second, [{**_row("something else entirely for the hard tier", noul), "expected": "a"}])

        report = audit(load_probes([first, second]), data_dir, 0.5)

        assert report["by_tier"]["easy"]["identical"] == 1
        assert report["by_tier"]["hard"]["identical"] == 0
        assert report["probes"] == 2

    def test_empty_probe_set_does_not_divide_by_zero(self, tmp_path):
        noul = {"type": "noul", "instructions": "is it true"}
        data_dir = self._setup(tmp_path, [_row("some training text", noul)])

        report = audit([], data_dir, 0.5)

        assert report["probes"] == 0
        assert report["max_containment"] == 0.0
        assert report["by_tier"] == {}
