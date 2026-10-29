"""Tests for the routed choice system's calibration measurement."""

from __future__ import annotations

import math

import pytest

from model.evaluation.routed_calibration import (
    LONG_MIN,
    buckets,
    by_path,
    item_logits,
    path_of,
    routed_report,
)


def _record(k, gold, head, native=None):
    return {"k": k, "gold": gold, "head": head, "native": native}


def _confident_record(k, gold, overconfident=0.9):
    """A two-option item whose head is confident and sometimes wrong."""
    other = 1.0 - overconfident
    head = [overconfident, other] if gold == 0 else [other, overconfident]
    return _record(k, gold, head)


class TestPathOf:
    def test_short_list_with_a_native_score_is_routed_native(self):
        assert path_of(_record(2, 0, [0.5, 0.5], [0.0, 0.0])) == "native"

    def test_short_list_without_a_native_score_falls_back_to_the_head(self):
        assert path_of(_record(2, 0, [0.5, 0.5], None)) == "head"

    def test_long_list_uses_the_head_even_if_native_was_dumped(self):
        assert path_of(_record(LONG_MIN, 0, [0.5, 0.5], [0.0, 0.0])) == "head"

    def test_an_empty_native_vector_counts_as_absent(self):
        assert path_of(_record(3, 0, [0.5, 0.5], [])) == "head"


class TestItemLogits:
    def test_head_logits_pass_through_unchanged(self):
        # The dump's `head` vector is a logit vector, not a probability vector;
        # treating it as probabilities was the first bug this module had.
        head = [0.7, 0.2, 0.1]
        assert item_logits(_record(3, 0, head)) == head

    def test_service_probabilities_are_a_softmax_over_head_logits(self):
        from model.evaluation.calibration import softmax

        head = [-1.0, 2.0]
        assert softmax(item_logits(_record(2, 0, head))) == pytest.approx(
            softmax(head), abs=1e-12
        )

    def test_native_log_odds_pass_through(self):
        assert item_logits(_record(2, 0, [0.5, 0.5], [1.5, -0.5])) == [1.5, -0.5]


class TestRoutedReport:
    def _mixed_records(self, n=200):
        records = []
        for index in range(n):
            k = 2 if index % 2 else 5
            gold = index % k
            head = [0.1] * k
            head[gold] = 0.6
            native = [0.0] * k
            native[gold] = 1.0
            records.append(_record(k, gold, head, native))
        return records

    def test_counts_both_paths_and_splits_every_item(self):
        report = routed_report(self._mixed_records())
        assert report["paths"] == {"native": 200, "head": 0}
        assert sum(fold["eval_n"] for fold in report["folds"]) == 200
        assert report["rule"]["folds"] == 2
        assert report["rule"]["long_min"] == LONG_MIN

    def test_each_fold_fits_on_the_items_it_does_not_score(self):
        report = routed_report(self._mixed_records())
        for fold in report["folds"]:
            assert fold["fit_n"] + fold["eval_n"] == 200

    def test_an_overconfident_path_is_brought_down_by_its_temperature(self):
        # Everything is right with confidence 0.6, so the fitted temperature
        # must soften the distribution and reduce ECE relative to T=1.
        report = routed_report(self._mixed_records())
        mean = report["mean"]
        assert mean["routed"]["ece"] < mean["routed_at_t1"]["ece"]
        assert mean["routed"]["accuracy"] == pytest.approx(1.0)

    def test_temperature_cannot_change_accuracy(self):
        for fold in routed_report(self._mixed_records())["folds"]:
            assert fold["routed"]["accuracy"] == pytest.approx(
                fold["routed_at_t1"]["accuracy"]
            )

    def test_forcing_every_item_onto_the_head_matches_the_calibrated_head(self):
        report = routed_report(self._mixed_records(), head_path=lambda r, i, n: "head")
        assert report["paths"] == {"native": 0, "head": 200}
        for fold in report["folds"]:
            assert fold["routed"]["ece"] == pytest.approx(fold["head_only_calibrated"]["ece"])
            assert fold["routed"]["brier"] == pytest.approx(
                fold["head_only_calibrated"]["brier"]
            )

    def test_head_only_is_reported_at_t1_beside_the_calibrated_one(self):
        for fold in routed_report(self._mixed_records())["folds"]:
            assert fold["head_only"]["ece"] >= fold["head_only_calibrated"]["ece"]

    def test_the_single_temperature_variant_is_reported(self):
        for fold in routed_report(self._mixed_records())["folds"]:
            assert fold["single_temperature"] > 0
            assert fold["routed_single_t"]["accuracy"] == pytest.approx(
                fold["routed"]["accuracy"]
            )
            assert fold["routed_single_t"]["ece"] >= fold["routed"]["ece"]

    def test_folds_must_be_at_least_two(self):
        with pytest.raises(ValueError):
            routed_report(self._mixed_records(), folds=1)

    def test_no_records_is_an_error_rather_than_a_zero(self):
        with pytest.raises(ValueError):
            routed_report([])

    def test_long_items_are_scored_by_the_head(self):
        records = [
            _record(20, 0, [0.9, 0.1], [5.0, -5.0]),
            _record(20, 0, [0.9, 0.1], [5.0, -5.0]),
        ]
        report = routed_report(records)
        assert report["paths"] == {"native": 0, "head": 2}


class TestByPathAndBuckets:
    def test_by_path_reports_one_row_per_serving_path(self):
        records = [
            _record(2, 0, [0.5, 0.5], [1.0, 0.0]),
            _record(25, 0, [0.9] + [0.1 / 24] * 24, None),
        ]
        rows = by_path(records, {"native": 1.0, "head": 1.0})
        assert sorted(rows) == ["head", "native"]
        assert rows["native"]["n"] == 1 and rows["head"]["n"] == 1

    def test_buckets_match_the_notes(self):
        records = [_record(2, 0, [1.0]), _record(6, 0, [1.0]), _record(20, 0, [1.0])]
        assert buckets(records) == {"<=5": 1, "6-19": 1, ">=20": 1}
