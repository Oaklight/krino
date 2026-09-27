"""Tests for probability distribution normalization."""

from __future__ import annotations

import pytest

from data.normalize_probs import (
    normalize_items_in_place,
    normalize_label_file,
    normalize_probs,
    normalize_teacher_probs,
)


class TestNormalizeProbs:
    def test_already_normalized(self):
        probs = {"true": 0.7, "false": 0.3}
        result = normalize_probs(probs)
        assert result is probs  # same object, no copy

    def test_near_one_within_tolerance(self):
        probs = {"true": 0.702, "false": 0.301}
        result = normalize_probs(probs)
        assert result is probs  # 1.003, within 0.005

    def test_double_sum(self):
        probs = {"a": 0.99, "b": 0.99, "c": 0.0, "d": 0.0}
        result = normalize_probs(probs)
        assert result is not probs
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["a"] - 0.5) < 0.01
        assert abs(result["b"] - 0.5) < 0.01

    def test_all_high(self):
        probs = {"a": 0.99, "b": 0.99, "c": 0.99, "d": 0.99}
        result = normalize_probs(probs)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["a"] - 0.25) < 0.01

    def test_truncated_low_sum(self):
        probs = {"only_one": 0.001}
        result = normalize_probs(probs)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result["only_one"] == 1.0

    def test_empty_dict(self):
        assert normalize_probs({}) == {}

    def test_zero_total(self):
        probs = {"a": 0.0, "b": 0.0}
        result = normalize_probs(probs)
        assert result is probs  # can't normalize all-zero

    def test_exact_one(self):
        probs = {"true": 0.5, "false": 0.5}
        result = normalize_probs(probs)
        assert result is probs


class TestNormalizeTeacherProbs:
    def test_single_teacher_needs_fix(self):
        tp = {"gpt_5_6_luna": {"a": 0.99, "b": 0.99, "c": 0.01, "d": 0.01}}
        result, count = normalize_teacher_probs(tp)
        assert count == 1
        assert abs(sum(result["gpt_5_6_luna"].values()) - 1.0) < 1e-9

    def test_multiple_teachers_one_bad(self):
        tp = {
            "jev": {"true": 0.8, "false": 0.2},
            "gpt_5_6_luna": {"a": 0.99, "b": 0.99, "c": 0.0, "d": 0.0},
        }
        result, count = normalize_teacher_probs(tp)
        assert count == 1
        assert result["jev"] is tp["jev"]  # jev untouched
        assert abs(sum(result["gpt_5_6_luna"].values()) - 1.0) < 1e-9

    def test_all_good(self):
        tp = {
            "jev": {"true": 0.9, "false": 0.1},
            "luna": {"a": 0.5, "b": 0.3, "c": 0.2},
        }
        result, count = normalize_teacher_probs(tp)
        assert count == 0

    def test_non_dict_probs_preserved(self):
        tp = {"jev": "invalid"}
        result, count = normalize_teacher_probs(tp)
        assert count == 0
        assert result["jev"] == "invalid"


class TestNormalizeItemsInPlace:
    def test_fixes_bad_items(self):
        items = [
            {"id": "good", "teacher_probs": {"jev": {"true": 0.7, "false": 0.3}}},
            {"id": "bad", "teacher_probs": {"luna": {"a": 0.99, "b": 0.99, "c": 0.01, "d": 0.01}}},
            {"id": "no_tp"},
        ]
        count = normalize_items_in_place(items)
        assert count == 1
        assert abs(sum(items[1]["teacher_probs"]["luna"].values()) - 1.0) < 1e-9

    def test_no_items_to_fix(self):
        items = [
            {"id": "ok", "teacher_probs": {"jev": {"true": 0.5, "false": 0.5}}},
        ]
        assert normalize_items_in_place(items) == 0


class TestNormalizeLabelFile:
    def test_normalizes_label_entries(self):
        labels = [
            {"id": "item-1", "teacher_probs": {"luna": {"a": 0.99, "b": 0.99, "c": 0.0, "d": 0.0}}},
            {"id": "item-2", "teacher_probs": {"luna": {"a": 0.8, "b": 0.2}}},
        ]
        result, count = normalize_label_file(labels)
        assert count == 1
        assert abs(sum(result[0]["teacher_probs"]["luna"].values()) - 1.0) < 1e-9
        assert result[1]["teacher_probs"]["luna"] is labels[1]["teacher_probs"]["luna"]


class TestNormalizeProbsEdgeCases:
    def test_slight_over(self):
        probs = {"a": 0.335, "b": 0.332, "c": 0.335}  # sum=1.002
        result = normalize_probs(probs)
        assert result is probs  # within 0.005 tolerance

    def test_slightly_under(self):
        probs = {"a": 0.333, "b": 0.333, "c": 0.332}  # sum=0.998
        result = normalize_probs(probs)
        assert result is probs  # within 0.005 tolerance

    def test_over_tolerance_triggers_fix(self):
        probs = {"a": 0.34, "b": 0.33, "c": 0.34}  # sum=1.01, outside 0.005
        result = normalize_probs(probs)
        assert result is not probs
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_moderate_deviation_triggers_fix(self):
        probs = {"a": 0.5, "b": 0.5, "c": 0.5}  # sum=1.5
        result = normalize_probs(probs)
        assert result is not probs
        assert abs(sum(result.values()) - 1.0) < 1e-9
        for v in result.values():
            assert abs(v - 1 / 3) < 1e-9
