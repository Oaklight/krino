"""Tests for Phase C: augmented noul training integration.

Covers teacher selection/flattening in pipeline.py and augmented noul
routing through the choice head in supervised.py.
"""

from __future__ import annotations

import pytest

from data.pipeline import _select_teacher_probs


# ---------------------------------------------------------------------------
# _select_teacher_probs tests
# ---------------------------------------------------------------------------


class TestSelectTeacherProbs:
    def test_augmented_noul_prefers_jev_augmented(self):
        cached = {
            "jev": {"true": 0.8, "false": 0.2},
            "jev_augmented": {"true": 0.6, "false": 0.1, "partial": 0.2, "unlikely": 0.05, "unrelated": 0.05},
            "gpt_5_6_luna_augmented": {"true": 0.5, "false": 0.2, "partial": 0.15, "unlikely": 0.1, "unrelated": 0.05},
        }
        question = {"type": "noul", "instructions": "test?", "augmented_options": {"true": "T", "false": "F", "partial": "P", "unlikely": "U", "unrelated": "X"}}
        result = _select_teacher_probs(cached, question)
        assert result == cached["jev_augmented"]

    def test_augmented_noul_falls_back_to_luna_augmented(self):
        cached = {
            "jev": {"true": 0.8, "false": 0.2},
            "gpt_5_6_luna_augmented": {"true": 0.5, "false": 0.2, "partial": 0.15, "unlikely": 0.1, "unrelated": 0.05},
        }
        question = {"type": "noul", "instructions": "test?", "augmented_options": {"true": "T", "false": "F", "partial": "P", "unlikely": "U", "unrelated": "X"}}
        result = _select_teacher_probs(cached, question)
        assert result == cached["gpt_5_6_luna_augmented"]

    def test_augmented_noul_falls_back_to_standard_if_no_augmented_teacher(self):
        cached = {
            "jev": {"true": 0.8, "false": 0.2},
        }
        question = {"type": "noul", "instructions": "test?", "augmented_options": {"true": "T", "false": "F", "partial": "P"}}
        result = _select_teacher_probs(cached, question)
        assert result == cached["jev"]

    def test_standard_noul_prefers_jev(self):
        cached = {
            "jev": {"true": 0.8, "false": 0.2},
            "gpt_5_6_luna": {"true": 0.7, "false": 0.3},
        }
        question = {"type": "noul", "instructions": "test?"}
        result = _select_teacher_probs(cached, question)
        assert result == cached["jev"]

    def test_standard_noul_falls_back_to_luna(self):
        cached = {
            "gpt_5_6_luna": {"true": 0.7, "false": 0.3},
        }
        question = {"type": "noul", "instructions": "test?"}
        result = _select_teacher_probs(cached, question)
        assert result == cached["gpt_5_6_luna"]

    def test_choice_prefers_jev(self):
        cached = {
            "jev": {"a": 0.5, "b": 0.3, "c": 0.2},
            "gpt_5_6_luna": {"a": 0.4, "b": 0.4, "c": 0.2},
        }
        question = {"type": "choice", "instructions": "pick one", "criteria": {"a": "A", "b": "B", "c": "C"}}
        result = _select_teacher_probs(cached, question)
        assert result == cached["jev"]

    def test_empty_cache_returns_none(self):
        result = _select_teacher_probs({}, {"type": "noul", "instructions": "x?"})
        assert result is None

    def test_empty_teacher_dict_skipped(self):
        cached = {"jev": {}, "gpt_5_6_luna": {"true": 0.6, "false": 0.4}}
        question = {"type": "noul", "instructions": "test?"}
        result = _select_teacher_probs(cached, question)
        assert result == cached["gpt_5_6_luna"]

    def test_qwen_fallback(self):
        cached = {"qwen3_8_27b": {"true": 0.9, "false": 0.1}}
        question = {"type": "noul", "instructions": "test?"}
        result = _select_teacher_probs(cached, question)
        assert result == cached["qwen3_8_27b"]
