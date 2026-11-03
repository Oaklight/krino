"""Tests for explicit teacher selection in the synthetic loader."""

from __future__ import annotations

from data.pipeline import (
    _label_teacher_key,
    _select_teacher_entry,
    _select_teacher_probs,
)


class TestExplicitTeacher:
    def test_jev_pinned_over_luna(self):
        cached = {
            "jev": {"a": 0.7, "b": 0.3},
            "gpt_5_6_luna": {"a": 0.2, "b": 0.8},
        }
        question = {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": "A", "b": "B"},
        }
        assert _select_teacher_probs(cached, question, teacher="jev") == cached["jev"]
        assert _select_teacher_entry(cached, question, teacher="jev")[0] == "jev"

    def test_missing_explicit_teacher_does_not_fall_back(self):
        cached = {"gpt_5_6_luna": {"true": 0.9, "false": 0.1}}
        question = {"type": "noul", "instructions": "x?"}
        assert _select_teacher_probs(cached, question, teacher="jev") is None
        assert _select_teacher_entry(cached, question, teacher="jev") == (None, None)

    def test_augmented_noul_prefers_augmented_bucket(self):
        cached = {
            "jev": {"true": 0.8, "false": 0.2},
            "jev_augmented": {
                "true": 0.6,
                "false": 0.1,
                "partial": 0.2,
                "unlikely": 0.05,
                "unrelated": 0.05,
            },
        }
        question = {
            "type": "noul",
            "instructions": "x?",
            "augmented_options": {"true": "T"},
        }
        assert _select_teacher_probs(cached, question, teacher="jev") == cached["jev_augmented"]
        assert _select_teacher_entry(cached, question, teacher="jev")[0] == "jev_augmented"

    def test_luna_alias(self):
        cached = {"gpt_5_6_luna": {"true": 0.9, "false": 0.1}}
        question = {"type": "noul", "instructions": "x?"}
        assert _select_teacher_probs(cached, question, teacher="luna") == cached["gpt_5_6_luna"]

    def test_default_priority_unchanged(self):
        cached = {
            "jev": {"a": 0.7, "b": 0.3},
            "gpt_5_6_luna": {"a": 0.2, "b": 0.8},
        }
        question = {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": "A", "b": "B"},
        }
        assert _select_teacher_probs(cached, question) == cached["jev"]


class TestLabelTeacherKey:
    def test_qwen_file_rekeyed(self):
        assert (
            _label_teacher_key(
                "medical_triage_qwen3_8_27b_labels.jsonl", "gpt_5_6_luna"
            )
            == "qwen3_8_27b"
        )

    def test_other_file_unchanged(self):
        assert (
            _label_teacher_key(
                "medical_triage_gpt_5_6_luna_labels.jsonl", "gpt_5_6_luna"
            )
            == "gpt_5_6_luna"
        )
