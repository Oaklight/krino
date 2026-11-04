"""Tests for the torch-free evaluation gate metrics."""

from __future__ import annotations

import pytest

from model.evaluation.gate import (
    Prediction,
    brier_skill_score,
    chance_corrected_accuracy,
    expected_calibration_error,
    k_bucket,
    multiclass_brier,
    order_sensitivity,
    quadratic_weighted_kappa,
    rank_probability_score,
    risk_coverage,
    schema_key,
    summarize,
)


def noul(item_id: str, prob_true: float, label: bool) -> Prediction:
    return Prediction(
        item_id=item_id,
        q_type="noul",
        label=label,
        probs=(1.0 - prob_true, prob_true),
        keys=("no", "yes"),
    )


def choice(
    item_id: str,
    probs: list[float],
    keys: list[str],
    label: str,
) -> Prediction:
    return Prediction(
        item_id=item_id,
        q_type="choice",
        label=label,
        probs=tuple(probs),
        keys=tuple(keys),
    )


def score(item_id: str, probs: list[float], label: int) -> Prediction:
    return Prediction(
        item_id=item_id,
        q_type="score",
        label=float(label),
        probs=tuple(probs),
        keys=tuple(str(i) for i in range(len(probs))),
    )


class TestPointMetrics:
    def test_perfect_calibration_has_zero_ece(self):
        assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)

    def test_ece_uses_the_confidence_gap(self):
        assert expected_calibration_error([0.9], [False]) == pytest.approx(0.9)

    def test_binary_brier(self):
        from model.evaluation.gate import binary_brier

        assert binary_brier(0.5, True) == pytest.approx(0.25)

    def test_multiclass_brier(self):
        assert multiclass_brier([0.7, 0.2, 0.1], 0) == pytest.approx(0.14)

    def test_brier_skill_against_a_reference(self):
        assert brier_skill_score(0.1, 0.2) == pytest.approx(0.5)
        assert brier_skill_score(0.1, 0.0) is None

    def test_qwk_is_one_for_a_perfect_ordinal_prediction(self):
        assert quadratic_weighted_kappa([0, 1, 2], [0, 1, 2], 3) == pytest.approx(1.0)

    def test_qwk_is_negative_for_a_reversed_ordinal_prediction(self):
        assert quadratic_weighted_kappa([0, 1, 2], [2, 1, 0], 3) == pytest.approx(-1.0)

    def test_rps_zero_for_a_confident_correct_forecast(self):
        assert rank_probability_score([1.0, 0.0, 0.0], 0) == pytest.approx(0.0)

    def test_rps_penalises_a_confident_wrong_forecast(self):
        assert rank_probability_score([0.0, 1.0, 0.0], 0) == pytest.approx(0.5)


class TestBuckets:
    @pytest.mark.parametrize(
        "k,expected",
        [
            (2, "2"),
            (3, "3-5"),
            (5, "3-5"),
            (6, "6-20"),
            (20, "6-20"),
            (21, "21-80"),
            (80, "21-80"),
            (81, "81-256"),
            (256, "81-256"),
            (257, ">256"),
        ],
    )
    def test_k_bucket_boundaries(self, k, expected):
        assert k_bucket(k) == expected


class TestSchemaKey:
    def test_formatting_does_not_change_the_schema(self):
        a = schema_key({"instructions": "Pick One", "criteria": {"a": "A", "b": "B"}})
        b = schema_key({"instructions": "  pick   one ", "criteria": {"b": "B", "a": "A"}})
        assert a == b

    def test_label_set_changes_the_schema(self):
        a = schema_key({"instructions": "Pick", "criteria": {"a": "A", "b": "B"}})
        b = schema_key({"instructions": "Pick", "criteria": {"a": "A", "c": "C"}})
        assert a != b


class TestOrderSensitivity:
    def test_unchanged_distribution_reports_zero_movement(self):
        before = [choice("1", [0.9, 0.1], ["a", "b"], "a")]
        after = [choice("1", [0.1, 0.9], ["b", "a"], "a")]
        result = order_sensitivity(before, after)
        assert result["changed_fraction"] == pytest.approx(0.0)
        assert result["mean_tv"] == pytest.approx(0.0)
        assert result["accuracy_after"] == pytest.approx(1.0)

    def test_a_flipped_answer_is_counted(self):
        before = [choice("1", [0.9, 0.1], ["a", "b"], "a")]
        after = [choice("1", [0.1, 0.9], ["a", "b"], "a")]
        result = order_sensitivity(before, after)
        assert result["changed_fraction"] == pytest.approx(1.0)
        assert result["mean_tv"] == pytest.approx(0.8)
        assert result["accuracy_after"] == pytest.approx(0.0)


class TestRiskCoverage:
    def test_confident_correct_items_are_kept_first(self):
        rows = [
            noul("1", 0.99, True),
            noul("2", 0.95, True),
            noul("3", 0.55, False),
            noul("4", 0.51, False),
        ]
        result = risk_coverage(rows, grid=(0.5, 1.0))
        assert result["curve"][0]["selective_accuracy"] == pytest.approx(1.0)
        assert result["curve"][-1]["selective_accuracy"] == pytest.approx(0.5)
        assert result["risk_at_80"] is not None


class TestSummarize:
    def test_noul_summary(self):
        rows = [noul("1", 0.9, True), noul("2", 0.2, False)]
        result = summarize(rows)
        assert result["n"] == 2
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["brier"] == pytest.approx(0.025)
        assert result["bss"] == pytest.approx(0.9)

    def test_score_summary_reports_ordinal_metrics(self):
        rows = [
            score("1", [0.1, 0.2, 0.7], 2),
            score("2", [0.7, 0.2, 0.1], 0),
            score("3", [0.1, 0.7, 0.2], 1),
        ]
        result = summarize(rows)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["qwk"] == pytest.approx(1.0)
        assert result["rps"] == pytest.approx(1.0 / 24.0)
        assert result["mae"] == pytest.approx(0.2 * 1 + 0.1 * 1)

    def test_unmapped_labels_are_excluded(self):
        rows = [choice("1", [0.5, 0.5], ["a", "b"], "missing")]
        assert summarize(rows) == {"n": 0}


class TestChanceCorrectedAccuracy:
    """Local Intelligence proxy: per-item chance normalisation."""

    def test_guessing_scores_zero(self):
        # Two options, always wrong.
        rows = [
            Prediction("a", "noul", True, (1.0, 0.0), ("false", "true")),
            Prediction("b", "noul", True, (1.0, 0.0), ("false", "true")),
        ]
        out = chance_corrected_accuracy(rows)
        assert out["accuracy"] == 0.0
        assert out["chance"] == pytest.approx(0.5)
        assert out["corrected"] == pytest.approx(-1.0)

    def test_perfect_scores_one(self):
        rows = [Prediction("a", "noul", True, (0.0, 1.0), ("false", "true"))]
        assert chance_corrected_accuracy(rows)["corrected"] == pytest.approx(1.0)

    def test_five_options_are_normalised_by_five(self):
        rows = [Prediction("a", "choice", "x", (0.4, 0.3, 0.2, 0.1, 0.0), ("x", "b", "c", "d", "e"))]
        out = chance_corrected_accuracy(rows)
        assert out["corrected"] == pytest.approx(1.0)
        assert out["chance"] == pytest.approx(0.2)

    def test_single_option_items_are_skipped(self):
        rows = [Prediction("a", "choice", "x", (1.0,), ("x",))]
        assert chance_corrected_accuracy(rows)["n"] == 0

    def test_matches_raw_accuracy_relation(self):
        # Chance-corrected = (acc - c) / (1 - c) when chance is uniform.
        rows = [
            Prediction("a", "choice", "x", (0.6, 0.4), ("x", "y")),
            Prediction("b", "choice", "x", (0.6, 0.4), ("x", "y")),
            Prediction("c", "choice", "y", (0.6, 0.4), ("x", "y")),
            Prediction("d", "choice", "y", (0.6, 0.4), ("x", "y")),
        ]
        out = chance_corrected_accuracy(rows)
        assert out["accuracy"] == pytest.approx(0.5)
        assert out["corrected"] == pytest.approx(0.0)


class TestResidualHeadQuestionOverride:
    """The K-sweep swaps in a shorter question; the head must follow it."""

    def test_head_logits_scores_the_substituted_question(self):
        pytest.importorskip("torch")
        from model.scripts.train_residual import head_logits

        seen: list[list[str]] = []

        class _Model:
            def forward_choice(self, state, instructions, texts):
                import torch

                seen.append(list(texts))
                return torch.zeros(1, len(texts))

        class _Item:
            id = "x"
            state = "s"
            question = {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"},
            }

        swept = {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}
        out = head_logits(_Model(), _Item(), swept)
        assert out.shape == (1, 2)
        assert seen == [["A", "B"]]

    def test_head_logits_defaults_to_the_item_question(self):
        pytest.importorskip("torch")
        from model.scripts.train_residual import head_logits

        seen: list[list[str]] = []

        class _Model:
            def forward_choice(self, state, instructions, texts):
                import torch

                seen.append(list(texts))
                return torch.zeros(1, len(texts))

        class _Item:
            id = "x"
            state = "s"
            question = {
                "type": "choice",
                "instructions": "pick",
                "criteria": {"a": "A", "b": "B"},
            }

        out = head_logits(_Model(), _Item())
        assert out.shape == (1, 2)
        assert seen == [["A", "B"]]


class TestSoftTarget:
    def test_namespaced_teacher_is_unwrapped(self):
        pytest.importorskip("torch")
        from model.scripts.train_residual import soft_target

        class _Item:
            teacher_name = "jev"
            teacher_probs = {"jev": {"false": 0.25, "true": 0.75}}
            question = {"type": "noul"}

        out = soft_target(_Item(), 2)
        assert out is not None
        assert out.tolist() == pytest.approx([0.25, 0.75])

    def test_flat_legacy_dict_still_works(self):
        pytest.importorskip("torch")
        from model.scripts.train_residual import soft_target

        class _Item:
            teacher_name = None
            teacher_probs = {"false": 0.5, "true": 0.5}
            question = {"type": "noul"}

        out = soft_target(_Item(), 2)
        assert out.tolist() == pytest.approx([0.5, 0.5])

    def test_mismatched_keys_fall_back(self):
        pytest.importorskip("torch")
        from model.scripts.train_residual import soft_target

        class _Item:
            teacher_name = "jev"
            teacher_probs = {"jev": {"a": 0.5, "b": 0.5}}
            question = {"type": "noul"}

        assert soft_target(_Item(), 2) is None
