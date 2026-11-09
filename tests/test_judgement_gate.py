"""The rerun's declared variants and validity gate (#324)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from model.scripts.train_judgement_head import (  # noqa: E402
    VARIANTS,
    paired_row_test,
    validity_gate,
)
from model.src.judgement_objective import TwoWayHead  # noqa: E402


def test_declared_variants_match_the_pre_registration():
    for name in ("plain", "calibrated", "weighted"):
        assert VARIANTS[name] == {
            "loss": "bce",
            "yes_bias": 0.0,
            "lr": 1e-3,
            "epochs": 5,
            "patience": 0,
        }


def test_paired_row_test_finds_a_gold_minus_non_gold_gap():
    # Three rows of two options; gold options are scored far above the others.
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    labels = np.array([1, 0, 1, 0, 1, 0])
    rows = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    result = paired_row_test(probabilities, labels, rows)
    assert result["rows"] == 3
    assert result["mean_diff"] == pytest.approx(0.6)
    assert result["p"] is not None and result["p"] < 0.01


def test_paired_row_test_skips_rows_without_both_classes():
    probabilities = np.array([0.9, 0.1, 0.2])
    labels = np.array([1, 0, 0])
    rows = [np.array([0, 1]), np.array([2])]
    assert paired_row_test(probabilities, labels, rows)["rows"] == 1


def test_constant_head_fails_the_gate():
    # A head that gives every option P(yes) = 0.5 carries no information about
    # the label: it must not beat the base-rate predictor, and it must fail the
    # positive-rate check.
    data = {
        "hidden": np.zeros((3, 4), dtype=np.float16),
        "labels": np.array([1, 0, 0]),
    }
    head = TwoWayHead(4, yes_bias=0.0)
    gate = validity_gate(head, data, [np.array([0, 1, 2])], base_rate=1 / 3)
    assert gate["checks"]["brier_beats_constant"] is False
    assert gate["checks"]["positive_rate_at_least_half_gold_fraction"] is False
    assert gate["passes"] is False
