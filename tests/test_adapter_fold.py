"""Tests for the adapter-folding benchmark helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.scripts.bench_adapter_fold import (  # noqa: E402
    _fill_adapter,
    _logit,
    _parity,
)
from model.src.lora import LoRALinear, inject_lora  # noqa: E402


def test_logit_is_zero_at_half_and_clamped():
    assert _logit(0.5) == pytest.approx(0.0)
    assert _logit(0.0) < 0
    assert _logit(1.0) > 0


def test_parity_counts_label_flips_and_max_difference():
    left = [[0.9, 0.1], [0.2, 0.8]]
    right = [[0.9, 0.1], [0.8, 0.2]]
    result = _parity(left, right)
    assert result["items"] == 2
    assert result["scored_options"] == 4
    assert result["label_flips"] == 1
    assert result["label_flip_fraction"] == 0.5
    assert result["max_abs_prob_diff"] == pytest.approx(0.6)
    assert result["max_abs_logit_gap_diff"] > 0
    assert result["flipped_top2_margin_max"] > 0


def test_parity_of_identical_scores_is_zero():
    scores = [[0.7, 0.2, 0.1]]
    result = _parity(scores, scores)
    assert result["max_abs_logit_gap_diff"] == 0.0
    assert result["label_flips"] == 0


def test_fill_adapter_makes_the_update_nonzero():
    base = torch.nn.Linear(8, 4)
    model = torch.nn.Sequential(base)
    inject_lora(model, targets=("0",), rank=2, alpha=4)
    adapter = model[0]
    assert isinstance(adapter, LoRALinear)
    assert torch.count_nonzero(adapter.lora_b.weight) == 0
    _fill_adapter(model, scale=0.01, seed=7)
    assert torch.count_nonzero(adapter.lora_b.weight) > 0
