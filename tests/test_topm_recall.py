"""Tests for the top-m recall helpers.

Torch-free on purpose: these run in the CI leg that skips torch.
"""

from __future__ import annotations

from model.evaluation.topm_recall import gold_rank, k_bucket, summarize


def test_ties_count_against_gold():
    assert gold_rank([0.4, 0.4, 0.2], 0) == 2
    assert gold_rank([0.4, 0.4, 0.2], 1) == 2
    assert gold_rank([0.5, 0.3, 0.2], 0) == 1
    assert gold_rank([0.2, 0.3, 0.5], 0) == 3


def test_k_bucket():
    assert k_bucket(3) == "3-5"
    assert k_bucket(5) == "3-5"
    assert k_bucket(6) == "6+"


def test_summarize_recall_and_se():
    rows = [{"rank": 1}, {"rank": 2}, {"rank": 3}, {"rank": 1}]
    out = summarize(rows, max_m=3)
    assert out["n"] == 4
    assert out["recall@1"] == 0.5
    assert out["recall@2"] == 0.75
    assert out["recall@3"] == 1.0
    assert out["se@1"] == 0.25
    assert out["se@3"] == 0.0


def test_empty_group_returns_none():
    out = summarize([], max_m=1)
    assert out == {
        "n": 0,
        "recall@1": None,
        "se@1": None,
        "n_informative@1": 0,
        "informative@1": None,
    }


def test_informative_drops_free_hits():
    # K=3 at m=3 is a guaranteed hit; only the K=6 row is informative.
    rows = [{"rank": 3, "k": 3}, {"rank": 5, "k": 6}]
    out = summarize(rows, max_m=3)
    assert out["recall@3"] == 0.5
    assert out["n_informative@3"] == 1
    assert out["informative@3"] == 0.0
