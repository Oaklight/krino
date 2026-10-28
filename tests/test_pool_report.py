"""Tests for the pool-vs-slots report. Torch-free, so CI runs them.

``print_pool_report`` used to live in ``model/scripts/train_multitask.py``,
which imports torch at module scope. CI installs the package with
``--no-deps``, so every test of it skipped there, including the one guarding
the empty-pool denominator.
"""

from __future__ import annotations

from collections import Counter

from data.pool_report import print_pool_report


def _report(capsys, pools: dict[str, int], sampling: dict) -> str:
    print_pool_report(Counter(pools), sampling)
    return capsys.readouterr().out


def _row(out: str, q_type: str) -> list[str]:
    return next(row.split() for row in out.splitlines() if row.strip().startswith(q_type))


def test_slots_split_evenly_when_every_type_has_items(capsys):
    out = _report(
        capsys,
        {"noul": 1_000, "score": 1_000},
        {"epoch_size": 10_000, "type_ratios": {"noul": 1, "score": 1}},
    )
    assert _row(out, "noul")[2] == "5,000"
    assert _row(out, "score")[2] == "5,000"


def test_an_empty_type_takes_no_share_of_the_epoch(capsys):
    """The sampler allocates over the types that have items, and so must this."""
    out = _report(
        capsys,
        {"noul": 9_000, "choice": 3_000},
        {
            "epoch_size": 15_000,
            "type_ratios": {"noul": 1, "choice": 1, "score": 1},
        },
    )
    assert _row(out, "noul")[2] == "7,500"
    assert _row(out, "choice")[2] == "7,500"
    # The old denominator (every ratio) printed 5,000 for each -- 33% low.
    assert "5,000" not in out


def test_empty_pool_is_na_not_a_division(capsys):
    out = _report(
        capsys,
        {"noul": 5},
        {"epoch_size": 100, "type_ratios": {"noul": 1, "score": 1}},
    )
    assert "n/a" in out
    assert _row(out, "noul")[2] == "100"


def test_difficulty_weights_take_precedence(capsys):
    out = _report(
        capsys,
        {"noul": 100, "score": 100},
        {
            "epoch_size": 1_000,
            "type_ratios": {"noul": 1, "score": 1},
            "difficulty_weights": {"noul": 3, "score": 1},
        },
    )
    assert _row(out, "noul")[2] == "750"
    assert _row(out, "score")[2] == "250"


def test_no_sampling_config_prints_nothing(capsys):
    assert _report(capsys, {"noul": 10}, {}) == ""


def test_resampling_is_pool_relative(capsys):
    out = _report(
        capsys,
        {"noul": 100_000, "score": 2_500},
        {"epoch_size": 10_000, "type_ratios": {"noul": 1, "score": 1}},
    )
    assert _row(out, "noul")[3] == "0.05x"
    assert _row(out, "score")[3] == "2.00x"
