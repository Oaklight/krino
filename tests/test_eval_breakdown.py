"""The eval breakdown must separate question types and rank cells honestly.

``by_source`` keyed on the source name and remembered only the first question
type it saw, so a multi-type source like ``synthetic`` (586 noul / 284 choice /
130 score in the eval pool) was reported as a single "noul" row whose accuracy
was a mix of all three. That is the bug these tests pin.

They also pin the two presentation decisions review asked for: "worst first"
must be ordered by a Wilson lower bound rather than raw accuracy, so a three-item
cell cannot outrank a thousand-item cell, and the printer must keep rendering
pre-existing ``history.json`` files, which have no ``by_source_type``.

Everything here is torch-free — this module is pure arithmetic — so the tests
run in CI, which installs only pytest.
"""

from __future__ import annotations

import pytest

from model.evaluation.breakdown import (
    build_eval_breakdown,
    format_detailed_eval,
    wilson_lower_bound,
    wilson_upper_bound,
)


def _rows(n, source, q_type, correct, loss=1.0):
    """`n` rows for a source/type, `correct` of them right."""
    return [(source, q_type, loss, i < correct) for i in range(n)]


def test_splits_a_multi_type_source_by_type():
    rows = (
        _rows(3, "synthetic", "noul", 2)
        + _rows(2, "synthetic", "choice", 1)
        + _rows(1, "synthetic", "score", 1)
        + _rows(1, "sst2", "noul", 1)
    )
    stats = build_eval_breakdown(rows)

    by_st = stats["by_source_type"]
    assert set(by_st) == {"synthetic", "sst2"}
    assert set(by_st["synthetic"]) == {"noul", "choice", "score"}

    assert by_st["synthetic"]["noul"]["n_items"] == 3
    assert by_st["synthetic"]["noul"]["correct"] == 2
    assert by_st["synthetic"]["noul"]["accuracy"] == pytest.approx(2 / 3)
    assert by_st["synthetic"]["choice"]["accuracy"] == pytest.approx(0.5)
    assert by_st["synthetic"]["score"]["accuracy"] == pytest.approx(1.0)
    assert by_st["sst2"]["noul"]["accuracy"] == pytest.approx(1.0)


def test_by_source_totals_are_the_sum_over_types():
    rows = _rows(1, "synthetic", "noul", 1) + _rows(1, "synthetic", "choice", 0)
    rows += _rows(1, "synthetic", "score", 0)
    stats = build_eval_breakdown(rows)

    merged = stats["by_source"]["synthetic"]
    per_type = stats["by_source_type"]["synthetic"]
    assert merged["n_items"] == sum(m["n_items"] for m in per_type.values())
    assert merged["correct"] == sum(m["correct"] for m in per_type.values())
    assert merged["correct"] == 1


def test_multi_type_source_reports_no_single_type():
    """The field that caused the original confusion must not answer falsely."""
    rows = _rows(2, "synthetic", "noul", 2) + _rows(2, "synthetic", "score", 1)
    stats = build_eval_breakdown(rows)
    assert stats["by_source"]["synthetic"]["type"] is None

    single = build_eval_breakdown(_rows(2, "sst2", "noul", 2))
    assert single["by_source"]["sst2"]["type"] == "noul"


def test_by_type_totals_match_the_sum_across_sources():
    rows = _rows(2, "sst2", "noul", 1) + _rows(1, "arc", "choice", 1) + _rows(3, "sst2", "choice", 2)
    stats = build_eval_breakdown(rows)

    assert stats["aggregate"]["n_items"] == 6
    assert stats["aggregate"]["correct"] == 4
    for q_type, metrics in stats["by_type"].items():
        rows_for_type = [
            m
            for types in stats["by_source_type"].values()
            for t, m in types.items()
            if t == q_type
        ]
        assert metrics["n_items"] == sum(r["n_items"] for r in rows_for_type)
        assert metrics["correct"] == sum(r["correct"] for r in rows_for_type)


def test_empty_input_is_all_zeros():
    stats = build_eval_breakdown([])
    assert stats["aggregate"]["n_items"] == 0
    assert stats["aggregate"]["accuracy"] == 0.0
    assert stats["by_type"] == {}
    assert stats["by_source_type"] == {}


@pytest.mark.parametrize(
    "correct,n,expected",
    [(0, 10, 0.0), (300, 1000, 0.2724), (1, 3, 0.0615), (5, 20, 0.1119)],
)
def test_wilson_lower_bound_matches_reference_values(correct, n, expected):
    assert wilson_lower_bound(correct, n) == pytest.approx(expected, abs=5e-4)


def test_wilson_orders_noise_below_signal():
    """Elena's case: 1/3 point estimate beats 300/1000, but must not outrank it."""
    assert 1 / 3 > 0.300
    assert wilson_lower_bound(1, 3) < wilson_lower_bound(300, 1000)


def test_ordering_key_is_the_upper_bound():
    """A thin cell must not lead the table just because it is thin."""
    assert wilson_upper_bound(1, 3) > wilson_upper_bound(300, 1000)
    assert wilson_upper_bound(60, 200) < wilson_upper_bound(180, 200)


def test_printer_orders_worst_reliably_first(capsys):
    rows = (
        _rows(3, "tiny", "choice", 1)          # 0.333 on n=3
        + _rows(1000, "solid", "choice", 300)  # 0.300 on n=1000
        + _rows(100, "good", "choice", 90)     # 0.900
    )
    stats = build_eval_breakdown(rows)

    print(format_detailed_eval(stats), end="")

    out = capsys.readouterr().out
    positions = {name: out.index(name) for name in ("tiny", "solid", "good")}
    # solid (0.300 on n=1000) is reliably the worst, so it leads; tiny (0.333 on
    # n=3) is merely unknown, so it must not lead; good (0.900) belongs last.
    assert positions["solid"] < positions["tiny"] < positions["good"]
    assert min(positions.values()) != positions["tiny"], "a 3-item cell must not lead"


def test_printer_renders_legacy_history_without_source_type(capsys):
    """A history.json written before this change has only by_source."""
    legacy = {
        "aggregate": {"mean_loss": 1.0, "accuracy": 0.5, "n_items": 4, "correct": 2},
        "by_type": {"choice": {"mean_loss": 1.0, "accuracy": 0.5, "n_items": 4, "correct": 2}},
        "by_source": {
            "arc": {
                "mean_loss": 1.0,
                "accuracy": 0.5,
                "n_items": 4,
                "correct": 2,
                "type": "choice",
            }
        },
    }

    print(format_detailed_eval(legacy), end="")

    out = capsys.readouterr().out
    assert "arc" in out
    assert "choice" in out
    assert "Aggregate" in out


def test_printer_handles_an_empty_breakdown(capsys):
    print(format_detailed_eval(build_eval_breakdown([])), end="")
    out = capsys.readouterr().out
    assert "Aggregate" in out


def test_printer_flags_cells_far_below_the_panel(capsys):
    rows = _rows(200, "good", "choice", 180) + _rows(200, "bad", "choice", 60)
    print(format_detailed_eval(build_eval_breakdown(rows)), end="")
    out = capsys.readouterr().out

    bad_line = next(line for line in out.splitlines() if "bad" in line)
    good_line = next(line for line in out.splitlines() if "good" in line)
    assert "<--" in bad_line
    assert "<--" not in good_line


class TestScoreColumns:
    """Score items carry (rps, mae); every other row stops at four columns."""

    def test_score_metrics_appear_only_on_score_buckets(self):
        from model.evaluation.breakdown import build_eval_breakdown

        stats = build_eval_breakdown(
            [
                ("sst5", "score", 1.0, True, 0.10, 0.25),
                ("sst5", "score", 1.2, False, 0.30, 0.75),
                ("sst2", "noul", 0.5, True),
            ]
        )
        score = stats["by_type"]["score"]
        assert score["rps"] == pytest.approx(0.20)
        assert score["mae"] == pytest.approx(0.50)
        assert score["n_scored"] == 2
        assert "rps" not in stats["by_type"]["noul"]

    def test_a_bucket_with_no_score_items_has_no_keys(self):
        from model.evaluation.breakdown import build_eval_breakdown

        stats = build_eval_breakdown([("boolq", "noul", 0.4, True)])
        assert "rps" not in stats["aggregate"]
        assert "n_scored" not in stats["aggregate"]

    def test_aggregate_mixes_only_the_score_rows(self):
        from model.evaluation.breakdown import build_eval_breakdown

        stats = build_eval_breakdown(
            [
                ("sst5", "score", 1.0, True, 0.10, 0.25),
                ("sst2", "noul", 0.5, True),
                ("sst2", "noul", 0.7, False),
            ]
        )
        # Mean over the one score item, not over all three.
        assert stats["aggregate"]["rps"] == pytest.approx(0.10)
        assert stats["aggregate"]["n_scored"] == 1
        assert stats["aggregate"]["n_items"] == 3

    def test_source_type_breakdown_carries_them_too(self):
        from model.evaluation.breakdown import build_eval_breakdown

        stats = build_eval_breakdown(
            [
                ("mixed", "score", 1.0, True, 0.20, 0.40),
                ("mixed", "noul", 0.5, True),
            ]
        )
        assert stats["by_source_type"]["mixed"]["score"]["rps"] == pytest.approx(0.20)
        assert "rps" not in stats["by_source_type"]["mixed"]["noul"]

    def test_the_report_prints_them_on_score_rows(self):
        from model.evaluation.breakdown import build_eval_breakdown, format_detailed_eval

        stats = build_eval_breakdown(
            [
                ("sst5", "score", 1.0, True, 0.10, 0.25),
                ("sst2", "noul", 0.5, True),
            ]
        )
        report = format_detailed_eval(stats)
        score_line = next(row for row in report.splitlines() if "score" in row)
        noul_line = next(row for row in report.splitlines() if "noul" in row)
        assert "rps=0.1000" in score_line and "mae=0.2500" in score_line
        assert "rps=" not in noul_line
