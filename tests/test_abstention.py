"""Tests for the abstention set builder and the risk-coverage evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from data.format import TypedQuestion
from data.pipeline import load_jsonl, save_jsonl
from model.scripts.build_abstention_sets import (
    _choice_without_gold,
    _insufficient,
    _out_of_scope,
    main as build_main,
)
from model.scripts.eval_abstention import _at_coverage, _aurc, _summary


def _choice(item_id: str = "c1", n_criteria: int = 4, **changes) -> TypedQuestion:
    criteria = {f"option_{i}": f"criterion {i}" for i in range(n_criteria)}
    defaults = dict(
        id=item_id,
        state="state text",
        question={"type": "choice", "instructions": "pick one", "criteria": criteria},
        label="option_0",
        source="fixture",
        split="train",
        teacher_probs={"option_0": 0.9, "option_1": 0.1},
        teacher_name="jev",
    )
    defaults.update(changes)
    return TypedQuestion(**defaults)


def _noul(item_id: str = "n1", label: bool = True, state: str = "A. B. C.", **changes) -> TypedQuestion:
    defaults = dict(
        id=item_id,
        state=state,
        question={"type": "noul", "instructions": "is this supported"},
        label=label,
        source="fixture",
        split="train",
        teacher_probs={"true": 0.9, "false": 0.1},
    )
    defaults.update(changes)
    return TypedQuestion(**defaults)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_choice_without_gold_removes_gold_and_strips_teacher_probs():
    item = _choice()
    built = _choice_without_gold(item)
    assert built is not None
    assert built.id == "c1-abstain-none_of_these"
    assert "option_0" not in built.question["criteria"]
    assert len(built.question["criteria"]) == 3
    assert built.label == "option_0"  # provenance only
    assert built.answerable is False
    assert built.reason == "none_of_these"
    assert built.teacher_probs is None
    assert built.teacher_name is None
    # The source item is untouched.
    assert "option_0" in item.question["criteria"]
    assert item.teacher_probs is not None


def test_choice_without_gold_rejects_small_or_invalid_choice():
    assert _choice_without_gold(_choice(n_criteria=2)) is None
    bad = _choice()
    bad.question["criteria"] = {"other": "x"}
    assert _choice_without_gold(bad) is None
    assert _choice_without_gold(_noul()) is None


def test_out_of_scope_uses_donor_state():
    item = _noul(state="original state")
    built = _out_of_scope(item, "donor state")
    assert built is not None
    assert built.id == "n1-abstain-out_of_scope"
    assert built.state == "donor state"
    assert built.label is False
    assert built.answerable is False
    assert built.reason == "out_of_scope"
    assert built.teacher_probs is None
    assert _out_of_scope(item, "original state") is None
    assert _out_of_scope(_choice(), "donor state") is None


def test_insufficient_truncates_state_and_requires_supported_noul():
    item = _noul(state="One sentence. Two sentence. Three sentence. Four sentence.")
    built = _insufficient(item, 0.5)
    assert built is not None
    assert built.state == "One sentence. Two sentence."
    assert built.label is False
    assert built.answerable is False
    assert built.reason == "insufficient"
    assert built.teacher_probs is None
    # Unsupported noul and choice items are not truncated into abstention.
    assert _insufficient(_noul(label=False), 0.5) is None
    assert _insufficient(_choice(), 0.5) is None


def test_insufficient_handles_single_sentence():
    state = "x" * 100
    built = _insufficient(_noul(state=state), 0.5)
    assert built is not None
    assert built.state == "x" * 50


def test_build_main_writes_sets_and_stats(tmp_path: Path):
    items: list[TypedQuestion] = []
    for i in range(10):
        items.append(_choice(f"c{i}"))
        items.append(_noul(f"n{i}", source=f"src{i % 2}"))
    items_path = tmp_path / "items.jsonl"
    out_path = tmp_path / "abstain.jsonl"
    save_jsonl(items, items_path)

    import sys

    argv = sys.argv
    sys.argv = [
        "build_abstention_sets.py",
        "--items", str(items_path),
        "--out", str(out_path),
        "--limit-per-reason", "5",
        "--seed", "7",
    ]
    try:
        assert build_main() == 0
    finally:
        sys.argv = argv

    built = load_jsonl(out_path)
    assert len(built) == 15  # 5 per reason
    assert {item.reason for item in built} == {"none_of_these", "out_of_scope", "insufficient"}
    assert all(item.answerable is False for item in built)
    assert all(item.teacher_probs is None for item in built)
    assert len({item.id for item in built}) == len(built)

    stats = json.loads(out_path.with_suffix(".stats.json").read_text())
    assert stats["selected"] == 15
    assert stats["by_reason"] == {"none_of_these": 5, "out_of_scope": 5, "insufficient": 5}
    assert stats["teacher_probs_present"] == 0
    assert stats["answerable_values"] == {"False": 15}


def test_build_main_rejects_unknown_reason(tmp_path: Path):
    items_path = tmp_path / "items.jsonl"
    save_jsonl([_choice()], items_path)

    import sys

    argv = sys.argv
    sys.argv = [
        "build_abstention_sets.py",
        "--items", str(items_path),
        "--out", str(tmp_path / "out.jsonl"),
        "--reasons", "made_up",
    ]
    try:
        try:
            build_main()
        except SystemExit as exc:
            assert "made_up" in str(exc)
        else:  # pragma: no cover - the call must exit
            raise AssertionError("expected SystemExit")
    finally:
        sys.argv = argv


# ---------------------------------------------------------------------------
# Risk-coverage evaluation
# ---------------------------------------------------------------------------


def _pred(item_id: str, correct: bool, p_answerable: float, max_prob: float, **changes) -> dict:
    record = {
        "id": item_id,
        "q_type": "choice",
        "gold": "a",
        "pred": "a" if correct else "b",
        "correct": correct,
        "p_answerable": p_answerable,
        "max_prob": max_prob,
    }
    record.update(changes)
    return record


def test_aurc_perfect_ordering():
    records = [
        _pred("1", True, 0.9, 0.9),
        _pred("2", True, 0.8, 0.9),
        _pred("3", False, 0.2, 0.9),
        _pred("4", False, 0.1, 0.9),
    ]
    # risks: 0, 0, 1/3, 1/2
    assert abs(_aurc(records, "p_answerable") - (0 + 0 + 1 / 3 + 1 / 2) / 4) < 1e-9


def test_aurc_reverse_ordering():
    records = [
        _pred("1", False, 0.9, 0.9),
        _pred("2", False, 0.8, 0.9),
        _pred("3", True, 0.2, 0.9),
        _pred("4", True, 0.1, 0.9),
    ]
    assert abs(_aurc(records, "p_answerable") - (1 + 1 + 2 / 3 + 1 / 2) / 4) < 1e-9


def test_aurc_missing_scores_are_skipped():
    records = [_pred("1", True, 0.9, 0.9), _pred("2", False, None, 0.2)]
    assert abs(_aurc(records, "p_answerable") - 0.0) < 1e-9


def test_at_coverage_reports_answered_and_abstain_zero_accuracy():
    records = [
        _pred("1", True, 0.9, 0.9),
        _pred("2", True, 0.8, 0.9),
        _pred("3", False, 0.2, 0.9),
        _pred("4", False, 0.1, 0.9),
    ]
    result = _at_coverage(records, "p_answerable", 50)
    assert result["answered"] == 2
    assert result["answered_accuracy"] == 1.0
    assert result["abstain_zero_accuracy"] == 0.5


def test_abstention_beats_max_prob_baseline():
    # max_prob points the wrong way: high confidence on the wrong answers.
    records = [
        _pred("1", True, 0.9, 0.55),
        _pred("2", True, 0.8, 0.54),
        _pred("3", False, 0.2, 0.99),
        _pred("4", False, 0.1, 0.98),
    ]
    model = _summary(records, "p_answerable")
    baseline = _summary(records, "max_prob")
    assert model["aurc"] < baseline["aurc"]
    model_50 = model["coverage"][0]
    baseline_50 = baseline["coverage"][0]
    assert model_50["answered_accuracy"] == 1.0
    assert baseline_50["answered_accuracy"] == 0.0


def test_derives_correct_from_pred_when_missing():
    record = {"id": "1", "gold": "a", "pred": "a", "p_answerable": 0.9}
    summary = _summary([record], "p_answerable")
    assert summary["coverage"][-1]["answered_accuracy"] == 1.0
