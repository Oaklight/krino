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
from model.scripts.eval_abstention import (
    _at_coverage,
    _aurc,
    _auroc,
    _summary,
    _threshold_accuracy,
)


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


def test_auroc_orders_answerable_above_unanswerable():
    positives = [_pred("p1", True, 0.9, 0.9), _pred("p2", True, 0.8, 0.9)]
    negatives = [
        _pred("n1", False, 0.2, 0.9, reason="insufficient"),
        _pred("n2", False, 0.1, 0.9, reason="insufficient"),
    ]
    assert _auroc(negatives + positives, "p_answerable") == 1.0
    assert _auroc(positives + negatives, "p_answerable") == 1.0
    assert _auroc(positives, "p_answerable") is None


def test_auroc_counts_ties_one_half():
    records = [
        _pred("p1", True, 0.5, 0.9),
        _pred("n1", False, 0.5, 0.9, reason="out_of_scope"),
    ]
    assert _auroc(records, "p_answerable") == 0.5


def test_threshold_accuracy_scores_abstention_as_wrong():
    records = [
        _pred("p1", True, 0.9, 0.9),
        _pred("n1", False, 0.7, 0.9, reason="insufficient"),
        _pred("p2", False, 0.4, 0.9),
        _pred("n2", False, 0.2, 0.9, reason="insufficient"),
    ]
    result = _threshold_accuracy(records, "p_answerable")
    assert result["answered"] == 2
    assert result["accuracy"] == 0.25
    assert result["answer_rate"] == 0.5


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


# ---------------------------------------------------------------------------
# Train/eval split
# ---------------------------------------------------------------------------


def _abstention_fixture() -> tuple[list[TypedQuestion], list[TypedQuestion]]:
    from model.scripts.build_abstention_sets import (
        _choice_without_gold,
        _insufficient,
        _out_of_scope,
    )

    corpus: list[TypedQuestion] = []
    negatives: list[TypedQuestion] = []
    for i in range(10):
        choice = _choice(f"c{i}")
        corpus.append(choice)
        negatives.append(_choice_without_gold(choice))
        noul = _noul(f"n{i}", state="One. Two. Three. Four.")
        corpus.append(noul)
        negatives.append(_insufficient(noul, 0.5))
        if i < 5:
            negatives.append(_out_of_scope(noul, "a donor state"))
    return corpus, [item for item in negatives if item is not None]


def test_split_keeps_base_ids_on_one_side():
    from model.scripts.split_abstention import base_id, split

    corpus, negatives = _abstention_fixture()
    parts = split(negatives, corpus, eval_fraction=0.2, seed=3)
    train_bases = {base_id(item) for item in parts["train"]}
    eval_bases = set(parts["eval_bases"])
    assert eval_bases
    assert not (train_bases & eval_bases)
    assert train_bases | eval_bases == {item.id for item in corpus}
    assert len(parts["train"]) + len(parts["eval_negatives"]) == len(negatives)


def test_split_eval_mix_pairs_positives_with_negatives():
    from model.scripts.split_abstention import base_id, split

    corpus, negatives = _abstention_fixture()
    parts = split(negatives, corpus, eval_fraction=0.2, seed=3)
    assert {item.id for item in parts["eval_positives"]} == set(parts["eval_bases"])
    for positive in parts["eval_positives"]:
        assert positive.answerable is True
        assert positive.reason is None
        assert positive.split == "test"
        assert positive.label is not None
    for negative in parts["eval_negatives"]:
        assert negative.answerable is False
        assert negative.split == "test"
        assert base_id(negative) in set(parts["eval_bases"])
    assert all(item.split == "train" for item in parts["train"])


def test_split_is_deterministic_and_honours_overrides():
    from model.scripts.split_abstention import split

    corpus, negatives = _abstention_fixture()
    first = split(negatives, corpus, 0.2, 7)
    second = split(negatives, corpus, 0.2, 7)
    assert first["eval_bases"] == second["eval_bases"]

    forced = split(negatives, corpus, 0.2, 7, eval_overrides={"c4", "n9"})
    assert {"c4", "n9"} <= set(forced["eval_bases"])
    assert len(forced["eval_bases"]) == len(first["eval_bases"])


def test_split_main_writes_files_and_stats(tmp_path: Path):
    from model.scripts.split_abstention import main as split_main

    corpus, negatives = _abstention_fixture()
    abstention_path = tmp_path / "abstention.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    save_jsonl(negatives, abstention_path)
    save_jsonl(corpus, corpus_path)

    import sys

    argv = sys.argv
    sys.argv = [
        "split_abstention.py",
        "--abstention", str(abstention_path),
        "--corpus", str(corpus_path),
        "--out-dir", str(tmp_path / "split"),
        "--eval-fraction", "0.2",
        "--seed", "3",
    ]
    try:
        assert split_main() == 0
    finally:
        sys.argv = argv

    out = tmp_path / "split"
    train = load_jsonl(out / "train.jsonl")
    eval_negatives = load_jsonl(out / "eval_negatives.jsonl")
    eval_positives = load_jsonl(out / "eval_positives.jsonl")
    eval_mix = load_jsonl(out / "eval_mix.jsonl")
    assert len(eval_mix) == len(eval_negatives) + len(eval_positives)
    assert len(train) + len(eval_negatives) == len(negatives)

    stats = json.loads((out / "split.stats.json").read_text())
    assert stats["base_id_overlap"] == 0
    assert stats["eval_negatives"] == len(eval_negatives)
    assert stats["eval_positives"] == len(eval_positives)
    assert stats["eval_bases_without_positive"] == []
    assert stats["eval_base_ids"] == len(eval_positives)

    ids = (out / "eval_base_ids.txt").read_text().splitlines()
    assert len(ids) == len(eval_positives)


def test_strict_counting_marks_answered_negative_incorrect():
    records = [
        _pred("1", True, 0.9, 0.9),
        _pred("2", True, 0.8, 0.8, reason="out_of_scope"),
        _pred("3", True, 0.2, 0.2, reason="insufficient"),
    ]
    strict = _summary(records, "p_answerable")
    task = _summary(records, "p_answerable", strict_abstention=False)
    # 50% of 3 -> ceil(1.5) = 2 answered: the positive plus one negative.
    assert strict["coverage"][0]["answered"] == 2
    assert strict["coverage"][0]["answered_accuracy"] == 0.5
    # At 100% coverage strict counts only the positive; task counting credits all.
    assert strict["coverage"][-1]["answered_accuracy"] == 1 / 3
    assert task["coverage"][-1]["answered_accuracy"] == 1.0
    assert strict["aurc"] > task["aurc"]


def test_recall_by_reason_uses_global_cutoff():
    from model.scripts.eval_abstention import _recall_by_reason

    records = [
        _pred("1", True, 0.9, 0.9),
        _pred("2", True, 0.8, 0.8),
        _pred("3", False, 0.4, 0.4, reason="none_of_these"),
        _pred("4", False, 0.3, 0.3, reason="none_of_these"),
        _pred("5", False, 0.2, 0.2, reason="insufficient"),
        _pred("6", False, 0.1, 0.1, reason="insufficient"),
    ]
    recall = _recall_by_reason(records, "p_answerable", 50)
    # cutoff = ceil(6*0.5) = 3 -> items 4-6 are abstained.
    assert recall["none_of_these"] == 0.5
    assert recall["insufficient"] == 1.0
    recall_70 = _recall_by_reason(records, "p_answerable", 70)
    # cutoff = ceil(6*0.7) = 5 -> only the last item is abstained.
    assert recall_70["none_of_these"] == 0.0
    assert recall_70["insufficient"] == 0.5
