"""The benchmark leakage audit must separate repeats from duplicates.

The three counters answer different questions and are easy to collapse into
one. A PIQA goal sentence recurs with different option pairs, so the state
repeats without any item being seen twice; counting that as leakage would
condemn a source that is clean, and counting only states would miss the case
that actually matters -- an eval row the model was trained on, byte for byte.
The tests below pin the distinction, and the whitespace/case normalisation that
keeps formatting from hiding a duplicate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_audit():
    """Import ``scripts/check_benchmark_leakage.py`` by path (scripts/ is not a package)."""
    path = ROOT / "scripts" / "check_benchmark_leakage.py"
    spec = importlib.util.spec_from_file_location("check_benchmark_leakage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(state: str, label: str, split: str, criteria: list[str] | None = None) -> dict:
    question = {"type": "choice", "criteria": criteria or ["a", "b"]}
    return {"state": state, "question": question, "label": label, "split": split}


def test_identical_item_is_counted_and_a_repeat_is_not(tmp_path: Path) -> None:
    audit = _load_audit()
    path = _write(
        tmp_path / "src.jsonl",
        [
            _row("the same state", "a", "train"),
            _row("the same state", "a", "test"),
            _row("recurring goal", "a", "train", ["x", "y"]),
            _row("recurring goal", "a", "test", ["z", "w"]),
            _row("only test", "b", "test"),
        ],
    )

    report = audit.audit_file(path)

    assert report["n_train"] == 2
    assert report["n_test"] == 3
    assert report["identical"] == 1
    # Both the identical row and the recurring-goal row repeat a train state.
    assert report["state_repeats"] == 2
    assert report["label_conflicts"] == 0
    assert report["identical_rate"] == 1 / 3


def test_a_relabelled_item_is_a_conflict_not_leakage(tmp_path: Path) -> None:
    audit = _load_audit()
    path = _write(
        tmp_path / "src.jsonl",
        [
            _row("state", "a", "train"),
            _row("state", "b", "test"),
        ],
    )

    report = audit.audit_file(path)

    assert report["identical"] == 0
    assert report["label_conflicts"] == 1
    assert report["state_repeats"] == 1


def test_whitespace_and_case_do_not_hide_a_duplicate(tmp_path: Path) -> None:
    audit = _load_audit()
    path = _write(
        tmp_path / "src.jsonl",
        [
            _row("A  plane is  taking off", "a", "train"),
            _row("  a plane is taking off\n", "a", "test"),
        ],
    )

    report = audit.audit_file(path)

    assert report["identical"] == 1


def test_option_order_does_not_hide_a_duplicate(tmp_path: Path) -> None:
    audit = _load_audit()
    path = _write(
        tmp_path / "src.jsonl",
        [
            _row("state", "a", "train", ["first", "second"]),
            _row("state", "a", "test", ["second", "first"]),
        ],
    )

    report = audit.audit_file(path)

    assert report["identical"] == 1


def test_rows_without_a_supported_type_are_ignored(tmp_path: Path) -> None:
    audit = _load_audit()
    path = _write(
        tmp_path / "src.jsonl",
        [
            {"state": "s", "question": {"type": "other"}, "label": "a", "split": "train"},
            {"state": "s", "question": {"type": "other"}, "label": "a", "split": "test"},
            _row("s", "a", "test"),
        ],
    )

    report = audit.audit_file(path)

    assert report["n_train"] == 0
    assert report["n_test"] == 1
    assert report["identical"] == 0


def test_rate_gate_exits_non_zero_above_the_threshold(tmp_path: Path, capsys) -> None:
    audit = _load_audit()
    _write(
        tmp_path / "dirty.jsonl",
        [_row("state", "a", "train"), _row("state", "a", "test")],
    )

    # main() reads sys.argv, so drive it the way a caller would and restore after.
    argv = sys.argv
    try:
        sys.argv = [
            "check_benchmark_leakage.py",
            "--data-dir",
            str(tmp_path),
            "--max-identical-rate",
            "0.5",
        ]
        code = audit.main()
    finally:
        sys.argv = argv

    assert code == 1
    out = capsys.readouterr().out
    assert "1 are the same item with the same label" in out
    assert "FAIL dirty" in out


def test_duplicates_alone_do_not_fail_the_run(tmp_path: Path) -> None:
    """Reporting is the default; gating is opt-in, so a dirty corpus still exits 0."""
    audit = _load_audit()
    _write(
        tmp_path / "dirty.jsonl",
        [_row("state", "a", "train"), _row("state", "a", "test")],
    )

    argv = sys.argv
    try:
        sys.argv = ["check_benchmark_leakage.py", "--data-dir", str(tmp_path)]
        code = audit.main()
    finally:
        sys.argv = argv

    assert code == 0


def test_a_clean_corpus_passes_the_gate(tmp_path: Path) -> None:
    audit = _load_audit()
    _write(
        tmp_path / "clean.jsonl",
        [_row("train state", "a", "train"), _row("test state", "a", "test")],
    )

    argv = sys.argv
    try:
        sys.argv = [
            "check_benchmark_leakage.py",
            "--data-dir",
            str(tmp_path),
            "--max-identical-rate",
            "0.005",
        ]
        code = audit.main()
    finally:
        sys.argv = argv

    assert code == 0
