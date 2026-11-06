"""Tests for the corpus dedup and contamination audit (#235).

The audit rule is fixed in the issue. These tests pin the rule, not the
current corpus counts.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.corpus_audit import (
    ContaminationIndex,
    JevItem,
    StateInfo,
    _state_digest,
    audit,
    load_jevbench,
    main,
    near_duplicate_report,
    normalize_state,
    word_shingles,
)

JEVBENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "jevbench"


def _jev_item(item_id: str, state: str) -> JevItem:
    return JevItem(
        id=item_id,
        tier="easy",
        state=state,
        normalized_state=normalize_state(state),
        shingles=frozenset(word_shingles(state)),
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(record_id: str, state: str, instructions: str, label: str = "a") -> dict:
    return {
        "id": record_id,
        "state": state,
        "question": {
            "type": "choice",
            "instructions": instructions,
            "criteria": {"a": "first", "b": "second"},
        },
        "label": label,
        "source": "fixture",
        "split": "train",
        "group": "fixture-group",
    }


def _audit(tmp_path: Path, **overrides):
    options = {
        "corpus_root": tmp_path,
        "jevbench_root": JEVBENCH_DIR,
        "include_synthetic": False,
        "include_typed_decisions": False,
        "skip_near_dup": True,
    }
    options.update(overrides)
    return audit(**options)


def test_normalize_state_folds_case_punctuation_and_unicode():
    assert normalize_state("  Hello,   WORLD!! ") == "hello world"
    assert normalize_state("Café au lait") == "café au lait"
    assert normalize_state("Café au lait") == "café au lait"
    assert normalize_state("a_b-c") == "a b c"


def test_word_shingles_uses_eight_word_windows():
    state = "one two three four five six seven eight nine"
    shingles = word_shingles(state)
    assert shingles == {
        "one two three four five six seven eight",
        "two three four five six seven eight nine",
    }
    assert word_shingles("one two three") == set()


def test_contamination_rule_fires_on_id_state_and_three_shingles():
    state = (
        "The support agent must decide whether the customer may return a "
        "product after thirty days without a receipt."
    )
    index = ContaminationIndex([_jev_item("jev-1", state)])

    state_hits = index.check_state(normalize_state(state), word_shingles(state))
    assert [(item.id, reason) for item, reason, _ in state_hits] == [("jev-1", "state")]

    same_run = (
        "The support agent must decide whether the customer may return a "
        "product after sixty days without a receipt."
    )
    shingle_hits = index.check_state(normalize_state(same_run), word_shingles(same_run))
    assert [(item.id, reason) for item, reason, _ in shingle_hits] == [
        ("jev-1", "shingle")
    ]
    assert shingle_hits[0][2] >= 3

    unrelated = "A different sentence about ocean tides and lunar gravity."
    assert index.check_state(normalize_state(unrelated), word_shingles(unrelated)) == []


def test_two_shared_shingles_do_not_fire():
    state = "one two three four five six seven eight nine ten"
    index = ContaminationIndex([_jev_item("jev-short", state)])
    variant = "one two three four five six seven eight nine eleven"
    assert len(word_shingles(state) & word_shingles(variant)) == 2
    assert index.check_state(normalize_state(variant), word_shingles(variant)) == []


def test_jevbench_loader_reads_the_231_public_items():
    items = load_jevbench(JEVBENCH_DIR)
    assert len(items) == 231
    assert len({item.id for item in items}) == 231
    counts = {
        tier: sum(item.tier == tier for item in items)
        for tier in ("easy", "hard", "original")
    }
    assert counts == {"easy": 48, "hard": 111, "original": 72}
    assert all(item.normalized_state for item in items)


def test_canonical_dedup_keeps_distinct_questions_on_one_passage(tmp_path: Path):
    passage = "A passage that several questions reuse as their state text."
    rows = [
        _record("race-1", passage, "What is the first question?"),
        _record("race-1", passage, "What is the second question?"),
        _record("race-1", passage, "What is the third question?"),
        _record("race-1", passage, "What is the first question?"),
    ]
    _write_jsonl(tmp_path / "race_short_passage_choice.jsonl", rows)

    report = _audit(tmp_path)
    coverage = report["coverage"]
    assert coverage["raw_rows_loaded"] == 4
    assert coverage["canonical_records"] == 3
    assert coverage["passage_ids"] == 1
    assert coverage["dropped_duplicate_rows"] == 1
    assert coverage["same_source_id_extra_rows"] == 2

    summary = report["dedup"]["per_source"]["race_short_passage_choice"]
    assert summary["rows"] == 3
    assert summary["raw_rows"] == 4
    assert summary["distinct_ids"] == 1
    assert summary["distinct_normalized_states"] == 1
    assert summary["exact_duplicate_rows"] == 0
    assert summary["rows_sharing_id"] == 2
    assert summary["ids_with_multiple_questions"] == 1
    assert summary["states_with_multiple_questions"] == 1


def test_strict_id_dedup_collapses_questions_on_one_passage(tmp_path: Path):
    passage = "A passage that several questions reuse as their state text."
    _write_jsonl(
        tmp_path / "race_short_passage_choice.jsonl",
        [
            _record("race-1", passage, "What is the first question?"),
            _record("race-1", passage, "What is the second question?"),
        ],
    )
    report = _audit(tmp_path, keep_same_source_id_reuse=False)
    assert report["coverage"]["canonical_records"] == 1
    assert report["coverage"]["passage_ids"] == 1


def test_canonical_priority_drops_the_lower_priority_copy(tmp_path: Path):
    _write_jsonl(
        tmp_path / "label_target_30k_jev.jsonl",
        [_record("shared-1", "State for the shared target id.", "Target question?")],
    )
    _write_jsonl(
        tmp_path / "residual_train_20000_jev.jsonl",
        [_record("residual-1", "State for the residual id.", "Residual question?")],
    )
    _write_jsonl(
        tmp_path / "typed_decisions.jsonl",
        [_record("shared-1", "State for the shared target id.", "Typed question?")],
    )
    _write_jsonl(
        tmp_path / "apigen_tool_choice.jsonl",
        [_record("shared-1", "State for the shared target id.", "Target question?")],
    )
    report = _audit(tmp_path, include_typed_decisions=True)
    coverage = report["coverage"]
    assert coverage["passage_ids"] == 2
    assert coverage["canonical_records"] == 2
    assert coverage["dropped_duplicate_rows"] == 2
    assert coverage["dropped_by_source"] == {
        "typed_decisions": 1,
        "apigen_tool_choice": 1,
    }
    pairs = {
        (entry["first"], entry["dropped"]): entry["rows"]
        for entry in coverage["dropped_cross_source_pairs"]
    }
    assert pairs == {
        ("label_target_30k_jev", "typed_decisions"): 1,
        ("label_target_30k_jev", "apigen_tool_choice"): 1,
    }


def test_check_exit_code_is_one_when_a_jevbench_state_is_reused(tmp_path: Path):
    item = load_jevbench(JEVBENCH_DIR)[0]
    _write_jsonl(
        tmp_path / "race_short_passage_choice.jsonl",
        [_record("not-a-jevbench-id", item.state, "Copied state question")],
    )
    code = main(
        [
            "--corpus-root",
            str(tmp_path),
            "--jevbench-root",
            str(JEVBENCH_DIR),
            "--no-synthetic",
            "--no-typed-decisions",
            "--skip-near-dup",
            "--check",
        ]
    )
    assert code == 1


def test_check_exit_code_is_zero_on_clean_corpus(tmp_path: Path):
    _write_jsonl(
        tmp_path / "race_short_passage_choice.jsonl",
        [_record("clean-1", "An unrelated state about migratory birds.", "Which bird?")],
    )
    code = main(
        [
            "--corpus-root",
            str(tmp_path),
            "--jevbench-root",
            str(JEVBENCH_DIR),
            "--no-synthetic",
            "--no-typed-decisions",
            "--skip-near-dup",
            "--allow-missing",
            "--check",
        ]
    )
    assert code == 0


def test_check_fails_when_a_canonical_source_is_missing(tmp_path: Path):
    _write_jsonl(
        tmp_path / "race_short_passage_choice.jsonl",
        [_record("clean-1", "An unrelated state about migratory birds.", "Which bird?")],
    )
    args = [
        "--corpus-root",
        str(tmp_path),
        "--jevbench-root",
        str(JEVBENCH_DIR),
        "--no-synthetic",
        "--no-typed-decisions",
        "--skip-near-dup",
    ]
    assert main([*args, "--check"]) == 2
    assert main([*args, "--check", "--allow-missing"]) == 0


def test_check_fails_below_the_record_floor(tmp_path: Path):
    _write_jsonl(
        tmp_path / "race_short_passage_choice.jsonl",
        [_record("clean-1", "An unrelated state about migratory birds.", "Which bird?")],
    )
    args = [
        "--corpus-root",
        str(tmp_path),
        "--jevbench-root",
        str(JEVBENCH_DIR),
        "--no-synthetic",
        "--no-typed-decisions",
        "--skip-near-dup",
        "--check",
        "--allow-missing",
    ]
    assert main([*args, "--min-records", "2"]) == 2
    assert main([*args, "--min-records", "1"]) == 0


def test_near_duplicate_report_clusters_a_paraphrase():
    base = (
        "The customer asks whether the order placed on Monday with standard "
        "shipping can arrive before the weekend, given the warehouse delay."
    )
    paraphrase = (
        "The customer asks whether the order placed on Monday with standard "
        "shipping can arrive before the weekend, given a warehouse delay."
    )
    unrelated = "A recipe for sourdough bread with a long fermentation step."
    info = {
        _state_digest(normalize_state(text)): StateInfo(
            text=normalize_state(text),
            sources={"fixture"},
            rows=1,
            example_id=f"id-{index}",
        )
        for index, text in enumerate((base, paraphrase, unrelated))
    }
    report = near_duplicate_report(info, workers=1, max_clusters=10)
    assert report["clusters"] == 1
    assert report["states_in_clusters"] == 2
    assert report["cluster_details"][0]["cluster_id"] == "nd-00000"
    assert report["cluster_details"][0]["size"] == 2
    assert len(report["state_to_cluster"]) == 2
    assert report["unclustered_states"] == 1


def test_committed_audit_artifact_records_zero_hits():
    artifact = (
        Path(__file__).resolve().parents[1] / "research" / "235-corpus-audit.json"
    )
    report = json.loads(artifact.read_text(encoding="utf-8"))
    assert report["contamination"]["hit_rows"] == 0
    assert report["coverage"]["canonical_records"] == 243366
    assert report["coverage"]["passage_ids"] == 243012


def test_canonical_list_has_sixteen_flat_files():
    from data.corpus_audit import CANONICAL_FILES

    assert len(CANONICAL_FILES) == 16
