"""Tests for the paired unanswerable-variant builder."""

from __future__ import annotations

import json
from pathlib import Path

from data.format import TypedQuestion
from data.pipeline import load_jsonl, save_jsonl
from model.scripts.build_unanswerable_pairs import (
    build_pairs,
    corrupt_entities,
    is_answerable_original,
    key_entities,
    main as build_main,
    pair_error,
    remove_supporting_sentences,
    split_sentences,
    split_state,
)


def _noul(
    item_id: str,
    state: str,
    group: str,
    label: bool = True,
    source: str = "boolq",
    instructions: str = "Based on the passage, is the answer yes?",
    **changes,
) -> TypedQuestion:
    defaults = dict(
        id=item_id,
        state=state,
        question={"type": "noul", "instructions": instructions},
        label=label,
        source=source,
        split="train",
        group=group,
        teacher_probs={"true": 0.9, "false": 0.1},
    )
    defaults.update(changes)
    return TypedQuestion(**defaults)


def _boolq(item_id: str, group: str, entity: str = "Acme", **changes) -> TypedQuestion:
    return _noul(
        item_id,
        f"Passage: {entity} makes widgets. {entity} ships widgets worldwide. "
        f"The firm was founded in 1990.\n\nQuestion: Did {entity} make widgets?",
        group,
        **changes,
    )


def _fever(item_id: str, group: str) -> TypedQuestion:
    return TypedQuestion(
        id=item_id,
        state=(
            "Claim: Alice visited Paris.\n\nEvidence:\n"
            "Alice traveled to Paris in May. She stayed for a week."
        ),
        question={
            "type": "choice",
            "instructions": "Based on the evidence, what is the verdict on this claim?",
            "criteria": {
                "supports": "The evidence supports the claim",
                "refutes": "The evidence refutes or contradicts the claim",
                "nei": "There is not enough information to verify the claim",
            },
        },
        label="supports",
        source="fever",
        split="train",
        group=group,
    )


# ---------------------------------------------------------------------------
# State parsing and sentence splitting
# ---------------------------------------------------------------------------


def test_split_state_extracts_evidence_and_question():
    item = _boolq("b1", "g1")
    split = split_state(item)
    assert split is not None
    assert split.support == (
        "Acme makes widgets. Acme ships widgets worldwide. The firm was founded in 1990."
    )
    assert split.query == "Did Acme make widgets?"
    assert split.prefix == "Passage: "
    assert split.suffix.endswith("Did Acme make widgets?")


def test_split_state_returns_none_for_unstructured_source():
    item = _noul("u1", "some text", "g1", source="sst2")
    assert split_state(item) is None


def test_split_sentences_keeps_abbreviation_inside_the_sentence():
    units = split_sentences("D.C. is the capital. The city hosts the government.")
    assert [unit[2] for unit in units] == [
        "D.C. is the capital.",
        "The city hosts the government.",
    ]


def test_split_sentences_handles_single_sentence():
    assert len(split_sentences("Only one sentence here.")) == 1


def test_split_sentences_keeps_title_abbreviations():
    units = split_sentences("The trail to Mt. Misen is a hike. It is easy.")
    assert [unit[2] for unit in units] == [
        "The trail to Mt. Misen is a hike.",
        "It is easy.",
    ]


# ---------------------------------------------------------------------------
# Support removal
# ---------------------------------------------------------------------------


def test_remove_supporting_sentence_keeps_the_sentence_with_query_overlap():
    split = split_state(_boolq("b1", "g1"))
    assert split is not None
    removed = remove_supporting_sentences(split, {}, limit=1)
    assert len(removed) == 1
    new_support, sentence = removed[0]
    assert "widgets" in sentence.lower()
    assert sentence not in new_support
    assert "founded in 1990" in new_support


def test_remove_supporting_sentence_needs_two_sentences():
    split = split_state(
        _noul("b2", "Passage: Acme makes widgets.\n\nQuestion: Did Acme make widgets?", "g1")
    )
    assert split is not None
    assert remove_supporting_sentences(split, {}, limit=2) == []


def test_is_answerable_original_only_accepts_supported_items():
    assert is_answerable_original(_boolq("b1", "g1")) is True
    assert is_answerable_original(_boolq("b2", "g1", label=False)) is False
    assert is_answerable_original(_fever("f1", "g1")) is True
    unsupported = _fever("f2", "g1")
    unsupported.label = "nei"
    assert is_answerable_original(unsupported) is False


# ---------------------------------------------------------------------------
# Entity corruption
# ---------------------------------------------------------------------------


def test_key_entities_are_shared_capitalised_spans():
    item = _noul(
        "b1",
        "Passage: Acme makes widgets.\n\nQuestion: Did Acme make widgets?",
        "g1",
    )
    split = split_state(item)
    assert split is not None
    assert key_entities(split) == ["Acme"]


def test_key_entities_ignore_sentence_initial_grammar():
    item = _noul(
        "b2",
        "Passage: Widgets are made by Acme.\n\nQuestion: To whom are widgets sold?",
        "g1",
    )
    split = split_state(item)
    assert split is not None
    assert "To" not in key_entities(split)


def test_corrupt_entities_replaces_every_mention():
    split = split_state(_boolq("b1", "g1"))
    assert split is not None
    donors = {(1, False): {"g2": ["Beta"]}}
    results = corrupt_entities(split, "g1", donors, __import__("random").Random(0), 2)
    assert len(results) == 1
    new_support, entity, donor = results[0]
    assert entity == "Acme" and donor == "Beta"
    assert "Acme" not in new_support
    assert new_support.count("Beta") == 2


def test_corrupt_entities_skips_when_donor_is_in_the_same_group():
    split = split_state(_boolq("b1", "g1"))
    assert split is not None
    donors = {(1, False): {"g1": ["Beta"]}}
    assert corrupt_entities(split, "g1", donors, __import__("random").Random(0), 2) == []


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def test_build_pairs_emits_original_first_and_variants_after():
    items = [_boolq("b1", "g1"), _boolq("b2", "g2", entity="Beta")]
    rows, variant_counts, original_counts, diagnostics = build_pairs(items, seed=7)
    assert diagnostics["pair_failures"] == []
    assert original_counts["answerable_originals"] == 2
    assert variant_counts == {"support_removed": 4, "out_of_scope": 2, "entity_corrupted": 2}
    ids = [row.id for row in rows]
    first_original = ids.index("b1")
    assert ids[first_original + 1].startswith("b1-unans-")
    originals = [row for row in rows if row.id in {"b1", "b2"}]
    assert all(row.answerable is True for row in originals)
    variants = [row for row in rows if row.id not in {"b1", "b2"}]
    assert all(row.answerable is False for row in variants)
    assert all(row.reason in {"support_removed", "out_of_scope", "entity_corrupted"} for row in variants)
    assert all(row.teacher_probs is None for row in variants)


def test_each_pair_keeps_the_question_and_changes_only_evidence():
    items = [_boolq("b1", "g1"), _boolq("b2", "g2", entity="Beta")]
    rows, _counts, _originals, _diagnostics = build_pairs(items, seed=7)
    by_id = {row.id: row for row in rows}
    for variant in rows:
        if variant.answerable is not False:
            continue
        original_id = variant.id.split("-unans-")[0]
        original = by_id[original_id]
        assert variant.question == original.question
        assert variant.source == original.source
        assert variant.group == original.group
        original_split = split_state(original)
        variant_split = split_state(variant)
        assert original_split is not None and variant_split is not None
        assert variant_split.prefix == original_split.prefix
        assert variant_split.suffix == original_split.suffix
        assert variant_split.support != original_split.support


def test_pair_error_detects_a_changed_question():
    original = _boolq("b1", "g1")
    split = split_state(original)
    assert split is not None
    variant = TypedQuestion(
        id="b1-unans-support_removed-1",
        state=split.with_support(original.state, "Acme ships widgets worldwide."),
        question={"type": "noul", "instructions": "A different question"},
        label=True,
        source="boolq",
        split="train",
        group="g1",
        answerable=False,
        reason="support_removed",
    )
    assert pair_error(original, variant, "support_removed", "Acme ships widgets worldwide.") == (
        "question changed"
    )


def test_out_of_scope_uses_a_donor_from_another_group():
    items = [_boolq("b1", "g1"), _boolq("b2", "g2", entity="Beta")]
    rows, _counts, _originals, _diagnostics = build_pairs(
        items, transforms=["out_of_scope"], seed=3
    )
    variants = [row for row in rows if row.answerable is False]
    assert len(variants) == 2
    for variant in variants:
        assert "Beta" in variant.state
        assert "Acme" in variant.state  # the question is unchanged
        assert variant.reason == "out_of_scope"


def test_fever_supports_item_is_paired():
    rows, counts, originals, diagnostics = build_pairs(
        [_fever("f1", "g1"), _boolq("b2", "g2", entity="Beta")],
        transforms=["support_removed"],
        seed=1,
    )
    assert diagnostics["pair_failures"] == []
    assert counts["support_removed"] >= 1
    fever_variants = [row for row in rows if row.id.startswith("f1-unans-")]
    assert fever_variants
    assert fever_variants[0].label == "supports"  # provenance is kept
    assert fever_variants[0].answerable is False


def test_split_abstention_recognises_paired_ids():
    from dataclasses import replace

    from model.scripts.split_abstention import base_id

    item = _boolq("b1", "g1")
    paired = replace(
        item, id="b1-unans-support_removed-1", answerable=False, reason="support_removed"
    )
    legacy = replace(
        item, id="b1-abstain-out_of_scope", answerable=False, reason="out_of_scope"
    )
    assert base_id(paired) == "b1"
    assert base_id(legacy) == "b1"
    assert base_id(item) == "b1"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_pairs_negatives_and_stats(tmp_path: Path):
    items = [_boolq("b1", "g1"), _boolq("b2", "g2", entity="Beta")]
    items_path = tmp_path / "items.jsonl"
    save_jsonl(items, items_path)
    out = tmp_path / "pairs.jsonl"
    negatives = tmp_path / "negatives.jsonl"
    stats_path = tmp_path / "stats.json"

    assert build_main([
        "--items", str(items_path),
        "--out", str(out),
        "--negatives-out", str(negatives),
        "--stats", str(stats_path),
        "--seed", "5",
    ]) == 0

    pairs = load_jsonl(out)
    negatives_rows = load_jsonl(negatives)
    assert {row.id for row in negatives_rows} == {row.id for row in pairs if row.answerable is False}
    assert len(pairs) == len(negatives_rows) + 2
    stats = json.loads(stats_path.read_text())
    assert stats["variants"] == len(negatives_rows)
    assert stats["answerable_originals"] == 2
    assert stats["pair_checks"] == {"passed": len(negatives_rows), "failed": 0}
    assert set(stats["by_transform"]) == {"support_removed", "out_of_scope", "entity_corrupted"}
    for detail in stats["by_transform"].values():
        assert detail["length_delta"]["count"] == detail["variants"]


def test_main_rejects_an_unknown_transform(tmp_path: Path):
    items_path = tmp_path / "items.jsonl"
    save_jsonl([_boolq("b1", "g1")], items_path)
    try:
        build_main([
            "--items", str(items_path),
            "--out", str(tmp_path / "pairs.jsonl"),
            "--transforms", "made_up",
        ])
    except SystemExit as exc:
        assert "made_up" in str(exc)
    else:  # pragma: no cover - the call must exit
        raise AssertionError("expected SystemExit")
