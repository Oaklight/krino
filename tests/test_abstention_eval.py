"""Tests for the abstention evaluation instrument builder (#237)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_abstention_eval import build, existing_pairs, load_heldout


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _noul(record_id: str, group: str, state: str, label: bool = True) -> dict:
    return {
        "id": record_id,
        "state": state,
        "question": {"type": "noul", "instructions": "Answer yes or no."},
        "label": label,
        "source": "boolq",
        "split": "train",
        "group": group,
    }


def _variant(donor: dict, reason: str, label: bool = True) -> dict:
    row = dict(donor)
    row["id"] = f"{donor['id']}-abstain-{reason}"
    row["answerable"] = False
    row["reason"] = reason
    row["label"] = label
    return row


USER_STATE = (
    "Passage: Alpha Corp hired Jane Doe as lead engineer. "
    "The board approved the hire.\n\n"
    "Question: Did Alpha Corp hire Jane Doe?\n"
)


@pytest.fixture()
def corpus(tmp_path: Path) -> dict:
    donor_a = _noul("a-1", "g1", USER_STATE)
    donor_b = _noul("b-1", "g2", USER_STATE.replace("Alpha Corp", "Beta Ltd"))
    donor_c = _noul("c-1", "g3", USER_STATE.replace("Alpha Corp", "Gamma Inc"))
    rows = [
        donor_a,
        _variant(donor_a, "none_of_these"),
        donor_b,
        _variant(donor_b, "insufficient"),
        donor_c,
    ]
    canonical = _write_jsonl(tmp_path / "canonical.jsonl", rows)
    heldout = tmp_path / "heldout.json"
    heldout.write_text(
        json.dumps(
            {
                "clusters": [{"source": "boolq", "group": "g2", "n": 2}],
                "ids": ["b-1", "b-1-abstain-insufficient"],
                "ids_sha256": "test",
            }
        )
    )
    return {"canonical": canonical, "heldout": heldout, "tmp": tmp_path}


def test_build_selects_existing_pairs_and_constructs_the_rest(corpus):
    out = corpus["tmp"] / "selection.json"
    payload = build(
        corpus["canonical"], corpus["heldout"], out, None, 2, 4, 237
    )

    assert payload["counts"]["pairs"] == 2
    assert payload["counts"]["rows"] == 4
    assert payload["counts"]["existing_pairs"] == 1
    assert payload["counts"]["constructed_pairs"] == 1
    assert payload["counts"]["by_reason"]["none_of_these"] == 1
    constructed_reasons = (
        payload["counts"]["by_reason"].get("support_removed", 0)
        + payload["counts"]["by_reason"].get("entity_corrupted", 0)
    )
    assert constructed_reasons == 1
    assert len(set(payload["ids"])) == 4
    assert "b-1" not in payload["ids"]
    assert "b-1-abstain-insufficient" not in payload["ids"]
    assert all(cluster["group"] != "g2" for cluster in payload["clusters"])


def test_build_is_deterministic(corpus):
    first = build(
        corpus["canonical"], corpus["heldout"], corpus["tmp"] / "a.json", None, 2, 4, 237
    )
    second = build(
        corpus["canonical"], corpus["heldout"], corpus["tmp"] / "b.json", None, 2, 4, 237
    )
    assert first["ids"] == second["ids"]
    assert first["ids_sha256"] == second["ids_sha256"]


def test_build_writes_paired_rows(corpus):
    out_rows = corpus["tmp"] / "rows.jsonl"
    build(
        corpus["canonical"],
        corpus["heldout"],
        corpus["tmp"] / "selection.json",
        out_rows,
        2,
        4,
        237,
    )
    rows = [json.loads(line) for line in out_rows.read_text().splitlines() if line]
    assert len(rows) == 4
    donors = {row["id"] for row in rows if row.get("answerable") is True}
    variants = {row["id"] for row in rows if row.get("reason")}
    assert donors == {"a-1", "c-1"}
    assert "a-1-abstain-none_of_these" in variants
    assert len(variants) == 2
    assert any(
        record_id.startswith("c-1-unans-") for record_id in variants
    )


def test_rows_floor_fails(corpus):
    with pytest.raises(SystemExit):
        build(corpus["canonical"], corpus["heldout"], corpus["tmp"] / "s.json", None, 1, 3, 237)


def test_shared_donor_is_written_once(tmp_path: Path):
    donor = _noul("s-1", "g1", USER_STATE)
    canonical = _write_jsonl(
        tmp_path / "canonical.jsonl",
        [donor, _variant(donor, "insufficient"), _variant(donor, "out_of_scope")],
    )
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"clusters": [], "ids": [], "ids_sha256": "test"}))
    payload = build(canonical, heldout, tmp_path / "s.json", None, 2, 3, 237)

    assert payload["counts"]["pairs"] == 2
    assert payload["counts"]["rows"] == 3
    assert payload["counts"]["distinct_donors"] == 1
    assert payload["counts"]["by_reason"] == {"insufficient": 1, "out_of_scope": 1}
    assert len(set(payload["ids"])) == 3


def test_heldout_for_another_corpus_fails(corpus):
    heldout = json.loads(corpus["heldout"].read_text())
    heldout["ids"].append("not-in-canonical")
    corpus["heldout"].write_text(json.dumps(heldout))
    with pytest.raises(SystemExit):
        build(corpus["canonical"], corpus["heldout"], corpus["tmp"] / "s.json", None, 1, 2, 237)


def test_heldout_needs_its_fields(tmp_path: Path):
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps({"clusters": []}))
    with pytest.raises(SystemExit):
        load_heldout(path)


def test_existing_pair_needs_an_answerable_donor_and_same_cluster(tmp_path: Path):
    donor = _noul("d-1", "g1", USER_STATE)
    donor["answerable"] = False
    rows = [
        donor,
        _variant(donor, "none_of_these"),
        _noul("e-1", "g9", USER_STATE),
        _variant(_noul("e-1", "g8", USER_STATE), "insufficient"),
    ]
    by_id = {row["id"]: row for row in rows}
    pairs, problems = existing_pairs(by_id)
    assert pairs == []
    assert problems["donor is not answerable"] == 1
    assert problems["donor and variant differ in group"] == 1
