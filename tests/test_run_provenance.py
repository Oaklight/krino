"""Tests for the provenance block a benchmark run embeds (#277)."""

from __future__ import annotations

import json
from pathlib import Path

from model.evaluation.run_provenance import (
    build_run_provenance,
    resolve_code_commit,
    resolve_snapshot_revision,
)

SNAPSHOT = "/cache/huggingface/hub/models--openbmb--MiniCPM5-2B/snapshots/f97400052a43d642bbc6e9975e2397e3ae6a6b52"


def test_revision_comes_from_the_snapshot_path():
    assert resolve_snapshot_revision(SNAPSHOT) == "f97400052a43d642bbc6e9975e2397e3ae6a6b52"


def test_code_commit_is_unresolved_outside_a_git_tree(tmp_path):
    assert resolve_code_commit(tmp_path) == "unresolved"


def test_explicit_code_commit_wins(tmp_path):
    assert resolve_code_commit(tmp_path, "abc123") == "abc123"


def test_block_records_the_fields_the_issue_requires(tmp_path):
    block = build_run_provenance(
        model_id="openbmb/MiniCPM5-2B",
        snapshot_path=SNAPSHOT,
        dtype="bfloat16",
        device="cuda:0",
        strategy="description",
        norm="mean",
        root=tmp_path,
        variant="instruct",
        code_commit="deadbee",
        command=["python", "eval_logit.py", "--model", "openbmb/MiniCPM5-2B"],
    )
    assert block["model_id"] == "openbmb/MiniCPM5-2B"
    assert block["revision"] == "f97400052a43d642bbc6e9975e2397e3ae6a6b52"
    assert block["snapshot_path"] == SNAPSHOT
    assert block["dtype"] == "bfloat16"
    assert block["device"] == "cuda:0"
    assert block["code_commit"] == "deadbee"
    assert block["strategy"] == "description"
    assert block["norm"] == "mean"
    assert block["variant"] == "instruct"
    assert block["date_utc"].endswith("Z")
    assert block["command"].startswith("python eval_logit.py")


def test_variant_is_absent_when_not_declared(tmp_path):
    block = build_run_provenance(
        model_id="m",
        snapshot_path=SNAPSHOT,
        dtype="bfloat16",
        device="cpu",
        strategy="description",
        norm="mean",
        root=tmp_path,
        code_commit="deadbee",
    )
    assert "variant" not in block


def test_block_is_json_serialisable(tmp_path):
    block = build_run_provenance(
        model_id="m",
        snapshot_path=SNAPSHOT,
        dtype="bfloat16",
        device="cpu",
        strategy="description",
        norm="mean",
        root=tmp_path,
        code_commit="deadbee",
    )
    assert json.loads(json.dumps(block))["model_id"] == "m"
