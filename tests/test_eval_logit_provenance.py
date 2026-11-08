"""The provenance block eval_logit.py embeds (#277).

The Phase A files carry no revision, no code commit and no dataset hashes, which
is why their numbers cannot be audited.  These tests pin the block that fixes it
for new files.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from model.scripts.eval_logit import build_provenance  # noqa: E402

SHA = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def _build(tmp_path, **overrides):
    data = tmp_path / "arc.jsonl"
    data.write_text('{"id": "a"}\n')
    kwargs = dict(
        model_id="Qwen/Qwen2.5-3B",
        revision=SHA,
        dtype="bfloat16",
        device="cuda:0",
        strategy="description",
        norm="mean",
        max_items=200,
        dataset_paths=[str(data)],
        license_name="qwen-research",
    )
    kwargs.update(overrides)
    return build_provenance(**kwargs)


def test_the_block_carries_every_field_277_asks_for(tmp_path):
    block = _build(tmp_path)
    for key in (
        "model_id",
        "revision",
        "dtype",
        "device",
        "code_commit",
        "date_utc",
        "strategy",
        "norm",
        "datasets",
    ):
        assert key in block, key
    assert block["model_id"] == "Qwen/Qwen2.5-3B"
    assert block["revision"] == SHA
    assert block["revision_is_commit"] is True
    assert block["license"] == "qwen-research"
    assert block["strategy"] == "description"
    assert block["norm"] == "mean"


def test_the_snapshot_path_is_cache_relative(tmp_path):
    block = _build(tmp_path)
    assert block["snapshot_path"] == f"models--Qwen--Qwen2.5-3B/snapshots/{SHA}"
    assert "pding" not in json.dumps(block)


def test_a_tag_is_not_recorded_as_a_commit(tmp_path):
    block = _build(tmp_path, revision="main")
    assert block["revision_is_commit"] is False


def test_each_dataset_carries_its_own_hash(tmp_path):
    data = tmp_path / "arc.jsonl"
    data.write_text('{"id": "a"}\n')
    block = _build(tmp_path)
    expected = hashlib.sha256(data.read_bytes()).hexdigest()
    assert block["datasets"]["arc"]["sha256"] == expected
    # A file outside the checkout is recorded by name, never by host path.
    assert block["datasets"]["arc"]["path"] == "arc.jsonl"
