"""Guard: no committed result file may carry an absolute home path.

The provenance block records where the weights and datasets lived, and the
repository is public. `scripts/redact_provenance.py` replaces every absolute
root with `<host>`; this test makes sure the next run cannot reintroduce one,
or that a file is committed without being redacted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import redact_provenance as redact  # noqa: E402

EXPERIMENTS = ROOT / "model" / "experiments"


def _result_files() -> list[Path]:
    return sorted(EXPERIMENTS.rglob("*.json"))


def test_no_committed_result_file_contains_a_home_path():
    offenders = []
    for path in _result_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "/home/" in text or "/Users/" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], (
        "committed result files carry an absolute home path; run "
        f"scripts/redact_provenance.py on: {offenders}"
    )


def test_the_guard_can_fail(tmp_path):
    """Positive control: a file that does carry a home path must be reported."""
    dirty = tmp_path / "dirty.json"
    dirty.write_text(json.dumps({"_provenance": {"snapshot_path": "/home/u/x"}}))
    text = dirty.read_text()
    assert "/home/" in text
    assert redact.absolute_home_paths(text) == ["/home/u"]
    offenders = [
        str(path) for path in sorted(tmp_path.rglob("*.json"))
        if "/home/" in path.read_text()
    ]
    assert offenders == [str(dirty)]


def test_home_path_regex_finds_the_shapes_we_have_seen():
    for text in (
        "/home/pding/.cache/huggingface/hub",
        "/homes/someone/data",
        "/Users/name/projects",
    ):
        assert redact.absolute_home_paths(text), text


def test_home_path_regex_ignores_repo_relative_text():
    for text in ("data/benchmarks/agnews.jsonl", "models--org--name/snapshots/abc"):
        assert redact.absolute_home_paths(text) == []


def test_snapshot_path_keeps_the_cache_tail():
    path = "/home/user/.cache/huggingface/hub/models--org--name/snapshots/abc123"
    assert redact.redact_text(path) == (
        "<host>/.cache/huggingface/hub/models--org--name/snapshots/abc123"
    )


def test_a_snapshot_path_outside_the_cache_still_loses_its_root():
    cleaned = redact.redact_text("/home/user/weights/local")
    assert cleaned == "<host>/weights/local"
    assert redact.absolute_paths(cleaned) == []


def test_command_has_no_absolute_path_left():
    command = (
        "/home/user/miniforge3/envs/jev/bin/python model/scripts/eval_logit.py "
        "--model org/name --datasets /home/user/projects/krino/data/benchmarks/agnews.jsonl "
        "/home/user/projects/krino/data/benchmarks/openjev/authored144.jsonl"
    )
    cleaned = redact.redact_text(command)
    assert cleaned == (
        "<host>/miniforge3/envs/jev/bin/python model/scripts/eval_logit.py "
        "--model org/name --datasets <host>/projects/krino/data/benchmarks/agnews.jsonl "
        "<host>/projects/krino/data/benchmarks/openjev/authored144.jsonl"
    )
    assert redact.absolute_paths(cleaned) == []


def test_an_absolute_path_outside_the_home_root_is_also_redacted():
    """`/tmp/...` is not a home path, but it is still an absolute path."""
    cleaned = redact.redact_text("--output /tmp/krino-out/x.json")
    assert cleaned == "--output <host>/tmp/krino-out/x.json"
    assert redact.absolute_paths(cleaned) == []


def test_a_named_scratch_root_is_replaced_whole():
    cleaned = redact.redact_text("--datasets /scratch/run42/data/x.jsonl", ["/scratch/run42"])
    assert cleaned == "--datasets <host>/data/x.jsonl"
    assert redact.absolute_paths(cleaned) == []


def test_relative_arguments_are_untouched():
    command = "model/scripts/eval_logit.py --datasets data/benchmarks/agnews.jsonl"
    assert redact.redact_text(command) == command


def test_a_url_is_not_rewritten():
    """``scheme://host/path`` is not a filesystem path and must survive intact."""
    for text in (
        "--endpoint hf://datasets/org/name",
        "--hub https://huggingface.co/org/name",
    ):
        assert redact.redact_text(text) == text
        assert redact.absolute_paths(text) == []


def test_the_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    """The result file is the only copy; a rewrite must not truncate it."""
    path = tmp_path / "result.json"
    path.write_text("old")
    redact.write_atomically(path, "new")
    assert path.read_text() == "new"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["result.json"]


def test_redaction_is_idempotent():
    block = {
        "snapshot_path": "/home/u/.cache/huggingface/hub/models--o--n/snapshots/a",
        "command": "python x --output /tmp/o/a.json --datasets /home/u/d/data/benchmarks/agnews.jsonl",
        "model_id": "o/n",
        "revision": "a",
    }
    once = redact.sanitize_block(block)
    twice = redact.sanitize_block(once)
    assert once == twice
    assert once["model_id"] == "o/n" and once["revision"] == "a"


def test_a_moved_field_outside_the_two_is_an_error():
    """The transform must refuse rather than normalise something unexpected."""
    before = {"model_id": "o/n", "dtype": "bfloat16"}
    after = {"model_id": "o/n", "dtype": "float16"}
    with pytest.raises(AssertionError):
        redact.assert_only_paths_changed(before, after)
    redact.assert_only_paths_changed(
        {"snapshot_path": "/home/u/x", "model_id": "o/n"},
        {"snapshot_path": "<host>/x", "model_id": "o/n"},
    )


def test_a_home_path_in_another_field_is_refused():
    """The rewrite covers two fields; a third field may not smuggle a root."""
    payload = {
        "_provenance": {"snapshot_path": "/home/u/x"},
        "agnews": {"note": "/home/u/private"},
    }
    with pytest.raises(ValueError, match="host roots survived redaction"):
        redact.sanitize_payload(payload)


def test_payload_without_provenance_is_refused():
    with pytest.raises(ValueError):
        redact.sanitize_payload({"agnews": {"items": 1}})


def test_result_files_are_json():
    """A sanitiser that rewrites a file must not leave it unparsable."""
    for path in _result_files():
        json.loads(path.read_text(encoding="utf-8"))
