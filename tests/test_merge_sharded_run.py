"""The sharded-run merge must refuse anything it cannot vouch for.

A merge that copies benchmark blocks together leaves a result file whose
provenance claims the whole run while its command names one shard's datasets.
These tests pin the checks that stop that, and the rewrite that fixes it.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import merge_sharded_run as merge_mod  # noqa: E402

DATASETS = ["/data/benchmarks/agnews.jsonl", "/data/benchmarks/arc.jsonl"]


def _provenance(datasets: list[str], date: str = "2026-11-08T05:35:54Z") -> dict:
    return {
        "model_id": "org/model",
        "revision": "abc",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "code_commit": "deadbeef",
        "date_utc": date,
        "command": "model/scripts/eval_logit.py --model org/model --dtype bfloat16"
        " --datasets "
        + " ".join(datasets),
    }


def _part(name: str, provenance: dict, correct: int = 3) -> dict:
    return {
        "_provenance": provenance,
        name: {
            "items": 10,
            "choice": {"accuracy": {"n": 10, "accuracy": correct / 10}},
        },
    }


def _shards() -> list[dict]:
    return [
        _part("agnews", _provenance(DATASETS[:1], "2026-11-08T05:36:10Z")),
        _part("arc", _provenance(DATASETS[1:], "2026-11-08T05:35:54Z")),
    ]


def test_the_merge_rebuilds_the_full_command():
    merged = merge_mod.merge(_shards(), DATASETS)
    command = merged["_provenance"]["command"]
    assert "--datasets" in command
    for dataset in DATASETS:
        assert dataset in command
    assert command.split("--datasets", 1)[1].split() == DATASETS


def test_the_merge_records_each_shard_by_name_only():
    merged = merge_mod.merge(_shards(), DATASETS)
    shards = merged["_provenance"]["shards"]
    assert [shard["index"] for shard in shards] == [0, 1]
    assert [shard["benchmarks"] for shard in shards] == [["agnews"], ["arc"]]
    text = str(shards)
    assert "/data/" not in text, "shard records must not carry paths for redaction to miss"


def test_a_different_start_time_is_not_a_different_run():
    """Shards start seconds apart; date_utc must not block the merge."""
    merged = merge_mod.merge(_shards(), DATASETS)
    assert merged["_provenance"]["date_utc"] == "2026-11-08T05:35:54Z"
    assert [shard["started_utc"] for shard in merged["_provenance"]["shards"]] == [
        "2026-11-08T05:36:10Z",
        "2026-11-08T05:35:54Z",
    ]


def test_a_different_setting_is_refused():
    parts = _shards()
    parts[1]["_provenance"]["dtype"] = "float16"
    with pytest.raises(ValueError, match="provenance differs"):
        merge_mod.merge(parts, DATASETS)


def test_a_different_code_commit_is_refused():
    parts = _shards()
    parts[1]["_provenance"]["code_commit"] = "cafebabe"
    with pytest.raises(ValueError, match="code_commit"):
        merge_mod.merge(parts, DATASETS)


def test_a_duplicate_benchmark_is_refused():
    parts = _shards()
    parts[1]["agnews"] = copy.deepcopy(parts[1]["arc"])
    with pytest.raises(ValueError, match="more than one shard"):
        merge_mod.merge(parts, DATASETS)


def test_a_missing_benchmark_is_refused():
    with pytest.raises(ValueError, match="do not match the dataset list"):
        merge_mod.merge(_shards()[:1], DATASETS)


def test_an_items_mismatch_is_refused():
    parts = _shards()
    parts[1]["arc"]["choice"]["accuracy"]["n"] = 9
    with pytest.raises(ValueError, match="items=10"):
        merge_mod.merge(parts, DATASETS)


def test_a_failed_shard_is_refused(tmp_path):
    (tmp_path / "part_0.status").write_text("part 0 exit=0 gpu=0")
    (tmp_path / "part_1.status").write_text("part 1 exit=1 gpu=1")
    with pytest.raises(ValueError, match="not ok"):
        merge_mod.merge(_shards(), DATASETS, tmp_path)


def test_a_missing_status_file_is_refused(tmp_path):
    (tmp_path / "part_0.status").write_text("part 0 exit=0 gpu=0")
    with pytest.raises(ValueError, match="not ok"):
        merge_mod.merge(_shards(), DATASETS, tmp_path)


def test_ok_status_files_are_recorded(tmp_path):
    for index in range(2):
        (tmp_path / f"part_{index}.status").write_text(f"part {index} exit=0 gpu={index}")
    merged = merge_mod.merge(_shards(), DATASETS, tmp_path)
    assert [shard["status"] for shard in merged["_provenance"]["shards"]] == ["ok", "ok"]
