"""Source-carrying plan, namespaced split, and the per-template cap."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from model.scripts.build_judgement_plan import (
    cap_per_template,
    namespace_of,
    synthetic_family,
    synthetic_template,
)
from model.training.group_split import build_group_keys, split_items_by_group

ROOT = Path(__file__).resolve().parents[1]


def _item(item_id: str, state: str, source: str = "s", group: str | None = None):
    return SimpleNamespace(id=item_id, state=state, source=source, group=group)


def _keys(items, **kwargs):
    return build_group_keys(
        items,
        id_of=lambda item: item.id,
        state_of=lambda item: item.state,
        group_of=lambda item: item.group,
        **kwargs,
    )


def test_state_relation_is_namespaced_by_source():
    items = [_item("a", "same state", "one"), _item("b", "same state", "two")]
    keys, groups = _keys(items, namespace_of=lambda item: item.source, use_group=False)
    assert keys[0] != keys[1]
    assert groups == 2

    keys, groups = _keys(items, use_group=False)
    assert keys[0] == keys[1]
    assert groups == 1


def test_cluster_relation_stays_global_across_namespaces():
    items = [_item("a", "state a", "one"), _item("b", "state b", "two")]
    keys, groups = _keys(
        items,
        cluster_of=lambda item: "nd-00000",
        namespace_of=lambda item: item.source,
        use_group=False,
    )
    assert keys[0] == keys[1]
    assert groups == 1


def test_forced_cluster_component_never_reaches_val():
    items = [
        SimpleNamespace(
            id=f"a{index}",
            state=f"state {index}",
            group=None,
            source="apigen" if index < 4 else "other",
        )
        for index in range(8)
    ]
    for seed in range(4):
        train, val, _report = split_items_by_group(
            items,
            0.5,
            seed,
            id_of=lambda item: item.id,
            state_of=lambda item: item.state,
            group_of=lambda item: item.group,
            cluster_of=lambda item: "nd-00000" if item.source == "apigen" else None,
            namespace_of=lambda item: item.source,
            use_group=False,
            force_train_clusters=["nd-00000"],
        )
        assert {item.source for item in train} >= {"apigen"}
        assert not any(item.source == "apigen" for item in val)


def _template_rows():
    item = SimpleNamespace(
        id="synthetic-fam-0001-choice-0",
        source="synthetic",
        group="synthetic-fam-0001",
    )
    records = []
    for q_type, count in (("choice", 4), ("score", 3), ("noul", 3)):
        for index in range(count):
            records.append(
                (
                    f"corpus:synthetic-fam-0001-{q_type}-{index}",
                    item,
                    1,
                    q_type,
                    "train",
                )
            )
    return records


def test_cap_keeps_six_rows_round_robin_over_q_type():
    records = _template_rows()
    kept = cap_per_template(records, 6)
    assert len(kept) == 6
    assert Counter(record[3] for record in kept) == {
        "choice": 2,
        "score": 2,
        "noul": 2,
    }
    assert [record[0] for record in kept] == [
        record[0] for record in records if record[0] in {r[0] for r in kept}
    ]

    reversed_kept = cap_per_template(list(reversed(records)), 6)
    assert sorted(record[0] for record in kept) == sorted(
        record[0] for record in reversed_kept
    )


def test_cap_leaves_non_synthetic_rows_alone():
    item = SimpleNamespace(id="agnews-1", source="agnews", group=None)
    records = [(f"corpus:agnews-{index}", item, 2, "noul", "train") for index in range(3)]
    assert len(cap_per_template(records, 1)) == 3


def test_helpers_read_item_source_and_synthetic_family():
    item = SimpleNamespace(
        id="synthetic-fam-0001-choice-0",
        source="synthetic",
        group="synthetic-fam-0001",
    )
    assert namespace_of(item) == "synthetic/fam"
    assert synthetic_family(item) == "fam"
    assert synthetic_template(item) == ("fam", "0001")

    plain = SimpleNamespace(id="agnews-1", source="agnews", group=None)
    assert namespace_of(plain) == "agnews"
    assert synthetic_family(plain) is None
    assert synthetic_template(plain) is None


def test_builder_writes_the_rows_own_source(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {
            "id": "agnews-1",
            "state": "A team won the match.",
            "question": {
                "type": "noul",
                "instructions": "Is it sports?",
                "criteria": {"true": "Yes", "false": "No"},
            },
            "label": True,
            "source": "agnews",
            "split": "train",
        },
        {
            "id": "synthetic-fam-0001-choice-0",
            "state": "A synthetic situation.",
            "question": {
                "type": "choice",
                "instructions": "Pick one.",
                "criteria": {"a": "First", "b": "Second"},
            },
            "label": "b",
            "source": "synthetic",
            "split": "train",
            "group": "synthetic-fam-0001",
        },
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    out = tmp_path / "plan.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "model/scripts/build_judgement_plan.py"),
            "--corpus",
            str(corpus),
            "--out",
            str(out),
            "--internal-split",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {record["source"] for record in records} == {"agnews", "synthetic"}
    synthetic = [record for record in records if record["source"] == "synthetic"]
    assert {record["family"] for record in synthetic} == {"fam"}
    assert {record["internal_split"] for record in records} <= {"train", "val", "external"}
    assert all("internal_split" in record for record in records)

    # One record per option prompt, not one per row.
    by_row: dict[str, list] = {}
    for record in records:
        by_row.setdefault(record["row_id"], []).append(record)
    for row_id, prompts in by_row.items():
        k = prompts[0]["n_options"]
        assert len(prompts) == k, row_id
        assert sorted(prompt["option_index"] for prompt in prompts) == list(range(k))
