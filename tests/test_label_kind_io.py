"""A canonical row carrying label_kind must load through the shared loader."""

from __future__ import annotations

import json

from data.pipeline import load_jsonl


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_jsonl_accepts_label_kind_and_round_trips_it(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write(
        path,
        [
            {
                "id": "c-1",
                "state": "A customer asks for a refund.",
                "question": {
                    "type": "choice",
                    "instructions": "What is the intent?",
                    "criteria": {"refund": "Ask for a refund", "other": "Other"},
                },
                "label": "refund",
                "source": "fixture",
                "split": "train",
                "label_kind": "option",
            },
            {
                "id": "s-1",
                "state": "The response mixes two policies.",
                "question": {
                    "type": "score",
                    "instructions": "How severe is the issue?",
                    "criteria": ["None", "Low", "Moderate", "High"],
                },
                "label": 1.5,
                "source": "fixture",
                "split": "train",
                "label_kind": "fractional",
            },
        ],
    )

    items = load_jsonl(path)

    assert [item.label_kind for item in items] == ["option", "fractional"]
    assert items[0].to_dict()["label_kind"] == "option"
    assert items[1].to_dict()["label_kind"] == "fractional"


def test_load_jsonl_leaves_label_kind_none_when_absent(tmp_path):
    path = tmp_path / "old.jsonl"
    _write(
        path,
        [
            {
                "id": "n-1",
                "state": "The sky is blue.",
                "question": {"type": "noul", "instructions": "Is this true?"},
                "label": True,
                "source": "fixture",
                "split": "train",
            }
        ],
    )

    item = load_jsonl(path)[0]

    assert item.label_kind is None
    assert "label_kind" not in item.to_dict()
