"""The panel-2 trainer's plan reader: options, gold, and the split field."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from model.scripts.train_panel2_lora import (  # noqa: E402
    build_rows,
    parse_prompt_text,
)

PREFIX = "State: A situation.\nQuestion: Is it true?\n"
CHOICE_PREFIX = "State: A situation.\nQuestion: Which one?\n"


def _write(tmp_path, records, name="plan.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _record(row_id, option_index, label, q_type, prefix, option, n_options,
            split="train", field="internal_split", polarity=None):
    payload = {
        "row_id": row_id, "option_index": option_index, "label": label,
        "q_type": q_type, "n_options": n_options, "prefix": prefix,
        "suffix": f"Option: {option}\nDoes this option correctly answer the "
                  "question? Answer:",
        "answerable": True, "source": "corpus", "split": split,
        "state_key": "k", "group": None, "text": prefix,
    }
    if field == "internal_split":
        payload["internal_split"] = split
    if polarity is not None:
        payload["polarity"] = polarity
    return payload


def test_parse_prompt_text_splits_state_and_instructions():
    state, instructions = parse_prompt_text(PREFIX)
    assert state == "A situation."
    assert instructions == "Is it true?"


def test_choice_row_keeps_option_order_and_the_gold_index(tmp_path):
    records = [
        _record("corpus:a", 0, 0, "choice", CHOICE_PREFIX, "First.", 3),
        _record("corpus:a", 1, 1, "choice", CHOICE_PREFIX, "Second.", 3),
        _record("corpus:a", 2, 0, "choice", CHOICE_PREFIX, "Third.", 3),
    ]
    rows, split_field = build_rows(_write(tmp_path, records))
    assert split_field == "internal_split"
    (row,) = rows
    assert row.texts == ["First.", "Second.", "Third."]
    assert row.gold == 1
    assert row.codes == ["A", "B", "C"]
    assert row.split == "train"


def test_noul_gold_reads_the_option_role_not_its_position(tmp_path):
    # The role marker survives a permutation; the plan position does not.
    permuted = [
        _record("corpus:b", 0, 0, "noul", PREFIX, "It does not.", 2,
                polarity="false"),
        _record("corpus:b", 1, 1, "noul", PREFIX, "It holds.", 2,
                polarity="true"),
    ]
    rows, _ = build_rows(_write(tmp_path, permuted))
    (row,) = rows
    assert row.texts == ["no", "yes"]
    assert row.gold == 1, "the gold option is the true wording, so 'yes' is gold"

    negated = [
        _record("corpus:c", 0, 1, "noul", PREFIX, "It does not.", 2,
                polarity="false"),
        _record("corpus:c", 1, 0, "noul", PREFIX, "It holds.", 2,
                polarity="true"),
    ]
    rows, _ = build_rows(_write(tmp_path, negated, "plan2.jsonl"))
    assert rows[0].gold == 0


def test_noul_role_mode_refuses_a_plan_without_the_marker(tmp_path):
    records = [
        _record("corpus:d", 0, 1, "noul", PREFIX, "It holds.", 2),
        _record("corpus:d", 1, 0, "noul", PREFIX, "It does not.", 2),
    ]
    with pytest.raises(ValueError, match="no polarity marker"):
        build_rows(_write(tmp_path, records))


def test_noul_position_mode_reads_the_legacy_unpermuted_order(tmp_path):
    records = [
        _record("corpus:e", 0, 1, "noul", PREFIX, "It holds.", 2),
        _record("corpus:e", 1, 0, "noul", PREFIX, "It does not.", 2),
    ]
    rows, _ = build_rows(_write(tmp_path, records), noul_gold="position")
    assert rows[0].gold == 1


def test_criteria_less_noul_row_reads_its_polarity_from_the_single_prompt(tmp_path):
    text = "State: A situation.\nQuestion: Is it true?\nAnswer:"
    record = _record("corpus:f", 0, 1, "noul", "", "", 1)
    record["prefix"] = ""
    record["suffix"] = text
    rows, _ = build_rows(_write(tmp_path, [record]))
    (row,) = rows
    assert row.state == "A situation."
    assert row.instructions == "Is it true?"
    assert row.texts == ["no", "yes"]
    assert row.gold == 1


def test_a_v2_plan_without_internal_split_falls_back_to_its_split_field(tmp_path):
    record = _record("corpus:g", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2,
                     field="split")
    rows, split_field = build_rows(_write(tmp_path, [record]))
    assert split_field == "split"
    assert rows[0].split == "train"


def test_a_record_without_any_split_field_is_an_error(tmp_path):
    record = _record("corpus:h", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2)
    del record["split"]
    del record["internal_split"]
    with pytest.raises(ValueError, match="neither split field"):
        build_rows(_write(tmp_path, [record]))


def test_a_plan_that_mixes_split_fields_is_an_error(tmp_path):
    first = _record("corpus:i", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2)
    second = _record("corpus:j", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2,
                     field="split")
    with pytest.raises(ValueError, match="mixes"):
        build_rows(_write(tmp_path, [first, second]))
