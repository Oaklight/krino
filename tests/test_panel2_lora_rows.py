"""The panel-2 trainer's plan reader: options, gold, and the split field."""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from model.scripts.train_panel2_lora import (  # noqa: E402
    agree_plan,
    build_rows,
    load_plan_rows,
    parse_prompt_text,
    read_plan,
)

PREFIX = "State: A situation.\nQuestion: Is it true?\n"
CHOICE_PREFIX = "State: A situation.\nQuestion: Which one?\n"


def _write(tmp_path, records, name="plan.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def _bytes(tmp_path, records, name="plan.jsonl"):
    """The plan's bytes, which is what the reader takes."""
    return _write(tmp_path, records, name).read_bytes()


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
    rows, split_field, _ = build_rows(_bytes(tmp_path, records),
                                      plan_unpermuted_noul=True)
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
    rows, _, _ = build_rows(_bytes(tmp_path, permuted), noul_gold="role")
    (row,) = rows
    assert row.texts == ["no", "yes"]
    assert row.gold == 1, "the gold option is the true wording, so 'yes' is gold"

    negated = [
        _record("corpus:c", 0, 1, "noul", PREFIX, "It does not.", 2,
                polarity="false"),
        _record("corpus:c", 1, 0, "noul", PREFIX, "It holds.", 2,
                polarity="true"),
    ]
    rows, _, _ = build_rows(_bytes(tmp_path, negated, "plan2.jsonl"),
                            noul_gold="role")
    assert rows[0].gold == 0


def test_the_default_reads_position_and_says_so(tmp_path):
    # The v3 plan keeps criteria-bearing noul rows unpermuted, so position is
    # the default; the payload records which mode produced the gold.
    records = [
        _record("corpus:d", 0, 1, "noul", PREFIX, "It holds.", 2),
        _record("corpus:d", 1, 0, "noul", PREFIX, "It does not.", 2),
    ]
    rows, _, _ = build_rows(_bytes(tmp_path, records),
                            plan_unpermuted_noul=True)
    assert rows[0].gold == 1


def test_noul_role_mode_refuses_a_plan_without_the_marker(tmp_path):
    records = [
        _record("corpus:e", 0, 1, "noul", PREFIX, "It holds.", 2),
        _record("corpus:e", 1, 0, "noul", PREFIX, "It does not.", 2),
    ]
    with pytest.raises(ValueError, match="no polarity marker"):
        build_rows(_bytes(tmp_path, records), noul_gold="role")


def test_position_mode_requires_the_unpermuted_contract(tmp_path):
    # The assumption must be stated on the command line, not inherited from a
    # default: a permuted plan would invert every noul gold silently.
    records = [
        _record("corpus:p", 0, 1, "noul", PREFIX, "It holds.", 2),
        _record("corpus:p", 1, 0, "noul", PREFIX, "It does not.", 2),
    ]
    with pytest.raises(ValueError, match="left criteria-bearing noul rows unpermuted"):
        build_rows(_bytes(tmp_path, records))


def test_rows_the_plan_masks_are_excluded_and_counted(tmp_path):
    masked = _record("corpus:m", 0, 0, "choice", CHOICE_PREFIX, "Only.", 2)
    masked["answerable"] = False
    kept = [
        _record("corpus:k", 0, 1, "choice", CHOICE_PREFIX, "First.", 2),
        _record("corpus:k", 1, 0, "choice", CHOICE_PREFIX, "Second.", 2),
    ]
    rows, _, masked_count = build_rows(_bytes(tmp_path, [masked, *kept]),
                                       plan_unpermuted_noul=True)
    assert masked_count == 1
    assert [row.row_id for row in rows] == ["corpus:k"]


def test_criteria_less_noul_row_reads_its_polarity_from_the_single_prompt(tmp_path):
    text = "State: A situation.\nQuestion: Is it true?\nAnswer:"
    record = _record("corpus:f", 0, 1, "noul", "", "", 1)
    record["prefix"] = ""
    record["suffix"] = text
    rows, _, _ = build_rows(_bytes(tmp_path, [record]), plan_unpermuted_noul=True)
    (row,) = rows
    assert row.state == "A situation."
    assert row.instructions == "Is it true?"
    assert row.texts == ["no", "yes"]
    assert row.gold == 1


def test_a_v2_plan_without_internal_split_falls_back_to_its_split_field(tmp_path):
    record = _record("corpus:g", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2,
                     field="split")
    rows, split_field, _ = build_rows(_bytes(tmp_path, [record]),
                                      plan_unpermuted_noul=True)
    assert split_field == "split"
    assert rows[0].split == "train"


def test_a_record_without_any_split_field_is_an_error(tmp_path):
    record = _record("corpus:h", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2)
    del record["split"]
    del record["internal_split"]
    with pytest.raises(ValueError, match="neither split field"):
        build_rows(_bytes(tmp_path, [record]), plan_unpermuted_noul=True)


def test_a_plan_that_mixes_split_fields_is_an_error(tmp_path):
    first = _record("corpus:i", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2)
    second = _record("corpus:j", 0, 1, "choice", CHOICE_PREFIX, "Only.", 2,
                     field="split")
    with pytest.raises(ValueError, match="mixes"):
        build_rows(_bytes(tmp_path, [first, second]),
                   plan_unpermuted_noul=True)


def _choice_plan(tmp_path, name="plan.jsonl"):
    return _write(tmp_path, [_record("corpus:n", 0, 1, "choice", CHOICE_PREFIX,
                                     "Only.", 2)])


def test_the_plan_reads_to_its_own_bytes_and_digest(tmp_path):
    path = _choice_plan(tmp_path)
    data, digest, error = read_plan(path, None, "plan")
    assert error is None
    assert data == path.read_bytes()
    assert len(digest) == 64


def test_the_pinned_digest_accepts_the_real_file(tmp_path):
    path = _choice_plan(tmp_path)
    digest = read_plan(path, None, "plan")[1]
    # The run logs print the first 16 characters; that form must also match.
    for pinned in (digest[:16], digest):
        rows, split_field, masked, seen = load_plan_rows(
            path, pinned, "plan", plan_unpermuted_noul=True
        )
        assert seen == digest
        assert len(rows) == 1 and split_field == "internal_split" and masked == 0


def test_a_different_digest_stops_before_any_row_is_read(tmp_path):
    path = _choice_plan(tmp_path)
    with pytest.raises(SystemExit, match="does not match the expected"):
        load_plan_rows(path, "0" * 64, "plan", plan_unpermuted_noul=True)


def test_a_fragment_too_short_is_refused_before_the_file_is_read(tmp_path, monkeypatch):
    path = _choice_plan(tmp_path)
    reads = []
    real_read_bytes = pathlib.Path.read_bytes

    def counting_read_bytes(self):
        reads.append(self)
        return real_read_bytes(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", counting_read_bytes)
    with pytest.raises(SystemExit, match="shorter than 16"):
        load_plan_rows(path, "abc", "plan", plan_unpermuted_noul=True)
    assert reads == [], "the fragment must be rejected without opening the file"


def test_the_pin_fails_before_the_bytes_are_parsed(tmp_path):
    """A wrong pin on unparseable bytes must report the pin, not the parse.

    This is the mutation check for the ordering: if the parse is moved above
    the check, the file below raises JSONDecodeError and this test goes red.
    """
    path = tmp_path / "garbage.jsonl"
    path.write_bytes(b"{not json\n")
    with pytest.raises(SystemExit, match="does not match the expected"):
        load_plan_rows(path, "0" * 64, "plan", plan_unpermuted_noul=True)


def test_the_plan_is_read_once_and_parsed_from_the_hashed_bytes(tmp_path, monkeypatch):
    """No second open: the bytes hashed are the bytes parsed.

    Counting reads and opens catches a reader that takes a path and reopens
    it, which would let the two see different files.
    """
    path = _choice_plan(tmp_path)
    digest = read_plan(path, None, "plan")[1]
    reads = []
    opens = []

    real_read_bytes = pathlib.Path.read_bytes
    real_open = pathlib.Path.open

    def counting_read_bytes(self):
        reads.append(self)
        return real_read_bytes(self)

    def counting_open(self, *args, **kwargs):
        opens.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_bytes", counting_read_bytes)
    monkeypatch.setattr(pathlib.Path, "open", counting_open)
    rows, split_field, masked, seen = load_plan_rows(
        path, digest, "plan", plan_unpermuted_noul=True
    )
    assert seen == digest
    assert len(rows) == 1 and split_field == "internal_split" and masked == 0
    assert len(reads) == 1, "the plan must be read exactly once"
    # One open: the one read_bytes performs. A reader that takes a path and
    # reopens it would make this two.
    assert len(opens) == 1, "the plan path must be opened once, by the read"


def test_a_read_failure_reaches_the_collective_instead_of_exiting_alone(
    tmp_path, monkeypatch
):
    """An unreadable file must be carried to the gather, not raised locally.

    A rank that exits before the collective leaves the other ranks blocked in
    it; the failure has to travel through the gather so every rank raises.
    """
    import torch.distributed as dist

    path = _choice_plan(tmp_path)

    def unreadable(self):
        raise OSError("disk gone")

    seen = []

    def gather(output, value):
        seen.append(value)
        output[:] = [value] * len(output)

    monkeypatch.setattr(pathlib.Path, "read_bytes", unreadable)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(SystemExit, match="cannot read the plan"):
        load_plan_rows(path, None, "plan", world=2, rank=1,
                       plan_unpermuted_noul=True)
    assert seen, "the gather must run before the failure is raised"
    digest, error = seen[0]
    assert digest is None and "cannot read the plan" in error


def test_ranks_that_hash_different_bytes_stop_the_run(tmp_path, monkeypatch):
    import torch.distributed as dist

    path = _choice_plan(tmp_path)

    def disagree(output, value):
        output[:] = [("a" * 64, None), ("b" * 64, None)]

    monkeypatch.setattr(dist, "all_gather_object", disagree)
    with pytest.raises(SystemExit, match="ranks disagree on the plan digest"):
        load_plan_rows(path, None, "plan", world=2, rank=0,
                       plan_unpermuted_noul=True)


def test_a_pin_mismatch_is_judged_after_the_gather(tmp_path, monkeypatch):
    """A mismatched pin must not exit one rank before the collective.

    Every rank reaches the gather whatever its pin says; only then does the
    comparison run, on a digest all ranks share. The patched gather records
    its call, so a comparison moved above ``agree_plan`` leaves the record
    empty and this test goes red.
    """
    import torch.distributed as dist

    path = _choice_plan(tmp_path)
    calls = []

    def gather(output, value):
        calls.append(value)
        output[:] = [value] * len(output)

    monkeypatch.setattr(dist, "all_gather_object", gather)
    with pytest.raises(SystemExit, match="does not match the expected"):
        load_plan_rows(path, "0" * 64, "plan", world=2, rank=0,
                       plan_unpermuted_noul=True)
    assert len(calls) == 1, "the gather must run before the pin is judged"


def test_ranks_that_agree_on_the_digest_proceed(monkeypatch):
    import torch.distributed as dist

    def agree(output, value):
        output[:] = [value] * len(output)

    monkeypatch.setattr(dist, "all_gather_object", agree)
    data, digest, error = agree_plan(b"x", "c" * 64, None, 8, 3)
    assert (data, digest, error) == (b"x", "c" * 64, None)
