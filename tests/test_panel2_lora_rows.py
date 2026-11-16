"""The panel-2 trainer's plan reader: options, gold, and the split field."""

from __future__ import annotations

import importlib
import json
import math
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from model.scripts.train_panel2_lora import (  # noqa: E402
    agree_plan,
    batch_indices,
    cap_rows,
    check_shared_non_negative,
    build_rows,
    load_plan_rows,
    parse_prompt_text,
    read_plan,
    reduce_gradients,
    shard_and_steps,
    split_by_token_budget,
    step_plan,
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


def test_every_rank_takes_the_same_number_of_steps_on_an_uneven_plan():
    """12,290 rows over 8 ranks: 1,537 and 1,536 rows must give one step count.

    A rank that runs one step fewer enters the epoch-end statistics
    all-reduce while its peers are still in a gradient all-reduce; the two
    pair, and the run hangs for ten minutes before the watchdog kills it.
    """
    n_rows, world, batch_rows = 12290, 8, 8
    steps, shard_rows = step_plan(n_rows, world, batch_rows)
    assert (steps, shard_rows) == (193, 1537)
    lengths = [len(range(rank, n_rows, world)) for rank in range(world)]
    assert sorted(set(lengths)) == [1536, 1537]
    # The shared step count must cover the longest shard, or the long ranks
    # would still need a step the short ranks never take.
    assert steps == max(math.ceil(length / batch_rows) for length in lengths)


def test_a_negative_cross_entropy_is_refused_with_rank_and_step():
    with pytest.raises(SystemExit, match=r"rank 3 step 7: mean loss is -0.0018"):
        check_shared_non_negative(-0.0018, "mean loss", 3, 1, "cpu", step=7)
    with pytest.raises(SystemExit, match="cannot be a cross-entropy mean"):
        check_shared_non_negative(float("nan"), "frozen sample CE", 0, 1, "cpu")


def test_a_zero_or_positive_cross_entropy_passes():
    assert check_shared_non_negative(0.0, "mean loss", 0, 1, "cpu") == 0.0
    assert check_shared_non_negative(2.5, "mean loss", 0, 1, "cpu") == 2.5


def test_another_ranks_failure_stops_this_rank_too(monkeypatch):
    """The verdict is reduced, so a rank with a healthy value still exits.

    One rank's pairing bug must not leave seven ranks training on into the
    next collective.
    """
    import torch.distributed as dist

    def all_reduce(tensor, op=None):
        tensor.add_(1.0)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    with pytest.raises(SystemExit, match=r"1 rank\(s\) failed"):
        check_shared_non_negative(1.0, "mean loss", 0, 2, "cpu", step=7)


def test_fewer_rows_than_ranks_is_refused_before_the_loop():
    """4 rows over 8 ranks: some shards are empty, so every rank refuses.

    Both inputs are identical on every rank, so this check fails the same way
    everywhere instead of leaving one rank dividing by zero while the rest
    wait in a collective.
    """
    with pytest.raises(SystemExit, match="cannot be spread over 8 ranks"):
        shard_and_steps(list(range(4)), 0, 8, 2)


def test_a_healthy_rank_does_not_name_its_own_value_as_the_offender(monkeypatch):
    """The abort must not claim this rank's value is bad when it is not."""
    import torch.distributed as dist

    def all_reduce(tensor, op=None):
        tensor.add_(1.0)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    with pytest.raises(SystemExit) as raised:
        check_shared_non_negative(1.25, "mean loss", 0, 2, "cpu", step=7)
    message = str(raised.value)
    assert "healthy here (1.2500)" in message
    assert "another rank" in message
    assert "1 rank(s) failed" in message


def _fake_training_modules(module, monkeypatch):
    """Patch every seam main touches so its own loop runs on CPU."""
    import torch

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.forwards = 0

        def forward(self, input_ids=None, **kwargs):
            self.forwards += 1
            # Must depend on a parameter or backward has nothing to do.
            logits = torch.zeros(1, 1, 200) + self.weight * 0.0
            return types.SimpleNamespace(logits=logits)

        def train(self, mode=True):
            return self

        def eval(self):
            return self

    fake = FakeModel()
    monkeypatch.setattr(module, "load_causal_lm",
                        lambda *a, **k: (fake, FakeTokenizer()))
    monkeypatch.setattr(module, "inject_lora", lambda *a, **k: 0)
    monkeypatch.setattr(
        module, "code_token_ids",
        lambda tokenizer, codes, form: ({code: index for index, code in enumerate(codes)}, []))
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: ({}, [], {}))
    monkeypatch.setattr(module, "evaluate_jevbench",
                        lambda *a, **k: {"items": [], "by_tier_natural": {}})
    monkeypatch.setattr(module.torch, "save", lambda *a, **k: None)
    monkeypatch.setattr(module.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(module.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(module.dist, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(module.dist, "destroy_process_group", lambda *a, **k: None)
    monkeypatch.setattr(
        module.dist, "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value] * len(output)))
    return fake


def test_full_ft_main_adapts_nothing_and_still_steps(tmp_path, monkeypatch):
    """--full-ft must skip the adapter and train the model it loaded."""
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:f{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(6) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    fake = _fake_training_modules(module, monkeypatch)
    adapted = []
    monkeypatch.setattr(module, "inject_lora",
                        lambda *a, **k: adapted.append(1) or 0)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2", "--full-ft",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
    ])
    assert module.main() == 0
    assert adapted == [], "full fine-tuning must not inject a LoRA adapter"
    # 2 frozen-sample rows + 3 steps of 2 rows on a 6-row plan.
    assert fake.forwards == 8


def test_main_takes_the_shared_step_count_from_its_own_loop(tmp_path, monkeypatch):
    """Runs main() itself at a rank whose shard is short.

    10 rows over 4 ranks: rank 2 holds 2 rows. The shared count is 2 steps;
    a count derived from ``len(shard)`` would be 1. The fake model counts
    forwards (2 frozen-sample rows + 2 steps x 2 rows), so a reversion changes
    the number this test sees.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:r{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(10) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    fake = _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "4", "--rank", "2",
        "--device", "cpu", "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
    ])
    assert module.main() == 0
    assert fake.forwards == 6, (
        "two frozen-sample rows plus two steps of two rows")


def test_gradients_are_reduced_in_buckets_and_averaged(monkeypatch):
    """Two replicas, two parameters, one bucket: the mean, not the sum."""
    import torch
    import torch.distributed as dist

    calls = []

    def all_reduce(tensor, op=None):
        calls.append(tensor.numel())
        tensor.mul_(2)  # the other replica holds identical gradients

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    first = torch.nn.Parameter(torch.zeros(4))
    first.grad = torch.full((4,), 3.0)
    second = torch.nn.Parameter(torch.zeros(2))
    second.grad = torch.full((2,), -1.5)

    count = reduce_gradients([first, second], 2, bucket_bytes=1 << 20)

    assert count == 1, "both parameters fit one bucket"
    assert calls == [6], "one collective over the flat buffer"
    assert torch.equal(first.grad, torch.full((4,), 3.0))
    assert torch.equal(second.grad, torch.full((2,), -1.5))


def test_a_small_bucket_makes_one_collective_per_group_of_parameters(monkeypatch):
    import torch
    import torch.distributed as dist

    calls = []

    def all_reduce(tensor, op=None):
        calls.append(tensor.numel())

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    params = []
    for _ in range(3):
        param = torch.nn.Parameter(torch.zeros(8))
        param.grad = torch.ones(8)
        params.append(param)

    count = reduce_gradients(params, 2, bucket_bytes=1)
    assert count == len(calls) == 3
    assert calls == [8, 8, 8]


def test_a_rank_with_one_replica_reduces_nothing(monkeypatch):
    import torch
    import torch.distributed as dist

    calls = []
    monkeypatch.setattr(dist, "all_reduce", lambda *a, **k: calls.append(1))
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.ones(2)
    assert reduce_gradients([param], 1) == 0
    assert calls == []


def test_the_token_budget_splits_a_step_without_dropping_rows():
    rows = [object() for _ in range(4)]
    counts = {id(rows[0]): 300, id(rows[1]): 300, id(rows[2]): 100, id(rows[3]): 400}
    chunks = split_by_token_budget(rows, counts, 500)
    assert [len(chunk) for chunk in chunks] == [1, 2, 1]
    assert sum(len(chunk) for chunk in chunks) == 4
    assert split_by_token_budget(rows, counts, 0) == [rows]


def test_the_row_cap_skips_and_counts_the_long_rows():
    rows = [object() for _ in range(4)]
    counts = {id(rows[0]): 100, id(rows[1]): 5000, id(rows[2]): 4096, id(rows[3]): 8000}
    kept, skipped, longest = cap_rows(rows, counts, 4096)
    assert kept == [rows[0], rows[2]], "the cap is inclusive"
    assert skipped == 2
    assert longest == 8000, "the report names the longest row it dropped"


def test_evaluate_reports_calibration_bins_that_add_up(tmp_path):
    """Every scored row lands in exactly one (type, K, decile) bin."""
    import torch

    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:e{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(2) for option in (0, 1)
    ]
    rows, _, _ = build_rows(_bytes(tmp_path, records), plan_unpermuted_noul=True)

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return types.SimpleNamespace(logits=torch.zeros(1, 1, 200))

        def train(self, mode=True):
            return self

        def eval(self):
            return self

    from model.scripts.train_panel2_lora import CODES

    code_ids = {code: index for index, code in enumerate(CODES)}
    counts, per_item, bins = module.evaluate(
        FakeModel(), FakeTokenizer(), rows, code_ids, "cpu")
    total = sum(slot[0] for cell in bins.values() for slot in cell.values())
    correct = sum(slot[1] for cell in bins.values() for slot in cell.values())
    assert total == len(rows) == 2
    assert correct == counts["all"][0]


def test_only_benchmark_rows_enter_the_calibration_bins(tmp_path):
    """Benchmark rows from three families count; synthetic rows do not.

    The LoRA arms' ECE is scored later over the 9,923 benchmark val rows only;
    the eval plan's val split is 31,948 synthetic rows on top of that. A test
    with one benchmark family and one synthetic family would still pass if the
    filter dropped every family but the first, so the plan here carries two
    benchmark families and one synthetic one.
    This is the mutation check for the filter: with it removed the bin total
    becomes 6 and the test goes red.
    """
    import torch

    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = []
    for index in range(2):
        for option in (0, 1):
            records.append(_record(f"corpus:bench{index}", option,
                                   1 if option == 0 else 0, "choice",
                                   CHOICE_PREFIX, f"Option {option}.", 2))
    for index in range(2):
        for option in (0, 1):
            records.append(_record(f"corpus:typed_decisions-t{index}", option,
                                   1 if option == 0 else 0, "choice",
                                   CHOICE_PREFIX, f"Option {option}.", 2))
    for index in range(2):
        for option in (0, 1):
            records.append(_record(f"corpus:synthetic-s{index}", option,
                                   1 if option == 0 else 0, "choice",
                                   CHOICE_PREFIX, f"Option {option}.", 2))
    rows, _, _ = build_rows(_bytes(tmp_path, records), plan_unpermuted_noul=True)
    sources = sorted({row.source for row in rows})
    assert "synthetic" in sources and "typed_decisions" in sources
    assert len(sources) == 4

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return types.SimpleNamespace(logits=torch.zeros(1, 1, 200))

        def train(self, mode=True):
            return self

        def eval(self):
            return self

    from model.scripts.train_panel2_lora import CODES

    code_ids = {code: index for index, code in enumerate(CODES)}
    counts, per_item, bins = module.evaluate(
        FakeModel(), FakeTokenizer(), rows, code_ids, "cpu")
    total = sum(slot[0] for cell in bins.values() for slot in cell.values())
    assert len(rows) == 6
    assert counts["all"][1] == 6, "accuracy still counts every row"
    assert total == 4, "both benchmark families may enter the bins, synthetic may not"


def test_every_rank_gets_the_same_steps_from_its_own_shard():
    """The wiring, not just the helper: the count must not come from the shard.

    This is the mutation check for the hang. If the step count is taken from
    ``len(shard)`` the two long ranks get 193 steps and the six short ones
    192, and the collectives desynchronise.
    """
    rows = list(range(12290))
    counts = {shard_and_steps(rows, rank, 8, 8)[1] for rank in range(8)}
    assert counts == {193}


def test_step_plan_of_a_divisible_plan_matches_the_old_count():
    assert step_plan(19392, 8, 8) == (303, 2424)
    assert step_plan(19392, 1, 8) == (2424, 19392)


def test_a_short_shard_wraps_its_last_batch_instead_of_dropping_it():
    # The 193rd batch starts at position 1,536 in a 1,536-row shard, so it
    # wraps to the first eight rows rather than being empty.
    indices = batch_indices(193, 8, 1536)
    assert indices == list(range(8))
    assert all(index < 1536 for index in indices), "every index must address the shard"
    assert batch_indices(1, 8, 1536) == list(range(8))


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


def test_eval_every_writes_a_checkpoint_before_the_last_step(tmp_path, monkeypatch):
    """A long run must produce a curve, not one number at the end.

    Six rows at two per step is three steps; --eval-every 1 records steps 1
    and 2 and leaves step 3 to the regular final payload, which scores the
    same rows. Removing the call leaves no checkpoint file and this fails.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:c{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(6) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--eval-every", "1",
    ])
    assert module.main() == 0
    out = tmp_path / "out"
    assert (out / "checkpoint_1_shard_0.json").exists()
    assert (out / "checkpoint_2_shard_0.json").exists()
    assert not (out / "checkpoint_3_shard_0.json").exists(), (
        "the last step is scored by the final payload, not a checkpoint")
    record = json.loads((out / "checkpoint_1_shard_0.json").read_text())
    assert record["step"] == 1
    assert record["ece_bins_population"] == "benchmark"
    assert "jevbench" in record


def test_the_one_hot_target_is_the_shipped_cross_entropy():
    import torch
    import torch.nn.functional as F

    from model.scripts.train_panel2_lora import target_loss

    values = torch.tensor([0.0, 2.0, -1.0])
    assert torch.isclose(target_loss(values, 1, "onehot"),
                         F.cross_entropy(values.unsqueeze(0),
                                         torch.tensor([1])))


def test_the_soft_target_charges_less_for_an_adjacent_miss():
    """The soft target must reward a near miss and still prefer the gold.

    With logits [0, 4, 0] and gold 0, one-hot charges the whole margin to the
    wrong level. Soft charges 0.37 of it to that level, so its loss is lower;
    and between two rows, the one whose mass sits on the gold must still be
    the cheaper. If the decay is removed the two losses coincide.
    """
    import torch

    from model.scripts.train_panel2_lora import target_loss

    near_miss = torch.tensor([0.0, 4.0, 0.0])
    on_gold = torch.tensor([4.0, 0.0, 0.0])
    onehot_near = float(target_loss(near_miss, 0, "onehot"))
    soft_near = float(target_loss(near_miss, 0, "soft"))
    assert soft_near < onehot_near, "a near miss must be cheaper under soft"
    assert float(target_loss(on_gold, 0, "soft")) < soft_near, (
        "the gold level must still be cheaper than its neighbour")


def test_the_expected_target_charges_a_far_miss_more_than_a_near_one():
    """Linear in level distance: the property the soft target lacks."""
    import torch

    from model.scripts.train_panel2_lora import target_loss

    # Gold 0 on K=4: mass on level 1 costs ~1, mass on level 3 costs ~3.
    near = torch.tensor([0.0, 6.0, 0.0, 0.0])
    far = torch.tensor([0.0, 0.0, 0.0, 6.0])
    near_loss = float(target_loss(near, 0, "expected"))
    far_loss = float(target_loss(far, 0, "expected"))
    assert 0.9 < near_loss < 1.1
    assert 2.9 < far_loss < 3.1
    assert far_loss > 2 * near_loss


def test_choice_rows_keep_the_one_hot_target_when_a_score_target_is_set(
    tmp_path, monkeypatch
):
    """--score-target must reach score rows and only score rows.

    Two choice rows and two score rows, one step each; the patched target
    function records the mode each row was charged under.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = []
    for index in range(2):
        for option in (0, 1):
            records.append(_record(f"corpus:ch{index}", option,
                                   1 if option == 0 else 0, "choice",
                                   CHOICE_PREFIX, f"Option {option}.", 2))
    for index in range(2):
        for option in range(3):
            records.append(_record(f"corpus:sc{index}", option,
                                   1 if option == 1 else 0, "score",
                                   CHOICE_PREFIX, f"Level {option}.", 3))
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)

    seen = []
    real = module.target_loss

    def recording(values, gold, mode, tau=1.0):
        seen.append(mode)
        return real(values, gold, mode, tau)

    monkeypatch.setattr(module, "target_loss", recording)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--score-target", "soft", "--no-stop-on-loss-rise",
    ])
    assert module.main() == 0
    assert "soft" in seen, "the score rows must take the flagged target"
    assert "onehot" in seen, "the choice rows must keep the shipped target"
    payload = json.loads((tmp_path / "out" / "shard_0.json").read_text())
    assert payload["score_target"] == "soft"
    assert payload["score_target_tau"] == 1.0


def test_a_resume_without_weights_is_refused(tmp_path, monkeypatch):
    """--start-step alone would restart the schedule while claiming step 543."""
    module = importlib.import_module("model.scripts.train_panel2_lora")
    plan = _weight_plan(tmp_path)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--epochs", "1", "--batch-rows", "2", "--plan-unpermuted-noul",
        "--start-step", "3",
    ])
    with pytest.raises(SystemExit, match="needs --load-weights"):
        module.main()


def test_a_resumed_leg_runs_only_the_remaining_steps(tmp_path, monkeypatch):
    """The second leg must continue the schedule, not restart it.

    Six rows at two per step is three steps. A leg that saved at step 1 and
    resumes with --start-step 1 and --max-steps 3 runs steps 2 and 3 only:
    three forwards for the frozen sample head plus two steps of two rows is
    six forwards, and a restart would make it eight.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:w{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(6) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(module.torch, "save", torch.serialization.save)

    def run(argv, out):
        monkeypatch.setattr(sys, "argv", [
            "train_panel2_lora.py", "--plan", str(plan), "--output", str(out),
            "--world", "1", "--device", "cpu", "--lr", "0", "--epochs", "1",
            "--batch-rows", "2", "--plan-unpermuted-noul",
            "--frozen-loss-sample", "2", *argv,
        ])
        return module.main()

    assert run(["--max-steps", "1", "--save-weights"], tmp_path / "leg1") == 0
    weights = tmp_path / "leg1" / "weights.pt"
    assert weights.exists()

    fake = _fake_training_modules(module, monkeypatch)
    assert run(["--max-steps", "3", "--start-step", "1",
                "--load-weights", str(weights)], tmp_path / "leg2") == 0
    assert fake.forwards == 6, "two frozen-sample rows plus steps 2 and 3"
    payload = json.loads((tmp_path / "leg2" / "shard_0.json").read_text())
    assert payload["start_step"] == 1
    assert payload["step_range"] == [2, 3]
    assert payload["weights_loaded_sha256"] == module.file_sha256(weights)


def test_a_resume_past_the_last_step_is_refused(tmp_path, monkeypatch):
    module = importlib.import_module("model.scripts.train_panel2_lora")
    plan = _weight_plan(tmp_path)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(module.torch, "save", torch.serialization.save)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "leg1"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--max-steps", "1", "--save-weights",
    ])
    assert module.main() == 0
    weights = tmp_path / "leg1" / "weights.pt"
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "leg2"), "--world", "1", "--device", "cpu",
        "--epochs", "1", "--batch-rows", "2", "--plan-unpermuted-noul",
        "--max-steps", "3", "--start-step", "3", "--load-weights", str(weights),
    ])
    with pytest.raises(SystemExit, match="already cover this leg"):
        module.main()


def test_a_checkpoint_failure_aborts_every_rank_through_the_collective(monkeypatch):
    """A rank-0 checkpoint failure must not leave the others in an all-reduce.

    The patched reduce records its call and reports a failure from another
    rank; this rank's own flag is healthy and it must still exit, after the
    reduce and not before it.
    """
    import torch
    import torch.distributed as dist

    from model.scripts.train_panel2_lora import abort_if_any_failed

    calls = []

    def all_reduce(tensor, op=None):
        calls.append(float(tensor.item()))
        tensor.add_(1.0)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    failed = torch.tensor([0.0])
    with pytest.raises(SystemExit, match=r"checkpoint step 125: 1 rank"):
        abort_if_any_failed(failed, "checkpoint step 125", 2)
    assert calls == [0.0], "the reduce must run before the raise"


def test_a_rank_zero_checkpoint_exception_raises_instead_of_stalling(
    tmp_path, monkeypatch
):
    """main() must abort cleanly when rank 0's checkpoint work throws."""
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:w{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(6) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(module, "evaluate_jevbench", boom)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--eval-every", "1",
    ])
    with pytest.raises(SystemExit, match=r"checkpoint step 1: 1 rank\(s\) failed"):
        module.main()


def test_ranks_that_agree_on_the_digest_proceed(monkeypatch):
    import torch.distributed as dist

    def agree(output, value):
        output[:] = [value] * len(output)

    monkeypatch.setattr(dist, "all_gather_object", agree)
    data, digest, error = agree_plan(b"x", "c" * 64, None, 8, 3)
    assert (data, digest, error) == (b"x", "c" * 64, None)


def _weight_plan(tmp_path):
    records = [
        _record(f"corpus:w{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(2) for option in (0, 1)
    ]
    return _write(tmp_path, records)


def test_save_weights_records_the_digest_of_the_saved_file(tmp_path, monkeypatch):
    """A full fine-tune's arm is its weights; the payload must name them.

    Without the file the cells cannot be re-scored on a later metric, and
    without the digest the file cannot be tied to the cells it produced.
    """
    import hashlib

    module = importlib.import_module("model.scripts.train_panel2_lora")
    plan = _weight_plan(tmp_path)
    _fake_training_modules(module, monkeypatch)
    # ``module.torch`` is the torch module, whose ``save`` the fake seaming
    # already replaced with a no-op; the real writer is still reachable here.
    monkeypatch.setattr(module.torch, "save", torch.serialization.save)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2", "--full-ft",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2", "--save-weights",
    ])
    assert module.main() == 0
    out = tmp_path / "out"
    payload = json.loads((out / "shard_0.json").read_text())
    saved = out / "weights.pt"
    assert saved.exists()
    assert payload["weights_sha256"] == hashlib.sha256(
        saved.read_bytes()).hexdigest()
    assert payload["weights_bytes"] == saved.stat().st_size


def test_load_weights_puts_the_saved_values_into_the_model(tmp_path, monkeypatch):
    """--load-weights must overwrite the loaded model, strictly.

    The saved tensor is bumped by one before loading; the value the model
    holds at evaluation time is the one this test reads.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    plan = _weight_plan(tmp_path)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(module.torch, "save", torch.serialization.save)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2", "--full-ft",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2", "--save-weights",
    ])
    assert module.main() == 0

    saved = tmp_path / "out" / "weights.pt"
    state = torch.load(saved)
    state["weight"] = state["weight"] + 1.0
    torch.serialization.save(state, saved)

    seen = {}
    monkeypatch.setattr(
        module, "evaluate",
        lambda model, *args, **kwargs: (
            seen.update(weight=float(model.state_dict()["weight"])),
            ({}, [], {}))[1])
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out2"), "--world", "1", "--device", "cpu",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--load-weights", str(saved), "--eval-only",
    ])
    assert module.main() == 0
    assert seen["weight"] == 1.0
    payload = json.loads((tmp_path / "out2" / "shard_0.json").read_text())
    assert payload["weights_loaded_sha256"] == module.file_sha256(saved)
