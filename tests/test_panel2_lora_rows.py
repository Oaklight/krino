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
        def __call__(self, text, return_tensors=None, add_special_tokens=False,
                     padding=False):
            texts = [text] if isinstance(text, str) else list(text)
            # Width depends on the text, so a chunk of unequal rows pads; the
            # 8-token floor keeps the spread inside the chunker's 1.25 ratio
            # for rows two apart, which the payload test relies on.
            widths = [8 + len(item) % 3 for item in texts]
            width = max(widths)
            ids = torch.zeros(len(texts), width, dtype=torch.long)
            mask = torch.zeros(len(texts), width, dtype=torch.long)
            for index, item_width in enumerate(widths):
                ids[index, width - item_width:] = 1
                mask[index, width - item_width:] = 1
            if return_tensors is None:
                # The trainer counts a prompt with len(input_ids), so the
                # tensor-free call must return the id list, as a real
                # tokenizer does.
                single = isinstance(text, str)
                return {"input_ids": ids[0].tolist() if single else ids.tolist(),
                        "attention_mask": (mask[0].tolist() if single
                                           else mask.tolist())}
            return {"input_ids": ids, "attention_mask": mask}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.forwards = 0

        def forward(self, input_ids=None, attention_mask=None,
                    position_ids=None, **kwargs):
            self.forwards += 1
            # Must depend on a parameter or backward has nothing to do.
            batch, width = input_ids.shape
            logits = torch.zeros(batch, width, 200) + self.weight * 0.0
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
    def fake_evaluate(model, tokenizer, rows, code_ids, device, limit=None):
        # The same shape as the real evaluate, without a forward: the bins and
        # their composition over the non-synthetic rows, so the payload wiring
        # is exercised. The filter itself is tested directly, on the real
        # evaluate, above.
        bins, bin_sources = {}, {"by_source": {}, "by_type": {}}
        for row in (rows[:limit] if limit else rows):
            if row.source == "synthetic":
                continue
            bins.setdefault(f"{row.q_type}|2", {}).setdefault(
                "5", [0, 0])[0] += 1
            by_source = bin_sources["by_source"]
            by_source[row.source] = by_source.get(row.source, 0) + 1
            by_type = bin_sources["by_type"]
            by_type[row.q_type] = by_type.get(row.q_type, 0) + 1
        return {"all": [0, 0]}, [], bins, bin_sources

    monkeypatch.setattr(module, "evaluate", fake_evaluate)
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
    counts, per_item, bins, bin_sources = module.evaluate(
        FakeModel(), FakeTokenizer(), rows, code_ids, "cpu")
    total = sum(slot[0] for cell in bins.values() for slot in cell.values())
    correct = sum(slot[1] for cell in bins.values() for slot in cell.values())
    assert total == len(rows) == 2
    assert correct == counts["all"][0]
    assert sum(bin_sources["by_source"].values()) == total


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
    counts, per_item, bins, bin_sources = module.evaluate(
        FakeModel(), FakeTokenizer(), rows, code_ids, "cpu")
    total = sum(slot[0] for cell in bins.values() for slot in cell.values())
    assert len(rows) == 6
    assert counts["all"][1] == 6, "accuracy still counts every row"
    assert total == 4, "both benchmark families may enter the bins, synthetic may not"
    # The composition is the evidence for that total: it names the families
    # that entered. bench0 and bench1 are two families under the row-id rule,
    # so each contributes its one row.
    assert bin_sources["by_source"] == {"bench0": 1, "bench1": 1,
                                        "typed_decisions": 2}
    assert bin_sources["by_type"] == {"choice": 4}
    assert "synthetic" not in bin_sources["by_source"]


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
    assert record["ece_bins_population"] == "non_synthetic"
    assert "jevbench" in record
    # The final payload carries two counts that can disagree: the rows binned
    # and the non-synthetic rows handed to evaluate. Six rows, none synthetic,
    # so a filter that admitted synthetic rows or dropped a family separates
    # them.
    final = json.loads((out / "shard_0.json").read_text())
    assert final["ece_bins_population_rule"] == 'source != "synthetic"'
    assert final["ece_bins_rows_this_rank"] == \
        final["val_rows_non_synthetic_this_rank"] == 6
    assert sum(final["ece_bins_sources"]["by_source"].values()) == 6
    assert final["ece_bins_sources"]["by_type"] == {"choice": 6}


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


def test_score_loss_ce_is_the_shipped_target():
    import torch

    from model.scripts.train_panel2_lora import score_loss, target_loss

    values = torch.tensor([0.5, 2.0, -1.0])
    assert torch.isclose(score_loss(values, 1, "ce", None),
                         target_loss(values, 1, "onehot"))


def test_the_ranked_score_orders_its_mistakes_by_distance():
    """RPS must charge a near miss less than a wrong-side miss.

    With the gold on level 0, mass on level 1 is one level away and mass on
    level 2 is two; the ordered score must price them in that order, which
    cross-entropy cannot do. If the CDF sum is replaced by a plain squared
    error over levels the second inequality still holds, but the first would
    not: the test pins the ordering, not only the total.
    """
    import torch

    from model.scripts.train_panel2_lora import score_loss

    on_gold = float(score_loss(torch.tensor([4.0, 0.0, 0.0]), 0, "rps", None))
    adjacent = float(score_loss(torch.tensor([0.0, 4.0, 0.0]), 0, "rps", None))
    far = float(score_loss(torch.tensor([0.0, 0.0, 4.0]), 0, "rps", None))
    assert on_gold < adjacent < far
    assert far <= 1.0, "RPS is normalised to at most 1"


def test_mix_is_the_cross_entropy_plus_the_weighted_ranked_score():
    import pytest
    import torch

    from model.scripts.train_panel2_lora import score_loss

    values = torch.tensor([0.3, 1.2, -0.4])
    ce = float(score_loss(values, 2, "ce", None))
    rps = float(score_loss(values, 2, "rps", None))
    assert float(score_loss(values, 2, "mix", 0.7)) == pytest.approx(
        ce + 0.7 * rps, rel=1e-6)


def test_the_score_loss_flags_refuse_unmeasured_combinations(tmp_path, monkeypatch):
    """A mix without a weight, or a weight nothing uses, must not launch.

    CE is unbounded and RPS is at most 1, so an unmeasured lambda would
    rescale score supervision; and rps with a soft target is a third axis
    nobody pre-registered. Each case is refused at argument time, before the
    plan is read.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"factor": 0.25, "overall": {
        "n": 1111, "mean_ce": 2.0, "mean_rps": 0.4, "ce_over_rps": 5.0,
        "lambda": 1.25}}))
    bad_ratio = tmp_path / "bad_ratio.json"
    bad_ratio.write_text(json.dumps({"factor": 0.25, "overall": {
        "n": 1111, "mean_ce": 2.0, "mean_rps": 0.4, "ce_over_rps": 6.0,
        "lambda": 1.5}}))
    bad_lambda = tmp_path / "bad_lambda.json"
    bad_lambda.write_text(json.dumps({"factor": 0.25, "overall": {
        "n": 1111, "mean_ce": 2.0, "mean_rps": 0.4, "ce_over_rps": 5.0,
        "lambda": 1.0}}))
    no_factor = tmp_path / "no_factor.json"
    no_factor.write_text(json.dumps({"overall": {
        "n": 1111, "mean_ce": 2.0, "mean_rps": 0.4, "ce_over_rps": 5.0,
        "lambda": 1.25}}))
    zero_rps = tmp_path / "zero_rps.json"
    zero_rps.write_text(json.dumps({"factor": 0.25, "overall": {
        "n": 1111, "mean_ce": 2.0, "mean_rps": 0.0, "ce_over_rps": 0.0,
        "lambda": 1.25}}))
    base = [
        "train_panel2_lora.py", "--plan", str(tmp_path / "plan.jsonl"),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
    ]
    cases = [
        (["--score-loss", "mix"], "needs --score-cramer-weight"),
        (["--score-loss", "rps", "--score-cramer-weight", "1.0"],
         "only with mix"),
        (["--score-loss", "rps", "--score-target", "soft"],
         "not pre-registered"),
        (["--score-loss", "rps", "--score-cramer-weight-source", str(probe)],
         "no effect under"),
        (["--score-loss", "mix", "--score-cramer-weight", "1.5",
          "--score-cramer-weight-source", str(probe)],
         "is not the measured lambda"),
        (["--score-loss", "mix", "--score-cramer-weight", "0"],
         "greater than zero"),
        (["--score-loss", "mix", "--score-cramer-weight", "-1"],
         "greater than zero"),
        (["--score-loss", "mix", "--score-cramer-weight", "inf"],
         "greater than zero"),
        (["--score-loss", "mix", "--score-cramer-weight", "nan"],
         "greater than zero"),
        (["--score-loss", "mix", "--score-cramer-weight", "1.5",
          "--score-cramer-weight-source", str(bad_ratio)],
         "internally inconsistent"),
        (["--score-loss", "mix", "--score-cramer-weight", "1.0",
          "--score-cramer-weight-source", str(bad_lambda)],
         "internally inconsistent"),
        (["--score-loss", "mix", "--score-cramer-weight", "1.25",
          "--score-cramer-weight-source", str(no_factor)],
         "records no factor"),
        (["--score-loss", "mix", "--score-cramer-weight", "1.25",
          "--score-cramer-weight-source", str(zero_rps)],
         "mean RPS is"),
    ]
    for extra, match in cases:
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(SystemExit, match=match):
            module.main()


def test_the_lambda_provenance_says_whether_it_was_measured(tmp_path, monkeypatch):
    """A weight passed by hand must not read as a measured one.

    With a probe file the payload copies the measured means, the row count and
    the file's digest, so the lambda claim is auditable; without one it says
    the weight was passed by hand. Both runs pin the same number, so the only
    difference is what the artifact can support.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:m{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(2) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"plan": str(plan), "weights": "-",
                                 "factor": 0.25,
                                 "by_gold": {"0": {"lambda": 0.5}},
                                 "overall": {
        "n": 1111, "mean_ce": 2.0, "mean_rps": 0.4, "ce_over_rps": 5.0,
        "lambda": 1.25}}))

    def run(out, extra):
        _fake_training_modules(module, monkeypatch)
        monkeypatch.setattr(sys, "argv", [
            "train_panel2_lora.py", "--plan", str(plan),
            "--output", str(out), "--world", "1", "--device", "cpu",
            "--lr", "0", "--epochs", "1", "--batch-rows", "2",
            "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
            "--no-stop-on-loss-rise", "--score-loss", "mix",
            "--score-cramer-weight", "1.25", *extra,
        ])
        assert module.main() == 0
        return json.loads((out / "shard_0.json").read_text())

    measured = run(tmp_path / "measured", [
        "--score-cramer-weight-source", str(probe)])[
            "score_cramer_weight_provenance"]
    assert measured["measured"] is True
    assert measured["rows"] == 1111
    assert measured["factor"] == 0.25
    assert measured["by_gold"] == {"0": {"lambda": 0.5}}
    assert measured["mean_ce"] == 2.0 and measured["mean_rps"] == 0.4
    assert measured["lambda"] == 1.25
    assert measured["source_sha256"] == module.file_sha256(probe)

    hand = run(tmp_path / "hand", [])["score_cramer_weight_provenance"]
    assert hand["measured"] is False
    assert "passed by hand" in hand["reason"]


def test_score_rows_take_the_flagged_loss_and_choice_rows_keep_one_hot(
    tmp_path, monkeypatch
):
    """--score-loss must reach score rows and only score rows.

    Two choice rows and two score rows, one step each, with both loss
    functions recorded as they are called. Under rps every score-row charge
    must be rps and every other row the shipped one-hot cross-entropy; if the
    flag leaks to choice or noul rows, the arm moves three heads at once and
    the paired endpoint stops meaning what it says.
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
    real_score = module.score_loss

    def recording(values, gold, loss, weight, target_mode="onehot", tau=1.0):
        seen.append(("score", loss))
        return real_score(values, gold, loss, weight, target_mode, tau)

    real_target = module.target_loss

    def recording_target(values, gold, mode, tau=1.0):
        seen.append(("target", mode))
        return real_target(values, gold, mode, tau)

    monkeypatch.setattr(module, "score_loss", recording)
    monkeypatch.setattr(module, "target_loss", recording_target)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--score-loss", "rps", "--no-stop-on-loss-rise",
    ])
    assert module.main() == 0
    assert ("score", "rps") in seen, "the score rows must take the flagged loss"
    assert ("target", "onehot") in seen, "other rows keep the shipped one-hot"
    assert all(loss == "rps" for kind, loss in seen if kind == "score")
    payload = json.loads((tmp_path / "out" / "shard_0.json").read_text())
    assert payload["score_loss"] == "rps"
    assert payload["score_cramer_weight"] is None


def test_the_stop_rule_compares_reweighted_references(tmp_path, monkeypatch):
    """Both sides are reweighted to the plan's composition, not mixed.

    Two choice items and four score items, interleaved so the two-row frozen
    sample holds one of each while the plan is 1/3 and 2/3. Six rows at two
    per step is three full batches, so no row wraps and the epoch block's own
    mix is the plan's; the frozen block's mix is not. With zero logits a choice
    row costs ln 2 and a score row ln 3, so the two differences separate.
    """
    import pytest

    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = []
    order = [("c0", "choice", 2), ("s0", "score", 3), ("c1", "choice", 2),
             ("s1", "score", 3), ("s2", "score", 3), ("s3", "score", 3)]
    for row_id, q_type, n_options in order:
        for option in range(n_options):
            gold = 1 if q_type == "score" else 0
            records.append(_record(f"corpus:{row_id}", option,
                                   1 if option == gold else 0, q_type,
                                   CHOICE_PREFIX, f"Option {option}.",
                                   n_options))
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
    ])
    assert module.main() == 0
    payload = json.loads((tmp_path / "out" / "shard_0.json").read_text())
    composition = payload["stop_composition"]
    assert composition["counts"] == {"choice": 2, "score": 4}
    assert composition["fractions"] == pytest.approx(
        {"choice": 1 / 3, "score": 2 / 3})
    reference = payload["stop_reference"]
    ln2, ln3 = 0.693147, 1.098612
    pinned = ln2 / 3 + 2 * ln3 / 3
    assert reference["frozen"]["per_type"]["score"]["rows"] == 1
    assert reference["frozen"]["mixed"] == pytest.approx((ln2 + ln3) / 2,
                                                         abs=1e-5)
    assert reference["frozen"]["reweighted"] == pytest.approx(pinned, abs=1e-5)
    assert reference["epoch"]["per_type"]["score"]["rows"] == 4
    assert reference["epoch"]["reweighted"] == pytest.approx(pinned, abs=1e-5)
    assert reference["reweighted_difference"] == pytest.approx(0.0, abs=1e-9)
    assert reference["mixed_difference"] == pytest.approx(
        pinned - (ln2 + ln3) / 2, abs=1e-5)
    assert payload["stop_rule"] == "armed_not_fired"
    assert payload["stopped"] is False


def test_a_frozen_sample_missing_a_weighted_type_does_not_default_to_no_stop(
    tmp_path, monkeypatch
):
    """An unevaluable rule must say so, not read as "no stop".

    The plan's first two rows are choice, so a two-row frozen sample has no
    score rows while the pinned composition weights them. The payload must
    record insufficient_sample_composition and name score; the run must not
    stop and must not raise.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = []
    order = [("c0", "choice", 2), ("c1", "choice", 2), ("s0", "score", 3),
             ("s1", "score", 3)]
    for row_id, q_type, n_options in order:
        for option in range(n_options):
            gold = 1 if q_type == "score" else 0
            records.append(_record(f"corpus:{row_id}", option,
                                   1 if option == gold else 0, q_type,
                                   CHOICE_PREFIX, f"Option {option}.",
                                   n_options))
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
    ])
    assert module.main() == 0
    payload = json.loads((tmp_path / "out" / "shard_0.json").read_text())
    assert payload["stop_rule"] == "insufficient_sample_composition"
    assert payload["stop_rule_missing_types"] == ["score"]
    assert payload["stop_reference"]["frozen"]["reweighted"] is None
    assert payload["stopped"] is False


def test_the_probe_summarise_reports_the_ratio_and_the_term_share():
    """lambda = 0.25 * CE/RPS makes the rps term 20% of the score-row loss."""
    import pytest

    from model.scripts.measure_score_cramer import summarise

    rows = [{"k": 5, "gold": 0, "ce": 2.0, "rps": 0.5},
            {"k": 5, "gold": 1, "ce": 4.0, "rps": 0.5},
            {"k": 4, "gold": 1, "ce": 3.0, "rps": 1.0}]
    summary = summarise(rows)
    overall = summary["overall"]
    assert overall["n"] == 3
    assert overall["mean_ce"] == pytest.approx(3.0)
    assert overall["mean_rps"] == pytest.approx(2.0 / 3.0)
    assert overall["ce_over_rps"] == pytest.approx(4.5)
    assert overall["lambda"] == pytest.approx(1.125)
    assert overall["rps_term_fraction_at_block_lambda"] == pytest.approx(0.2)
    assert overall["rps_term_fraction_at_overall_lambda"] == pytest.approx(0.2)
    assert summary["by_k"]["5"]["n"] == 2
    assert summary["by_k"]["4"]["lambda"] == pytest.approx(0.75)
    # The share is heterogeneous by gold position, so the probe reports it
    # there too, computed at the overall lambda: one global lambda cannot make
    # it uniform. The block-local value is 0.2 in every block by construction.
    assert summary["by_gold"]["0"]["n"] == 1
    assert summary["by_gold"]["1"]["n"] == 2
    assert summary["by_gold"]["0"]["ce_over_rps"] == pytest.approx(4.0)
    assert summary["by_gold"]["1"]["ce_over_rps"] == pytest.approx(14.0 / 3.0)
    assert summary["by_gold"]["0"][
        "rps_term_fraction_at_block_lambda"] == pytest.approx(0.2)
    assert summary["by_gold"]["0"][
        "rps_term_fraction_at_overall_lambda"] == pytest.approx(0.219512, abs=1e-5)
    assert summary["by_gold"]["1"][
        "rps_term_fraction_at_overall_lambda"] == pytest.approx(0.194245, abs=1e-5)


def test_the_probe_scores_only_the_rows_own_codes(tmp_path):
    """A row's logits must span its own K codes, not the global code list.

    The training loss sees K levels; scoring all codes would normalise CE and
    RPS over a distribution the model is never trained on and inflate the
    ratio. The plan here has a 2-option item while the code map holds every
    code, so passing the global map returns far more than 2 values.
    """
    import torch

    from model.scripts.measure_score_cramer import row_logits

    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [_record("corpus:k2", option, 1 if option == 0 else 0, "choice",
                       CHOICE_PREFIX, f"Option {option}.", 2)
               for option in (0, 1)]
    rows, _, _ = build_rows(_bytes(tmp_path, records),
                            plan_unpermuted_noul=True)

    class FakeTokenizer:
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, **kwargs):
            return types.SimpleNamespace(logits=torch.zeros(1, 1, 200))

    code_ids = {code: index for index, code in enumerate(module.CODES)}
    values = row_logits(FakeModel(), FakeTokenizer(), rows[0], code_ids, "cpu")
    assert len(rows[0].codes) == 2
    assert len(values) == 2, "the probe must score the row's own K codes"


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


def test_a_resume_with_more_than_one_epoch_is_refused(tmp_path, monkeypatch):
    """--start-step names steps of one pass; two passes have no reading.

    The check runs at argument time, before any weights or plan are read, so
    the path need not exist. Without the refusal the run would either skip
    only epoch 1's steps or repeat the skip, and the payload could not say
    which.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(tmp_path / "plan.jsonl"),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--epochs", "2", "--start-step", "1",
        "--load-weights", str(tmp_path / "weights.pt"),
    ])
    with pytest.raises(SystemExit, match="ambiguous"):
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
    assert payload["planned_step_range"] == [2, 3]
    assert payload["steps_run"] == 2, "two steps executed, not the planned three"
    assert payload["weights_loaded_sha256"] == module.file_sha256(weights)
    # The two legs' frozen samples must be the same rows in the same order, or
    # the stop rule's references are not comparable across arms.
    leg1 = json.loads((tmp_path / "leg1" / "shard_0.json").read_text())
    assert leg1["frozen_sample_rows"] == payload["frozen_sample_rows"] == 2
    assert leg1["frozen_sample_row_ids_sha256"] == \
        payload["frozen_sample_row_ids_sha256"]
    assert "frozen_sample_ce" not in payload
    assert "frozen_sample_loss" in payload


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


def test_a_rank_zero_final_exception_aborts_through_the_collective(
    tmp_path, monkeypatch
):
    """The final JevBench read and weight write sit inside the same bracket.

    With no --eval-every, the only JevBench call is the final one, which is
    exactly the work that used to run before the barrier with nothing around
    it. The failure must come out as a group abort, so a peer never waits.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:w{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(6) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)

    def boom(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(module, "evaluate_jevbench", boom)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
    ])
    with pytest.raises(SystemExit,
                       match=r"final evaluation or weight write: 1 rank\(s\) failed"):
        module.main()
    assert not (tmp_path / "out" / "shard_0.json").exists(), (
        "a failed final write must not be recorded as a completed payload")


def test_phase_profile_times_every_phase_of_the_first_steps(tmp_path, monkeypatch):
    """--phase-profile N writes one record per timed step, with all six phases."""
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:w{index}", option, 1 if option == 0 else 0, "choice",
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
        "--phase-profile", "1",
    ])
    assert module.main() == 0
    profile = json.loads(
        (tmp_path / "out" / "phase_profile_rank0.json").read_text())
    assert profile["path_this_rank"] == "phase_profile_rank0.json"
    assert not (tmp_path / "out" / "phase_profile.json").exists(), (
        "a shared filename is what let eight ranks tear the file")
    assert profile["steps_this_rank"] == 1
    assert profile["first_step"] == 1
    assert profile["includes_startup"] is True
    assert set(profile["mean_seconds_this_rank"]) == {
        "tokenise", "forward", "backward", "allreduce", "clip", "optimizer"}
    assert profile["mean_profiled_seconds"] > 0
    assert profile["mean_profiled_seconds"] >= profile["mean_measured_phases_seconds"]
    assert all(value >= 0 for value in profile["mean_seconds_this_rank"].values())


def test_a_second_rank_writes_its_own_file_and_leaves_rank0_alone(tmp_path,
                                                                  monkeypatch):
    """The filename must carry the writing rank, not a constant.

    One rank proves nothing: with a constant rank in the name, rank 1 would
    overwrite rank 0's file and a rank-0-only test would still pass.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:p{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(8) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    out = tmp_path / "out"

    def run(rank):
        monkeypatch.setattr(sys, "argv", [
            "train_panel2_lora.py", "--plan", str(plan), "--output", str(out),
            "--world", "2", "--rank", str(rank), "--device", "cpu", "--lr", "0",
            "--epochs", "1", "--batch-rows", "2", "--plan-unpermuted-noul",
            "--frozen-loss-sample", "2", "--phase-profile", "1",
        ])
        assert module.main() == 0

    run(0)
    zero_path = out / "phase_profile_rank0.json"
    zero_before = zero_path.read_bytes()
    run(1)
    one_path = out / "phase_profile_rank1.json"
    assert one_path.exists()
    body = json.loads(one_path.read_text())
    assert body["rank_this_process"] == 1
    assert body["path_this_rank"] == one_path.name
    assert zero_path.read_bytes() == zero_before, (
        "a rank-1 run must not touch rank 0's file")


def test_a_failed_phase_profile_write_aborts_the_group(tmp_path, monkeypatch):
    """A local write failure must reach the group, not vanish in a traceback.

    Rank 0's write is local, so an exception there would leave the other ranks
    at the next collective with nothing to see.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:f{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(6) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    real_write = pathlib.Path.write_text

    def refuse(self, *args, **kwargs):
        if self.name.startswith("phase_profile_rank"):
            raise OSError("no space left on device")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", refuse)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--phase-profile", "1",
    ])
    with pytest.raises(SystemExit,
                       match=r"phase profile write: 1 rank\(s\) failed"):
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
            ({}, [], {}, {}))[1])
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


def _tiny_lm(vocab=64, hidden=16, intermediate=32, layers=1, seed=0):
    """A real attention stack, small enough to run on CPU in a test."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128)
    return Qwen2ForCausalLM(config)


class _CharTokenizer:
    """One token per character, deterministic, with the padding side selectable."""

    def __init__(self, pad_side="right"):
        self.pad_side = pad_side

    def __call__(self, text, return_tensors=None, padding=False,
                 add_special_tokens=False):
        import torch

        texts = [text] if isinstance(text, str) else list(text)
        sequences = [[1 + (ord(char) % 60) for char in item] for item in texts]
        width = max(len(sequence) for sequence in sequences)
        ids, mask = [], []
        for sequence in sequences:
            pad = [0] * (width - len(sequence))
            ones = [1] * len(sequence)
            if self.pad_side == "left":
                ids.append(pad + sequence)
                mask.append([0] * len(pad) + ones)
            else:
                ids.append(sequence + pad)
                mask.append(ones + [0] * len(pad))
        return {"input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(mask)}


def _batch_rows(count=6, growth=3):
    from model.scripts.train_panel2_lora import Row

    rows = []
    for index in range(count):
        texts = [f"Option {option} " + "x" * (index * growth + option + 1)
                 for option in range(2)]
        rows.append(Row(f"corpus:b{index}", "train", "choice", "State: s",
                        "Question: q", texts, 0, ["+", "-"], "corpus"))
    return rows


def test_deal_rows_partitions_the_step_and_balances_its_token_sums():
    """Every row goes to exactly one rank, and no rank trails by a whole row."""
    from model.scripts.train_panel2_lora import deal_rows

    rows = _batch_rows(8, growth=7)
    counts = {id(row): len(row.texts[0]) + len(row.texts[1]) for row in rows}
    loads = deal_rows(rows, counts, 4)
    assert sorted(row.row_id for load in loads for row in load) == sorted(
        row.row_id for row in rows)
    totals = [sum(counts[id(row)] for row in load) for load in loads]
    assert max(totals) - min(totals) <= max(counts.values())


def test_the_step_gradient_does_not_depend_on_chunking():
    """Same six rows, one per forward versus three chunks: the same gradient.

    The loop divides each chunk's summed row losses by the step's nominal rows
    per rank. A mean over the chunk instead would give a one-row chunk the
    same weight as a four-row chunk and change the gradient, so this is the
    precondition for any chunked arm.
    """
    import torch
    from model.scripts.train_panel2_lora import (
        chunk_code_logits, chunk_loss, target_loss,
    )

    model = _tiny_lm()
    tokenizer = _CharTokenizer()
    rows = _batch_rows(6)
    code_ids = {"+": 0, "-": 1}
    nominal = 6

    def gradients(chunks):
        model.zero_grad(set_to_none=True)
        for chunk in chunks:
            values = chunk_code_logits(model, tokenizer, chunk, code_ids, "cpu")
            loss = chunk_loss(
                chunk, values,
                lambda row, value: target_loss(value, row.gold, "onehot"),
                nominal)
            loss.backward()
        return {name: parameter.grad.clone()
                for name, parameter in model.named_parameters()}

    singles = gradients([[row] for row in rows])
    grouped = gradients([rows[:1], rows[1:4], rows[4:]])
    for name in singles:
        assert torch.allclose(singles[name], grouped[name], atol=1e-7,
                              rtol=1e-5), name


def test_a_later_row_cannot_change_an_earlier_rows_logits():
    """Perturb the chunk's last row; the earlier rows must not move at all.

    A mask-shaped check can pass while a concatenated sequence still leaks, so
    this asserts equality of the logits themselves.
    """
    import torch
    from model.scripts.train_panel2_lora import chunk_code_logits

    model = _tiny_lm().eval()
    tokenizer = _CharTokenizer()
    rows = _batch_rows(3)
    code_ids = {"+": 0, "-": 1}
    with torch.no_grad():
        before = chunk_code_logits(model, tokenizer, rows, code_ids, "cpu")
        # Same length, different tokens: a length change would alter the
        # batch's padding width and with it the kernel tiling, which moves
        # other rows' logits in the last bits without any attention leak.
        rows[-1].texts = [text.replace("x", "y") for text in rows[-1].texts]
        after = chunk_code_logits(model, tokenizer, rows, code_ids, "cpu")
    for index in (0, 1):
        assert torch.equal(before[index], after[index]), (
            f"row {index}'s logits moved when the last row's tokens changed")
    assert not torch.equal(before[2], after[2]), (
        "the perturbation must reach its own row, or the test proves nothing")


def test_position_ids_restart_at_zero_for_every_row():
    """A chunked forward must not inherit the previous row's positions."""
    import torch
    from model.scripts.train_panel2_lora import chunk_inputs

    rows = _batch_rows(3, growth=5)
    _, attention, positions = chunk_inputs(_CharTokenizer("left"), rows, "cpu")
    for index in range(len(rows)):
        real = attention[index].bool()
        count = int(real.sum())
        first = int(torch.argmax(real.long()))
        assert positions[index][first].item() == 0
        assert positions[index][first:first + count].tolist() == list(range(count))


def test_the_payload_records_the_tokens_the_chunks_carried(tmp_path,
                                                           monkeypatch):
    """Real and padded tokens per step, and the assignment rule, are recorded."""
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:c{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX,
                f"Option {option}.{'z' * (2 * index + 2 * option)}", 2)
        for index in range(8) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "2",
        "--chunk-tokens", "4096", "--balance-rows", "--phase-profile", "4",
    ])
    assert module.main() == 0
    profile = json.loads(
        (tmp_path / "out" / "phase_profile_rank0.json").read_text())
    chunking = profile["chunking_this_rank"]
    payload = json.loads((tmp_path / "out" / "shard_0.json").read_text())
    steps = len(profile["steps"])
    assert chunking["tokens"] == 4096
    assert len(chunking["real_tokens_per_step"]) == steps
    assert len(profile["padded_cost_this_rank"]) == steps
    assert "padded_cost_this_rank" not in chunking, (
        "the padded series is emitted once, at the profile payload's top level")
    pairs = list(zip(chunking["real_tokens_per_step"],
                     profile["padded_cost_this_rank"]))
    assert all(padded >= real for real, padded in pairs)
    assert any(padded > real for real, padded in pairs), (
        "the fake tokenizer's uneven lengths must produce some padding")
    assert len(chunking["forwards_per_step"]) == steps
    assert all(count >= 1 for count in chunking["forwards_per_step"])
    assert all(sum(sizes) == 2 for sizes in chunking["chunk_sizes"]), (
        "each step holds two rows, so its chunks must account for two")
    assert all(rows == 2 for rows in chunking["rows_per_step"])
    assert sum(chunking["chunk_decision_counts"].values()) == steps
    assert set(chunking["chunk_decision_counts"]) <= {
        "chunk_tokens", "ratio", "chunk_rows", "budget_non_binding"}
    assert len(chunking["chunk_decision_per_step"]) == steps
    assert payload["row_assignment"]["rule"] == "balanced_by_length"
    assert payload["row_assignment"]["digest_this_rank"] is not None


def test_chunking_does_not_change_the_step_losses(tmp_path, monkeypatch):
    """Same plan at lr 0: chunked and one-row forwards give the same history.

    This pins the call site's divisor. With the chunk's own row count as the
    divisor, a step of two one-row chunks records twice the loss of the same
    step with one two-row forward.
    """
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:h{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(4) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)

    def history(chunk_rows):
        argv = [
            "train_panel2_lora.py", "--plan", str(plan),
            "--output", str(tmp_path / f"out{chunk_rows}"), "--world", "1",
            "--device", "cpu", "--lr", "0", "--epochs", "1",
            "--batch-rows", "2", "--plan-unpermuted-noul",
            "--frozen-loss-sample", "1",
        ]
        if chunk_rows:
            argv.extend(["--chunk-rows", str(chunk_rows)])
        monkeypatch.setattr(sys, "argv", argv)
        assert module.main() == 0
        payload = json.loads(
            (tmp_path / f"out{chunk_rows}" / "shard_0.json").read_text())
        return payload["step_losses"]

    chunked = history(1)
    whole = history(0)
    assert len(chunked) == len(whole) == 2
    assert chunked == pytest.approx(whole, rel=1e-9)


def test_the_chunk_decision_reports_the_knob_that_bound():
    """A step's label names the constraint that split its chunks, not the config."""
    from model.scripts.train_panel2_lora import (chunk_by_length,
                                                 chunk_decision_label)

    rows = _batch_rows(3, growth=1)
    counts = {id(rows[0]): 100, id(rows[1]): 90, id(rows[2]): 60}
    decisions = []
    chunk_by_length(rows, counts, 10 ** 6, 1.25, 0, decisions)
    assert decisions == ["ratio"]
    assert chunk_decision_label(decisions) == "ratio"
    decisions = []
    chunk_by_length(rows, counts, 120, 1.0, 0, decisions)
    assert decisions == ["chunk_tokens", "chunk_tokens"]
    assert chunk_decision_label(decisions) == "chunk_tokens"
    assert chunk_decision_label([]) == "budget_non_binding"
    rows = _batch_rows(4, growth=1)
    counts = {id(row): 100 for row in rows}
    decisions = []
    chunk_by_length(rows, counts, 10 ** 6, 1.0, 2, decisions)
    assert chunk_decision_label(decisions) == "chunk_rows"


def test_the_memory_budget_is_reported_not_silently_outranked(tmp_path,
                                                              monkeypatch):
    """Both budgets may be passed; the payload names which one decided."""
    module = importlib.import_module("model.scripts.train_panel2_lora")
    records = [
        _record(f"corpus:m{index}", option, 1 if option == 0 else 0, "choice",
                CHOICE_PREFIX, f"Option {option}.", 2)
        for index in range(4) for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "train_panel2_lora.py", "--plan", str(plan),
        "--output", str(tmp_path / "out"), "--world", "1", "--device", "cpu",
        "--lr", "0", "--epochs", "1", "--batch-rows", "2",
        "--plan-unpermuted-noul", "--frozen-loss-sample", "1",
        "--max-batch-tokens", "1000", "--chunk-rows", "1",
        "--chunk-ratio", "1.0", "--phase-profile", "2",
    ])
    assert module.main() == 0
    profile = json.loads(
        (tmp_path / "out" / "phase_profile_rank0.json").read_text())
    assert profile["chunking_this_rank"]["length_ratio"] == 1.0
    counts = profile["chunking_this_rank"]["chunk_decision_counts"]
    assert set(counts) <= {"chunk_tokens", "ratio", "chunk_rows",
                           "budget_non_binding"}
    assert sum(counts.values()) == len(profile["steps"])


def test_chunk_rows_fixes_the_forwards_per_rank():
    """--chunk-rows splits by row count whatever the token budget would allow."""
    from model.scripts.train_panel2_lora import chunk_by_length

    rows = _batch_rows(8, growth=1)
    counts = {id(row): 100 for row in rows}
    assert len(chunk_by_length(rows, counts, 10 ** 6, 1.0)) == 1
    assert len(chunk_by_length(rows, counts, 10 ** 6, 1.0, max_rows=2)) == 4
    assert len(chunk_by_length(rows, counts, 10 ** 6, 1.0, max_rows=1)) == 8


def test_the_ratio_branch_splits_a_short_row_from_a_long_one():
    """100/90/60 at ratio 1.25: 60 splits, 90 stays with 100; 1.0 cannot fire."""
    from model.scripts.train_panel2_lora import chunk_by_length

    rows = _batch_rows(3, growth=1)
    counts = {id(rows[0]): 100, id(rows[1]): 90, id(rows[2]): 60}
    chunks = chunk_by_length(rows, counts, 10 ** 6, 1.25)
    assert [[counts[id(row)] for row in chunk] for chunk in chunks] == [
        [100, 90], [60]]
    assert len(chunk_by_length(rows, counts, 10 ** 6, 1.0)) == 1


def test_a_balanced_step_trains_the_strided_batches_union():
    """The deal's rows are the strided step's rows, wrapped the same way.

    Ten rows over four ranks: shards of 3, 3, 2 and 2. On step 2 the shards
    wrap by their own lengths, so the union holds rows the global slice of the
    same size would not; a balanced run that trained a global window would be
    a different step on the wrapped step and nowhere else.
    """
    from model.scripts.train_panel2_lora import balanced_step_batch, batch_indices

    rows = _batch_rows(10, growth=3)
    counts = {id(row): len(row.texts[0]) + len(row.texts[1]) for row in rows}
    for step in (1, 2):
        expected = []
        for other in range(4):
            shard = rows[other::4]
            expected.extend(shard[index] for index in
                            batch_indices(step, 2, len(shard)))
        union = []
        for rank in range(4):
            union.extend(balanced_step_batch(step, rows, rank, 4, 2, counts))
        assert len(union) == 8
        assert sorted(row.row_id for row in union) == sorted(
            row.row_id for row in expected)


def test_the_recorded_step_loss_uses_the_gradient_divisor(tmp_path,
                                                          monkeypatch):
    """One step, two ranks, an uneven deal: the loss divides by nominal rows.

    Rank 0 is dealt one long row and rank 1 the three short ones. Divided by
    the nominal two rows the recorded losses are ln2/2 and 3*ln2/2; divided by
    each rank's own row count they would both be ln2, the mean of per-rank
    means that the fixed divisor exists to avoid.
    """
    import torch

    module = importlib.import_module("model.scripts.train_panel2_lora")
    specs = [("long", CHOICE_PREFIX + "x" * 600), ("s1", CHOICE_PREFIX),
             ("s2", CHOICE_PREFIX), ("s3", CHOICE_PREFIX)]
    records = [
        _record(f"corpus:{name}", option, 1 if option == 0 else 0, "choice",
                prefix, f"Option {option}.", 2)
        for name, prefix in specs for option in (0, 1)
    ]
    plan = _write(tmp_path, records)
    _fake_training_modules(module, monkeypatch)

    class WidthTokenizer:
        """One token per character, so the long prefix is a long prompt."""

        def __call__(self, text, return_tensors=None, padding=False,
                     add_special_tokens=False):
            texts = [text] if isinstance(text, str) else list(text)
            width = max(len(item) for item in texts)
            ids = [[1] * len(item) + [0] * (width - len(item)) for item in texts]
            mask = [[1] * len(item) + [0] * (width - len(item)) for item in texts]
            if return_tensors is None:
                single = isinstance(text, str)
                return {"input_ids": ids[0] if single else ids,
                        "attention_mask": mask[0] if single else mask}
            return {"input_ids": torch.tensor(ids),
                    "attention_mask": torch.tensor(mask)}

    fake, _ = module.load_causal_lm("unused", device="cpu", dtype=None)
    monkeypatch.setattr(module, "load_causal_lm",
                        lambda *a, **k: (fake, WidthTokenizer()))

    def run(rank, balance):
        argv = [
            "train_panel2_lora.py", "--plan", str(plan),
            "--output", str(tmp_path / f"out{rank}{balance}"), "--world", "2",
            "--rank", str(rank), "--device", "cpu", "--lr", "0", "--epochs", "1",
            "--batch-rows", "2", "--plan-unpermuted-noul",
            "--frozen-loss-sample", "1",
        ]
        if balance:
            argv.append("--balance-rows")
        monkeypatch.setattr(sys, "argv", argv)
        assert module.main() == 0
        payload = json.loads(
            (tmp_path / f"out{rank}{balance}" / f"shard_{rank}.json").read_text())
        return payload["epoch_mean_loss"]

    value = math.log(2.0)
    paired = sorted(run(rank, True) for rank in (0, 1))
    assert paired == pytest.approx([value / 2, 3 * value / 2], rel=1e-6)
    strided = [run(rank, False) for rank in (0, 1)]
    assert strided == pytest.approx([value, value], rel=1e-6)
