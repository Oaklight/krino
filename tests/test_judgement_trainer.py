"""The judgement trainer's loss, answerable mask and positive-rate treatment."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from model.evaluation.judgement_prompt import render, render_parts  # noqa: E402
from model.src.judgement_objective import (  # noqa: E402
    ScaledMargin,
    build_prompts,
    fit_bias,
    masked_row_normalised_loss,
    positive_base_rate,
    row_weights,
)
from model.src.judgement_readout import subsample_prompts  # noqa: E402
from model.scripts.train_judgement import _record_to_prompt  # noqa: E402
from model.scripts.train_judgement import draw_subset  # noqa: E402
from model.scripts.train_judgement import step_chunks  # noqa: E402


def test_subset_draw_is_seeded_nested_and_in_plan_order():
    rows = [(f"row-{index}", []) for index in range(100)]
    first = draw_subset(rows, 20, 42)
    again = draw_subset(rows, 20, 42)
    assert [row[0] for row in first] == [row[0] for row in again]
    assert len(first) == 20
    assert {row[0] for row in first} <= {row[0] for row in rows}
    indices = [int(row[0].split("-")[1]) for row in first]
    assert indices == sorted(indices)
    assert draw_subset(rows, 20, 43) != first
    assert draw_subset(rows, 100, 42) == rows


def _item(
    q_type="choice", criteria=None, label="b", answerable=None, state="A situation."
):
    return SimpleNamespace(
        id="row-1",
        state=state,
        question={"type": q_type, "instructions": "Pick one.", "criteria": criteria},
        label=label,
        split="train",
        answerable=answerable,
    )


# -- prompt construction -----------------------------------------------------


def test_prefix_plus_suffix_is_the_rendered_prompt():
    state, instructions, option = "A situation.", "Pick one.", "Second"
    assert "".join(render_parts(state, instructions, option)) == render(
        state, instructions, option
    )


def test_every_option_of_a_row_shares_one_prefix():
    prompts = build_prompts(_item(criteria={"a": "First", "b": "Second"}, label="b"))
    assert len({prompt.prefix for prompt in prompts}) == 1
    assert [prompt.suffix for prompt in prompts] == [
        "Option: First\nDoes this option correctly answer the question? Answer:",
        "Option: Second\nDoes this option correctly answer the question? Answer:",
    ]
    assert [prompt.label for prompt in prompts] == [0, 1]


def test_noul_is_a_two_option_row_with_one_positive():
    prompts = build_prompts(
        _item(
            q_type="noul",
            criteria={"true": "The text states that this is so", "false": "not so"},
            label=True,
        )
    )
    assert len(prompts) == 2
    assert [prompt.label for prompt in prompts] == [1, 0]


def test_criteria_less_noul_is_one_two_way_polarity_prompt():
    prompts = build_prompts(_item(q_type="noul", criteria=None, label=False))
    assert len(prompts) == 1
    assert prompts[0].label == 0
    assert not prompts[0].has_option_span
    assert prompts[0].prefix == ""
    assert prompts[0].suffix == prompts[0].text


def test_unanswerable_rows_are_built_not_dropped():
    prompts = build_prompts(
        _item(criteria={"a": "First", "b": "Second"}, label="b", answerable=False)
    )
    assert len(prompts) == 2
    assert all(prompt.answerable is False for prompt in prompts)


def test_plan_round_trip_keeps_the_mask_and_the_prompt_parts():
    prompts = build_prompts(
        _item(criteria={"a": "First", "b": "Second"}, label="b", answerable=False)
    )
    for prompt in prompts:
        record = json.loads(json.dumps(prompt.__dict__))
        rebuilt = _record_to_prompt(record)
        assert rebuilt == prompt


# -- the internal split ------------------------------------------------------


def _planned_row(row_id, state, split="train"):
    prompts = build_prompts(
        _item(criteria={"a": "First", "b": "Second"}, label="b", state=state),
        row_id=row_id,
        split=split,
    )
    return row_id, prompts


def test_internal_split_is_group_aware_and_excludes_heldout():
    from model.scripts.train_judgement import split_rows

    rows = [
        _planned_row("r:p1", "Same  state"),
        _planned_row("r:p2", "Same state"),
        _planned_row("r:a", "alpha"),
        _planned_row("r:b", "beta"),
        _planned_row("r:eval", "held out"),
        _planned_row("r:test", "external", split="test"),
    ]
    train, val, external, report = split_rows(rows, {"r:eval"}, None, 0.5, 0)
    train_ids = {row_id for row_id, _ in train}
    val_ids = {row_id for row_id, _ in val}
    assert ("r:p1" in train_ids) == ("r:p2" in train_ids)
    assert ("r:p1" in val_ids) == ("r:p2" in val_ids)
    assert "r:eval" not in train_ids | val_ids
    assert [row_id for row_id, _ in external] == ["r:test"]
    assert report["groups"] == 3  # the pair + two singletons
    assert report["items"] == 4


def test_batch_shard_partitions_every_global_batch():
    from model.scripts.train_judgement import _batch_shard

    rows = list(range(8))
    left = _batch_shard(rows, 0, 2, 2)
    right = _batch_shard(rows, 1, 2, 2)
    assert left == [0, 1, 4, 5]
    assert right == [2, 3, 6, 7]
    assert sorted(left + right) == rows


def test_batch_shard_pads_only_the_last_block():
    from model.scripts.train_judgement import _batch_shard

    rows = list(range(6))
    left = _batch_shard(rows, 0, 2, 2)
    right = _batch_shard(rows, 1, 2, 2)
    # A short final block repeats its last row; the first block is exact.
    assert left[:2] == [0, 1] and right[:2] == [2, 3]
    assert len(left) == len(right) == 4


# -- the row-normalised loss -------------------------------------------------


def test_row_normalised_loss_weights_rows_not_options():
    # Two rows with different per-row losses.  Tripling row B's options leaves
    # B's per-row loss and the row-normalised batch loss unchanged; an option
    # mean moves, because it weights B by its option count.
    labels_a = torch.tensor([1.0, 0.0])
    labels_b = torch.tensor([1.0, 0.0])
    row_a = torch.tensor([2.0, -2.0])
    row_b = torch.tensor([-1.0, -1.0])
    loss_a = torch.nn.functional.binary_cross_entropy_with_logits(row_a, labels_a)
    loss_b = torch.nn.functional.binary_cross_entropy_with_logits(row_b, labels_b)

    narrow = masked_row_normalised_loss(
        torch.cat([row_a, row_b]),
        torch.cat([labels_a, labels_b]),
        [torch.arange(0, 2), torch.arange(2, 4)],
        torch.ones(2),
    )
    wide_b = row_b.repeat(3)
    wide = masked_row_normalised_loss(
        torch.cat([row_a, wide_b]),
        torch.cat([labels_a, labels_b.repeat(3)]),
        [torch.arange(0, 2), torch.arange(2, 8)],
        torch.ones(2),
    )
    assert torch.allclose(narrow, wide)
    assert torch.allclose(narrow, (loss_a + loss_b) / 2)

    option_narrow = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.cat([row_a, row_b]), torch.cat([labels_a, labels_b])
    )
    option_wide = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.cat([row_a, wide_b]), torch.cat([labels_a, labels_b.repeat(3)])
    )
    assert not torch.allclose(option_narrow, option_wide)


def test_masked_row_is_excluded_from_the_denominator():
    margins = torch.tensor([0.0, -1.0, 2.0, 2.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    rows = [torch.arange(0, 2), torch.arange(2, 4)]
    masked = masked_row_normalised_loss(margins, labels, rows, torch.tensor([1.0, 0.0]))
    alone = masked_row_normalised_loss(margins, labels, rows[:1], torch.tensor([1.0]))
    assert torch.allclose(masked, alone)


@pytest.mark.feature_guard("KRINO_DISABLE_ANSWERABLE_MASK")
def test_unanswerable_row_contributes_zero_gradient():
    labels = torch.tensor([1.0, 0.0])
    rows = [torch.arange(2)]
    raw = torch.tensor([-2.0, -2.0], requires_grad=True)
    loss = masked_row_normalised_loss(raw, labels, rows, row_weights([False]))
    loss.backward()
    assert torch.count_nonzero(raw.grad) == 0

    # The same row unmasked does move the gradient: the zero above is the
    # mask's doing, not an inert loss.  This is what fails when the kill switch
    # disables the mask.
    unmasked = torch.tensor([-2.0, -2.0], requires_grad=True)
    masked_row_normalised_loss(unmasked, labels, rows, torch.ones(1)).backward()
    assert torch.count_nonzero(unmasked.grad) > 0


def test_row_weights_flag_only_explicit_false():
    weights = row_weights([True, None, False])
    assert weights.tolist() == [1.0, 1.0, 0.0]


# -- the positive-base-rate treatment ----------------------------------------


def test_scaled_margin_applies_scale_and_bias():
    scaling = ScaledMargin(scale=2.0, bias=-1.0)
    raw = torch.tensor([0.0, 1.0])
    assert torch.allclose(scaling(raw), torch.tensor([-1.0, 1.0]))
    assert torch.allclose(
        ScaledMargin.probability(scaling(raw)),
        torch.sigmoid(torch.tensor([-1.0, 1.0])),
    )


def test_bias_calibration_can_lift_a_gold_option_above_half():
    # Gold raw margin +3 and two negatives at -1: separable enough that a bias
    # fitted to the 1/3 positive rate puts the gold above 0.5.
    raw = torch.tensor([3.0, -1.0, -1.0])
    rows = [torch.arange(3)]
    target = 1 / 3
    offset = fit_bias(raw, rows, target)
    probability = torch.sigmoid(raw + offset)
    assert probability[0] > 0.5
    assert probability[1:].max() < 0.5
    assert abs(float(probability.mean()) - target) < 1e-4


def test_anti_correlated_margins_leave_no_gold_above_half():
    # The gold is scored below the negatives.  A bias fitted to the positive
    # rate lifts every option together, so no gold reaches 0.5: this is the
    # collapse #299's check must catch.
    raw = torch.tensor([-5.0, -1.0, -1.0])  # gold first
    rows = [torch.arange(3)]
    offset = fit_bias(raw, rows, 1 / 3)
    probability = torch.sigmoid(raw + offset)
    assert probability[0] < 0.5
    assert (probability > 0.5).sum() == 0


def test_positive_base_rate_excludes_masked_rows():
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0])
    rows = [torch.arange(0, 2), torch.arange(2, 4)]
    rate = positive_base_rate(labels, rows, torch.tensor([1.0, 0.0]))
    assert rate == 0.5


# -- option subsampling ------------------------------------------------------


def test_step_chunks_cover_every_row_exactly_once():
    rows = [
        build_prompts(_item(criteria={"a": "alpha", "b": "beta"}, label="a"))
        for _ in range(40)
    ]
    chunks = step_chunks(rows, 512)
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(rows)
    for (_, stop), (start, _) in zip(chunks, chunks[1:]):
        assert stop == start
    assert all(start < stop for start, stop in chunks)


def test_step_chunks_charge_the_padded_bucket():
    # A 4000-character state is about 1000 tokens, which pads to the 1024
    # bucket, so a 4096-token budget holds three rows, not four by raw length.
    rows = [
        build_prompts(
            _item(criteria={"a": "alpha", "b": "beta"}, label="a", state="x" * 4000)
        )
        for _ in range(8)
    ]
    assert [stop - start for start, stop in step_chunks(rows, 4096)] == [3, 3, 2]


def test_step_chunks_keep_a_long_row_whole():
    long_row = build_prompts(
        _item(criteria={"a": "alpha", "b": "beta"}, label="a", state="y" * 16000)
    )
    short_rows = [
        build_prompts(
            _item(criteria={"a": "alpha", "b": "beta"}, label="a", state="z" * 40)
        )
        for _ in range(5)
    ]
    chunks = step_chunks([long_row] + short_rows, 512)
    assert chunks[0] == (0, 1)
    assert all(start >= 1 for start, _ in chunks[1:])


def test_subsampling_keeps_the_positive_and_caps_the_row():
    import random

    prompts = build_prompts(
        _item(criteria={str(i): f"option {i}" for i in range(30)}, label="7")
    )
    kept = subsample_prompts(prompts, 10, random.Random(0))
    assert len(kept) == 10
    assert sum(prompt.label for prompt in kept) == 1
    assert [prompt.option_index for prompt in kept] == sorted(
        prompt.option_index for prompt in kept
    )
