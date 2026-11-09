"""The judgement objective's prompt builder, head and row-normalised loss (#324)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from model.src.judgement_objective import (  # noqa: E402
    TwoWayHead,
    build_prompts,
    contiguous_rows,
    full_attention_mask,
    row_normalised_loss,
)


def _item(q_type: str, criteria, label, instructions="Pick one."):
    return SimpleNamespace(
        id="easy-1",
        state="A situation.",
        question={"type": q_type, "instructions": instructions, "criteria": criteria},
        label=label,
    )


def test_choice_row_is_one_prompt_per_option_with_a_single_positive():
    prompts = build_prompts(_item("choice", {"a": "First", "b": "Second"}, "b"))
    assert [p.label for p in prompts] == [0, 1]
    assert [p.option_index for p in prompts] == [0, 1]
    assert all(p.has_option_span for p in prompts)
    assert prompts[0].text.startswith("State: A situation.\nQuestion: Pick one.\nOption: First\n")
    assert prompts[0].text[prompts[0].option_char_end :].startswith("\nDoes this option")


def test_noul_uses_the_questions_own_true_false_wording_true_first():
    prompts = build_prompts(_item("noul", {"true": "Sí", "false": "No"}, True))
    assert [p.text.split("Option: ")[1].split("\n")[0] for p in prompts] == ["Sí", "No"]
    assert [p.label for p in prompts] == [1, 0]


def test_criteria_less_noul_is_one_polarity_prompt():
    prompts = build_prompts(_item("noul", None, False))
    assert len(prompts) == 1
    assert prompts[0].label == 0
    assert not prompts[0].has_option_span
    assert prompts[0].text.endswith("Answer:")


def test_item_with_neither_options_nor_polarity_yields_nothing():
    assert build_prompts(_item("choice", {"a": "First"}, "zzz")) == []
    assert build_prompts(_item("noul", {}, "not-a-bool")) == []


def test_row_normalised_loss_weighs_rows_not_options():
    # Row 0 has four options, all confidently right (near-zero loss); row 1 has
    # two options, one maximally wrong. The result must be half of that wrong
    # option's loss, not a third of it as an option-mean would give.
    logits = torch.tensor(
        [[10.0, -10.0], [10.0, -10.0], [10.0, -10.0], [10.0, -10.0],
         [10.0, -10.0], [-10.0, 10.0]]
    )
    labels = torch.tensor([0, 0, 0, 0, 0, 0])
    rows = contiguous_rows(torch.tensor([0, 0, 0, 0, 1, 1]))
    loss = row_normalised_loss(logits, labels, rows)
    wrong = torch.nn.functional.cross_entropy(logits[4:], labels[4:])
    assert torch.allclose(loss, wrong / 2, atol=1e-6)


def test_contiguous_rows_splits_runs():
    rows = contiguous_rows(torch.tensor([7, 7, 7, 9, 9, 12]))
    assert [row.tolist() for row in rows] == [[0, 1, 2], [3, 4], [5]]


def test_two_way_head_starts_biased_towards_yes_and_scores_the_difference():
    head = TwoWayHead(8, yes_bias=2.0)
    logits = head(torch.zeros(3, 8))
    assert torch.allclose(logits[:, 1], torch.full((3,), 2.0))
    assert torch.allclose(TwoWayHead.score(logits), torch.full((3,), 2.0))
    assert torch.all(TwoWayHead.score(logits) > 0)


def test_full_attention_mask_masks_padding_keys_only():
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    mask = full_attention_mask(attention)
    assert mask.shape == (2, 1, 4, 4)
    assert (mask[0, 0, :, :3] == 0).all()
    assert torch.isinf(mask[0, 0, :, 3]).all() and (mask[0, 0, :, 3] < 0).all()
    assert (mask[1, 0, :, :2] == 0).all()
