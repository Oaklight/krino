"""Script-level helpers for the listwise arm, without a model.

The readout itself is covered by ``test_listwise.py`` on a tiny hybrid model;
these tests pin the arithmetic around it: subset restriction of the base
distribution, the residual fusion, subset-to-option index mapping and the
subset file format the combined arm feeds in.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from model.scripts.eval_listwise import _load_subsets, score_item  # noqa: E402
from model.scripts.train_listwise import (  # noqa: E402
    _subset_probs,
    fused_logits,
    split_items,
)


class _Item:
    def __init__(self, q_type="choice", criteria=None, state="s", instructions="q"):
        self.state = state
        self.question = {
            "type": q_type,
            "instructions": instructions,
            "criteria": criteria or {"a": "alpha", "b": "beta", "c": "gamma"},
        }
        self.label = "b"


class _StubReadout:
    """Returns a fixed distribution over whatever subset it is handed."""

    def __init__(self, probs):
        self.probs = probs
        self.calls = []

    def score_shifted(self, state, instructions, options, shifts=4):
        self.calls.append((list(options), shifts))
        probs = torch.tensor([self.probs[len(options)]], dtype=torch.float32)
        return probs


def test_subset_probs_renormalises_the_picked_options():
    base = torch.tensor([0.1, 0.2, 0.7])
    picked = _subset_probs(base, [0, 2])
    assert picked.tolist() == pytest.approx([0.125, 0.875], abs=1e-6)


def test_subset_probs_keeps_the_full_set_unchanged():
    base = torch.tensor([0.1, 0.2, 0.7])
    assert _subset_probs(base, [0, 1, 2]).tolist() == pytest.approx(
        base.tolist(), abs=1e-6
    )


def test_fusion_is_the_residual_rule():
    base = torch.tensor([0.5, 0.5])
    head = torch.tensor([2.0, 0.0])
    alpha = torch.tensor(0.5)
    got = fused_logits(base, head, alpha)
    expected = torch.log(base) + 0.5 * head
    assert got.tolist() == pytest.approx(expected.tolist(), abs=1e-6)


def test_a_zero_alpha_reproduces_the_base():
    base = torch.tensor([0.3, 0.7])
    head = torch.tensor([5.0, -5.0])
    fused = torch.softmax(fused_logits(base, head, torch.tensor(0.0)), dim=-1)
    assert fused.tolist() == pytest.approx(base.tolist(), abs=1e-6)


def test_score_item_maps_a_local_choice_back_to_the_option_index():
    readout = _StubReadout({2: [0.9, 0.1]})
    item = _Item()
    pred, probs = score_item(readout, item, [1, 2], None, {}, shifts=4)
    # The stub puts 0.9 on the first entry of the subset, which is option 1.
    assert pred == 1
    assert probs == pytest.approx([0.9, 0.1], abs=1e-6)


def test_score_item_uses_the_configured_shifts_for_choice_only():
    readout = _StubReadout({2: [0.5, 0.5]})
    score_item(readout, _Item(q_type="choice"), [0, 1], None, {}, shifts=4)
    score_item(readout, _Item(q_type="score", criteria=["1", "2"]), [0, 1], None, {}, shifts=4)
    assert readout.calls[0][1] == 4
    assert readout.calls[1][1] == 1


def test_score_item_fuses_the_base_distribution():
    readout = _StubReadout({2: [0.5, 0.5]})
    item = _Item()
    base = torch.tensor([0.1, 0.45, 0.45])
    pred, probs = score_item(readout, item, [1, 2], base, {"choice": 0.0}, shifts=1)
    # alpha 0 reproduces the renormalised base over [1, 2].
    assert probs == pytest.approx([0.5, 0.5], abs=1e-6)
    assert pred in (1, 2)


def test_subsets_file_is_parsed_and_validated(tmp_path):
    path = tmp_path / "subsets.json"
    path.write_text(json.dumps({"item-1": [0, 2], "item-2": [1]}))
    assert _load_subsets(path) == {"item-1": [0, 2], "item-2": [1]}
    assert _load_subsets(None) == {}


def test_a_non_object_subsets_file_is_rejected(tmp_path):
    path = tmp_path / "subsets.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SystemExit):
        _load_subsets(path)


def test_the_val_split_reuses_the_residual_split():
    """The listwise arm must share D20K's seed-42 val ids."""
    from model.scripts import train_residual

    assert split_items is train_residual.split_items
