"""Script-level helpers for the listwise arm, without a model.

The readout itself is covered by ``test_listwise.py`` on a tiny hybrid model;
these tests pin the arithmetic around it: subset restriction of the base
distribution, the residual fusion, subset-to-option index mapping and the
subset file format the combined arm feeds in.
"""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from model.scripts.eval_listwise import _load_subsets, score_item  # noqa: E402
from model.scripts.train_listwise import (  # noqa: E402
    _subset_probs,
    build_parser,
    evaluate,
    fused_logits,
    item_loss,
    split_items,
    unanswerable_excluded,
)


class _Item:
    def __init__(
        self,
        q_type="choice",
        criteria=None,
        state="s",
        instructions="q",
        answerable=None,
    ):
        self.state = state
        self.question = {
            "type": q_type,
            "instructions": instructions,
            "criteria": criteria or {"a": "alpha", "b": "beta", "c": "gamma"},
        }
        self.label = "b"
        self.answerable = answerable


class _StubReadout:
    """Returns a fixed distribution over whatever subset it is handed."""

    def __init__(self, probs):
        self.probs = probs
        self.calls = []

    def score_shifted(self, state, instructions, options, shifts=4):
        self.calls.append((list(options), shifts))
        probs = torch.tensor([self.probs[len(options)]], dtype=torch.float32)
        return probs


class _LossReadout:
    """A readout with a flat score, for the loss-shape checks."""

    def score(self, state, instructions, options):
        return torch.zeros(len(options))


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


def test_sampling_honours_a_max_size():
    from model.src.listwise import ListwiseMarkerReadout

    rng = random.Random(0)
    for _ in range(50):
        subset = ListwiseMarkerReadout.sample_subset(30, 0, rng, max_size=4)
        assert len(subset) <= 4 and 0 in subset


def test_the_split_seed_is_separate_from_the_training_seed():
    """A replication seed must not reshuffle which items are held out."""
    args = build_parser().parse_args(
        ["--items", "a.jsonl", "--base-probs", "b.jsonl", "--output", "out"]
    )
    assert args.split_seed == 42
    assert args.seed == 42
    rerun = build_parser().parse_args(
        [
            "--items", "a.jsonl", "--base-probs", "b.jsonl", "--output", "out",
            "--seed", "43", "--split-seed", "42",
        ]
    )
    assert rerun.seed == 43
    assert rerun.split_seed == 42


@pytest.mark.feature_guard("KRINO_DISABLE_LISTWISE_ANSWERABLE_EXCLUSION")
def test_an_unanswerable_row_is_excluded():
    assert unanswerable_excluded(_Item(answerable=False)) is True


@pytest.mark.feature_guard("KRINO_DISABLE_LISTWISE_ANSWERABLE_EXCLUSION")
def test_item_loss_is_a_graphless_zero_for_an_unanswerable_row():
    """The row must not reach backward() with no graph."""
    item = _Item(answerable=False)
    loss, kl, ce = item_loss(
        _LossReadout(),
        item,
        torch.tensor([0.2, 0.3, 0.5]),
        torch.zeros((), requires_grad=True),
        random.Random(0),
        1.0,
        0.2,
        2,
        False,
    )
    assert kl == 0.0
    assert ce == 0.0
    assert loss.item() == 0.0
    assert not loss.requires_grad


class _NoulItem:
    def __init__(self, label=True):
        self.id = "noul-1"
        self.state = "s"
        self.question = {"type": "noul", "instructions": "does it hold"}
        self.label = label


class _EvalReadout:
    """Returns a fixed distribution for whatever option set it is handed."""

    class _Model:
        def eval(self):
            return self

        def train(self):
            return self

    def __init__(self, probs):
        self.probs = probs
        self.model = self._Model()
        self.calls = 0

    def fits(self, options):
        return True

    def score_shifted(self, state, instructions, options, shifts=4, **kwargs):
        self.calls += 1
        return torch.tensor([self.probs], dtype=torch.float32)


def test_the_validation_scores_noul_through_the_readout():
    """A moved noul head must move the validation number.

    Noul is in ``TRAINED_TYPES``, so a validation that answered from the base
    distribution would report the frozen base under the arm's name and no
    amount of noul training could show up.  The guard is the disagreement
    between the two heads below: with the base shortcut in place both return
    the same value.
    """
    item = _NoulItem(label=True)          # gold index 1, "yes"
    base = {"noul-1": torch.tensor([0.5, 0.5])}
    alphas = {"noul": torch.tensor(1.0)}

    right = evaluate(_EvalReadout([0.02, 0.98]), [item], base, alphas, 4)
    wrong = evaluate(_EvalReadout([0.98, 0.02]), [item], base, alphas, 4)

    assert right["by_type"]["noul"] == pytest.approx(1.0)
    assert wrong["by_type"]["noul"] == pytest.approx(-1.0)
    assert right["n"] == 1 and wrong["n"] == 1


def test_the_validation_asks_the_readout_for_noul():
    item = _NoulItem()
    base = {"noul-1": torch.tensor([0.5, 0.5])}
    readout = _EvalReadout([0.5, 0.5])
    evaluate(readout, [item], base, {"noul": torch.tensor(0.0)}, 4)
    assert readout.calls == 1
