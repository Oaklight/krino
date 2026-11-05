"""Two-stage pairwise arm on a tiny hybrid model.

Covers the parts that are easy to get subtly wrong without a real model: the
shortlist is deterministic and stable on ties, the gold is force-included when
training needs it, choice pairs average both presentation orders while rubric
pairs do not, options outside the shortlist get probability 0, and the pair
loss backpropagates into the marker head.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
qwen35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from model.src.decision_model import DecisionModel  # noqa: E402
from model.src.gpu_config import GPUConfig  # noqa: E402
from model.src.pairwise import (  # noqa: E402
    PairwiseMarkerReadout,
    stage1_shortlist,
)

VOCAB = 128
MARKER_ID = 1


class _Encoding(dict):
    def to(self, device):
        return self


class _CharTokenizer:
    eos_token_id = MARKER_ID
    pad_token_id = 0

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=False,
        **kwargs,
    ):
        ids = [2 + (b % (VOCAB - 4)) for b in str(text).encode()]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return _Encoding(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )
        return {"input_ids": ids}


def _tiny_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return qwen35.Qwen3_5ForCausalLM(config).eval()


def _readout() -> PairwiseMarkerReadout:
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=256,
        gradient_checkpointing=False,
    )
    model = DecisionModel(
        backbone=_tiny_model(),
        tokenizer=_CharTokenizer(),
        rank=8,
        gpu_config=gpu,
        option_marker=True,
        option_chunk_size=1,
    ).eval()
    return PairwiseMarkerReadout(model)


STATE = "a fairly long state " * 4
INSTRUCTIONS = "pick one"
OPTIONS = ["alpha", "beta", "gamma", "delta", "epsilon"]


@pytest.fixture(scope="module")
def readout() -> PairwiseMarkerReadout:
    return _readout()


class TestShortlist:
    def test_is_deterministic_and_sorted(self):
        base = torch.tensor([0.1, 0.4, 0.2, 0.3])
        assert stage1_shortlist(base, 2) == [1, 3]
        assert stage1_shortlist(base, 2) == stage1_shortlist(base, 2)

    def test_ties_break_on_index(self):
        base = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert stage1_shortlist(base, 2) == [0, 1]

    def test_the_gold_is_forced_in_for_training(self):
        base = torch.tensor([0.7, 0.1, 0.1, 0.1])
        assert stage1_shortlist(base, 2) == [0, 1]
        assert stage1_shortlist(base, 2, must_include=3) == [0, 3]

    def test_the_gold_is_already_present(self):
        base = torch.tensor([0.7, 0.1, 0.1, 0.1])
        assert stage1_shortlist(base, 2, must_include=0) == [0, 1]

    def test_m_larger_than_the_set_returns_everything(self):
        base = torch.tensor([0.5, 0.5])
        assert stage1_shortlist(base, 5) == [0, 1]

    def test_a_zero_size_is_rejected(self):
        with pytest.raises(ValueError):
            stage1_shortlist(torch.tensor([0.5, 0.5]), 0)


class TestDuels:
    def test_a_choice_pair_averages_both_orders(self, readout):
        forward = readout.duel(STATE, INSTRUCTIONS, "alpha", "beta")
        backward = 1.0 - readout.duel(STATE, INSTRUCTIONS, "beta", "alpha")
        averaged = readout.pair_win(
            STATE, INSTRUCTIONS, "alpha", "beta", order_sensitive=True
        )
        assert averaged == pytest.approx(0.5 * (forward + backward), abs=1e-9)

    def test_a_rubric_pair_keeps_its_order(self, readout):
        forward = readout.duel(STATE, INSTRUCTIONS, "alpha", "beta")
        kept = readout.pair_win(
            STATE, INSTRUCTIONS, "alpha", "beta", order_sensitive=False
        )
        assert kept == pytest.approx(forward, abs=1e-9)

    def test_the_order_average_is_not_the_single_order(self, readout):
        single = readout.duel(STATE, INSTRUCTIONS, "alpha", "beta")
        averaged = readout.pair_win(
            STATE, INSTRUCTIONS, "alpha", "beta", order_sensitive=True
        )
        assert abs(averaged - single) > 1e-6


class TestShortlistProbabilities:
    def test_mass_sits_on_the_shortlist(self, readout):
        base = torch.tensor([0.1, 0.1, 0.1, 0.4, 0.3])
        probs, shortlist = readout.shortlist_probabilities(
            STATE, INSTRUCTIONS, OPTIONS, base, 0.0,
            m=2, order_sensitive=True,
        )
        assert shortlist == [3, 4]
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert probs[0].item() == 0.0 and probs[1].item() == 0.0
        assert probs[2].item() == 0.0

    def test_a_zero_alpha_reproduces_the_shortlist_base(self, readout):
        base = torch.tensor([0.1, 0.1, 0.1, 0.4, 0.3])
        probs, _ = readout.shortlist_probabilities(
            STATE, INSTRUCTIONS, OPTIONS, base, 0.0,
            m=2, order_sensitive=True,
        )
        expected = torch.tensor([0.4, 0.3]) / 0.7
        assert probs[3].item() == pytest.approx(expected[0].item(), abs=1e-6)
        assert probs[4].item() == pytest.approx(expected[1].item(), abs=1e-6)

    def test_a_single_option_shortlist_is_deterministic(self, readout):
        base = torch.tensor([0.1, 0.6, 0.3])
        probs, shortlist = readout.shortlist_probabilities(
            STATE, INSTRUCTIONS, OPTIONS[:3], base, 0.5,
            m=1, order_sensitive=True,
        )
        assert shortlist == [1]
        assert probs[1].item() == 1.0


class TestTraining:
    def test_the_pair_loss_backpropagates_to_the_head(self, readout):
        readout.model.zero_grad(set_to_none=True)
        base = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0])
        alpha = torch.nn.Parameter(torch.tensor(1.0))
        loss, kl, ce = readout.pair_loss(
            STATE,
            INSTRUCTIONS,
            OPTIONS,
            gold_index=2,
            distractor_index=0,
            base=base,
            alpha=alpha,
            rng=random.Random(0),
            kl_weight=1.0,
            ce_weight=0.2,
            order_sensitive=True,
        )
        loss.backward()
        assert kl >= 0 and ce >= 0
        assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in readout.model.marker_head.parameters()
        )

    def test_the_pair_order_is_seeded(self, readout):
        """The same rng state presents the pair in the same order."""
        base = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0])
        alpha = torch.zeros(())
        first = readout.pair_loss(
            STATE, INSTRUCTIONS, OPTIONS, 1, 3, base, alpha, random.Random(7),
            kl_weight=1.0, ce_weight=0.2, order_sensitive=True,
        )
        second = readout.pair_loss(
            STATE, INSTRUCTIONS, OPTIONS, 1, 3, base, alpha, random.Random(7),
            kl_weight=1.0, ce_weight=0.2, order_sensitive=True,
        )
        assert float(first[0].detach()) == pytest.approx(
            float(second[0].detach()), abs=1e-9
        )
