"""Tests for the cross-encoder choice head.

The point of the mode is that an option's score depends on the state, because
the state is inside the option's prompt. These tests pin that property, the
zero-initialised read-out, and the checkpoint inference that lets evaluation
rebuild the same head.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from model.evaluation.checkpoint_readout import infer_choice_mode  # noqa: E402
from model.src.heads import CrossEncoderHead  # noqa: E402


class _Config:
    hidden_size = 16
    num_hidden_layers = 2
    max_position_embeddings = 128
    is_decoder = True


class _Block(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.layers = nn.ModuleList([_Block(16) for _ in range(2)])


def _model(choice_mode: str):
    from model.src.decision_model import DecisionModel
    from model.src.gpu_config import GPUConfig

    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=128,
        gradient_checkpointing=False,
    )
    return DecisionModel(
        backbone=_FakeBackbone(),
        tokenizer=None,
        rank=8,
        gpu_config=gpu,
        choice_mode=choice_mode,
    )


class TestCrossEncoderHead:
    def test_it_returns_one_logit_per_option(self):
        head = CrossEncoderHead(16)
        assert head(torch.randn(5, 16)).shape == (5,)

    def test_a_fresh_head_scores_every_option_identically(self):
        # Zero-initialised read-out: the first training step starts from a
        # uniform distribution, not from whichever option a random projection
        # happens to favour.
        head = CrossEncoderHead(16).eval()
        scores = head(torch.randn(7, 16))
        assert torch.allclose(scores, scores[0].expand_as(scores))

    def test_it_scores_options_differently_once_the_weights_move(self):
        head = CrossEncoderHead(16).eval()
        with torch.no_grad():
            head.score.weight.normal_()
        scores = head(torch.randn(7, 16))
        assert not torch.allclose(scores, scores[0].expand_as(scores))


class TestForwardChoice:
    def _stubbed(self, choice_mode: str, recorder: list[str]):
        model = _model(choice_mode)

        def fake_encode(text, max_length=None):
            texts = [text] if isinstance(text, str) else list(text)
            recorder.extend(texts)
            # Deterministic embedding that depends on the whole string, so a
            # prompt containing the state embeds differently from one without.
            rows = []
            for value in texts:
                seed = sum(ord(char) for char in value) % 997
                generator = torch.Generator().manual_seed(seed)
                rows.append(torch.randn(model.hidden_size, generator=generator))
            return torch.stack(rows)

        model._encode_text = fake_encode  # type: ignore[assignment]
        model._encode_with_sequence = lambda text, max_length=None: torch.randn(  # type: ignore[assignment]
            1, 4, model.hidden_size
        )
        return model

    def test_cross_encoder_puts_the_state_into_every_option_prompt(self):
        recorder: list[str] = []
        model = self._stubbed("cross_encoder", recorder)
        logits = model.forward_choice(
            "the customer is angry", "which intent?", ["refund", "cancel"]
        )
        assert logits.shape == (1, 2)
        assert len(recorder) == 2
        for prompt in recorder:
            assert "the customer is angry" in prompt
            assert "which intent?" in prompt

    def test_tower_mode_keeps_the_two_encode_calls(self):
        recorder: list[str] = []
        model = self._stubbed("tower", recorder)
        model._encode_with_sequence = lambda text, max_length=None: (  # type: ignore[assignment]
            recorder.append(text) or torch.randn(1, 4, model.hidden_size)
        )
        model.forward_choice("state", "instructions", ["a", "b"])
        # One context encode plus one batched option encode.
        assert recorder[0] == "state instructions"
        assert recorder[1:] == ["a", "b"]

    def test_the_score_depends_on_the_state_in_cross_encoder_mode(self):
        recorder: list[str] = []
        model = self._stubbed("cross_encoder", recorder)
        # The read-out starts at zero, so every score is identical until the
        # weights move; move them to observe the dependency under test.
        with torch.no_grad():
            model.choice_head.score.weight.normal_()
        first = model.forward_choice("state one", "q", ["yes", "no"])
        second = model.forward_choice("state two", "q", ["yes", "no"])
        assert not torch.allclose(first, second)

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            _model("bilinear")


class TestInferChoiceMode:
    def test_cross_encoder_tensors_are_recognised(self):
        keys = ["choice_head.norm.weight", "choice_head.score.weight"]
        assert infer_choice_mode(keys) == "cross_encoder"

    def test_attention_head_tensors_are_recognised(self):
        keys = ["choice_head.attention.query.weight", "noul_head.proj.0.weight"]
        assert infer_choice_mode(keys) == "tower"

    def test_a_checkpoint_without_a_choice_head_is_not_guessed(self):
        assert infer_choice_mode(["noul_head.proj.0.weight"]) is None
