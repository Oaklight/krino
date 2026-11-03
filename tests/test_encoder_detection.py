"""Encoder-vs-causal detection for DecisionModel.

``config.is_decoder`` is unset on both Qwen3 causal configs and ModernBERT
encoder configs, so the class name and the declared architecture decide.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.src.decision_model import DecisionModel, _is_encoder_model
from model.src.gpu_config import GPUConfig


class _Cfg:
    hidden_size = 8
    num_hidden_layers = 1
    max_position_embeddings = 64
    model_type = "unknown"


class _BareModel(nn.Module):
    """A class name that carries no causal suffix."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # DecisionModel.device reads the first parameter; give it one.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = torch.zeros(
            1, input_ids.shape[1], self.config.hidden_size, dtype=torch.float32
        )
        return type(
            "Out", (), {"hidden_states": (hidden,) * (self.config.num_hidden_layers + 1)}
        )()


class TestEncoderDetection:
    def test_modernbert_config_is_an_encoder(self):
        cfg = _Cfg()
        cfg.architectures = ["ModernBertModel"]
        assert _is_encoder_model(_BareModel(cfg)) is True

    def test_causal_architecture_is_not_an_encoder(self):
        cfg = _Cfg()
        cfg.architectures = ["Qwen3ForCausalLM"]
        assert _is_encoder_model(_BareModel(cfg)) is False

    def test_causal_class_name_is_not_an_encoder(self):
        cfg = _Cfg()
        cfg.architectures = None
        model = type("Qwen3ForCausalLM", (nn.Module,), {})()
        model.config = cfg
        model.forward = lambda **kw: None
        assert _is_encoder_model(model) is False

    def test_explicit_decoder_flag_wins(self):
        cfg = _Cfg()
        cfg.architectures = None
        cfg.is_decoder = True
        assert _is_encoder_model(_BareModel(cfg)) is False

    def test_model_takes_the_mean_pooling_path(self):
        cfg = _Cfg()
        cfg.architectures = ["ModernBertModel"]
        model = DecisionModel(
            backbone=_BareModel(cfg),
            tokenizer=object(),
            rank=4,
            gpu_config=GPUConfig(False, False, 64, False),
        )
        assert model.is_encoder is True
