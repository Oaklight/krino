"""Tests for effective max_length clamping in DecisionModel.

`max_length` is resolved in two stages: the CLI/config request, then a clamp
to the GPU's safe length and the backbone's native maximum.  The startup
banner reports the clamped value, so the clamping itself has to be pinned --
an explicit request above the backbone's positional-embedding limit would
otherwise overflow the position table at forward time.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig


class _Config:
    hidden_size = 8
    num_hidden_layers = 1
    max_position_embeddings = 4096
    is_decoder = True


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.layers = nn.ModuleList([nn.Linear(8, 8)])


def _model(max_length: int | None) -> DecisionModel:
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=8192,
        gradient_checkpointing=False,
    )
    return DecisionModel(
        backbone=_Backbone(),
        tokenizer=None,
        rank=4,
        gpu_config=gpu,
        max_length=max_length,
    )


class TestMaxLengthClamp:
    def test_explicit_length_above_backbone_capacity_is_clamped(self) -> None:
        assert _model(32768).max_length == 4096

    def test_explicit_length_within_capacity_is_kept(self) -> None:
        assert _model(2048).max_length == 2048

    def test_auto_uses_the_smaller_of_gpu_and_backbone(self) -> None:
        # gpu_config says 8192; the backbone caps at 4096.
        assert _model(None).max_length == 4096
