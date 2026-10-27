"""Tests for LoRA adapters and multi-layer readout."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.src.heads import LayerMixer
from model.src.lora import LoRALinear, inject_lora, lora_parameters


class _Block(nn.Module):
    """Stand-in for a transformer block, named like a real backbone's."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)


class _Backbone(nn.Module):
    """Minimal backbone exposing ``layers.N.<linear>`` module names."""

    def __init__(self, n_layers: int = 4, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim) for _ in range(n_layers)])


class TestLoRALinear:
    def test_identity_at_initialisation(self) -> None:
        base = nn.Linear(8, 8)
        lora = LoRALinear(base, rank=4, alpha=8)
        x = torch.randn(3, 8)
        # lora_b starts at zero, so the adapter must not change the output.
        assert torch.equal(lora(x), base(x))

    def test_base_is_frozen_and_factors_train(self) -> None:
        lora = LoRALinear(nn.Linear(8, 8), rank=4)
        assert not lora.base.weight.requires_grad
        assert lora.lora_a.weight.requires_grad
        assert lora.lora_b.weight.requires_grad

    def test_gradients_reach_adapters_only(self) -> None:
        base = nn.Linear(8, 8)
        lora = LoRALinear(base, rank=4)
        lora(torch.randn(3, 8)).sum().backward()
        assert lora.lora_a.weight.grad is not None
        assert lora.lora_b.weight.grad is not None
        assert base.weight.grad is None

    def test_scaling_is_alpha_over_rank(self) -> None:
        lora = LoRALinear(nn.Linear(8, 8), rank=4, alpha=8)
        assert lora.scaling == pytest.approx(2.0)


class TestInjectLora:
    def test_adapts_every_matching_layer(self) -> None:
        backbone = _Backbone(n_layers=4)
        adapted = inject_lora(backbone, targets=("q_proj", "v_proj"))
        assert adapted == 8  # 4 blocks x 2 targets
        assert isinstance(backbone.layers[0].q_proj, LoRALinear)
        assert isinstance(backbone.layers[3].v_proj, LoRALinear)
        # Untargeted layers keep their original type.
        assert isinstance(backbone.layers[0].mlp, nn.Linear)
        assert not isinstance(backbone.layers[0].mlp, LoRALinear)

    def test_last_n_layers_only(self) -> None:
        backbone = _Backbone(n_layers=4)
        adapted = inject_lora(
            backbone, targets=("q_proj",), last_n_layers=2
        )
        assert adapted == 2
        assert not isinstance(backbone.layers[1].q_proj, LoRALinear)
        assert isinstance(backbone.layers[2].q_proj, LoRALinear)
        assert isinstance(backbone.layers[3].q_proj, LoRALinear)

    def test_unmatched_targets_raise(self) -> None:
        # Silently adapting nothing would look like a working run that
        # trains no backbone parameters at all.
        with pytest.raises(ValueError, match="No linear layers match"):
            inject_lora(_Backbone(), targets=("nonexistent",))

    def test_lora_parameters_lists_only_adapters(self) -> None:
        backbone = _Backbone(n_layers=2)
        inject_lora(backbone, targets=("q_proj",))
        params = lora_parameters(backbone)
        assert len(params) == 4  # 2 blocks x (lora_a + lora_b)


class TestLayerMixer:
    def _hidden_states(self) -> tuple[torch.Tensor, ...]:
        # Three layers, each a constant tensor, so mixtures are predictable.
        return tuple(
            torch.full((1, 2, 3), float(i)) for i in range(3)
        )

    def test_last_returns_final_layer(self) -> None:
        mixer = LayerMixer(3, "last")
        out = mixer(self._hidden_states())
        assert torch.equal(out, self._hidden_states()[-1])

    def test_mean_last_n(self) -> None:
        mixer = LayerMixer(3, "mean_last_n", n=2)
        out = mixer(self._hidden_states())
        expected = (self._hidden_states()[1] + self._hidden_states()[2]) / 2
        assert torch.allclose(out, expected)

    def test_weighted_starts_near_last_layer(self) -> None:
        mixer = LayerMixer(3, "weighted")
        out = mixer(self._hidden_states())
        last = self._hidden_states()[-1]
        # Initialised with a strong bias to the final layer; every value must
        # sit within a small distance of it rather than midway between layers.
        assert torch.allclose(out, last, atol=0.5)

    def test_weighted_is_trainable(self) -> None:
        mixer = LayerMixer(3, "weighted")
        mixer(self._hidden_states()).sum().backward()
        assert mixer.layer_logits.grad is not None

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown readout mode"):
            LayerMixer(3, "bogus")

    def test_mean_window_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            LayerMixer(3, "mean_last_n", n=5)
