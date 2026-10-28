"""Tests for LoRA adapters and multi-layer readout."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

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

    def test_weighted_init_share_is_depth_independent(self) -> None:
        # A constant logit would make the starting share e^L/(e^L + N - 1),
        # i.e. 0.67 at 29 layers and 0.85 at 37 -- the "starts at the last
        # layer" property would silently depend on backbone depth.
        for n_layers in (2, 3, 29, 37):
            mixer = LayerMixer(n_layers, "weighted")
            share = torch.softmax(mixer.layer_logits, dim=0)[-1].item()
            assert share == pytest.approx(0.95, abs=1e-6), f"n_layers={n_layers}"

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

    def test_layer_returns_the_selected_layer(self) -> None:
        mixer = LayerMixer(3, "layer", layer=1)
        result = mixer(self._hidden_states())
        assert torch.equal(result, self._hidden_states()[1])
        assert mixer.out_mult == 1

    def test_layer_index_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            LayerMixer(3, "layer", layer=3)

    def test_fuse_pair_concat_fuses_width_and_out_mult(self) -> None:
        mixer = LayerMixer(3, "fuse_pair", pair=(0, 2), fusion="concat")
        result = mixer(self._hidden_states())
        expected = torch.cat(
            [self._hidden_states()[0], self._hidden_states()[2]], dim=-1
        )
        assert torch.equal(result, expected)
        assert result.shape[-1] == 6
        assert mixer.out_mult == 2

    def test_fuse_pair_sum_keeps_width(self) -> None:
        mixer = LayerMixer(3, "fuse_pair", pair=(0, 2), fusion="sum")
        result = mixer(self._hidden_states())
        expected = self._hidden_states()[0] + self._hidden_states()[2]
        assert torch.equal(result, expected)
        assert mixer.out_mult == 1

    def test_fuse_pair_needs_two_indices(self) -> None:
        with pytest.raises(ValueError, match="two layer indices"):
            LayerMixer(3, "fuse_pair", pair=None)
        with pytest.raises(ValueError, match="outside"):
            LayerMixer(3, "fuse_pair", pair=(0, 3))
        with pytest.raises(ValueError, match="fusion"):
            LayerMixer(3, "fuse_pair", pair=(0, 1), fusion="bogus")


class _MixedBackbone(nn.Module):
    """Matching layers under an indexed block, plus one outside any block."""

    def __init__(self, n_layers: int = 4, dim: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim) for _ in range(n_layers)])
        self.extra = _Block(dim)


class TestInjectLoraMixedNames:
    def test_unindexed_layers_do_not_lower_the_cutoff(self) -> None:
        # `extra.q_proj` matches the target but carries no block index.  It
        # must be dropped, not adapted and not allowed to drag max() down.
        backbone = _MixedBackbone(n_layers=4)
        adapted = inject_lora(backbone, targets=("q_proj",), last_n_layers=2)
        assert adapted == 2
        assert isinstance(backbone.layers[2].q_proj, LoRALinear)
        assert isinstance(backbone.layers[3].q_proj, LoRALinear)
        assert not isinstance(backbone.layers[1].q_proj, LoRALinear)
        assert not isinstance(backbone.extra.q_proj, LoRALinear)


class _FakeConfig:
    """Minimal backbone config, enough for DecisionModel to construct."""

    hidden_size = 16
    num_hidden_layers = 2
    max_position_embeddings = 128
    is_decoder = True


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.layers = nn.ModuleList([_Block(16) for _ in range(2)])


class TestCheckpointRoundTrip:
    """LoRA adapters must survive save_heads/load_heads.

    save_heads() filters out everything under ``backbone.``, which is where
    the adapters live.  Without the trainable-parameter exception a
    checkpoint loads cleanly under ``strict=False`` and silently drops every
    adapter, so the failure is invisible without a round-trip test.
    """

    def _model(self, **kwargs):
        from model.src.decision_model import DecisionModel
        from model.src.gpu_config import GPUConfig

        gpu = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=128,
            gradient_checkpointing=False,
        )
        return DecisionModel(
            backbone=_FakeBackbone(), tokenizer=None, rank=8, gpu_config=gpu, **kwargs
        )

    def test_lora_adapters_are_saved(self, tmp_path) -> None:
        model = self._model(lora_layers=-1)
        path = tmp_path / "heads.pt"
        model.save_heads(path)
        state = torch.load(path, weights_only=True)

        assert [key for key in state if "lora_" in key], (
            "LoRA adapters were dropped from the checkpoint"
        )
        # Frozen pretrained weights must still be excluded.
        assert not [key for key in state if "backbone" in key and "lora_" not in key]

    def test_lora_adapters_load_back(self, tmp_path) -> None:
        model = self._model(lora_layers=-1)
        path = tmp_path / "heads.pt"
        model.save_heads(path)

        fresh = self._model(lora_layers=-1)
        with torch.no_grad():
            for name, param in fresh.named_parameters():
                if "lora_b" in name:
                    param.fill_(1.0)

        fresh.load_heads(path)

        # The saved lora_b factors are zero at initialisation, so a successful
        # load must overwrite the 1.0s.
        for name, param in fresh.named_parameters():
            if "lora_b" in name:
                assert torch.count_nonzero(param) == 0, f"{name} did not load"


class _ForwardConfig:
    """Config for a backbone that can actually run a forward pass."""

    def __init__(self, hidden_size: int, num_hidden_layers: int = 2) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = 128
        self.is_decoder = True


class _ForwardBackbone(nn.Module):
    """Deterministic backbone returning per-layer hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = _ForwardConfig(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        use_cache: bool = False,
    ):
        from types import SimpleNamespace

        batch, seq = input_ids.shape
        base = torch.arange(seq, dtype=torch.float32)
        states = []
        for layer in range(self.config.num_hidden_layers + 1):
            value = base.view(1, seq, 1) + float(layer)
            states.append(value.expand(batch, seq, self.config.hidden_size).contiguous())
        return SimpleNamespace(hidden_states=tuple(states))


class TestTwoBackboneFusion:
    def _model(self, **kwargs):
        from model.src.decision_model import DecisionModel
        from model.src.gpu_config import GPUConfig

        gpu = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=128,
            gradient_checkpointing=False,
        )
        return DecisionModel(
            backbone=_ForwardBackbone(16),
            tokenizer=None,
            rank=8,
            gpu_config=gpu,
            **kwargs,
        )

    def test_concat_fusion_projects_to_head_width(self) -> None:
        model = self._model(
            second_backbone=_ForwardBackbone(8), fusion="concat"
        )
        inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        out = model._hidden_sequence(inputs)
        assert out.shape == (1, 4, 16)
        assert isinstance(model.fusion_project, nn.Linear)
        assert model.fusion_project.in_features == 24

    def test_fuse_pair_readout_round_trips_width(self) -> None:
        model = self._model(readout="fuse_pair", readout_pair=(0, 2))
        inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        out = model._hidden_sequence(inputs)
        assert out.shape == (1, 4, 16)
        # The two selected layers are concatenated before the projection.
        assert isinstance(model.readout_project, nn.Linear)
        assert model.readout_project.in_features == 32

    def test_sum_fusion_requires_equal_widths(self) -> None:
        with pytest.raises(ValueError, match="equal hidden sizes"):
            self._model(second_backbone=_ForwardBackbone(8), fusion="sum")

