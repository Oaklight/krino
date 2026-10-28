"""Tests for the option-marker readout.

The readout's expensive part is the backbone, so these tests use a stub
backbone that returns deterministic hidden states and a fake KV cache.  That
lets the tests assert the properties that matter without downloading a model:

- one prefix forward and ``ceil(K / chunk_size)`` suffix forwards;
- every suffix starts from the same prefix (the cache is cropped);
- the marker positions point at the marker token, not at option text;
- 256 options produce a 256-way distribution without building one
  context-length sequence per option.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig
from model.src.heads import OptionMarkerHead

MARKER_ID = 2


class _Encoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    """Whitespace-free character tokenizer with a fixed marker id."""

    eos_token_id = MARKER_ID

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
        ids = [3 + (ord(ch) % 5) for ch in str(text)]
        if max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return _Encoding(
                input_ids=torch.tensor([ids], dtype=torch.long),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            )
        return {"input_ids": ids}


class _FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, n: int) -> None:
        # transformers 5.x uses a negative count to remove tokens.
        self.length = self.length + n if n < 0 else n


class _Output:
    def __init__(self, hidden_states, past_key_values):
        self.hidden_states = hidden_states
        self.past_key_values = past_key_values


class _Config:
    hidden_size = 16
    num_hidden_layers = 2
    max_position_embeddings = 512
    is_decoder = True


class _MarkerBackbone(nn.Module):
    """Stub causal backbone: hidden states depend only on input ids."""

    def __init__(self, return_cache: bool = True) -> None:
        super().__init__()
        self.config = _Config()
        self.embed = nn.Embedding(100, self.config.hidden_size)
        self.return_cache = return_cache
        self.calls: list[tuple[str, int]] = []
        self.cache_lengths: list[int] = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_hidden_states=True,
        use_cache=False,
        **kwargs,
    ):
        kind = "prefix" if use_cache else "suffix"
        self.calls.append((kind, input_ids.shape[1]))
        hidden = self.embed(input_ids)
        hidden_states = (
            tuple(hidden for _ in range(self.config.num_hidden_layers + 1))
            if output_hidden_states
            else None
        )
        cache = past_key_values
        if use_cache:
            cache = _FakeCache(input_ids.shape[1]) if self.return_cache else None
        elif cache is not None:
            self.cache_lengths.append(cache.length)
            cache.length += input_ids.shape[1]
        return _Output(hidden_states, cache)


def _build_model(**kwargs) -> DecisionModel:
    gpu = GPUConfig(
        flash_attention=False,
        bf16_compute=False,
        max_length=512,
        gradient_checkpointing=False,
    )
    chunk_size = kwargs.pop("option_chunk_size", 32)
    return DecisionModel(
        backbone=_MarkerBackbone(**kwargs.pop("backbone_kwargs", {})),
        tokenizer=_FakeTokenizer(),
        rank=8,
        gpu_config=gpu,
        option_marker=True,
        option_chunk_size=chunk_size,
        **kwargs,
    )


class TestOptionMarkerHead:
    def test_shape_and_mask(self) -> None:
        head = OptionMarkerHead(16, rank=8)
        hidden = torch.randn(1, 4, 16)
        logits = head(hidden)
        assert logits.shape == (1, 4)

        masked = head(hidden, torch.tensor([[True, True, False, True]]))
        assert masked[0, 2].item() == float("-inf")
        assert torch.isfinite(masked[0, 0])

    def test_gradients_reach_every_parameter(self) -> None:
        head = OptionMarkerHead(16, rank=8)
        hidden = torch.randn(1, 3, 16, requires_grad=True)
        head(hidden).sum().backward()
        assert hidden.grad is not None
        for param in head.parameters():
            assert param.grad is not None

    def test_param_count_matches_attention_head(self) -> None:
        head = OptionMarkerHead(1024, rank=64)
        hidden = 64 * 3
        expected = (
            2 * 1024  # LayerNorm
            + 1024 * hidden + hidden  # scorer first layer
            + hidden + 1  # scorer second layer
        )
        assert sum(p.numel() for p in head.parameters()) == expected
        # AttentionHead(1024, rank=64) is 200,704 params; the marker scorer
        # is within one percent of it so head capacity is not confounded.
        assert abs(expected - 200_704) / 200_704 < 0.01


class TestChunking:
    def test_marker_positions_point_at_marker_tokens(self) -> None:
        model = _build_model()
        chunk = [[10, 11], [12], [13, 14, 15]]
        ids, positions = model._build_marker_suffix(chunk)
        assert ids.shape == (1, 9)
        assert positions.tolist() == [2, 4, 8]
        assert ids[0, positions].tolist() == [MARKER_ID] * 3

    def test_chunks_respect_size(self) -> None:
        model = _build_model(option_chunk_size=4)
        chunks = model._chunk_option_ids([[i] for i in range(10)])
        assert [len(c) for c in chunks] == [4, 4, 2]

    def test_256_options_use_prefix_plus_eight_suffix_forwards(self) -> None:
        model = _build_model()
        options = [f"o{i}" for i in range(256)]
        logits = model.forward_choice("state", "instructions", options)
        assert logits.shape == (1, 256)
        kinds = [kind for kind, _ in model.backbone.calls]
        assert kinds.count("prefix") == 1
        assert kinds.count("suffix") == math.ceil(256 / 32)

    def test_cache_is_cropped_so_chunks_share_the_prefix(self) -> None:
        model = _build_model()
        options = [f"o{i}" for i in range(70)]
        model.forward_choice("state", "instructions", options)
        assert model.backbone.cache_lengths == [
            # Every suffix forward starts from the same prefix length.
            model.backbone.calls[0][1]
        ] * math.ceil(70 / 32)

    def test_missing_cache_falls_back_to_reencoding(self) -> None:
        model = _build_model(backbone_kwargs={"return_cache": False})
        options = [f"o{i}" for i in range(40)]
        logits = model.forward_choice("state", "instructions", options)
        assert logits.shape == (1, 40)
        # One discarded prefix attempt plus one full-sequence forward per chunk.
        assert len(model.backbone.calls) == 1 + math.ceil(40 / 32)

    def test_suffix_larger_than_max_length_raises(self) -> None:
        model = _build_model()
        model.max_length = 20
        with pytest.raises(ValueError, match="option_marker needs room"):
            model.forward_choice("state", "instructions", ["x" * 40] * 4)

    def test_unified_and_marker_are_mutually_exclusive(self) -> None:
        gpu = GPUConfig(False, False, 512, False)
        with pytest.raises(ValueError, match="mutually exclusive"):
            DecisionModel(
                backbone=_MarkerBackbone(),
                tokenizer=_FakeTokenizer(),
                rank=8,
                gpu_config=gpu,
                option_marker=True,
                unified_head=True,
            )

    def test_context_truncation_keeps_instructions(self) -> None:
        model = _build_model()
        state = "state " * 100
        instructions = "which intent"
        inputs = model._tokenize_context(state, instructions, max_length=20)
        assert inputs["input_ids"].shape[1] == 20
        instruction_ids = _FakeTokenizer()(" " + instructions)["input_ids"]
        assert inputs["input_ids"][0, -len(instruction_ids):].tolist() == instruction_ids

    def test_context_truncation_with_no_state_room(self) -> None:
        model = _build_model()
        instructions = "a" * 40
        inputs = model._tokenize_context("state", instructions, max_length=20)
        assert inputs["input_ids"].shape[1] == 20

    def test_softmax_is_a_distribution(self) -> None:
        model = _build_model()
        logits = model.forward_choice(
            "state", "instructions", [f"o{i}" for i in range(256)]
        )
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


class TestExistingReadoutUnchanged:
    def test_default_model_has_no_marker_head(self) -> None:
        gpu = GPUConfig(False, False, 512, False)
        model = DecisionModel(
            backbone=_MarkerBackbone(),
            tokenizer=_FakeTokenizer(),
            rank=8,
            gpu_config=gpu,
        )
        assert model.option_marker is False
        assert not hasattr(model, "marker_head")
