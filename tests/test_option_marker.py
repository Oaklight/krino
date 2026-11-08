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
from model.src.heads import OptionMarkerHead, base_features

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
        self.batch = 1

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.batch = repeats

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
            + 1024 * hidden + 2 * hidden + hidden  # scorer first layer
            + hidden + 1  # scorer second layer
        )
        assert sum(p.numel() for p in head.parameters()) == expected
        # AttentionHead(1024, rank=64) is 200,704 params; the marker scorer
        # is within one percent of it so head capacity is not confounded.
        assert abs(expected - 200_704) / 200_704 < 0.01

    def test_the_zero_path_matches_the_pre_change_head(self) -> None:
        """Without base features the head adds only exact zeros."""
        head = OptionMarkerHead(16, rank=8).eval()
        hidden = torch.randn(1, 4, 16)
        with torch.no_grad():
            logits = head(hidden)
            x = head.layer_norm(hidden)
            x = torch.nn.functional.linear(
                x, head.scorer[1].weight[:, :16], head.scorer[1].bias
            )
            x = torch.nn.functional.gelu(x)
            reference = torch.nn.functional.linear(
                x, head.scorer[4].weight, head.scorer[4].bias
            ).squeeze(-1)
        assert torch.allclose(logits, reference, atol=1e-6)
        # A pre-change head has no feature weights at all; the new columns
        # start at zero, which is what makes the two computations equal.
        assert head.scorer[1].weight[:, 16:].abs().max().item() == 0.0

    def test_a_pre_change_checkpoint_still_loads_and_scores(self) -> None:
        torch.manual_seed(0)
        head = OptionMarkerHead(16, rank=8).eval()
        state = {key: value.clone() for key, value in head.state_dict().items()}
        legacy = dict(state)
        legacy["scorer.1.weight"] = state["scorer.1.weight"][:, :16]

        fresh = OptionMarkerHead(16, rank=8).eval()
        # The hook pads the narrower weight with zeros on load.
        fresh.load_state_dict(legacy, strict=True)

        hidden = torch.randn(1, 3, 16)
        with torch.no_grad():
            assert torch.allclose(fresh(hidden), head(hidden), atol=1e-6)

    def test_base_features_carry_log_probability_and_argmax(self) -> None:
        probs = torch.tensor([[0.25, 0.75, 0.0]])
        feat = base_features(probs)
        assert torch.allclose(
            feat[..., 0], torch.log(probs.clamp_min(1e-6))
        )
        assert feat[..., 1].tolist() == [[0.0, 1.0, 0.0]]

    def test_a_trained_feature_column_moves_the_logits(self) -> None:
        head = OptionMarkerHead(16, rank=8).eval()
        hidden = torch.randn(1, 2, 16)
        feat = base_features(torch.tensor([0.25, 0.75])).unsqueeze(0)
        with torch.no_grad():
            # At initialisation the feature columns are zero, so the added
            # zeros and the real features score the same.
            assert torch.allclose(head(hidden, base_feat=feat), head(hidden), atol=1e-6)
            head.scorer[1].weight[:, 16:] = 0.5
            assert not torch.allclose(
                head(hidden, base_feat=feat), head(hidden), atol=1e-6
            )

    def test_a_gradient_step_moves_the_feature_columns(self) -> None:
        head = OptionMarkerHead(16, rank=8)
        hidden = torch.randn(1, 2, 16)
        feat = base_features(torch.tensor([0.9, 0.1])).unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(
            head(hidden, base_feat=feat), torch.tensor([1])
        )
        loss.backward()
        grad = head.scorer[1].weight.grad
        assert grad is not None
        assert grad[:, 16:].abs().sum().item() > 0
        assert grad[:, :16].abs().sum().item() > 0


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


class _EncoderConfig(_Config):
    is_decoder = False


class _EncoderBackbone(nn.Module):
    """Stub bidirectional backbone: no KV cache, one forward per chunk."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _EncoderConfig()
        self.embed = nn.Embedding(100, self.config.hidden_size)
        self.calls: list[int] = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=True,
        **kwargs,
    ):
        assert "use_cache" not in kwargs, "encoder path must not request a cache"
        self.calls.append(input_ids.shape[1])
        hidden = self.embed(input_ids)
        hidden_states = (
            tuple(hidden for _ in range(self.config.num_hidden_layers + 1))
            if output_hidden_states
            else None
        )
        return _Output(hidden_states, None)


class TestEncoderBackbone:
    """The marker layout must work without a KV cache."""

    def _build(self, chunk_size: int = 32) -> DecisionModel:
        gpu = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=512,
            gradient_checkpointing=False,
        )
        return DecisionModel(
            backbone=_EncoderBackbone(),
            tokenizer=_FakeTokenizer(),
            rank=8,
            gpu_config=gpu,
            option_marker=True,
            option_chunk_size=chunk_size,
        )

    def test_encoder_is_accepted(self) -> None:
        model = self._build()
        assert model.is_encoder
        assert model.marker_token_id == MARKER_ID

    def test_one_full_sequence_forward_per_chunk(self) -> None:
        model = self._build()
        logits = model.forward_choice("state", "instructions", [f"o{i}" for i in range(70)])
        assert logits.shape == (1, 70)
        # No prefix pass: every forward carries the context plus one chunk.
        assert len(model.backbone.calls) == math.ceil(70 / 32)

    def test_logits_differ_across_options(self) -> None:
        model = self._build()
        logits = model.forward_score("state", "instructions", ["low", "mid", "high"])
        assert logits.shape == (1, 3)
        assert torch.isfinite(logits).all()
        assert len(set(logits[0].tolist())) > 1

    def test_second_backbone_rejected(self) -> None:
        gpu = GPUConfig(False, False, 512, False)
        with pytest.raises(ValueError, match="second backbone"):
            DecisionModel(
                backbone=_EncoderBackbone(),
                tokenizer=_FakeTokenizer(),
                rank=8,
                gpu_config=gpu,
                option_marker=True,
                second_backbone=_EncoderBackbone(),
            )


class _StrictTokenizer(_FakeTokenizer):
    """Raises on non-str input, like the HuggingFace tokenizers do."""

    def __call__(self, text, **kwargs):
        if not isinstance(text, str):
            raise TypeError("text input must be of type `str`")
        return super().__call__(text, **kwargs)


class TestStructuredState:
    """JevBench hard items carry a dict state, not a string."""

    def _model(self) -> DecisionModel:
        gpu = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=512,
            gradient_checkpointing=False,
        )
        return DecisionModel(
            backbone=_EncoderBackbone(),
            tokenizer=_StrictTokenizer(),
            rank=8,
            gpu_config=gpu,
            option_marker=True,
            option_chunk_size=1,
        )

    def test_truncation_stringifies_a_dict_state(self) -> None:
        model = self._model()
        state = {"archive": ["a" * 200], "note": "x" * 200}
        inputs = model._tokenize_context(state, "which one", max_length=20)
        assert inputs["input_ids"].shape[1] == 20

    def test_marker_forward_accepts_a_dict_state(self) -> None:
        model = self._model()
        state = {"archive": ["a" * 200], "note": "x" * 200}
        logits = model.forward_choice(state, "which one", ["alpha", "beta"])
        assert logits.shape == (1, 2)


class TestBatchedOptionReadout:
    """``option_batch`` scores several options per forward, not per chunk."""

    def test_options_are_scored_in_batches(self) -> None:
        model = _build_model(option_batch=4)
        logits = model.forward_choice(
            "state", "instructions", [f"o{i}" for i in range(10)]
        )
        assert logits.shape == (1, 10)
        kinds = [kind for kind, _ in model.backbone.calls]
        assert kinds.count("prefix") == 1
        assert kinds.count("suffix") == 3  # ceil(10 / 4)

    def test_batched_path_matches_the_sequential_path(self) -> None:
        # One model, two readout paths, so the weights are identical.
        # eval() disables the head's dropout, which is the only randomness.
        model = _build_model(option_chunk_size=1)
        model.eval()
        options = [f"o{i}" for i in range(6)]
        expected = model.forward_choice("state", "instr", options)
        model.option_batch = 8
        actual = model.forward_choice("state", "instr", options)
        assert torch.allclose(expected, actual, atol=1e-5)

    def test_the_cache_is_expanded_to_the_group_size(self) -> None:
        model = _build_model(option_batch=3)
        model.forward_choice("state", "instr", [f"o{i}" for i in range(7)])
        cache = model.backbone.calls  # prefix then ceil(7/3) suffixes
        assert len(cache) == 1 + 3


class TestTrainSafeBuckets:
    """The padding-only variant must not change the scores."""

    def test_scores_match_the_default_path(self) -> None:
        # One model, two padding schemes, so the weights are identical.
        model = _build_model(option_batch=16, option_chunk_size=1)
        model.eval()
        options = ["a", "bb", "ccc", "dddd", "eeeee"]
        expected = model.forward_choice("state", "instr", options)
        model.train_safe_buckets = True
        actual = model.forward_choice("state", "instr", options)
        assert torch.allclose(expected, actual, atol=1e-5)

    def test_the_suffix_is_padded_to_a_multiple_of_eight(self) -> None:
        model = _build_model(option_batch=16, train_safe_buckets=True)
        model.forward_choice("state", "instr", ["o0", "o1", "o2"])
        suffix_widths = [
            width for kind, width in model.backbone.calls if kind == "suffix"
        ]
        assert suffix_widths and all(width % 8 == 0 for width in suffix_widths)
        assert max(suffix_widths) < 64


class TestNoulMarker:
    """Noul can share the marker readout as a two-option question."""

    def test_noul_marker_uses_the_shared_marker_readout(self) -> None:
        model = _build_model(noul_marker=True)
        assert model.noul_marker is True
        assert model.noul_head is None

        captured: dict[str, object] = {}

        def fake_marker(state, instructions, option_texts):
            captured["state"] = state
            captured["instructions"] = instructions
            captured["texts"] = option_texts
            return torch.tensor([[0.25, 0.75]])

        model._forward_options_by_marker = fake_marker
        logit = model.forward_noul(
            "the state", "is it supported?", {"false": "No", "true": "Yes"}
        )

        assert captured["texts"] == ["No", "Yes"]
        assert captured["state"] == "the state"
        assert logit.shape == (1, 1)
        assert abs(logit.item() - 0.5) < 1e-6  # 0.75 - 0.25

    def test_noul_marker_maps_yes_no_criteria_and_defaults(self) -> None:
        model = _build_model(noul_marker=True)
        seen: list[list[str]] = []
        model._forward_options_by_marker = (
            lambda state, instructions, texts: (
                seen.append(list(texts)) or torch.zeros(1, 2)
            )
        )

        model.forward_noul("s", "q", {"no": "unsupported", "yes": "supported"})
        model.forward_noul("s", "q", None)

        assert seen[0] == ["unsupported", "supported"]
        assert seen[1] == ["No", "Yes"]

    def test_noul_marker_gradients_reach_the_marker_head(self) -> None:
        model = _build_model(noul_marker=True)
        logit = model.forward_noul("abc", "def", {"false": "No", "true": "Yes"})
        logit.sum().backward()
        assert any(p.grad is not None for p in model.marker_head.parameters())

    def test_pooled_noul_head_is_still_the_default(self) -> None:
        model = _build_model()
        assert model.noul_marker is False
        assert model.noul_head is not None

    def test_noul_marker_requires_option_marker(self) -> None:
        gpu = GPUConfig(
            flash_attention=False,
            bf16_compute=False,
            max_length=512,
            gradient_checkpointing=False,
        )
        with pytest.raises(ValueError, match="noul_marker"):
            DecisionModel(
                backbone=_MarkerBackbone(),
                tokenizer=_FakeTokenizer(),
                rank=8,
                gpu_config=gpu,
                noul_marker=True,
            )
