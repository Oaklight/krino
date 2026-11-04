"""Qwen3.5 backbone loading: text-only and multimodal (4B) paths.

The 4B family is ``Qwen3_5ForConditionalGeneration`` (multimodal), which
``AutoModelForCausalLM`` rejects.  These tests mock the transformers auto
classes so the loader's dispatch, text-tower extraction and config resolution
run offline and never touch the 4B weights.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from model.src import backbone as backbone_module
from model.src.backbone import (
    _extract_lm_head,
    _extract_text_backbone,
    _resolve_text_config,
    is_hybrid_model,
    load_causal_lm,
    load_qwen35_base,
    load_qwen35_base_with_head,
)
from model.src.decision_model import DecisionModel
from model.src.gpu_config import GPUConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Config:
    """Minimal stand-in for a transformers config object."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_text_config(self, *args, **kwargs):
        nested = getattr(self, "text_config", None)
        return nested if nested is not None else self


class _RaisingConfig(_Config):
    """A composite config whose ``get_text_config`` is ambiguous."""

    def get_text_config(self, *args, **kwargs):
        raise ValueError("Multiple valid text configs were found")


class _Tower(torch.nn.Module):
    """Text tower: returns hidden states and no vocabulary logits."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append(dict(kwargs))
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.config.hidden_size)
        return SimpleNamespace(
            hidden_states=(hidden,) * (self.config.num_hidden_layers + 1),
            last_hidden_state=hidden,
        )


class _Wrapper(torch.nn.Module):
    """A causal-LM or conditional-generation wrapper."""

    def __init__(self, model, config, lm_head=None):
        super().__init__()
        self.model = model
        self.config = config
        if lm_head is not None:
            self.lm_head = lm_head


class _FakeAutoClass:
    """Records ``from_pretrained`` calls and returns a canned object or raises."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def from_pretrained(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class _FakeTokenizerClass:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return _FakeTokenizer(name, **kwargs)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, name="", **kwargs):
        self.name = name
        self.kwargs = kwargs


def _install(monkeypatch, *, causal, image=None, concrete=None):
    monkeypatch.setattr(backbone_module, "AutoTokenizer", _FakeTokenizerClass)
    monkeypatch.setattr(backbone_module, "AutoModelForCausalLM", causal)
    monkeypatch.setattr(backbone_module, "AutoModelForImageTextToText", image)
    monkeypatch.setattr(backbone_module, "Qwen3_5ForConditionalGeneration", concrete)


def _text_config(**overrides):
    values = dict(
        hidden_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        vocab_size=248320,
        max_position_embeddings=64,
        model_type="qwen3_5_text",
    )
    values.update(overrides)
    return _Config(**values)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestQwen35Dispatch:
    def test_text_only_checkpoint_uses_auto_model_for_causal_lm(self, monkeypatch):
        config = _text_config(architectures=["Qwen3_5ForCausalLM"])
        tower = _Tower(config)
        head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        wrapper = _Wrapper(tower, config, lm_head=head)
        causal = _FakeAutoClass(result=wrapper)
        _install(monkeypatch, causal=causal)

        backbone, tokenizer = load_qwen35_base(
            "Qwen/Qwen3.5-0.8B-Base", device="cpu", dtype=torch.float32
        )

        assert backbone is tower
        assert tokenizer.name == "Qwen/Qwen3.5-0.8B-Base"
        assert len(causal.calls) == 1
        assert causal.calls[0][0] == "Qwen/Qwen3.5-0.8B-Base"
        # The public contract is a two-tuple; the head variant is opt-in.
        assert isinstance((backbone, tokenizer), tuple)
        with_head = load_qwen35_base_with_head(
            "Qwen/Qwen3.5-0.8B-Base", device="cpu", dtype=torch.float32
        )
        assert len(with_head) == 3

    def test_multimodal_checkpoint_falls_back_to_image_text_loader(self, monkeypatch):
        text_config = _text_config()
        full_config = _Config(
            hidden_size=4096,
            num_hidden_layers=99,
            vocab_size=7,
            text_config=text_config,
            architectures=["Qwen3_5ForConditionalGeneration"],
            model_type="qwen3_5",
        )
        tower = _Tower(text_config)
        multimodal = _Wrapper(tower, full_config)
        multimodal.language_model = tower
        head = torch.nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        wrapper = _Wrapper(multimodal, full_config, lm_head=head)
        causal = _FakeAutoClass(
            error=ValueError(
                "Unrecognized configuration class Qwen3_5Config for this kind of AutoModel"
            )
        )
        image = _FakeAutoClass(result=wrapper)
        _install(monkeypatch, causal=causal, image=image)

        backbone, _, lm_head = load_qwen35_base_with_head(
            "Qwen/Qwen3.5-4B", device="cpu", dtype=torch.float32
        )

        assert backbone is tower
        assert lm_head is head
        assert len(causal.calls) == 1
        assert len(image.calls) == 1

    def test_concrete_conditional_generation_class_is_the_last_fallback(
        self, monkeypatch
    ):
        text_config = _text_config()
        full_config = _Config(
            text_config=text_config,
            architectures=["Qwen3_5ForConditionalGeneration"],
            model_type="qwen3_5",
        )
        tower = _Tower(text_config)
        multimodal = _Wrapper(tower, full_config)
        multimodal.language_model = tower
        wrapper = _Wrapper(multimodal, full_config)
        causal = _FakeAutoClass(error=ValueError("Unrecognized configuration class"))
        image = _FakeAutoClass(error=ValueError("Unrecognized configuration class"))
        concrete = _FakeAutoClass(result=wrapper)
        _install(monkeypatch, causal=causal, image=image, concrete=concrete)

        backbone, _ = load_qwen35_base(
            "Qwen/Qwen3.5-4B", device="cpu", dtype=torch.float32
        )

        assert backbone is tower
        assert len(concrete.calls) == 1

    def test_all_loaders_failing_reports_every_attempt(self, monkeypatch):
        causal = _FakeAutoClass(error=ValueError("first failure"))
        image = _FakeAutoClass(error=ValueError("second failure"))
        _install(monkeypatch, causal=causal, image=image)

        with pytest.raises(RuntimeError, match="first failure.*second failure"):
            load_qwen35_base("Qwen/Qwen3.5-4B", device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Text-tower and head extraction
# ---------------------------------------------------------------------------


class TestTextTowerExtraction:
    def test_causal_lm_model_shape(self):
        tower = torch.nn.Linear(1, 1)
        wrapper = _Wrapper(tower, _Config())
        assert _extract_text_backbone(wrapper) is tower

    def test_conditional_generation_model_language_model_shape(self):
        tower = torch.nn.Linear(1, 1)
        multimodal = _Wrapper(tower, _Config())
        multimodal.language_model = tower
        wrapper = _Wrapper(multimodal, _Config())
        assert _extract_text_backbone(wrapper) is tower

    def test_nested_model_model_language_model_shape(self):
        tower = torch.nn.Linear(1, 1)
        inner = _Wrapper(tower, _Config())
        inner.language_model = tower
        outer = _Wrapper(inner, _Config())
        wrapper = _Wrapper(outer, _Config())
        assert _extract_text_backbone(wrapper) is tower

    def test_lm_head_comes_from_the_wrapper_when_the_tower_has_none(self):
        tower = torch.nn.Linear(1, 1)
        head = torch.nn.Linear(1, 1)
        multimodal = _Wrapper(tower, _Config())
        multimodal.language_model = tower
        wrapper = _Wrapper(multimodal, _Config(), lm_head=head)
        assert _extract_lm_head(wrapper, tower) is head

    def test_missing_lm_head_returns_none(self):
        tower = torch.nn.Linear(1, 1)
        wrapper = _Wrapper(tower, _Config())
        assert _extract_lm_head(wrapper, tower) is None

    def test_loader_returns_none_when_the_checkpoint_has_no_head(self, monkeypatch):
        config = _text_config()
        tower = _Tower(config)
        wrapper = _Wrapper(tower, config)
        _install(monkeypatch, causal=_FakeAutoClass(result=wrapper))

        _, _, lm_head = load_qwen35_base_with_head(
            "Qwen/Qwen3.5-0.8B-Base", device="cpu", dtype=torch.float32
        )

        assert lm_head is None


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


class TestTextConfigResolution:
    def test_nested_text_config_values_win(self):
        text_config = _text_config(
            hidden_size=2560,
            num_hidden_layers=36,
            vocab_size=248320,
            layer_types=["linear_attention", "full_attention"],
        )
        full_config = _Config(
            hidden_size=4096,
            num_hidden_layers=99,
            vocab_size=7,
            layer_types=["vision_attention"],
            text_config=text_config,
            architectures=["Qwen3_5ForConditionalGeneration"],
        )

        resolved = _resolve_text_config(full_config)

        assert resolved is text_config
        assert resolved.hidden_size == 2560
        assert resolved.num_hidden_layers == 36
        assert resolved.vocab_size == 248320
        assert resolved.layer_types == ["linear_attention", "full_attention"]

    def test_parent_causal_architecture_is_copied_to_a_bare_text_config(self):
        text_config = _text_config(architectures=["Qwen3_5TextModel"])
        full_config = _Config(
            text_config=text_config,
            architectures=["Qwen3_5ForConditionalGeneration"],
        )

        resolved = _resolve_text_config(full_config)

        assert resolved.architectures == ["Qwen3_5ForConditionalGeneration"]

    def test_explicit_text_config_is_used_when_get_text_config_raises(self):
        text_config = _text_config()
        full_config = _RaisingConfig(text_config=text_config)

        assert _resolve_text_config(full_config) is text_config

    def test_loader_attaches_the_text_config(self, monkeypatch):
        text_config = _text_config()
        full_config = _Config(
            hidden_size=4096,
            text_config=text_config,
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
        tower = _Tower(text_config)
        multimodal = _Wrapper(tower, full_config)
        multimodal.language_model = tower
        wrapper = _Wrapper(multimodal, full_config)
        _install(monkeypatch, causal=_FakeAutoClass(result=wrapper))

        backbone, _ = load_qwen35_base(
            "Qwen/Qwen3.5-4B", device="cpu", dtype=torch.float32
        )

        assert backbone.config is text_config
        assert backbone.config.hidden_size == 32
        assert backbone.config.num_hidden_layers == 2
        assert backbone.config.vocab_size == 248320
        assert backbone.config.layer_types == [
            "linear_attention",
            "full_attention",
        ]


# ---------------------------------------------------------------------------
# Hybrid detection
# ---------------------------------------------------------------------------


class TestHybridDetection:
    def test_text_only_config(self):
        assert is_hybrid_model(
            _Config(layer_types=["linear_attention", "full_attention"])
        )

    def test_multimodal_config_checks_the_nested_text_config(self):
        text_config = _Config(layer_types=["linear_attention", "full_attention"])
        assert is_hybrid_model(_Config(layer_types=None, text_config=text_config))

    def test_full_attention_only_is_not_hybrid(self):
        assert not is_hybrid_model(_Config(layer_types=["full_attention"]))

    def test_missing_layer_types_is_not_hybrid(self):
        assert not is_hybrid_model(_Config())


# ---------------------------------------------------------------------------
# Regression: the Qwen3 path is unchanged
# ---------------------------------------------------------------------------


class TestCausalLMRegression:
    def test_qwen3_still_goes_through_auto_model_for_causal_lm(self, monkeypatch):
        config = _Config(
            hidden_size=1024,
            num_hidden_layers=28,
            vocab_size=151669,
            max_position_embeddings=32768,
            model_type="qwen3",
            architectures=["Qwen3ForCausalLM"],
        )
        model = _Wrapper(torch.nn.Linear(1, 1), config)
        causal = _FakeAutoClass(result=model)
        _install(monkeypatch, causal=causal)

        loaded, tokenizer = load_causal_lm(
            "Qwen/Qwen3-0.6B", device="cpu", dtype=torch.float32
        )

        assert loaded is model
        assert tokenizer.name == "Qwen/Qwen3-0.6B"
        assert len(causal.calls) == 1

    def test_decision_model_uses_the_text_tower_without_an_lm_head(self):
        config = _text_config(
            hidden_size=16,
            num_hidden_layers=2,
            vocab_size=32,
            architectures=["Qwen3_5TextModel"],
        )
        tower = _Tower(config)

        model = DecisionModel(
            backbone=tower,
            tokenizer=object(),
            rank=4,
            gpu_config=GPUConfig(False, False, 64, False),
        )

        assert model.is_encoder is False
        assert model.is_hybrid is True
        model._backbone_forward({"input_ids": torch.zeros(1, 5, dtype=torch.long)})
        assert "logits_to_keep" not in tower.calls[-1]
        assert tower.calls[-1]["use_cache"] is False


# ---------------------------------------------------------------------------
# Raw reranker (eval_gate)
# ---------------------------------------------------------------------------


class _RerankerTokenizer:
    pad_token_id = 0

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def convert_tokens_to_ids(self, token):
        return {"no": 2, "yes": 3}.get(token, 0)

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class _ScoringTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config(
            hidden_size=4,
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            vocab_size=8,
            max_position_embeddings=64,
            model_type="qwen3_5_text",
        )
        self.calls: list[dict] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        )
        batch, seq = input_ids.shape
        return SimpleNamespace(
            last_hidden_state=torch.ones(batch, seq, self.config.hidden_size)
        )


class _CausalRerankerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config(layer_types=["full_attention"], hidden_size=4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.calls: list[dict] = []

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append(
            {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        )
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 8)
        logits[..., 3] = 1.0
        logits[..., 2] = -1.0
        return SimpleNamespace(logits=logits)


class TestRawRerankerQwen35:
    def test_hybrid_text_tower_scores_yes_no_logits(self, monkeypatch):
        eval_gate = pytest.importorskip("model.scripts.eval_gate")
        tower = _ScoringTower()
        lm_head = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            lm_head.weight.zero_()
            lm_head.weight[3] = 1.0
            lm_head.weight[2] = -1.0

        monkeypatch.setattr(eval_gate, "AutoTokenizer", _RerankerTokenizer)
        monkeypatch.setattr(
            eval_gate,
            "AutoModelForCausalLM",
            _FakeAutoClass(error=ValueError("Unrecognized configuration class")),
        )
        monkeypatch.setattr(
            eval_gate,
            "load_qwen35_base_with_head",
            lambda *args, **kwargs: (tower, _RerankerTokenizer(), lm_head),
        )

        reranker = eval_gate.RawReranker("Qwen/Qwen3.5-4B", "cpu")

        assert reranker.model is tower
        assert reranker.lm_head is lm_head
        assert reranker.is_hybrid is True
        score = reranker._score_option("state", "instr", "noul", "yes", "Yes")
        assert score == pytest.approx(0.99966466, rel=1e-5)
        assert set(tower.calls[-1]) == {"input_ids", "attention_mask"}

        scores = reranker.score_many(
            "state",
            "instr",
            "noul",
            [("no", "No"), ("yes", "Yes")],
        )
        assert len(scores) == 2
        for call in tower.calls:
            assert "past_key_values" not in call
            assert "use_cache" not in call

    def test_qwen3_causal_path_is_unchanged(self, monkeypatch):
        eval_gate = pytest.importorskip("model.scripts.eval_gate")
        model = _CausalRerankerModel()
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[3] = 1.0
            model.lm_head.weight[2] = -1.0
        monkeypatch.setattr(eval_gate, "AutoTokenizer", _RerankerTokenizer)
        monkeypatch.setattr(
            eval_gate, "AutoModelForCausalLM", _FakeAutoClass(result=model)
        )

        reranker = eval_gate.RawReranker("Qwen/Qwen3-Reranker-0.6B", "cpu")

        assert reranker.model is model
        assert reranker.lm_head is model.lm_head
        assert reranker.is_hybrid is False
        score = reranker._score_option("state", "instr", "noul", "yes", "Yes")
        assert score == pytest.approx(0.880797, rel=1e-5)
        assert set(model.calls[-1]) == {"input_ids", "attention_mask"}

    def test_unsupported_architecture_error_is_recognised(self):
        eval_gate = pytest.importorskip("model.scripts.eval_gate")
        assert eval_gate._is_unsupported_architecture_error(
            ValueError("Unrecognized configuration class Qwen3_5Config")
        )
        assert not eval_gate._is_unsupported_architecture_error(
            ValueError("CUDA out of memory")
        )


# ---------------------------------------------------------------------------
# Opt-in integration test (small model only; never the 4B weights)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("KRINO_QWEN35_INTEGRATION") != "1",
    reason="set KRINO_QWEN35_INTEGRATION=1 to run the small-model download",
)
def test_small_qwen35_checkpoint_loads():
    model_id = os.environ.get("KRINO_QWEN35_MODEL", "Qwen/Qwen3.5-0.8B-Base")
    assert "4B" not in model_id, "the integration test must never download 4B weights"

    backbone, tokenizer = load_qwen35_base(
        model_id, device="cpu", dtype=torch.float32
    )

    assert backbone.config.hidden_size > 0
    assert backbone.config.num_hidden_layers > 0
    assert tokenizer is not None
