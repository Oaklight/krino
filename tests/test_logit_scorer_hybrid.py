"""The logit scorer must build a cache the backbone can actually use.

A GDN hybrid (Qwen3.5) reads a linear-attention state from the cache on its very
first forward. A config-less ``DynamicCache`` allocates none, so scoring raises
``IndexError`` in ``get_conv_state`` before the first item is answered — which
is what happened when the Qwen3.5 benchmark sweep was first run.

A tiny random Qwen3.5 text model runs on CPU in float32, so this is a real
end-to-end check of scoring rather than a mock of the cache.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
qwen35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from model.src.backbone import is_hybrid_model  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402

VOCAB = 128


class _Ids:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _CharTokenizer:
    """One token per byte, with a fixed pad id and no special tokens."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [2 + (byte % (VOCAB - 4)) for byte in text.encode()] or [2]
        return _Ids(torch.tensor([ids], dtype=torch.long))


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
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    return qwen35.Qwen3_5ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def scorer():
    model = _tiny_model()
    return LogitScorer(
        model, _CharTokenizer(), device="cpu", norm="mean", strategy="description"
    )


def test_the_tiny_model_is_hybrid(scorer):
    assert scorer.is_hybrid


def test_a_hybrid_model_scores_options(scorer):
    """The end-to-end path: no exception, one finite score per option."""
    scores = scorer.score_options("State: a b c\nQuestion: pick\nAnswer:", [" first", " second"])
    assert len(scores) == 2
    assert all(isinstance(value, float) and value == value for value in scores)


def test_the_scores_depend_on_the_context(scorer):
    """A scorer that returned a constant would pass the previous test."""
    first = scorer.score_options("State: alpha\nQuestion: pick\nAnswer:", [" first", " second"])
    second = scorer.score_options("State: omega\nQuestion: pick\nAnswer:", [" first", " second"])
    assert first != second


def test_the_hybrid_branch_is_what_makes_it_work(scorer):
    """Mutation check: without it, the first forward raises ``IndexError``.

    Flipping ``is_hybrid`` off makes the scorer build the config-less cache and
    take the dense repeat path — exactly the code that failed on Qwen3.5.
    """
    original = scorer.is_hybrid
    scorer.is_hybrid = False
    try:
        with pytest.raises(IndexError):
            scorer.score_options("State: a b c\nQuestion: pick\nAnswer:", [" first", " second"])
    finally:
        scorer.is_hybrid = original


def test_the_hybrid_detector_sees_the_tiny_model(scorer):
    assert is_hybrid_model(scorer.model.config)
