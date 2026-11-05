"""Marker readout on a hybrid (GDN) cache matches one forward per option.

A Qwen3.5 cache holds gated-delta-net layers whose conv and recurrent states
are overwritten in place, so the full-attention trick of repeating the prefix
cache and cropping the suffix back does not apply.  ``DecisionModel`` forks
the one-row prefix cache per row group instead.  A tiny random model on CPU
in float32 makes the batched and sequential paths comparable to float
tolerance; more options than rows forces several groups over the same prefix,
which is where an in-place state update would leak.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
qwen35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from model.src.decision_model import DecisionModel  # noqa: E402
from model.src.gpu_config import GPUConfig  # noqa: E402

VOCAB = 128
MARKER_ID = 1


class _Encoding(dict):
    def to(self, device):
        return self


class _CharTokenizer:
    """One token per byte, with ``marker`` as its own id."""

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


def _decision_model(
    option_batch: int, chunk_size: int = 1, noul_marker: bool = False
) -> DecisionModel:
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
        option_chunk_size=chunk_size,
        option_batch=option_batch,
        noul_marker=noul_marker,
    )
    # The marker head carries dropout; evaluation must be deterministic.
    return model.eval()


def _no_cache_logits(model: DecisionModel, state, instructions, options):
    """Exact readout: re-encode the whole context for every option chunk."""
    option_ids = model._tokenize_option_ids(options)
    chunks = model._chunk_option_ids(option_ids)
    max_suffix = max(sum(len(ids) + 1 for ids in chunk) for chunk in chunks)
    prefix_inputs = model._tokenize_context(
        state, instructions, model.max_length - max_suffix
    )
    with torch.no_grad():
        return model._forward_options_no_cache(prefix_inputs, chunks, warn=False)


@pytest.fixture(scope="module")
def model() -> DecisionModel:
    # chunk=1 is the only exact setting: a multi-option chunk lets an option
    # see the options before it in the same sequence.
    return _decision_model(option_batch=1, chunk_size=1)


@pytest.fixture(scope="module")
def sample():
    state = "a fairly long state " * 6
    instructions = "pick one"
    # Ten options over fewer rows: several groups share one prefix cache.
    options = [(chr(ord("A") + i), "option text " * (i + 1)) for i in range(10)]
    return state, instructions, options


def test_the_tiny_model_is_hybrid(model):
    assert model.is_hybrid


def test_sequential_marker_matches_the_no_cache_readout(model, sample):
    state, instructions, options = sample
    expected = _no_cache_logits(model, state, instructions, options)
    with torch.no_grad():
        actual = model.forward_choice(state, instructions, options)
    assert actual.shape == (1, len(options))
    assert actual[0].tolist() == pytest.approx(expected[0].tolist(), abs=1e-4)


def test_batched_marker_matches_the_sequential_path(model, sample):
    state, instructions, options = sample
    model.option_batch = 1
    with torch.no_grad():
        sequential = model.forward_choice(state, instructions, options)
    model.option_batch = 3
    try:
        with torch.no_grad():
            batched = model.forward_choice(state, instructions, options)
    finally:
        model.option_batch = 1
    assert batched.shape == sequential.shape
    assert batched[0].tolist() == pytest.approx(sequential[0].tolist(), abs=1e-4)


def test_batched_marker_matches_the_no_cache_readout(model, sample):
    state, instructions, options = sample
    expected = _no_cache_logits(model, state, instructions, options)
    with torch.no_grad():
        batched = model._forward_options_batched(
            state, instructions, options, group_size=3
        )
    assert batched[0].tolist() == pytest.approx(expected[0].tolist(), abs=1e-4)


def test_a_single_row_fork_matches_too(model, sample):
    state, instructions, options = sample
    expected = _no_cache_logits(model, state, instructions, options)
    with torch.no_grad():
        actual = model._forward_options_batched(
            state, instructions, options, group_size=1
        )
    assert actual[0].tolist() == pytest.approx(expected[0].tolist(), abs=1e-4)


def test_the_cache_path_is_used_not_the_re_encode_fallback(model, sample, monkeypatch):
    """A hybrid cache must fork, not fall back to re-encoding the prefix."""
    state, instructions, options = sample

    def boom(*args, **kwargs):
        raise AssertionError("hybrid marker path re-encoded the prefix")

    monkeypatch.setattr(model, "_forward_options_no_cache", boom)
    model.option_batch = 1
    with torch.no_grad():
        model.forward_choice(state, instructions, options)
    model.option_batch = 3
    try:
        with torch.no_grad():
            model.forward_choice(state, instructions, options)
    finally:
        model.option_batch = 1


def test_repeated_calls_do_not_drift(model, sample):
    state, instructions, options = sample
    model.option_batch = 2
    try:
        with torch.no_grad():
            first = model.forward_choice(state, instructions, options)
            second = model.forward_choice(state, instructions, options)
    finally:
        model.option_batch = 1
    assert torch.equal(first, second)


def test_noul_marker_reads_through_the_same_fork():
    """The noul marker readout is the two-option case of the same path."""
    model = _decision_model(option_batch=2, noul_marker=True)
    state, instructions = "a fairly long state " * 4, "is this true"
    texts = ["false", "true"]
    with torch.no_grad():
        through_noul = model.forward_noul(state, instructions)
        through_choice = model._forward_options_by_marker(
            state, instructions, texts
        )
    expected = through_choice[:, 1:2] - through_choice[:, 0:1]
    assert through_noul.shape == (1, 1)
    assert through_noul[0].tolist() == pytest.approx(expected[0].tolist(), abs=1e-4)
