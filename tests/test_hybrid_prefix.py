"""Shared-prefix scoring on a hybrid (GDN) cache matches one forward per option.

A tiny random Qwen3.5 text model runs on CPU in float32, so the batched,
left-padded, forked-cache path can be compared with the per-option path to
float tolerance.  More options than rows forces several groups over the same
prefix, which is where an in-place recurrent update would leak.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
qwen35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from model.scripts.eval_gate import RawReranker  # noqa: E402
from model.src.backbone import is_hybrid_model  # noqa: E402
from model.src.hybrid_cache import fork_prefix_cache  # noqa: E402

VOCAB = 128


class _CharTokenizer:
    """One token per byte, with ``yes``/``no`` as their own ids."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (b % (VOCAB - 4)) for b in text.encode()]

    def convert_tokens_to_ids(self, token):
        return {"yes": VOCAB - 1, "no": VOCAB - 2}[token]


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


def _reranker(model):
    scorer = RawReranker.__new__(RawReranker)
    scorer.device = "cpu"
    scorer.tokenizer = _CharTokenizer()
    scorer.model = model
    scorer.lm_head = model.lm_head
    scorer.is_hybrid = is_hybrid_model(model.config)
    scorer.true_id = scorer.tokenizer.convert_tokens_to_ids("yes")
    scorer.false_id = scorer.tokenizer.convert_tokens_to_ids("no")
    scorer.prefix_tokens = scorer.tokenizer.encode("<sys>")
    scorer.suffix_tokens = scorer.tokenizer.encode("</s>")
    scorer.max_length = 1024
    scorer.shared_prefix = True
    return scorer


@pytest.fixture(scope="module")
def scorer():
    return _reranker(_tiny_model())


def test_the_tiny_model_is_hybrid(scorer):
    assert scorer.is_hybrid


def test_shared_prefix_matches_one_forward_per_option(scorer):
    # Ten options over eight rows: two groups share one prefix cache.
    options = [(chr(ord("A") + i), "option text " * (i + 1)) for i in range(10)]
    state, instructions = "a fairly long state " * 6, "pick one"
    expected = [
        scorer._score_option(state, instructions, "choice", label, text)
        for label, text in options
    ]
    actual = scorer.score_many(state, instructions, "choice", options)
    assert actual == pytest.approx(expected, abs=1e-4)


def test_the_prefix_cache_is_not_mutated_by_a_fork(scorer):
    ids = torch.tensor([scorer.tokenizer.encode("some prefix text")])
    cache = scorer.model(input_ids=ids, use_cache=True).past_key_values
    before = [
        {k: v.clone() for k, v in layer.recurrent_states.items() if v is not None}
        for layer in cache.layers
        if isinstance(getattr(layer, "recurrent_states", None), dict)
    ]
    fork = fork_prefix_cache(cache, 3)
    rows = torch.tensor([scorer.tokenizer.encode("suffix")] * 3)
    scorer.model(input_ids=rows, past_key_values=fork, use_cache=True)
    after = [
        {k: v for k, v in layer.recurrent_states.items() if v is not None}
        for layer in cache.layers
        if isinstance(getattr(layer, "recurrent_states", None), dict)
    ]
    assert before and len(before) == len(after)
    for old, new in zip(before, after):
        for key in old:
            assert new[key].shape[0] == 1
            assert torch.equal(old[key], new[key])
