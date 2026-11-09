"""The full-attention path is real, not a causal pass with a silent mask (#324).

A causal pass cannot let an earlier token see a later change; a full pass can.
The test uses a randomly initialised tiny Qwen3 so it runs in CI without the
4B checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from model.src.judgement_objective import run_backbone  # noqa: E402


def _tiny():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config).eval()


def test_a_later_change_reaches_an_earlier_token_only_under_the_full_mask():
    model = _tiny()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 4, 9, 8]])
    attention = torch.ones_like(a)

    with torch.no_grad():
        causal_a = run_backbone(model, a, attention, bidirectional=False)
        causal_b = run_backbone(model, b, attention, bidirectional=False)
        full_a = run_backbone(model, a, attention, bidirectional=True)
        full_b = run_backbone(model, b, attention, bidirectional=True)

    assert torch.equal(causal_a[0, 2], causal_b[0, 2])
    assert not torch.equal(full_a[0, 2], full_b[0, 2])


def test_padded_batch_matches_the_unpadded_sequence_under_the_full_mask():
    model = _tiny()
    short = torch.tensor([[1, 2, 3, 4]])
    long = torch.tensor([[5, 6, 7, 8, 9, 10]])
    padded = torch.zeros(2, 6, dtype=torch.long)
    padded[0, :4] = short[0]
    padded[1] = long[0]
    attention = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])

    with torch.no_grad():
        single = run_backbone(model, short, torch.ones_like(short), bidirectional=True)
        batched = run_backbone(model, padded, attention, bidirectional=True)

    assert torch.allclose(single[0, :4], batched[0, :4], atol=1e-5)
