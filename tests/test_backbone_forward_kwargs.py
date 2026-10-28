"""The backbone forward must pass ``logits_to_keep`` exactly where it is valid.

The vocabulary projection this avoids is invisible when it regresses: the model
stays numerically correct and only the memory peak comes back, so nothing short
of a long-context run on a large batch would notice. These tests pin the decision
instead, with a stub backbone that records the kwargs it was called with.

Two properties have to hold together:

- a causal backbone **with** a vocabulary head gets ``logits_to_keep=1``;
- a backbone without one — ``load_qwen35_base`` returns
  ``AutoModelForCausalLM(...).model``, which has no ``lm_head`` — must not, or the
  call raises instead of running;
- an encoder gets neither this kwarg nor ``use_cache``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace

from model.src.decision_model import DecisionModel


class _RecordingBackbone(torch.nn.Module):
    """Minimal stand-in that reports hidden states and records its kwargs."""

    def __init__(self, *, lm_head: bool, hidden: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, vocab_size=128)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []
        if lm_head:
            # A plain attribute is enough: the guard only asks whether the head
            # exists, matching a real CausalLM's ``lm_head`` submodule.
            self.lm_head = torch.nn.Linear(hidden, 128, bias=False)

    def forward(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs.get("input_ids")
        batch, seq = input_ids.shape
        hidden = torch.zeros(batch, seq, self.config.hidden_size)
        out = SimpleNamespace(hidden_states=[hidden] * 3, last_hidden_state=hidden)
        if getattr(self, "lm_head", None) is not None:
            keep = kwargs.get("logits_to_keep", seq)
            out.logits = torch.zeros(batch, keep if isinstance(keep, int) else seq, self.config.vocab_size)
        return out


class _StubTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", **kwargs):
        # 5 tokens per string, whatever it says
        rows = text if isinstance(text, list) else [text]
        return {"input_ids": torch.zeros(len(rows), 5, dtype=torch.long)}


def _model(backbone, *, is_encoder: bool) -> DecisionModel:
    model = DecisionModel(backbone, _StubTokenizer(), hidden_size=16, rank=8)
    model.is_encoder = is_encoder  # force the branch under test
    return model


def test_causal_backbone_with_a_vocab_head_gets_the_kwarg():
    backbone = _RecordingBackbone(lm_head=True)
    model = _model(backbone, is_encoder=False)

    model._backbone_forward({"input_ids": torch.zeros(2, 5, dtype=torch.long)})

    assert backbone.calls, "backbone was never called"
    kwargs = backbone.calls[-1]
    assert kwargs.get("logits_to_keep") == 1
    assert kwargs.get("use_cache") is False
    assert kwargs.get("output_hidden_states") is True


def test_causal_backbone_without_a_vocab_head_does_not_get_the_kwarg():
    """``load_qwen35_base`` hands over ``CausalLM.model``, which has no lm_head."""
    backbone = _RecordingBackbone(lm_head=False)
    model = _model(backbone, is_encoder=False)

    model._backbone_forward({"input_ids": torch.zeros(2, 5, dtype=torch.long)})

    kwargs = backbone.calls[-1]
    assert "logits_to_keep" not in kwargs
    assert kwargs.get("use_cache") is False


def test_encoder_gets_neither_kwarg():
    backbone = _RecordingBackbone(lm_head=True)
    model = _model(backbone, is_encoder=True)

    model._backbone_forward({"input_ids": torch.zeros(2, 5, dtype=torch.long)})

    kwargs = backbone.calls[-1]
    assert "logits_to_keep" not in kwargs
    assert "use_cache" not in kwargs
    assert kwargs.get("output_hidden_states") is True


def test_hidden_states_are_unaffected_by_the_slice():
    """The fix must be numerically inert; this is the property it relies on."""
    backbone = _RecordingBackbone(lm_head=True)
    model = _model(backbone, is_encoder=False)
    inputs = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    with_keep = model._backbone_forward(inputs)
    del backbone.calls[:]
    model.backbone.lm_head  # noqa: B018  (still present)
    # Compare against a call that would not have been sliced, by asking the stub
    # directly: the stub returns hidden states independent of the kwarg.
    unsliced = backbone(input_ids=inputs["input_ids"], output_hidden_states=True, use_cache=False)

    assert torch.equal(with_keep.hidden_states[-1], unsliced.hidden_states[-1])
    assert with_keep.logits.shape[1] == 1
    assert unsliced.logits.shape[1] == 5
