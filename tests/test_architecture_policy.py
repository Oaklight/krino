"""Tests for the architecture allowlist that decides the model loader (#277)."""

from __future__ import annotations

from types import SimpleNamespace

from model.src.architecture_policy import (
    declared_architecture,
    routes_to_conditional_generation,
)


def _config(*architectures):
    return SimpleNamespace(architectures=list(architectures) or None)


def test_mistral3_routes_to_its_declared_class():
    assert routes_to_conditional_generation(_config("Mistral3ForConditionalGeneration"))


def test_qwen35_does_not_route():
    """The incumbent declares a ConditionalGeneration class; it must not match.

    A suffix rule would send every Qwen3.5 load through
    Qwen3_5ForConditionalGeneration and pull in a vision tower the text paths do
    not use.  This assertion is the one a widened rule fails.
    """
    assert not routes_to_conditional_generation(_config("Qwen3_5ForConditionalGeneration"))
    assert not routes_to_conditional_generation(_config("Qwen3_5MoeForConditionalGeneration"))


def test_plain_causal_lm_architectures_do_not_route():
    for arch in ("LlamaForCausalLM", "MiniCPM3ForCausalLM", "Qwen2ForCausalLM"):
        assert not routes_to_conditional_generation(_config(arch))


def test_missing_architectures_do_not_route():
    assert not routes_to_conditional_generation(_config())
    assert declared_architecture(_config()) is None


def test_declared_architecture_reads_the_first_entry():
    assert declared_architecture(_config("A", "B")) == "A"
