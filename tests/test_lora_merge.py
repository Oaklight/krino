"""Tests for folding a LoRA adapter into its base weights.

Torch-dependent, so they are skipped on a bare interpreter and in CI; the merge
maths is checked here rather than in the script that uses it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.src.lora import LoRALinear, merge_lora  # noqa: E402


def adapted_layer(seed: int = 0, in_features: int = 8, out_features: int = 4):
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_features, out_features)
    layer = LoRALinear(base, rank=2, alpha=4)
    with torch.no_grad():
        # A freshly initialised adapter is B=0, so it would merge to nothing and
        # the test would pass for the wrong reason.
        layer.lora_a.weight.normal_()
        layer.lora_b.weight.normal_()
    return base, layer


def test_merged_model_reproduces_the_adapter_output():
    _, layer = adapted_layer()
    holder = torch.nn.Sequential(layer)
    x = torch.randn(3, 8)
    before = holder(x)

    merged = merge_lora(holder)

    assert merged == 1
    assert isinstance(holder[0], torch.nn.Linear)
    assert torch.allclose(before, holder(x), atol=1e-5)


def test_merge_is_a_noop_without_an_adapter():
    plain = torch.nn.Sequential(torch.nn.Linear(4, 2))
    x = torch.randn(2, 4)
    before = plain(x)
    assert merge_lora(plain) == 0
    assert torch.allclose(before, plain(x))


def test_nested_layers_are_all_merged_and_the_parent_is_kept():
    torch.manual_seed(3)
    block = torch.nn.Sequential(adapted_layer(1)[1], torch.nn.ReLU(), adapted_layer(2)[1])
    x = torch.randn(2, 8)
    before = block(x)

    assert merge_lora(block) == 2
    assert all(isinstance(module, (torch.nn.Linear, torch.nn.ReLU)) for module in block)
    assert torch.allclose(before, block(x), atol=1e-5)


def test_a_zero_adapter_leaves_the_base_untouched():
    base = torch.nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, alpha=2)  # lora_b initialised to zeros
    before = base.weight.detach().clone()
    merge_lora(torch.nn.Sequential(layer))
    assert torch.allclose(before, layer.weight, atol=1e-6)


def test_merging_twice_does_not_apply_the_update_twice():
    _, layer = adapted_layer(4)
    holder = torch.nn.Sequential(layer)
    x = torch.randn(2, 8)
    expected = holder(x).detach()

    merge_lora(holder)
    again = merge_lora(holder)  # the wrapper is gone, so there is nothing to merge
    assert again == 0
    assert torch.allclose(expected, holder(x), atol=1e-5)
