"""Tests for the set-function head (#340)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.src.set_head import SetHead  # noqa: E402


def make_head(seed: int = 0, hidden: int = 16, width: int = 32) -> SetHead:
    torch.manual_seed(seed)
    return SetHead(hidden, width=width, layers=2, heads=4, ffn=64)


def test_head_is_equivariant_to_option_order():
    head = make_head()
    states = torch.randn(2, 5, 16)
    valid = torch.ones(2, 5, dtype=torch.bool)
    order = torch.tensor([3, 0, 4, 1, 2])

    straight = head(states, valid)
    permuted = head(states[:, order], valid)

    assert torch.allclose(permuted, straight[:, order], atol=1e-5)


def test_masked_slots_get_no_probability_mass():
    head = make_head()
    states = torch.randn(1, 6, 16)
    valid = torch.tensor([[True, True, True, False, False, False]])
    logits = head(states, valid)
    probabilities = head.probabilities(states, valid)

    assert torch.isinf(logits[0, 3:]).all()
    assert (logits[0, 3:] < 0).all()
    assert torch.allclose(probabilities[0, 3:], torch.zeros(3))
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_dropping_a_masked_slot_changes_nothing():
    head = make_head()
    states = torch.randn(1, 5, 16)

    shorter = head.probabilities(states[:, :3], torch.ones(1, 3, dtype=torch.bool))
    padded = head.probabilities(
        torch.cat([states[:, :3], torch.randn(1, 2, 16)], dim=1),
        torch.tensor([[True, True, True, False, False]]),
    )

    # The masked slots get no mass, so the real options keep their
    # probabilities exactly; the summary is pooled over the same three states.
    assert torch.allclose(shorter[0], padded[0, :3], atol=1e-6)
    assert torch.allclose(padded[0, 3:], torch.zeros(2))


def test_no_parameter_depends_on_the_option_count():
    head = make_head()
    for name, parameter in head.named_parameters():
        # The learned pooling query is (1, 1, width); everything else is at
        # most a 2-D weight. No shape carries K.
        assert parameter.shape[:2] == (1, 1) or parameter.dim() <= 2, (
            f"{name} has shape {tuple(parameter.shape)}"
        )
    logits_three = head(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    logits_seven = head(torch.randn(1, 7, 16), torch.ones(1, 7, dtype=torch.bool))
    assert logits_three.shape == (1, 3)
    assert logits_seven.shape == (1, 7)
