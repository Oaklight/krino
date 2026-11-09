"""Tests for the panel 2 read configurations (#337)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from model.scripts.eval_jevbench_panel2 import check_read_layout, read_logits


class FakeModel:
    """Identity norm and identity head, so a read returns its hidden state."""

    def __init__(self, hidden_states):
        self.config = SimpleNamespace(num_hidden_layers=len(hidden_states) - 1)
        self.model = SimpleNamespace(norm=lambda tensor: tensor)
        self.lm_head = lambda tensor: tensor


def hidden_states(values):
    states = []
    for value in values:
        tensor = torch.zeros(1, 2, 3)
        tensor[0, -1] = torch.tensor([value, -value, 0.0])
        states.append(tensor)
    return tuple(states)


def test_last_read_returns_the_model_logits_unchanged():
    states = hidden_states([1.0, 2.0, 3.0])
    model = FakeModel(states)
    last = torch.tensor([9.0, 8.0, 7.0])
    assert read_logits(model, states, ("last",), last) is last


def test_layer_read_uses_the_named_hidden_state():
    states = hidden_states([1.0, 2.0, 3.0])
    model = FakeModel(states)
    result = read_logits(model, states, ("layer", 1), torch.zeros(3))
    assert torch.equal(result, states[1][0, -1])


def test_mean_last_n_excludes_the_final_normed_entry():
    # Five entries; the last is the normed state and must not be averaged in.
    states = hidden_states([1.0, 2.0, 3.0, 100.0, 4.0])
    model = FakeModel(states)
    result = read_logits(model, states, ("mean_last_n", 4), torch.zeros(3))
    expected = torch.stack([state[0, -1] for state in states[:4]]).mean(0)
    assert torch.equal(result, expected)


def test_unknown_read_spec_is_rejected():
    states = hidden_states([1.0, 2.0])
    model = FakeModel(states)
    with pytest.raises(ValueError):
        read_logits(model, states, ("unknown",), torch.zeros(3))


def test_layout_check_accepts_the_expected_layout():
    states = hidden_states([1.0, 2.0])
    model = FakeModel(states)
    output = SimpleNamespace(hidden_states=states, logits=model.lm_head(states[-1]))
    check_read_layout(model, output)


def test_layout_check_rejects_a_tuple_of_the_wrong_length():
    states = hidden_states([1.0, 2.0])
    model = FakeModel(hidden_states([1.0, 2.0, 3.0]))
    output = SimpleNamespace(hidden_states=states, logits=states[-1])
    with pytest.raises(RuntimeError):
        check_read_layout(model, output)


def test_layout_check_rejects_a_head_that_does_not_read_the_last_entry():
    states = hidden_states([1.0, 2.0])
    model = FakeModel(states)
    output = SimpleNamespace(
        hidden_states=states, logits=torch.full_like(states[-1], 5.0)
    )
    with pytest.raises(RuntimeError):
        check_read_layout(model, output)
