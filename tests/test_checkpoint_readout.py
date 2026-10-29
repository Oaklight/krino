"""Tests for reading a checkpoint's readout off its tensor names."""

from __future__ import annotations

from model.evaluation.checkpoint_readout import infer_readout


def test_weighted_is_detected_from_its_layer_mixture():
    keys = [
        "backbone.layers.0.self_attn.q_proj.lora_A.weight",
        "readout.layer_logits",
        "noul_head.proj.0.weight",
    ]
    assert infer_readout(keys) == "weighted"


def test_a_last_checkpoint_returns_none():
    keys = [
        "backbone.layers.0.self_attn.q_proj.lora_A.weight",
        "noul_head.proj.0.weight",
        "choice_head.proj.weight",
    ]
    assert infer_readout(keys) is None


def test_mean_last_n_is_not_guessed():
    # `mean_last_n` has no parameters at all, so it is indistinguishable from
    # `last` in a checkpoint. Reporting either would be a guess.
    assert infer_readout(["noul_head.proj.0.weight"]) is None


def test_empty_and_none_inputs_are_safe():
    assert infer_readout([]) is None
    assert infer_readout(iter(())) is None


def test_an_unrelated_readout_tensor_is_not_the_mixture():
    # `readout.*` without `layer_logits` is not the weighted readout; a future
    # readout with its own parameters must not be silently reported as this one.
    assert infer_readout(["readout.weight"]) is None
    assert infer_readout(["readout.layer_weights"]) is None


def test_non_string_keys_do_not_raise():
    assert infer_readout([0, 1, 2]) is None
