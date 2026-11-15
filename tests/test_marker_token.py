"""Tests for the marker-token selection and the checkpoint schema (#373)."""

from __future__ import annotations

import argparse

import pytest

from model.src.marker import (
    checkpoint_config,
    marker_token_id_for,
    resolve_checkpoint_marker,
    resolve_training_marker,
)


class Tok:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_the_override_wins_over_the_tokenizer_default():
    tok = Tok(eos_token_id=10, sep_token_id=11, pad_token_id=12)
    assert marker_token_id_for(tok, 151645) == 151645
    assert marker_token_id_for(tok, 0) == 0


def test_the_fallback_is_eos_then_sep_then_pad():
    assert marker_token_id_for(Tok(eos_token_id=10, sep_token_id=11, pad_token_id=12)) == 10
    assert (
        marker_token_id_for(Tok(eos_token_id=None, sep_token_id=11, pad_token_id=12))
        == 11
    )
    assert (
        marker_token_id_for(Tok(eos_token_id=None, sep_token_id=None, pad_token_id=12))
        == 12
    )
    # A zero id is falsy and must not stop the fallback.
    assert (
        marker_token_id_for(Tok(eos_token_id=0, sep_token_id=11, pad_token_id=12))
        == 11
    )


def test_a_tokenizer_without_any_marker_raises():
    with pytest.raises(ValueError):
        marker_token_id_for(Tok(eos_token_id=None, sep_token_id=None, pad_token_id=None))


SHIPPED = Tok(eos_token_id=151645, sep_token_id=None, pad_token_id=151643)


def make_args(**overrides):
    values = {
        "model": "Qwen/Qwen3-Reranker-4B",
        "option_marker": True,
        "noul_marker": False,
        "marker_token": 151651,
        "rank": 32,
        "max_length": 2048,
        "kl_weight": 1.0,
        "ce_weight": 0.2,
        "seed": 42,
        "option_chunk_size": 1,
        "option_max_tokens": 64,
        "option_batch": 16,
        "train_safe_buckets": False,
        "lora_layers": None,
        "lora_rank": 16,
        "lora_alpha": 32,
        "answerable_head": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def saved(args, tokenizer=SHIPPED, weight: float = 0.0) -> dict:
    return checkpoint_config(args, weight, resolve_training_marker(args, tokenizer))


class TestCheckpointMarker:
    """A checkpoint must carry the marker it trained against (#373)."""

    def test_the_stored_marker_resolves(self):
        config = {"option_marker": True, "marker_token": 151651}
        assert resolve_checkpoint_marker(config) == 151651

    def test_the_explicit_override_wins(self):
        config = {"option_marker": True, "marker_token": 151651}
        assert resolve_checkpoint_marker(config, 151649) == 151649

    def test_an_override_that_disagrees_with_the_record_warns(self, capsys):
        config = {"option_marker": True, "marker_token": 151651}
        assert resolve_checkpoint_marker(config, 151645) == 151645
        assert "checkpoint records marker_token 151651" in capsys.readouterr().err

    def test_a_missing_marker_raises_instead_of_falling_back_to_eos(self):
        # A legacy checkpoint has no field.  Rebuilding at the tokenizer eos
        # would evaluate a 151651 head as 151645 under a 151651 name.
        with pytest.raises(ValueError, match="records no marker_token"):
            resolve_checkpoint_marker({"option_marker": True})

    def test_a_present_but_null_marker_raises_with_its_own_message(self):
        # Present-and-null is a different fact from absent: the save failed to
        # resolve the id, rather than predating the field.
        with pytest.raises(ValueError, match="records marker_token: null"):
            resolve_checkpoint_marker({"option_marker": True, "marker_token": None})

    def test_a_non_marker_checkpoint_resolves_to_none(self):
        assert resolve_checkpoint_marker({"option_marker": False}) is None
        assert resolve_checkpoint_marker({}) is None

    def test_an_override_without_the_marker_readout_raises(self):
        with pytest.raises(ValueError, match="option_marker=True"):
            resolve_checkpoint_marker({"option_marker": False}, 151645)


class TestCheckpointConfig:
    def test_the_checkpoint_records_its_marker_token(self):
        assert saved(make_args())["marker_token"] == 151651

    def test_the_saved_marker_round_trips_through_the_consumer(self):
        assert resolve_checkpoint_marker(saved(make_args())) == 151651

    def test_the_default_arm_records_the_resolved_default(self):
        # The shipped configuration passes no --marker-token.  The checkpoint
        # must record the tokenizer eos (151645), not null: null is what a
        # failed save looks like, and it is what the guard rejects.
        config = saved(make_args(marker_token=None))
        assert config["marker_token"] == 151645
        assert resolve_checkpoint_marker(config) == 151645

    def test_the_default_arm_resolves_from_the_tokenizer_not_the_flag(self):
        assert resolve_training_marker(make_args(marker_token=None), SHIPPED) == 151645

    def test_a_non_marker_checkpoint_records_no_marker(self):
        config = saved(make_args(option_marker=False, marker_token=None))
        assert config["marker_token"] is None
        assert resolve_checkpoint_marker(config) is None
