"""Tests for the marker-token selection (#marker ablation)."""

from __future__ import annotations

import pytest

from model.src.marker import marker_token_id_for, resolve_checkpoint_marker


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


class TestCheckpointMarker:
    """A checkpoint must carry the marker it trained against (#373)."""

    def test_the_stored_marker_resolves(self):
        config = {"option_marker": True, "marker_token": 151651}
        assert resolve_checkpoint_marker(config) == 151651

    def test_the_explicit_override_wins(self):
        config = {"option_marker": True, "marker_token": 151651}
        assert resolve_checkpoint_marker(config, 151649) == 151649

    def test_a_missing_marker_raises_instead_of_falling_back_to_eos(self):
        # A legacy checkpoint has no field.  Rebuilding at the tokenizer eos
        # would evaluate a 151651 head as 151645 under a 151651 name.
        with pytest.raises(ValueError, match="records no marker_token"):
            resolve_checkpoint_marker({"option_marker": True})

    def test_a_non_marker_checkpoint_resolves_to_none(self):
        assert resolve_checkpoint_marker({"option_marker": False}) is None
        assert resolve_checkpoint_marker({}) is None

    def test_an_override_without_the_marker_readout_raises(self):
        with pytest.raises(ValueError, match="option_marker=True"):
            resolve_checkpoint_marker({"option_marker": False}, 151645)
