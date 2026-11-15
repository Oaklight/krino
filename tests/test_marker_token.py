"""Tests for the marker-token selection (#marker ablation)."""

from __future__ import annotations

import pytest

from model.src.marker import marker_token_id_for


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
