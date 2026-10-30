"""The padding used by the state-length probe stays honest about what it adds."""

from __future__ import annotations

import pytest

from model.evaluation.state_padding import CLOSING, pad_state


class TestPadState:
    def test_reaches_the_target_and_no_further(self):
        padded = pad_state("Short state.", 500)
        assert len(padded) >= 500
        # The filler is written in whole sentences of fixed length, so overshoot
        # is bounded by one copy plus the closing sentence.
        assert len(padded) < 500 + len(CLOSING) + 200

    def test_keeps_the_original_state_as_a_prefix(self):
        """Suffix padding only, so no original token moves."""
        state = "The release ships on Friday."
        assert pad_state(state, 400).startswith(state)

    def test_leaves_a_long_enough_state_alone(self):
        state = "x" * 900
        assert pad_state(state, 500) == state

    def test_zero_target_is_the_untouched_state(self):
        state = "anything"
        assert pad_state(state, 0) == state

    def test_closing_sentence_is_always_last(self):
        padded = pad_state("state", 2000)
        assert padded.endswith(CLOSING)

    def test_target_just_above_the_state_is_still_met(self):
        state = "x" * 100
        padded = pad_state(state, 110)
        assert len(padded) >= 110

    def test_empty_filler_is_rejected(self):
        with pytest.raises(ValueError):
            pad_state("state", 100, filler="")

    def test_padding_is_contentless(self):
        """The filler must not repeat the item's own text."""
        state = "The bank declined the transfer."
        padded = pad_state(state, 600)
        assert padded.count("bank declined") == 1
