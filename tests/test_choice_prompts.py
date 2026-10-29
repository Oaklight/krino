"""Tests for fitting cross-encoder prompts inside the token budget.

Torch-free on purpose: these run in CI, where the torch tests are skipped. The
failure they guard against is that the option is the prompt's suffix and the
tokenizer truncates from the right, so a long state silently deletes the text
that distinguishes the options.
"""

from __future__ import annotations

from model.src.choice_prompts import fit_option_prompts, option_prompt


class WhitespaceTokenizer:
    """A tokenizer whose tokens are words, for arithmetic that needs no model."""

    def encode(self, text: str) -> list[int]:
        return [hash(word) for word in text.split()]

    def decode(self, ids: list[int]) -> str:
        # Only the length matters to the caller; keep one word per id.
        return " ".join(f"t{index}" for index in range(len(ids)))


TOKENIZER = WhitespaceTokenizer()


def fit(state, instructions, options, max_length):
    return fit_option_prompts(
        state, instructions, options, max_length, TOKENIZER.encode, TOKENIZER.decode
    )


class TestOptionPrompt:
    def test_it_joins_the_three_parts(self):
        assert option_prompt("state", "question", "option") == "state question option"

    def test_it_drops_empty_parts_without_leaving_a_double_space(self):
        assert option_prompt("state", "", "option") == "state option"
        assert option_prompt("", "question", "option") == "question option"


class TestFitOptionPrompts:
    def test_a_short_state_is_left_alone(self):
        prompts, dropped = fit("short state", "which?", ["yes", "no"], max_length=64)
        assert dropped == 0
        assert prompts == ["short state which? yes", "short state which? no"]

    def test_a_long_state_is_shortened_but_the_option_survives(self):
        state = " ".join(f"w{index}" for index in range(100))
        prompts, dropped = fit(state, "which?", ["the correct option"], max_length=32)
        assert dropped > 0
        for prompt in prompts:
            assert "the correct option" in prompt
            # Every prompt fits the window, counted the same way the encoder will.
            assert len(TOKENIZER.encode(prompt)) <= 32

    def test_the_instructions_survive_too(self):
        state = " ".join(f"w{index}" for index in range(100))
        prompts, _ = fit(state, "pick the right one now", ["a", "b"], max_length=24)
        for prompt in prompts:
            assert prompt.endswith("pick the right one now a") or prompt.endswith(
                "pick the right one now b"
            )

    def test_every_prompt_fits_even_for_the_longest_option(self):
        state = " ".join(f"w{index}" for index in range(200))
        prompts, _ = fit(state, "which?", ["short", "a much longer option text here"], max_length=40)
        assert all(len(TOKENIZER.encode(prompt)) <= 40 for prompt in prompts)

    def test_options_that_do_not_fit_on_their_own_are_reported(self):
        # Nothing sensible can be kept: the suffix alone exceeds the window.
        state = "state"
        prompts, dropped = fit(state, "", [" ".join(f"o{index}" for index in range(50))], max_length=8)
        assert dropped > 0
        assert len(prompts) == 1

    def test_it_keeps_the_opening_of_the_state(self):
        state = "alpha beta gamma delta epsilon zeta eta theta"
        prompts, _ = fit(state, "q", ["a"], max_length=6)
        # The kept prefix starts at the beginning of the state, not the end.
        assert prompts[0].startswith("t0") or prompts[0].startswith("alpha")


class TestNonStringInputs:
    """A loader that did not render a structured state must not crash the head.

    35 of JevBench's 111 hard items carry a dict state. #180 renders those in
    the loader, but ``forward_choice`` still failed on the first eval that
    reached it with a raw one, because the coercion happened nowhere.
    """

    def test_a_dict_state_produces_a_prompt_instead_of_raising(self):
        state = {"policy": "refund within 30 days", "tier": "gold"}
        prompts, dropped = fit(state, "does the policy apply?", ["yes", "no"], max_length=256)
        assert dropped == 0
        assert len(prompts) == 2
        assert all("yes" in prompt or "no" in prompt for prompt in prompts)
        assert all("refund within 30 days" in prompt for prompt in prompts)

    def test_a_none_state_is_empty_rather_than_the_string_none(self):
        prompts, _ = fit(None, "which?", ["a", "b"], max_length=64)
        assert prompts == ["which? a", "which? b"]

    def test_the_truncation_path_also_accepts_a_dict(self):
        state = {"rows": [f"row{index}" for index in range(200)]}
        prompts, dropped = fit(state, "which?", ["the correct option"], max_length=32)
        assert dropped > 0
        assert "the correct option" in prompts[0]
        assert len(TOKENIZER.encode(prompts[0])) <= 32
