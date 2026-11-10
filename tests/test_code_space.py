"""Tests for the code-space survey and the gluing check (#350 follow-up)."""

from __future__ import annotations

import pytest

from model.evaluation.code_space import (
    candidate_strings,
    code_forms,
    sample_tier,
    string_survey,
    survey,
    text_forms,
    tier_of,
    usable_chars,
)
from model.evaluation.panel2 import glued_codes


class DictTokenizer:
    """Longest-match tokenizer over a fixed vocabulary, enough for these tests."""

    def __init__(self, tokens: list[str]):
        self.tokens = list(tokens)
        self.vocab = {token: index for index, token in enumerate(self.tokens)}
        self.all_special_ids = [self.vocab["<s>"]] if "<s>" in self.vocab else []

    def get_added_vocab(self) -> dict:
        return {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        index = 0
        while index < len(text):
            for length in range(len(text) - index, 0, -1):
                piece = text[index : index + length]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    index += length
                    break
            else:
                raise ValueError(f"cannot encode {text!r} at {index}")
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.tokens[index] for index in ids)


def tokenizer() -> DictTokenizer:
    return DictTokenizer(
        [
            "<s>",
            "A",
            "B",
            "C",
            "D",
            " A",
            " B",
            "中",
            "가",
            "!",
            "!.",
            "*",
            " ",
            ".",
            "\n",
            "\nOptions:\n",
        ]
    )


def test_glued_codes_finds_the_symbol_that_merges_with_the_period():
    tok = tokenizer()
    assert glued_codes(tok, ["A", "B", "*"]) == []
    assert glued_codes(tok, ["!", "A"]) == ["!"]


def test_glued_codes_also_flags_a_code_that_is_not_one_token():
    tok = tokenizer()
    # "中" is one token, " !" is two, so neither is glued by the separator.
    assert glued_codes(tok, ["中"]) == []
    assert glued_codes(tok, ["中", "!"]) == ["!"]


def test_code_forms_reports_the_two_token_forms_separately():
    tok = tokenizer()
    latin = code_forms(tok, "A")
    assert (latin["bare"], latin["space"]) == (True, True)
    assert latin["left_ok"] and latin["right_ok"]

    cjk = code_forms(tok, "中")
    assert cjk["bare"] is True
    assert cjk["space"] is False
    assert cjk["tier"] == "cjk_ideograph"

    symbol = code_forms(tok, "!")
    assert symbol["space"] is False
    assert symbol["right_ok"] is False


def test_usable_chars_splits_by_tier_and_drops_excluded_tiers():
    tok = tokenizer()
    rows = survey(tok, ["A", "B", "C", "D", "中", "가", "!", " "])
    usable = usable_chars(rows)
    assert usable["ascii_alnum"] == ["A", "B"]
    # Neither CJK nor Hangul has a space-prefixed single token here, so the
    # space form cannot name them at all.
    assert "cjk_ideograph" not in usable
    assert "hangul" not in usable
    assert " " not in usable.get("space_control", [])
    # Bare form admits them; the excluded tier stays out either way.
    bare = usable_chars(rows, form="bare")
    assert bare["cjk_ideograph"] == ["中"]
    assert bare["hangul"] == ["가"]
    assert bare["ascii_alnum"] == ["A", "B", "C", "D"]
    assert " " not in bare.get("space_control", [])


def test_sample_tier_is_deterministic_and_sorted():
    tok = tokenizer()
    rows = survey(tok, ["A", "B"])
    assert sample_tier(rows, "ascii_alnum", 2, 342) == ["A", "B"]
    assert sample_tier(rows, "ascii_alnum", 2, 7) == ["A", "B"]
    with pytest.raises(ValueError):
        sample_tier(rows, "ascii_alnum", 3, 342)


def test_tier_of_covers_the_tiers_an_alphabet_may_draw_from():
    assert tier_of("A") == "ascii_alnum"
    assert tier_of("7") == "ascii_alnum"
    assert tier_of("!") == "ascii_symbol"
    assert tier_of("中") == "cjk_ideograph"
    assert tier_of("가") == "hangul"
    assert tier_of("é") == "other_letter"
    assert tier_of("€") == "other_symbol"
    assert tier_of(" ") == "space_control"


def string_tokenizer() -> DictTokenizer:
    return DictTokenizer(
        [
            "<s>",
            "A",
            "B",
            "AA",
            " AA",
            "AB",
            " ",
            ".",
            "\n",
            "\nOptions:\n",
        ]
    )


def test_candidate_strings_enumerates_the_level_in_lexical_order():
    codes = candidate_strings(2, "AB")
    assert codes == ["AA", "AB", "BA", "BB"]
    assert len(candidate_strings(2)) == 676
    assert candidate_strings(3, "AB")[-1] == "BBB"


def test_text_forms_apply_the_same_four_tests_to_a_string():
    tok = string_tokenizer()
    stable = text_forms(tok, "AA")
    assert (stable["bare"], stable["space"]) == (True, True)
    assert stable["left_ok"] and stable["right_ok"]

    # One token bare, two with the leading space.
    no_space = text_forms(tok, "AB")
    assert (no_space["bare"], no_space["space"]) == (True, False)

    # Two tokens bare.
    no_bare = text_forms(tok, "BA")
    assert (no_bare["bare"], no_bare["space"]) == (False, False)


def test_string_survey_counts_each_level():
    level = string_survey(string_tokenizer(), 2, "AB")
    assert level["candidates"] == 4
    assert level["bare"] == 2
    assert level["space"] == 1
    assert level["both"] == 1
    assert level["stable"] == 1
    assert level["stable_codes"] == ["AA"]
