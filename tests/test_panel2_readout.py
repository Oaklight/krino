"""Tests for the panel 2 prompt, option construction, and diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from model.evaluation.panel2 import (
    CODES,
    PANEL2_TEMPLATE,
    code_token_ids,
    codes_for,
    option_lines,
    options_for,
    order_effect,
    pick_code_form,
    position_diagnostics,
    render_panel2,
)


def item(q_type: str, label, criteria=None) -> SimpleNamespace:
    question = {"type": q_type, "instructions": "Which one?"}
    if criteria is not None:
        question["criteria"] = criteria
    return SimpleNamespace(id="easy-x", state="The state.", question=question, label=label)


class FakeTokenizer:
    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.mapping[text]


def test_prompt_is_exactly_the_recorded_template():
    prompt = render_panel2("S", "Q", ["no", "yes"], ["A", "B"])
    assert prompt == (
        "State: S\n"
        "Question: Q\n"
        "Options:\n"
        "A. no\n"
        "B. yes\n"
        "Answer:"
    )
    assert PANEL2_TEMPLATE.format(
        state="S", instructions="Q", options="A. no\nB. yes"
    ) == prompt


def test_option_lines_pairs_codes_with_texts():
    assert option_lines(["x", "y", "z"], codes_for(3)) == "A. x\nB. y\nC. z"


def test_codes_reject_out_of_range_option_counts():
    assert codes_for(2) == ["A", "B"]
    assert codes_for(6) == list(CODES[:6])
    with pytest.raises(ValueError):
        codes_for(1)
    with pytest.raises(ValueError):
        codes_for(len(CODES) + 1)


def test_codes_extend_past_26_for_wide_items():
    wide = codes_for(80)
    assert len(wide) == 80
    assert len(set(wide)) == 80
    assert wide[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_noul_is_two_options_no_then_yes():
    texts, gold = options_for(item("noul", True, {"false": "f", "true": "t"}))
    assert texts == ["no", "yes"]
    assert gold == 1

    texts, gold = options_for(item("noul", False))
    assert texts == ["no", "yes"]
    assert gold == 0


def test_choice_uses_the_criteria_order():
    texts, gold = options_for(
        item("choice", "b", {"a": "First", "b": "Second", "c": "Third"})
    )
    assert texts == ["First", "Second", "Third"]
    assert gold == 1


def test_choice_label_outside_the_options_is_unscorable():
    assert options_for(item("choice", "z", {"a": "First", "b": "Second"})) is None


def test_score_levels_are_named_and_the_label_is_rounded():
    texts, gold = options_for(item("score", 1.0, ["none", "one", "two"]))
    assert texts == ["Level 0: none", "Level 1: one", "Level 2: two"]
    assert gold == 1


def test_code_token_ids_marks_multi_token_codes():
    tokenizer = FakeTokenizer({" A": [10], " B": [11], " C": [12, 13], "B": [21]})
    ids, multi = code_token_ids(tokenizer, ["A", "B", "C"], "space")
    assert ids == {"A": 10, "B": 11}
    assert multi == ["C"]

    ids, multi = code_token_ids(tokenizer, ["B"], "bare")
    assert ids == {"B": 21}
    assert multi == []


def test_pick_code_form_prefers_space_and_falls_back_to_bare():
    space_ok = FakeTokenizer({" A": [1], " B": [2], "A": [1], "B": [2]})
    assert pick_code_form([space_ok], ["A", "B"]) == "space"

    bare_only = FakeTokenizer({" A": [1, 2], " B": [3], "A": [1], "B": [2]})
    assert pick_code_form([bare_only], ["A", "B"]) == "bare"


def rows(pred_positions, gold_positions, correct):
    return [
        {"pred_pos": pred, "gold_pos": gold, "correct": hit}
        for pred, gold, hit in zip(pred_positions, gold_positions, correct)
    ]


def test_position_diagnostics_flags_an_all_a_model():
    # Ten items, all answered A (position 0). Four have A as the gold answer.
    data = rows([0] * 10, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], [True, False, False,
                                                              True, False, False,
                                                              True, False, False,
                                                              True])
    diagnostics = position_diagnostics(data)

    assert diagnostics["predicted_position_histogram"] == {"0": 10}
    assert diagnostics["modal_predicted_position"] == 0
    assert diagnostics["accuracy_raw"] == pytest.approx(0.4)
    # Excluding the four gold-A items leaves six wrong answers.
    assert diagnostics["accuracy_excluding_modal_gold"] == pytest.approx(0.0)
    assert diagnostics["position_dominated"] is True
    assert diagnostics["accuracy_by_gold_position"]["0"]["correct"] == 4


def test_position_diagnostics_is_not_dominated_when_the_two_agree():
    # Position 0 is the modal answer; removing gold-0 items leaves the same
    # accuracy, so the raw number is not position-dominated.
    data = rows([0, 1, 1, 0], [0, 1, 0, 1], [True, True, False, False])
    diagnostics = position_diagnostics(data)

    assert diagnostics["modal_predicted_position"] == 0
    assert diagnostics["accuracy_raw"] == pytest.approx(0.5)
    assert diagnostics["accuracy_excluding_modal_gold"] == pytest.approx(0.5)
    assert diagnostics["position_dominated"] is False


def test_position_diagnostics_on_empty_rows():
    diagnostics = position_diagnostics([])
    assert diagnostics["n"] == 0
    assert diagnostics["position_dominated"] is False


def test_order_effect_maps_reversed_positions_back():
    natural = [
        {"id": "a", "pred_pos": 0, "correct": True},
        {"id": "b", "pred_pos": 2, "correct": False},
    ]
    reversed_order = [
        # Three options: natural 0 is reversed 2, natural 2 is reversed 0.
        {"id": "a", "pred_pos": 2, "correct": True},
        {"id": "b", "pred_pos": 0, "correct": False},
    ]
    effect = order_effect(natural, reversed_order, {"a": 3, "b": 3})

    assert effect["n"] == 2
    assert effect["accuracy_natural"] == pytest.approx(0.5)
    assert effect["accuracy_reversed"] == pytest.approx(0.5)
    assert effect["difference_pp"] == pytest.approx(0.0)
    assert effect["changed_fraction"] == 0.0


def test_order_effect_counts_a_changed_choice():
    natural = [{"id": "a", "pred_pos": 0, "correct": True}]
    reversed_order = [{"id": "a", "pred_pos": 0, "correct": False}]
    effect = order_effect(natural, reversed_order, {"a": 3})

    assert effect["changed_fraction"] == 1.0
    assert effect["accuracy_reversed"] == 0.0
    assert effect["difference_pp"] == pytest.approx(-100.0)
