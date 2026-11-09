"""Tests for the attention-mass span alignment and combination (#341)."""

from __future__ import annotations

import math

import pytest

from model.evaluation.panel2 import order_effect, render_panel2
from model.evaluation.panel2_mass import (
    MASS_FLOOR,
    combine_scores,
    length_guard,
    mean_chance,
    option_spans,
    span_masses,
)


class CharTokenizer:
    """One token per character, so offsets are character indices."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        payload = {
            "input_ids": list(range(len(text))),
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload


def prompt_for(texts, codes):
    return render_panel2("The state.", "Which one?", texts, codes)


def test_option_spans_map_each_line_to_its_characters():
    texts = ["no", "yes"]
    codes = ["A", "B"]
    prompt = prompt_for(texts, codes)
    alignment = option_spans(CharTokenizer(), prompt, texts, codes)

    spans = alignment["spans"]
    assert len(spans) == 2
    for (lo, hi), line in zip(spans, ["A. no", "B. yes"]):
        assert "".join(prompt[lo:hi]) == line
    # The newline between the lines is inside the block but outside both spans.
    assert alignment["straddling_tokens"] == [prompt.index("A. no") + len("A. no")]
    assert prompt[alignment["straddling_tokens"][0]] == "\n"
    assert alignment["n_tokens"] == len(prompt)


def test_option_spans_reject_a_prompt_without_the_block():
    with pytest.raises(ValueError):
        option_spans(CharTokenizer(), "State: S\nAnswer:", ["no", "yes"], ["A", "B"])


def test_option_spans_reject_a_duplicated_block():
    texts, codes = ["no", "yes"], ["A", "B"]
    prompt = prompt_for(texts, codes)
    doubled = prompt + "\n" + prompt
    with pytest.raises(ValueError):
        option_spans(CharTokenizer(), doubled, texts, codes)


def test_span_masses_sum_spans_and_split_the_residual():
    texts, codes = ["no", "yes"], ["A", "B"]
    prompt = prompt_for(texts, codes)
    alignment = option_spans(CharTokenizer(), prompt, texts, codes)
    row = [0.0] * len(prompt)
    row[alignment["spans"][0][0]] = 0.25
    row[alignment["spans"][1][0]] = 0.5
    row[0] = 0.25

    masses = span_masses(row, alignment["spans"])
    assert masses["masses"] == pytest.approx([0.25, 0.5])
    assert masses["preamble_mass"] == pytest.approx(0.25)
    assert masses["span_sum"] == pytest.approx(0.75)
    assert masses["residual"] == pytest.approx(0.25)
    assert masses["total"] == pytest.approx(1.0)


def test_combination_follows_the_log_probability_when_mass_disagrees():
    # Equal code logits, so the mass decides: option 0 has most of it.
    combined = combine_scores([0.0, 0.0], [0.9, 0.1])
    assert combined[0] > combined[1]

    # A large code-logit gap overrides a small mass gap.
    combined = combine_scores([10.0, 0.0], [0.1, 0.9])
    assert combined[0] > combined[1]


def test_combination_is_finite_at_zero_mass():
    combined = combine_scores([0.0, 1.0], [0.0, 1.0])
    assert all(math.isfinite(value) for value in combined)
    expected = math.log(1.0 / (1.0 + math.e)) + math.log(MASS_FLOOR)
    assert combined[0] == pytest.approx(expected)


def test_combination_rejects_length_mismatch():
    with pytest.raises(ValueError):
        combine_scores([0.0, 1.0], [1.0])


def test_mean_chance_averages_over_items():
    assert mean_chance({"a": 2, "b": 4}) == pytest.approx(0.375)
    assert mean_chance({}) == 0.0


def test_length_guard_reports_baseline_accuracy_and_agreement():
    guard = length_guard(
        [
            {"texts": ["short", "longer", "longest"], "n_options": 3, "gold": 0, "chosen": 0},
            {"texts": ["a", "bb"], "n_options": 2, "gold": 1, "chosen": 1},
        ]
    )
    assert guard["items"] == 2
    # Item 1 picks the longest (index 2) and is wrong; item 2 picks index 1.
    assert guard["longest_option_baseline"]["correct"] == 1
    assert guard["mass_accuracy"]["correct"] == 2
    assert guard["agreement_with_longest"]["count"] == 1


def test_length_guard_ties_go_to_the_lowest_natural_index():
    guard = length_guard(
        [{"texts": ["aa", "bb"], "n_options": 2, "gold": 0, "chosen": 0}]
    )
    assert guard["longest_option_baseline"]["correct"] == 1
    assert guard["agreement_with_longest"]["fraction"] == 1.0


def test_length_guard_rejects_a_width_mismatch():
    with pytest.raises(ValueError):
        length_guard([{"texts": ["a", "b"], "n_options": 3, "gold": 0, "chosen": 0}])


def test_reversed_mapping_is_the_display_inverse():
    # Sanity anchor for the reversed order used by the mass cells.
    from model.evaluation.panel2 import option_index_for_display

    assert option_index_for_display(3, False) == [0, 1, 2]
    assert option_index_for_display(3, True) == [2, 1, 0]
    effect = order_effect(
        [{"id": "a", "pred_pos": 0, "correct": True}],
        [{"id": "a", "pred_pos": 2, "correct": True}],
        {"a": 3},
    )
    assert effect["changed_fraction"] == 0.0
