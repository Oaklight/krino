"""Tests for the reranker prompt helpers.

Torch-free by design: the prompt format and option extraction are the parts that
must not drift between the JevBench harness and the general eval, so they are
importable and testable without a GPU stack.
"""

from __future__ import annotations

from types import SimpleNamespace

from model.src.reranker_prompt import build_query_doc, get_options


def test_query_doc_carries_the_three_sections_in_order():
    text = build_query_doc("The cat sat.", "What happened?", "choice", "A", "The cat")
    assert text.startswith("<Instruct>: ")
    assert "\n<Query>: The cat sat.\n\nQuestion: What happened?" in text
    assert text.endswith("<Document>: Answer option A: The cat")


def test_score_rubrics_are_named_rubric_not_answer():
    text = build_query_doc("s", "q", "score", "2", "Somewhat positive")
    assert "Rubric level 2: Somewhat positive" in text
    assert "Answer option" not in text


def test_choice_options_keep_the_criteria_order():
    item = SimpleNamespace(
        question={"type": "choice", "criteria": {"b": "second", "a": "first"}}
    )
    assert get_options(item) == [("b", "second"), ("a", "first")]


def test_noul_options_are_no_then_yes():
    item = SimpleNamespace(question={"type": "noul", "criteria": {"true": "Yes", "false": "No"}})
    assert get_options(item) == [("no", "No"), ("yes", "Yes")]


def test_score_options_accept_a_list_of_rubrics():
    item = SimpleNamespace(question={"type": "score", "criteria": ["bad", "ok", "good"]})
    assert get_options(item) == [("0", "bad"), ("1", "ok"), ("2", "good")]


def test_an_unknown_type_yields_no_options():
    assert get_options(SimpleNamespace(question={"type": "other"})) == []
