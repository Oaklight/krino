"""Tests for the panel-2 option spans and the state-cache key (#340)."""

from __future__ import annotations

import pytest

from model.evaluation.panel2 import render_panel2
from model.evaluation.panel2_states import (
    build_batches,
    cache_key,
    data_digest,
    last_token_index,
    option_token_indices,
    render_with_spans,
)


class WordTokenizer:
    """One token per whitespace-separated word, enough for batching tests."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": text.split()}


def cache_row(
    item_id: str, n_options: int, state_words: int, split: str = "train"
) -> dict:
    return {
        "id": item_id,
        "split": split,
        "gold": 0,
        "state": " ".join(["state"] * state_words),
        "instructions": "question",
        "texts": [f"option {index}" for index in range(n_options)],
        "order": "natural",
    }


def test_spans_cover_each_option_text_exactly():
    state = "The user is tired and asks for a refund."
    instructions = "Which response is better?"
    texts = ["no", "yes", "Maybe later", "Ask again"]
    prompt, codes, spans = render_with_spans(state, instructions, texts)

    assert codes == ["A", "B", "C", "D"]
    assert prompt == render_panel2(state, instructions, texts, codes)
    for text, (start, end) in zip(texts, spans):
        assert prompt[start:end] == text


def test_spans_survive_a_multi_line_state_and_option_text():
    state = "line one\nline two"
    texts = ["first", "second\nwith a break", "third"]
    prompt, _, spans = render_with_spans(state, "Q", texts)
    for text, (start, end) in zip(texts, spans):
        assert prompt[start:end] == text


def test_last_token_index_takes_the_last_overlapping_token():
    # Offsets are character spans: "A" is part of the word "Answer", which is a
    # different option than the standalone "answer".
    offsets = [(0, 3), (4, 8), (9, 14), (15, 20)]
    assert last_token_index(offsets, (9, 14)) == 2
    # A token that straddles the span boundary still counts.
    assert last_token_index(offsets, (10, 13)) == 2
    assert last_token_index(offsets, (21, 25)) == -1


def test_option_token_indices_rejects_an_option_with_no_token():
    offsets = [(0, 3), (4, 8)]
    assert option_token_indices(offsets, [(0, 3), (4, 8)]) == [0, 1]
    with pytest.raises(ValueError):
        option_token_indices(offsets, [(0, 3), (9, 12)])


def test_data_digest_tracks_ids_texts_and_gold():
    rows = [{"id": "a", "texts": ["no", "yes"], "gold": 1}]
    baseline = data_digest(rows)
    assert data_digest([dict(rows[0])]) == baseline
    assert data_digest([{**rows[0], "gold": 0}]) != baseline
    assert data_digest([{**rows[0], "texts": ["yes", "no"]}]) != baseline
    assert data_digest([{**rows[0], "id": "b"}]) != baseline


def test_cache_key_covers_model_tokenizer_template_and_order():
    base = cache_key("rev-1", "tok-1", "natural")
    assert base == cache_key("rev-1", "tok-1", "natural")
    assert base != cache_key("rev-2", "tok-1", "natural")
    assert base != cache_key("rev-1", "tok-2", "natural")
    assert base != cache_key("rev-1", "tok-1", "reversed")


def test_batches_are_grouped_by_option_count_and_respect_the_caps():
    tokenizer = WordTokenizer()
    rows = [cache_row(f"a{i}", 2, 5 + i) for i in range(5)]
    rows += [cache_row(f"b{i}", 3, 5 + i) for i in range(3)]

    batches, skipped = build_batches(
        rows, tokenizer, batch_size=2, max_tokens=200, max_prompt_tokens=1000
    )

    assert skipped == []
    assert {len(batch[0]["spans"]) for batch in batches} <= {2, 3}
    for batch in batches:
        assert len({len(item["spans"]) for item in batch}) == 1
        assert len(batch) <= 2
        assert sum(item["length"] for item in batch) <= 200
    assert sum(len(batch) for batch in batches) == len(rows)
    # Longest prompts come first inside a group.
    assert [item["length"] for item in batches[0]] == sorted(
        (item["length"] for item in batches[0]), reverse=True
    )


def test_items_over_the_prompt_cap_are_skipped_and_counted():
    tokenizer = WordTokenizer()
    rows = [
        cache_row("short", 2, 1),
        cache_row("long", 2, 500),
        cache_row("medium", 2, 20),
    ]
    batches, skipped = build_batches(
        rows, tokenizer, batch_size=8, max_tokens=10000, max_prompt_tokens=200
    )

    assert [item["id"] for item in skipped] == ["long"]
    assert skipped[0]["split"] == "train"
    assert skipped[0]["tokens"] > 200
    ids = [item["id"] for batch in batches for item in batch]
    assert sorted(ids) == ["medium", "short"]


def test_one_item_over_the_token_budget_forms_its_own_batch():
    tokenizer = WordTokenizer()
    rows = [cache_row("big", 2, 300), cache_row("small", 2, 1)]
    batches, _ = build_batches(
        rows, tokenizer, batch_size=8, max_tokens=50, max_prompt_tokens=10000
    )

    assert len(batches) == 2
    assert [item["id"] for item in batches[0]] == ["big"]
