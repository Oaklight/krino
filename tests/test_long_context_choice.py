"""Tests for long-context multi-hop choice construction (no network)."""

from __future__ import annotations

import random

from data.format import TypedQuestion
from data.long_context_choice import (
    acceptable_distractor,
    answer_type,
    build_item,
    hotpot_row,
    pick_distractors,
    select_paragraphs,
)


def test_answer_type():
    assert answer_type("yes") == "yes_no"
    assert answer_type("No") == "yes_no"
    assert answer_type("1984") == "date"
    assert answer_type("3 May 1975") == "date"
    assert answer_type("42") == "number"
    assert answer_type("Scott Derrickson") == "proper"
    assert answer_type("Paramount Pictures") == "organization"


def test_acceptable_distractor_rejects_substrings_and_state_hits():
    gold = "Scott Derrickson"
    clean_state = "Scott Derrickson directed the film."
    hit_state = "Ed Wood wrote the film."
    assert acceptable_distractor("Ed Wood", gold, clean_state, "proper")
    assert not acceptable_distractor("Scott Derrickson", gold, clean_state, "proper")
    assert not acceptable_distractor("Derrickson", gold, clean_state, "proper")
    # A candidate that appears in the state is not a clean distractor.
    assert not acceptable_distractor("Ed Wood", gold, hit_state, "proper")
    # Wrong type and wild length ratio.
    assert not acceptable_distractor("1984", gold, clean_state, "proper")
    assert not acceptable_distractor("A" * 100, gold, clean_state, "proper")


def test_pick_distractors_returns_k_minus_one_unique_same_type():
    pool = ["Ed Wood", "Tim Burton", "Joel Coen", "Ethan Coen", "David Lynch", "1984"]
    picked = pick_distractors("Scott Derrickson", pool, "no state hit", k=4, rng=random.Random(1))
    assert picked is not None
    assert len(picked) == 3
    assert len(set(picked)) == 3
    assert "1984" not in picked
    assert all(answer_type(value) == "proper" for value in picked)


def test_pick_distractors_rejects_yes_no_and_small_pools():
    assert pick_distractors("yes", ["no", "maybe"], "state", 4, random.Random(1)) is None
    assert pick_distractors("Scott Derrickson", ["Ed Wood"], "state", 4, random.Random(1)) is None


def test_select_paragraphs_puts_supporting_first_and_controls_length():
    paragraphs = [(f"T{i}", "x" * 500) for i in range(10)]
    state = select_paragraphs(
        paragraphs, {"T7", "T8"}, target_chars=1500, max_chars=4000
    )
    assert state.startswith("T7:")
    assert "T8:" in state
    assert len(state) >= 1500
    assert len(state) <= 4000


def test_build_item_makes_a_valid_choice_question():
    paragraphs = [(f"T{i}", "evidence " * 100) for i in range(10)]
    example = {
        "id": "hotpotqa-long-abc",
        "question": "Which director was born first?",
        "paragraphs": paragraphs,
        "supporting_titles": {"T0", "T1"},
        "answer": "Scott Derrickson",
        "group": "comparison",
        "split": "validation",
    }
    pool = ["Ed Wood", "Tim Burton", "Joel Coen", "Ethan Coen", "David Lynch"]
    item = build_item(
        example, pool, k=6, min_chars=3000, target_chars=4000, max_chars=8000,
        rng=random.Random(2),
    )
    assert isinstance(item, TypedQuestion)
    assert item.id == "hotpotqa-long-abc"
    assert item.question["type"] == "choice"
    assert len(item.question["criteria"]) == 6
    assert item.question["criteria"][item.label] == "Scott Derrickson"
    assert item.state.startswith("T0:")
    assert len(item.state) >= 3000
    assert item.split == "validation"


def test_build_item_skips_short_state_and_yes_no():
    paragraphs = [("T0", "short")]
    example = {
        "id": "x",
        "question": "q",
        "paragraphs": paragraphs,
        "supporting_titles": {"T0"},
        "answer": "yes",
        "group": "comparison",
        "split": "validation",
    }
    assert build_item(example, ["yes", "no"], 6, 100, 200, 500, random.Random(1)) is None


def test_hotpot_row_normalises_context():
    row = {
        "id": "abc",
        "question": "q?",
        "answer": "Scott Derrickson",
        "type": "bridge",
        "context": {
            "title": ["A", "B"],
            "sentences": [["one.", "two."], ["three."]],
        },
        "supporting_facts": {"title": ["B"], "sent_id": [0]},
    }
    example = hotpot_row(row, "validation")
    assert example["paragraphs"] == [("A", "one. two."), ("B", "three.")]
    assert example["supporting_titles"] == {"B"}
    assert example["group"] == "bridge"


def test_question_implied_type_prefers_titles_for_position_questions():
    from data.long_context_choice import expected_type

    assert answer_type("Chief of Protocol") == "title"
    assert expected_type(
        "What government position was held by the woman who portrayed X?", "Chief of Protocol"
    ) == "title"
    assert expected_type("Who directed the film?", "Scott Derrickson") == "proper"
    assert expected_type("When was it released?", "1984") == "date"
    assert expected_type("How many episodes were there?", "42") == "number"


def test_pick_distractors_uses_the_question_type_not_only_the_gold_surface():
    pool = [
        "Chief of Staff", "Press Secretary", "Attorney General", "Ambassador",
        "Scott Derrickson", "1984",
    ]
    picked = pick_distractors(
        "Chief of Protocol", pool, "clean state", k=4, rng=random.Random(3),
        question="What government position was held by that person?",
    )
    assert picked is not None
    assert len(picked) == 3
    assert all(answer_type(value) == "title" for value in picked)


def test_musique_row_normalises_hops_and_supporting_paragraphs():
    from data.long_context_choice import musique_row

    row = {
        "id": "2hop__1_2",
        "question": "Who is the spouse of the Green performer?",
        "answer": "Miquette Giraudy",
        "answer_aliases": ["Miquette"],
        "answerable": True,
        "question_decomposition": [{"id": 1}, {"id": 2}],
        "paragraphs": [
            {"title": "A", "paragraph_text": "text a", "is_supporting": False},
            {"title": "B", "paragraph_text": "text b", "is_supporting": True},
        ],
    }
    example = musique_row(row, "validation")
    assert example["id"] == "musique-long-2hop__1_2"
    assert example["supporting_titles"] == {"B"}
    assert example["group"] == "2hop"
    assert example["aliases"] == ["Miquette"]

    row["answerable"] = False
    assert musique_row(row, "validation") is None


def test_gold_aliases_are_not_used_as_distractors():
    pool = ["Miquette Giraudy", "Miquette", "Steve Hillage", "Alice Coltrane"]
    picked = pick_distractors(
        "Miquette Giraudy", pool, "clean state", k=3, rng=random.Random(4),
        forbidden=["Miquette"],
    )
    assert picked is not None
    assert "Miquette" not in picked
    assert "Miquette Giraudy" not in picked
    assert len(picked) == 2


def test_build_item_accepts_a_musique_example():
    example = {
        "id": "musique-long-x",
        "question": "Who is the spouse of the Green performer?",
        "paragraphs": [(f"T{i}", "evidence " * 80) for i in range(10)],
        "supporting_titles": {"T0", "T1"},
        "answer": "Miquette Giraudy",
        "aliases": ["Miquette"],
        "group": "2hop",
        "split": "validation",
    }
    pool = ["Steve Hillage", "Alice Coltrane", "Miquette", "John McLaughlin"]
    item = build_item(
        example, pool, k=4, min_chars=4000, target_chars=5000, max_chars=8000,
        rng=random.Random(5),
    )
    assert item is not None
    assert item.id == "musique-long-x"
    assert item.group == "2hop"
    assert item.question["criteria"][item.label] == "Miquette Giraudy"
    assert "Miquette" not in item.question["criteria"].values()


def test_place_suffixes_are_not_person_distractors():
    from data.long_context_choice import expected_type

    assert answer_type("Green Bay") == "place"
    assert answer_type("Santa Clara County") == "place"
    assert expected_type("Where was the film shot?", "Paris") == "place"
    pool = ["Green Bay", "Santa Clara County", "George Benson", "Steve Hillage"]
    picked = pick_distractors(
        "Miquette Giraudy", pool, "clean state", k=3, rng=random.Random(6),
        question="Who is the spouse of the Green performer?",
    )
    assert sorted(picked) == ["George Benson", "Steve Hillage"]  # places excluded


def test_case_duplicate_candidates_are_deduplicated():
    pool = ["Ed Wood", "ed wood", "Tim Burton", "Joel Coen"]
    picked = pick_distractors(
        "Scott Derrickson", pool, "clean state", k=3, rng=random.Random(7),
        question="Who directed the film?",
    )
    assert picked is not None
    lowered = [value.lower() for value in picked]
    assert len(lowered) == len(set(lowered))


def test_title_words_only_count_as_positions_when_they_behave_like_them():
    assert answer_type("Chief of Protocol") == "title"
    assert answer_type("Israeli Ambassador") == "title"
    assert answer_type("a governor") == "title"
    assert answer_type("King Diamond") == "proper"
    assert answer_type("Captain Marvel") == "proper"


def test_organizations_are_not_person_distractors():
    assert answer_type("Digital First Media") == "organization"
    assert answer_type("Gmina Bystra-Sidzina") == "place"
    pool = ["Digital First Media", "Clio Goldsmith", "J. Robert Oppenheimer"]
    picked = pick_distractors(
        "Miquette Giraudy", pool, "clean state", k=3, rng=random.Random(8),
        question="Who is the spouse of the Green performer?",
    )
    assert sorted(picked) == ["Clio Goldsmith", "J. Robert Oppenheimer"]
