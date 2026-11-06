"""Tests for data/provenance.py."""

from __future__ import annotations

import pytest

from data.format import TypedQuestion
from data.provenance import (
    EXECUTION,
    GOLD,
    TEACHER,
    classify,
    may_drive_hard_label,
    normalise_label,
    tag,
)


def _noul(label=True, source="multirc", **kwargs):
    item = TypedQuestion.noul(
        id=kwargs.pop("id", "multirc-train-00001"),
        state="s",
        instructions="q",
        label=label,
        source=source,
        split="train",
        **kwargs,
    )
    return item


def test_dataset_source_is_gold():
    c = classify(_noul())
    assert (c.tier, c.origin) == (GOLD, "dataset")


def test_long_context_and_task_sources_are_gold():
    for source in (
        "hotpotqa_long_choice",
        "longbench_long_choice",
        "2wiki_long_choice",
        "mmlu_pro_short_reasoning_choice",
        "race_short_passage_choice",
    ):
        assert classify(_noul(source=source)).tier == GOLD


def test_community_authored_is_gold():
    assert classify(_noul(source="typed_decisions")).origin == "community_authored"
    assert classify(_noul(source="openjev_authored144")).origin == "community_authored"


def test_synthetic_is_gold_synthetic_authored():
    c = classify(_noul(source="synthetic", id="synthetic-x-0001-noul-fact"))
    assert (c.tier, c.origin) == (GOLD, "synthetic_authored")


def test_apigen_is_execution_trajectory():
    item = _noul(source="apigen_tool_choice", id="apigen-tool-00001")
    assert classify(item).tier == EXECUTION
    assert classify(item, corpus="apigen_tool_choice.jsonl").origin == "trajectory"


def test_abstention_is_execution_by_id_suffix():
    item = _noul(id="arc-train-1-abstain-none_of_these", source="arc")
    assert classify(item).tier == EXECUTION
    assert classify(item).origin == "constructed"


def test_abstention_is_execution_by_corpus_name():
    item = _noul(source="arc")
    assert classify(item, corpus="abstention_train_20k.jsonl").tier == EXECUTION


def test_declared_teacher_is_respected():
    item = _noul()
    item.provenance = TEACHER
    assert classify(item).tier == TEACHER


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        classify(_noul(source="mystery_dataset"))


def test_tag_sets_field_once():
    item = _noul()
    tag(item)
    assert item.provenance == GOLD
    assert tag(item).tier == GOLD


def test_tag_passes_corpus_to_classify():
    item = _noul(source="arc", id="arc-train-1")
    tag(item, "abstention_train_20k.jsonl")
    assert item.provenance == EXECUTION
    assert classify(item).tier == EXECUTION


def test_tag_stamps_dict_with_corpus():
    item = {
        "id": "arc-train-2",
        "source": "arc",
        "question": {"type": "noul", "instructions": "q"},
        "label": False,
    }
    tag(item, "abstention_train_20k.jsonl")
    assert item["provenance"] == EXECUTION


def test_normalise_label_coerces_strings():
    assert normalise_label(_noul(label="true")) is True
    assert normalise_label(_noul(label="false")) is False
    score = TypedQuestion.score(
        id="sst5-train-1",
        state="s",
        instructions="q",
        criteria=["a", "b", "c"],
        label="2",
        source="sst5",
        split="train",
    )
    assert normalise_label(score) == 2.0


def test_hard_label_eligible_for_valid_gold():
    assert may_drive_hard_label(_noul())
    assert may_drive_hard_label(_noul(label="true"))
    choice = TypedQuestion.choice(
        id="winogrande-train-1",
        state="s",
        instructions="q",
        criteria={"1": "a", "2": "b"},
        label="2",
        source="winogrande",
        split="train",
    )
    assert may_drive_hard_label(choice)


def test_hard_label_ineligible_for_teacher():
    item = _noul()
    item.provenance = TEACHER
    assert not may_drive_hard_label(item)


def test_hard_label_ineligible_for_abstention():
    item = _noul(id="arc-train-1-abstain-none_of_these", source="arc")
    item.answerable = False
    item.reason = "none_of_these"
    assert not may_drive_hard_label(item)


def test_hard_label_ineligible_when_choice_label_absent():
    item = TypedQuestion.choice(
        id="x",
        state="s",
        instructions="q",
        criteria={"1": "a", "2": "b"},
        label="3",
        source="winogrande",
        split="train",
    )
    assert not may_drive_hard_label(item)


def test_hard_label_ineligible_for_unknown_source():
    assert not may_drive_hard_label(_noul(source="mystery_dataset"))
