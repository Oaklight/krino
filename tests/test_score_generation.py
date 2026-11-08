"""Tests for score-question generation (#156)."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from data.score_generation import (
    Candidate,
    allocate_quotas,
    build_item,
    candidate_id,
    evidence_problem,
    expected_level,
    iter_candidates,
    label_level,
    level0_allowed,
    load_rubric_config,
    normalise_state,
    order_candidates,
    select_candidates,
    validate_distribution,
)

RUBRIC = {
    "instructions": "How severe is this situation?",
    "criteria": ["None", "Low", "Moderate", "High"],
}


def _item(item_id, state, source="fixture", q_type="noul", split="train"):
    return {
        "id": item_id,
        "state": state,
        "question": {"type": q_type, "instructions": "q", "criteria": ["a", "b"]},
        "label": True,
        "source": source,
        "split": split,
    }


def _candidate(state, source="fixture"):
    return Candidate(source=source, state=state, split="train", group=None, rubric=RUBRIC)


def test_normalise_and_id_are_stable_and_distinct():
    a = candidate_id("mednli", "Patient  has a fever.")
    b = candidate_id("mednli", "patient has a fever .")
    c = candidate_id("contractnli", "Patient  has a fever.")
    assert a == b
    assert a != c
    assert a.startswith("scoregen-mednli-")
    assert normalise_state("  Café  AU lait ") == "café au lait"


def test_iter_candidates_skips_existing_score_states_and_duplicates():
    items = [
        _item("1", "State one.", source="a"),
        _item("2", "State one.", source="a"),
        _item("3", "State two.", source="b", q_type="score"),
        _item("4", "State three.", source="b"),
    ]
    score_states = {normalise_state("State two.")}
    rubrics = {
        "a": {**RUBRIC, "evidence": "state text"},
        "b": {**RUBRIC, "evidence": "state text"},
    }
    candidates = list(iter_candidates(items, score_states, rubrics))
    assert [candidate.original_id for candidate in candidates] == ["1", "4"]
    assert [candidate.source for candidate in candidates] == ["a", "b"]


def test_iter_candidates_errors_on_a_source_with_no_rubric():
    items = [_item("1", "State one.", source="undeclared")]
    with pytest.raises(ValueError, match="undeclared"):
        list(iter_candidates(items, set(), {}))


def test_iter_candidates_skips_states_without_the_declared_evidence():
    items = [
        _item("1", "Contract clause: the supplier is liable.\n\nProposition: yes.", source="c"),
        _item("2", "Proposition: no clause is given.", source="c"),
        _item("3", "", source="c"),
    ]
    rubrics = {"c": {**RUBRIC, "evidence": "Contract clause"}}
    skipped = Counter()
    candidates = list(iter_candidates(items, set(), rubrics, skipped))
    assert [candidate.original_id for candidate in candidates] == ["1"]
    assert skipped["evidence:c"] == 1


def test_evidence_problem_reads_sections_json_keys_and_whole_state():
    assert evidence_problem("Contract clause: x\n\nProposition: y", "Contract clause") is None
    assert evidence_problem("Proposition: y", "Contract clause") is not None
    assert evidence_problem("Contract clause:   \n\nProposition: y", "Contract clause") is not None
    assert evidence_problem('{"System": "policy text"}', "System") is None
    assert evidence_problem('{"System": ""}', "System") is not None
    assert evidence_problem("anything", "state text") is None
    assert evidence_problem("   ", "state text") is not None


def test_allocate_quotas_sums_to_target_and_keeps_coverage():
    pools = {"small": 10, "medium": 500, "large": 5000}
    quota = allocate_quotas(pools, target=300, floor=50, cap=200)
    assert sum(quota.values()) == 300
    assert quota["small"] == 10
    assert quota["large"] <= 200
    assert all(value >= 0 for value in quota.values())


def test_allocate_quotas_is_deterministic_and_caps_at_pool_size():
    pools = {"a": 3, "b": 4}
    assert allocate_quotas(pools, target=100, floor=5, cap=10) == {"a": 3, "b": 4}
    first = allocate_quotas({"a": 10, "b": 90}, target=40, floor=5, cap=40)
    second = allocate_quotas({"a": 10, "b": 90}, target=40, floor=5, cap=40)
    assert first == second


def test_level0_allowed_enforces_the_ceiling_invariant():
    assert level0_allowed(0.5, 0, 0) is True
    assert level0_allowed(0.5, 1, 1) is False
    # n=3: ceil(0.5 * 3) = 2, so a second level-0 row is allowed and the share
    # that results is 66.7%, above the nominal 50%. The invariant is the
    # ceiling, not a flat share.
    assert level0_allowed(0.5, 2, 1) is True
    assert level0_allowed(0.5, 3, 2) is False
    assert level0_allowed(0.0, 10, 10) is True


def test_select_candidates_respects_quotas_and_is_seeded():
    candidates = [_candidate(f"state {source} {index}", source) for source in ("a", "b") for index in range(20)]
    first = select_candidates(candidates, target=30, floor=5, cap=20, seed=7)
    second = select_candidates(candidates, target=30, floor=5, cap=20, seed=7)
    assert [candidate.state for candidate in first] == [candidate.state for candidate in second]
    assert len(first) == 30
    assert {candidate.source for candidate in first} == {"a", "b"}


def test_order_candidates_interleaves_sources():
    candidates = [_candidate(f"a{i}", "a") for i in range(3)] + [
        _candidate(f"b{i}", "b") for i in range(2)
    ]
    ordered = order_candidates(candidates, ["a", "b"])
    assert [candidate.source for candidate in ordered] == ["a", "b", "a", "b", "a"]


@pytest.mark.parametrize(
    ("probabilities", "reason"),
    [
        ({"0": 0.2, "1": 0.3, "2": 0.5}, None),
        ({"0": 0.2, "1": 0.3}, "keys"),
        ({"0": 0.2, "1": 0.3, "2": 0.5, "3": 0.0}, "keys"),
        ({"0": 0.2, "1": 0.3, "2": -0.1}, "range"),
        ({"0": 0.2, "1": 0.3, "2": 0.4}, "sum"),
        ({"0": 0.995, "1": 0.005, "2": 0.0}, "degenerate"),
        ({"0": 0.0, "1": 0.0, "2": 1.0}, "degenerate"),
        ({"0": "x", "1": 0.3, "2": 0.7}, "nonnumeric"),
    ],
)
def test_validate_distribution(probabilities, reason):
    assert validate_distribution(probabilities, 3) == reason


def test_label_is_argmax_and_expected_is_reported_separately():
    probabilities = {"0": 0.1, "1": 0.55, "2": 0.35}
    assert label_level(probabilities) == 1.0
    assert expected_level(probabilities) == pytest.approx(1.25)


def test_build_item_carries_teacher_provenance_and_distribution():
    candidate = _candidate("A state.")
    probabilities = {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1}
    item = build_item(candidate, probabilities)
    assert item["label"] == 1.0
    assert item["provenance"] == "teacher"
    assert item["teacher_probs"] == {"jev": probabilities}
    assert item["question"]["criteria"] == RUBRIC["criteria"]
    assert item["id"] == candidate.id


def test_load_rubric_config_resolves_templates_and_carries_evidence(tmp_path):
    path = tmp_path / "rubrics.json"
    path.write_text(
        json.dumps(
            {
                "sources": {
                    "fixture": {"template": "fake", "evidence": "state text"},
                    "inline": {
                        "instructions": "Inline?",
                        "criteria": ["a", "b"],
                        "evidence": "Docstring",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    templates = {"fake": {"score_template": {"instructions": "From template?", "criteria": ["x", "y"]}}}
    rubrics = load_rubric_config(str(path), templates)
    assert rubrics["fixture"]["instructions"] == "From template?"
    assert rubrics["fixture"]["evidence"] == "state text"
    assert rubrics["inline"]["instructions"] == "Inline?"
    assert rubrics["inline"]["evidence"] == "Docstring"


def test_load_rubric_config_requires_evidence_on_every_source(tmp_path):
    path = tmp_path / "rubrics.json"
    path.write_text(
        json.dumps({"sources": {"fixture": {"template": "fake"}}}),
        encoding="utf-8",
    )
    templates = {"fake": {"score_template": {"instructions": "From template?", "criteria": ["x", "y"]}}}
    with pytest.raises(ValueError, match="evidence"):
        load_rubric_config(str(path), templates)


def test_load_rubric_config_rejects_a_default_rubric(tmp_path):
    path = tmp_path / "rubrics.json"
    path.write_text(
        json.dumps(
            {
                "sources": {"fixture": {"template": "fake", "evidence": "state text"}},
                "default": {"instructions": "Default?", "criteria": ["a", "b"]},
            }
        ),
        encoding="utf-8",
    )
    templates = {"fake": {"score_template": {"instructions": "From template?", "criteria": ["x", "y"]}}}
    with pytest.raises(ValueError, match="default"):
        load_rubric_config(str(path), templates)
