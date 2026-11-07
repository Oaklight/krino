"""Tests for the ability taxonomy and the corpus labeler (#247)."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest

from data.abilities import (
    ABILITIES,
    ABILITY_DEFINITIONS,
    CLASSIFICATION,
    DETECTION,
    HEURISTIC,
    ROUTING,
    SCORING,
    SEARCH,
    SOURCE_RULE,
    STRUCTURED_DATA_EXTRACTION,
    VERIFICATION,
    classify,
    classify_question,
)
from scripts.label_abilities import (
    Coverage,
    demand_report,
    gradient_eligible,
    jevbench_ability_map,
    main,
    option_count,
    score_handcheck,
    score_predictions,
)

JEVBENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "jevbench"


def _question(q_type: str, instructions: str, criteria=None) -> dict:
    question = {"type": q_type, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return question


def test_taxonomy_has_the_ten_documented_names():
    assert len(ABILITIES) == 10
    assert len(set(ABILITIES)) == 10
    assert set(ABILITY_DEFINITIONS) == set(ABILITIES)
    assert all(ABILITY_DEFINITIONS[name] for name in ABILITIES)


def test_score_questions_are_scoring_by_format():
    ability, method, rule = classify(
        "banking77", "b-1", _question("score", "How risky is this?", ["low", "high"])
    )
    assert (ability, method, rule) == (SCORING, SOURCE_RULE, "question_type:score")


@pytest.mark.parametrize(
    ("source", "ability"),
    [
        ("banking77", CLASSIFICATION),
        ("agnews", CLASSIFICATION),
        ("anli", VERIFICATION),
        ("contractnli", VERIFICATION),
        ("multirc", VERIFICATION),
        ("fever", VERIFICATION),
        ("codesearchnet", SEARCH),
    ],
)
def test_upstream_sources_use_the_source_rule(source: str, ability: str):
    got, method, rule = classify(
        source, f"{source}-1", _question("choice", "Which one?", {"A": "a", "B": "b"})
    )
    assert got == ability
    assert method == SOURCE_RULE
    assert rule == f"upstream_source:{source}"


@pytest.mark.parametrize(
    ("source", "ability"),
    [
        ("apigen_policy_compliance", VERIFICATION),
        ("apigen_tool_action", ROUTING),
        ("apigen_tool_argument", STRUCTURED_DATA_EXTRACTION),
        ("apigen_tool_choice", ROUTING),
        ("apigen_tool_next_action", ROUTING),
    ],
)
def test_apigen_files_use_the_file_rule(source: str, ability: str):
    got, method, _ = classify(
        source, f"{source}-1", _question("noul", "Does it comply?")
    )
    assert (got, method) == (ability, SOURCE_RULE)


@pytest.mark.parametrize(
    ("instruction", "ability"),
    [
        ("What should the observability system do with this trace?", ROUTING),
        (
            "The evidence indicates a credential or account has been compromised.",
            DETECTION,
        ),
        (
            "The invoice reconciles with the purchase order and the recorded delivery.",
            VERIFICATION,
        ),
        ("How did this agent run turn out?", CLASSIFICATION),
        ("How risky was the agent's behaviour in this trace?", SCORING),
    ],
)
def test_typed_decisions_instruction_table(instruction: str, ability: str):
    got, method, _ = classify(
        "typed_decisions", "typed-1", _question("choice", instruction)
    )
    assert (got, method) == (ability, SOURCE_RULE)


def test_typed_unknown_instruction_falls_to_the_type_default():
    got, method, rule = classify(
        "typed_decisions",
        "typed-1",
        _question("choice", "A new question form?", {"A": "a", "B": "b"}),
    )
    assert (got, method) == (CLASSIFICATION, HEURISTIC)
    assert rule.startswith("type_default:choice")


def test_synthetic_template_is_a_source_rule():
    got, method, _ = classify(
        "synthetic",
        "synthetic-game_strategy-0001-choice-1",
        _question(
            "choice",
            "Which action is the strongest strategic move?",
            {"A": "a", "B": "b"},
        ),
    )
    assert (got, method) == (ROUTING, SOURCE_RULE)


@pytest.mark.parametrize(
    ("instruction", "q_type", "ability", "rule_prefix"),
    [
        (
            "Does the passage support the claim that the policy changed?",
            "noul",
            VERIFICATION,
            "pattern:",
        ),
        (
            "How would you classify the accuracy of the claim?",
            "choice",
            CLASSIFICATION,
            "type_default:",
        ),
        (
            "What value should the agent pass for the 'zip' argument?",
            "choice",
            STRUCTURED_DATA_EXTRACTION,
            "pattern:",
        ),
    ],
)
def test_synthetic_free_form_uses_the_heuristic(
    instruction: str, q_type: str, ability: str, rule_prefix: str
):
    got, method, rule = classify(
        "synthetic",
        "synthetic-legal_judgment-0001-choice-1",
        _question(q_type, instruction, {"A": "a", "B": "b"}),
    )
    assert (got, method) == (ability, HEURISTIC)
    assert rule.startswith(rule_prefix)


def test_evidence_sufficiency_is_verification():
    for instruction in (
        "Is there sufficient detail in the passage to establish the rule?",
        "Is there evidence that the symptom developed before treatment?",
        "Is there enough detail in the scenario to confirm the action?",
    ):
        got, method, _ = classify(
            "synthetic",
            "synthetic-multi_hop_reasoning-0001-noul",
            _question("noul", instruction),
        )
        assert (got, method) == (VERIFICATION, HEURISTIC)


@pytest.mark.parametrize(
    "instruction",
    [
        "Did the recovery fail to end the comeback threat?",
        "Which step in the student's solution was executed correctly?",
        "Is it possible for the rover to retrieve the sample before the storm?",
        "Does the passage indicate that the highway connects two cities?",
        "Does this serum contain more than 10% vitamin C?",
    ],
)
def test_incidental_verbs_do_not_force_detection_or_routing(instruction: str):
    got, method, _ = classify(
        "synthetic",
        "synthetic-sequential_action-0001-choice-1",
        _question("choice", instruction, {"A": "a", "B": "b"}),
    )
    assert got == CLASSIFICATION
    assert method == HEURISTIC


def test_unknown_source_fails_closed():
    with pytest.raises(ValueError, match="no ability rule"):
        classify("mystery_source", "x-1", _question("choice", "Which one?"))


def _manifest() -> dict:
    return {
        "groups": {
            "banking77": {
                "verdict": "permissive",
                "included_in": [
                    "full",
                    "exclude_restrictive",
                    "exclude_restrictive_and_unresolved",
                ],
            },
            "anli": {
                "verdict": "restrictive",
                "included_in": ["full"],
            },
        }
    }


def test_coverage_counts_exclusions_and_gradient():
    coverage = Coverage(_manifest())
    coverage.add(
        {
            "id": "b-1",
            "source": "banking77",
            "question": _question("choice", "Which intent?", {"a": "A", "b": "B"}),
        },
        CLASSIFICATION,
        SOURCE_RULE,
        "upstream_source:banking77",
    )
    coverage.add(
        {
            "id": "a-1",
            "source": "anli",
            "question": _question("noul", "Does the premise entail?"),
        },
        VERIFICATION,
        SOURCE_RULE,
        "upstream_source:anli",
    )
    report = coverage.report(Path("canonical.jsonl"), "0" * 64)
    assert report["canonical_records"] == 2
    assert report["matrix"][CLASSIFICATION]["total"] == 1
    assert report["abilities"][CLASSIFICATION]["after_exclude_restrictive"] == 1
    assert report["abilities"][VERIFICATION]["after_exclude_restrictive"] == 0
    assert report["abilities"][CLASSIFICATION]["single_source"] is True
    assert report["abilities"][CLASSIFICATION]["licence_single_point"] is False
    assert report["abilities"][VERIFICATION]["licence_single_point"] is True
    assert "single_source" in report["abilities"][VERIFICATION]["flags"]
    assert "licence_single_point" in report["abilities"][VERIFICATION]["flags"]
    assert report["abilities"][CLASSIFICATION]["cookbook"].startswith("demonstrated")
    assert report["abilities"][VERIFICATION]["gradient_eligible_listwise"] == 0
    assert report["abilities"][VERIFICATION]["gradient_share"] == 0.0
    assert report["abilities"][CLASSIFICATION]["gradient_share"] == 1.0
    assert "under_2000" in report["abilities"][CLASSIFICATION]["flags"]
    assert "no_listwise_gradient" in report["abilities"][VERIFICATION]["flags"]
    assert report["types"]["noul"]["gradient_eligible_listwise"] == 0
    assert report["trainers"]["listwise_pairwise"]["eligible_rows"] == 1


def test_option_count_and_gradient_eligibility():
    assert option_count({"type": "noul"}) == 2
    assert option_count({"type": "choice", "criteria": {"a": "A"}}) == 1
    assert option_count({"type": "score", "criteria": ["low", "high"]}) == 2
    assert gradient_eligible({"type": "noul"}, "listwise_pairwise") is False
    assert gradient_eligible({"type": "noul"}, "residual") is True
    assert (
        gradient_eligible({"type": "choice", "criteria": {"a": "A"}}, "residual")
        is False
    )


def test_labeler_end_to_end(tmp_path: Path):
    canonical = tmp_path / "canonical.jsonl"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    rows = [
        {
            "id": "b-1",
            "source": "banking77",
            "question": _question("choice", "Which intent?", {"a": "A", "b": "B"}),
            "label": "a",
        },
        {
            "id": "a-1",
            "source": "anli",
            "question": _question("noul", "Does the premise entail?"),
            "label": True,
        },
    ]
    canonical.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    out = tmp_path / "labeled.jsonl"
    map_out = tmp_path / "ability-map.jsonl"
    coverage = tmp_path / "coverage.json"
    code = main(
        [
            "--canonical",
            str(canonical),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--map-out",
            str(map_out),
            "--coverage",
            str(coverage),
        ]
    )
    assert code == 0
    labeled = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(row["ability"] in ABILITIES for row in labeled)
    assert all(row["label_method"] == SOURCE_RULE for row in labeled)
    assert labeled[0]["ability_rule"] == "upstream_source:banking77"
    sidecar = [json.loads(line) for line in map_out.read_text().splitlines()]
    assert {row["id"] for row in sidecar} == {"b-1", "a-1"}
    assert map_out.with_name(map_out.name + ".sha256").exists()
    report = json.loads(coverage.read_text())
    assert report["canonical_records"] == 2
    assert (
        report["canonical_sha256"] == hashlib.sha256(canonical.read_bytes()).hexdigest()
    )
    assert report["sidecar"]["records"] == 2
    assert report["label_methods"] == {"source_rule": 2, "heuristic": 0, "model": 0}


def test_classify_question_uses_the_question_form_only():
    ability, method, rule = classify_question(
        _question("choice", "What should the agent do next?", {"A": "a", "B": "b"})
    )
    assert (ability, method) == (ROUTING, HEURISTIC)
    assert rule.startswith("pattern:")


def test_jevbench_items_have_an_ability_by_question_form():
    mapping = jevbench_ability_map(JEVBENCH_DIR)
    assert len(mapping) == 231
    assert set(mapping.values()) <= set(ABILITIES)
    assert mapping["easy-intent-00"] == CLASSIFICATION


def test_demand_report_matches_the_jevbench_mix():
    mapping = jevbench_ability_map(JEVBENCH_DIR)
    totals = collections.Counter({ability: 0 for ability in ABILITIES})
    totals[CLASSIFICATION] = 150_571
    totals[SCORING] = 29_515
    report = demand_report(mapping, totals)
    assert report["by_ability"][CLASSIFICATION] == 108
    assert report["by_ability"][SCORING] == 18
    assert report["by_ability"][ROUTING] == 37
    assert report["by_ability"][VERIFICATION] == 44
    assert report["by_ability"][DETECTION] == 0
    assert report["share"][CLASSIFICATION] == pytest.approx(108 / 231, abs=1e-6)
    assert len(report["notes"]) == 4


def test_score_predictions_joins_by_id(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "b-1", "q_type": "choice", "n_options": 4, "correct": True},
        {"id": "b-1", "q_type": "choice", "n_options": 4, "correct": False},
        {"id": "a-1", "q_type": "noul", "correct": True},
        {"id": "not-in-corpus", "q_type": "noul", "correct": True},
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = score_predictions(
        predictions, {"b-1": CLASSIFICATION, "a-1": VERIFICATION}
    )
    assert report["matched"] == 3
    assert report["unmatched"] == 1
    assert report["by_ability"][CLASSIFICATION]["accuracy"] == 0.5
    assert report["by_ability"][CLASSIFICATION]["chance_corrected"] == pytest.approx(
        ((1 - 0.25) / 0.75 + (0 - 0.25) / 0.75) / 2, abs=1e-6
    )
    assert report["by_ability"][VERIFICATION]["chance_corrected"] == 1.0


def test_labeler_rejects_duplicate_ids(tmp_path: Path):
    canonical = tmp_path / "canonical.jsonl"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    row = {
        "id": "b-1",
        "source": "banking77",
        "question": _question("choice", "Which intent?", {"a": "A", "b": "B"}),
        "label": "a",
    }
    canonical.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--canonical",
                str(canonical),
                "--manifest",
                str(manifest),
                "--coverage",
                str(tmp_path / "coverage.json"),
            ]
        )


def test_handcheck_scores_agreement(tmp_path: Path):
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            {
                "seed": 1,
                "per_ability": 2,
                "records": [
                    {
                        "id": "b-1",
                        "classifier_ability": CLASSIFICATION,
                        "human_ability": CLASSIFICATION,
                    },
                    {
                        "id": "b-2",
                        "classifier_ability": CLASSIFICATION,
                        "human_ability": ROUTING,
                    },
                    {
                        "id": "a-1",
                        "classifier_ability": VERIFICATION,
                        "human_ability": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = score_handcheck(sample, None)
    assert report["labelled"] == 2
    assert report["unlabelled"] == 1
    assert report["agreement"] == 0.5
    assert report["by_ability"][CLASSIFICATION]["disagreements"] == {ROUTING: 1}
