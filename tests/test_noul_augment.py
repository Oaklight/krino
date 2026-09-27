"""Tests for noul-to-choice augmentation pipeline stage.

Covers templates, matching, LLM response parsing, and propagation through
family_to_typed_questions. All tests run offline — no API calls.
"""

from __future__ import annotations

import json

import pytest

from data.format import TypedQuestion
from data.synthetic_noul_augment import (
    COGNITIVE_TYPE_AUGMENT_TEMPLATES,
    _build_augment_prompt,
    _build_full_options,
    match_template,
    parse_augment_response,
)
from data.synthetic_templates import COGNITIVE_TYPES


# ---------------------------------------------------------------------------
# Template quality
# ---------------------------------------------------------------------------


class TestTemplateDefinitions:
    def test_all_cognitive_types_have_templates(self):
        for ct in COGNITIVE_TYPES:
            assert ct in COGNITIVE_TYPE_AUGMENT_TEMPLATES, f"Missing template for {ct}"

    def test_all_templates_include_true_false(self):
        for ct, template in COGNITIVE_TYPE_AUGMENT_TEMPLATES.items():
            assert "true" in template, f"{ct} template missing 'true'"
            assert "false" in template, f"{ct} template missing 'false'"

    def test_all_templates_have_4_to_6_options(self):
        for ct, template in COGNITIVE_TYPE_AUGMENT_TEMPLATES.items():
            n = len(template)
            assert 4 <= n <= 6, f"{ct} template has {n} options, expected 4-6"

    def test_template_keys_are_snake_case(self):
        import re

        pat = re.compile(r"^[a-z][a-z0-9_]*$")
        for ct, template in COGNITIVE_TYPE_AUGMENT_TEMPLATES.items():
            for key in template:
                assert pat.match(key), f"{ct} template has non-snake_case key: {key}"

    def test_template_descriptions_are_nonempty_strings(self):
        for ct, template in COGNITIVE_TYPE_AUGMENT_TEMPLATES.items():
            for key, desc in template.items():
                assert isinstance(desc, str), f"{ct}.{key} description is not str"
                assert desc.strip(), f"{ct}.{key} description is empty"


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


class TestTemplateMatching:
    def test_match_by_cognitive_type(self):
        nq = {"cognitive_type": "entailment", "instructions": "Does X entail Y?"}
        result = match_template(nq)
        assert result is not None
        assert "true" in result
        assert "false" in result
        assert len(result) >= 4

    def test_match_all_types(self):
        for ct in COGNITIVE_TYPES:
            nq = {"cognitive_type": ct, "instructions": "test"}
            result = match_template(nq)
            assert result is not None, f"match_template returned None for {ct}"

    def test_unknown_type_returns_none(self):
        nq = {"cognitive_type": "unknown", "instructions": "something"}
        assert match_template(nq) is None

    def test_missing_type_returns_none(self):
        nq = {"instructions": "something"}
        assert match_template(nq) is None

    def test_empty_type_returns_none(self):
        nq = {"cognitive_type": "", "instructions": "something"}
        assert match_template(nq) is None


# ---------------------------------------------------------------------------
# LLM prompt building
# ---------------------------------------------------------------------------


class TestBuildAugmentPrompt:
    def test_contains_state(self):
        prompt = _build_augment_prompt("patient has fever", "Is it serious?", True, "safety")
        assert "patient has fever" in prompt

    def test_contains_instructions(self):
        prompt = _build_augment_prompt("state", "Is it serious?", True, "safety")
        assert "Is it serious?" in prompt

    def test_specifies_json_output(self):
        prompt = _build_augment_prompt("state", "question", False, "causal")
        assert "JSON" in prompt

    def test_excludes_true_false_instruction(self):
        prompt = _build_augment_prompt("state", "question", True, "entailment")
        assert "Do NOT include" in prompt or "Do not include" in prompt


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


class TestParseAugmentResponse:
    def test_valid_response(self):
        response = json.dumps(
            {
                "additional_options": {
                    "partially_true": "The fact is partly correct",
                    "unverifiable": "Cannot be verified",
                    "outdated": "No longer current",
                }
            }
        )
        result = parse_augment_response(response)
        assert result is not None
        assert "true" in result
        assert "false" in result
        assert "partially_true" in result
        assert len(result) == 5

    def test_response_with_code_fences(self):
        inner = json.dumps(
            {
                "additional_options": {
                    "partial": "Partially correct",
                    "uncertain": "Cannot determine",
                    "context_dependent": "Depends on context",
                }
            }
        )
        response = f"```json\n{inner}\n```"
        result = parse_augment_response(response)
        assert result is not None
        assert len(result) == 5

    def test_filters_true_false_from_additional(self):
        response = json.dumps(
            {
                "additional_options": {
                    "true": "should be filtered",
                    "false": "should be filtered",
                    "partial": "Partially correct",
                    "uncertain": "Cannot determine",
                }
            }
        )
        result = parse_augment_response(response)
        assert result is not None
        # "true" and "false" come from _build_full_options, not the LLM
        assert "partial" in result
        assert "uncertain" in result

    def test_malformed_json_returns_none(self):
        assert parse_augment_response("not json at all") is None

    def test_missing_key_returns_none(self):
        assert parse_augment_response(json.dumps({"wrong_key": {}})) is None

    def test_too_few_options_returns_none(self):
        response = json.dumps({"additional_options": {"only_one": "just one"}})
        assert parse_augment_response(response) is None

    def test_non_snake_case_filtered(self):
        response = json.dumps(
            {
                "additional_options": {
                    "ValidButCamel": "should be filtered",
                    "has spaces": "should be filtered",
                    "good_key": "Valid option",
                    "another_good": "Also valid",
                }
            }
        )
        result = parse_augment_response(response)
        assert result is not None
        assert "good_key" in result
        assert "another_good" in result
        assert "ValidButCamel" not in result


# ---------------------------------------------------------------------------
# Build full options
# ---------------------------------------------------------------------------


class TestBuildFullOptions:
    def test_includes_true_false(self):
        result = _build_full_options({"partial": "desc"})
        assert "true" in result
        assert "false" in result
        assert "partial" in result

    def test_additional_overwrites_nothing(self):
        result = _build_full_options({"extra_a": "A", "extra_b": "B"})
        assert len(result) == 4


# ---------------------------------------------------------------------------
# TypedQuestion.noul augmented_options
# ---------------------------------------------------------------------------


class TestTypedQuestionAugmentedOptions:
    def test_noul_without_augmented_options(self):
        tq = TypedQuestion.noul(
            id="test-1",
            state="test state",
            instructions="Is it true?",
            label=True,
            source="test",
            split="train",
        )
        assert "augmented_options" not in tq.question

    def test_noul_with_augmented_options(self):
        opts = {"true": "Yes", "false": "No", "partial": "Maybe"}
        tq = TypedQuestion.noul(
            id="test-1",
            state="test state",
            instructions="Is it true?",
            label=True,
            source="test",
            split="train",
            augmented_options=opts,
        )
        assert tq.question["augmented_options"] == opts

    def test_roundtrip_augmented_typed_question(self):
        opts = {"true": "Yes", "false": "No", "partial": "Maybe", "unknown": "Unclear"}
        tq = TypedQuestion.noul(
            id="test-1",
            state="test state",
            instructions="Is it true?",
            label=True,
            source="test",
            split="train",
            augmented_options=opts,
        )
        d = tq.to_dict()
        restored = TypedQuestion(**d)
        assert restored.question["augmented_options"] == opts


# ---------------------------------------------------------------------------
# family_to_typed_questions propagation
# ---------------------------------------------------------------------------


class TestFamilyToTypedQuestionsPropagation:
    """Test that augmented_options flows through all noul emission points."""

    @pytest.fixture()
    def augmented_family(self):
        return {
            "state": "Patient has high fever and cough",
            "noul_questions": [
                {
                    "instructions": "Is the patient at risk?",
                    "label": True,
                    "cognitive_type": "safety",
                    "augmented_options": {
                        "true": "Risk exists",
                        "false": "No risk",
                        "minor_risk": "Low risk",
                        "context_dependent": "Depends",
                        "insufficient_safety_data": "Cannot assess",
                    },
                }
            ],
            "choice_questions": [],
            "score_questions": [],
        }

    @pytest.fixture()
    def variants_with_negation(self):
        return {
            "counterfactual": {
                "state": "Patient has mild headache only",
                "noul_labels": [{"cognitive_type": "safety", "label": False}],
            },
            "paraphrase": {
                "state_paraphrase": "A patient presents with elevated temperature and coughing",
                "question_paraphrases": [
                    {
                        "original_instructions": "Is the patient at risk?",
                        "paraphrased_instructions": "Does the patient face danger?",
                    }
                ],
            },
            "negation": {
                "negated_questions": [
                    {
                        "cognitive_type": "safety",
                        "negated_instructions": "Is the patient NOT at risk?",
                        "negated_label": False,
                    }
                ]
            },
        }

    def _run_conversion(self, family, variants, enabled_stages=None):
        from data.synthetic import family_to_typed_questions
        from data.synthetic_templates import DOMAIN_TEMPLATES

        # Use a real domain template
        domain = list(DOMAIN_TEMPLATES.keys())[0]
        return family_to_typed_questions(
            family, variants, domain, 0, enabled_stages=enabled_stages
        )

    def test_base_noul_has_augmented_options(self, augmented_family):
        items = self._run_conversion(augmented_family, {}, enabled_stages={"base"})
        noul_items = [i for i in items if i.get("question", {}).get("type") == "noul"]
        assert len(noul_items) >= 1
        assert "augmented_options" in noul_items[0]["question"]
        assert len(noul_items[0]["question"]["augmented_options"]) == 5

    def test_counterfactual_noul_has_augmented_options(
        self, augmented_family, variants_with_negation
    ):
        items = self._run_conversion(
            augmented_family, variants_with_negation, enabled_stages=None
        )
        cf_items = [i for i in items if "-cf-noul-" in i.get("id", "")]
        assert len(cf_items) >= 1
        assert "augmented_options" in cf_items[0]["question"]

    def test_paraphrase_state_noul_has_augmented_options(
        self, augmented_family, variants_with_negation
    ):
        items = self._run_conversion(
            augmented_family, variants_with_negation, enabled_stages=None
        )
        para_items = [i for i in items if "-para-state-noul-" in i.get("id", "")]
        assert len(para_items) >= 1
        assert "augmented_options" in para_items[0]["question"]

    def test_negation_noul_has_augmented_options(
        self, augmented_family, variants_with_negation
    ):
        items = self._run_conversion(
            augmented_family, variants_with_negation, enabled_stages=None
        )
        neg_items = [i for i in items if "-neg-noul-" in i.get("id", "")]
        assert len(neg_items) >= 1
        assert "augmented_options" in neg_items[0]["question"]

    def test_non_augmented_noul_has_no_augmented_options(self):
        family = {
            "state": "Test state",
            "noul_questions": [
                {
                    "instructions": "Is it true?",
                    "label": True,
                    "cognitive_type": "entailment",
                }
            ],
            "choice_questions": [],
            "score_questions": [],
        }
        items = self._run_conversion(family, {}, enabled_stages={"base"})
        noul_items = [i for i in items if i.get("question", {}).get("type") == "noul"]
        assert len(noul_items) >= 1
        assert "augmented_options" not in noul_items[0]["question"]


# ---------------------------------------------------------------------------
# augment_noul_questions function
# ---------------------------------------------------------------------------


class TestAugmentNoulQuestions:
    @pytest.mark.asyncio
    async def test_template_augmentation(self):
        from data.synthetic_noul_augment import augment_noul_questions

        families = [
            {
                "state": "Test state",
                "noul_questions": [
                    {
                        "instructions": "Does X entail Y?",
                        "label": True,
                        "cognitive_type": "entailment",
                    },
                    {
                        "instructions": "Is the fact correct?",
                        "label": False,
                        "cognitive_type": "fact_verification",
                    },
                ],
            }
        ]

        updated, count, failed = await augment_noul_questions(None, families, "", "")
        assert count == 2
        assert failed == 0
        for nq in updated[0]["noul_questions"]:
            assert "augmented_options" in nq
            assert nq["_augment_source"] == "template"
            assert len(nq["augmented_options"]) == 5

    @pytest.mark.asyncio
    async def test_already_augmented_skipped(self):
        from data.synthetic_noul_augment import augment_noul_questions

        families = [
            {
                "state": "Test state",
                "noul_questions": [
                    {
                        "instructions": "test",
                        "label": True,
                        "cognitive_type": "entailment",
                        "augmented_options": {"true": "T", "false": "F", "p": "P", "q": "Q"},
                    }
                ],
            }
        ]

        updated, count, failed = await augment_noul_questions(None, families, "", "")
        assert count == 0
        assert failed == 0
        # Original options preserved
        assert updated[0]["noul_questions"][0]["augmented_options"]["p"] == "P"

    @pytest.mark.asyncio
    async def test_unknown_type_without_client_fails(self):
        from data.synthetic_noul_augment import augment_noul_questions

        families = [
            {
                "state": "Test state",
                "noul_questions": [
                    {
                        "instructions": "weird question",
                        "label": True,
                        "cognitive_type": "unknown",
                    }
                ],
            }
        ]

        updated, count, failed = await augment_noul_questions(None, families, "", "")
        assert count == 0
        assert failed == 1

    @pytest.mark.asyncio
    async def test_augment_count_matches_noul_count(self):
        from data.synthetic_noul_augment import augment_noul_questions

        families = [
            {
                "state": f"State {i}",
                "noul_questions": [
                    {"instructions": f"Q{i}", "label": True, "cognitive_type": ct}
                ],
            }
            for i, ct in enumerate(COGNITIVE_TYPES.keys())
        ]

        updated, count, failed = await augment_noul_questions(None, families, "", "")
        assert count == len(COGNITIVE_TYPES)
        assert failed == 0
