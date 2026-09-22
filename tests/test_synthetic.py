"""Tests for the synthetic data generation pipeline.

Tests template building, family conversion, validation logic, and label extraction.
All tests run offline — no API calls.
"""

from __future__ import annotations

import json

import pytest

from data.format import TypedQuestion
from data.lsh import LSHIndex, MinHash, char_ngrams
from data.synthetic_dedup import dedup_families
from data.synthetic_label import (
    CONFIDENCE_BUCKETS,
    LabelResult,
    _bucket_confidence,
    _extract_probs,
)
from data.synthetic_templates import (
    COGNITIVE_TYPE_DESCRIPTIONS,
    COGNITIVE_TYPES,
    DOMAIN_TEMPLATES,
    GENERATED_DOMAINS,
    SEEDED_DOMAINS,
    build_generated_family_prompt,
    build_seeded_family_prompt,
    build_validation_prompt,
    build_variant_prompt,
    json_compact,
)
from data.synthetic_validate import ValidationResult


# --- Template tests ---


class TestDomainTemplates:
    def test_all_domains_have_required_fields(self):
        required = {"description", "state_prompt", "choice_template", "score_template", "seed_source"}
        for domain, template in DOMAIN_TEMPLATES.items():
            missing = required - set(template.keys())
            assert not missing, f"{domain} missing fields: {missing}"

    def test_choice_template_has_criteria(self):
        for domain, template in DOMAIN_TEMPLATES.items():
            ct = template["choice_template"]
            assert "instructions" in ct, f"{domain} choice missing instructions"
            assert "criteria" in ct, f"{domain} choice missing criteria"
            assert isinstance(ct["criteria"], dict), f"{domain} choice criteria not dict"
            assert len(ct["criteria"]) >= 2, f"{domain} choice needs >= 2 options"

    def test_score_template_has_criteria_list(self):
        for domain, template in DOMAIN_TEMPLATES.items():
            st = template["score_template"]
            assert "instructions" in st, f"{domain} score missing instructions"
            assert "criteria" in st, f"{domain} score missing criteria"
            assert isinstance(st["criteria"], list), f"{domain} score criteria not list"
            assert len(st["criteria"]) >= 2, f"{domain} score needs >= 2 levels"

    def test_seeded_domains_have_valid_source(self):
        from data.pipeline import LOADERS
        for domain in SEEDED_DOMAINS:
            source = DOMAIN_TEMPLATES[domain]["seed_source"]
            assert source in LOADERS, f"{domain} seed_source {source!r} not in LOADERS"

    def test_generated_domains_have_no_source(self):
        for domain in GENERATED_DOMAINS:
            assert DOMAIN_TEMPLATES[domain]["seed_source"] is None

    def test_domain_partition_is_complete(self):
        all_domains = set(DOMAIN_TEMPLATES.keys())
        assert set(SEEDED_DOMAINS) | set(GENERATED_DOMAINS) == all_domains

    def test_no_overlap_between_seeded_and_generated(self):
        assert not set(SEEDED_DOMAINS) & set(GENERATED_DOMAINS)


class TestCognitiveTypes:
    def test_types_and_descriptions_match(self):
        assert set(COGNITIVE_TYPES.keys()) == set(COGNITIVE_TYPE_DESCRIPTIONS.keys())

    def test_all_12_types_present(self):
        assert len(COGNITIVE_TYPES) == 12


class TestPromptBuilding:
    def test_seeded_family_prompt_contains_state(self):
        seed_state = "Patient presents with chest pain and elevated troponin."
        prompt = build_seeded_family_prompt(
            "medical_triage", seed_state, ["entailment", "causal"]
        )
        assert seed_state in prompt
        assert "do not modify" in prompt.lower()
        assert "entailment" in prompt
        assert "causal" in prompt

    def test_generated_family_prompt_has_state_requirements(self):
        prompt = build_generated_family_prompt(
            "spatial_reasoning", ["temporal", "comparison"]
        )
        assert "STATE REQUIREMENTS" in prompt
        assert "spatial" in prompt.lower()

    def test_variant_prompt_counterfactual(self):
        prompt = build_variant_prompt('{"state": "test"}', "counterfactual")
        assert "COUNTERFACTUAL" in prompt
        assert "changed_fact" in prompt

    def test_variant_prompt_paraphrase(self):
        prompt = build_variant_prompt('{"state": "test"}', "paraphrase")
        assert "paraphrase" in prompt.lower()
        assert "state_paraphrase" in prompt

    def test_variant_prompt_negation(self):
        prompt = build_variant_prompt('{"state": "test"}', "negation")
        assert "NEGATION" in prompt
        assert "negated_instructions" in prompt

    def test_variant_prompt_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown variant type"):
            build_variant_prompt('{"state": "test"}', "invalid")

    def test_validation_prompt(self):
        prompt = build_validation_prompt("some state", "Is X true?", "true")
        assert "some state" in prompt
        assert "Is X true?" in prompt
        assert '"correct"' in prompt


# --- Family conversion tests ---


class TestFamilyConversion:
    SAMPLE_FAMILY = {
        "state": "Patient has BP 180/110 and chest pain.",
        "noul_questions": [
            {"cognitive_type": "entailment", "instructions": "Is BP elevated?", "label": True},
            {"cognitive_type": "causal", "instructions": "Could this cause stroke?", "label": True},
        ],
        "choice_questions": [
            {
                "instructions": "What is the triage level?",
                "criteria": {"immediate": "Life-threatening", "urgent": "Serious"},
                "label": "immediate",
            },
            {
                "instructions": "What department should handle this?",
                "criteria": {"cardiology": "Heart-related", "er": "Emergency room", "primary": "Primary care"},
                "label": "cardiology",
            },
        ],
        "score_questions": [
            {
                "instructions": "Rate urgency.",
                "criteria": ["Low", "Medium", "High", "Critical"],
                "label": 4.0,
            },
            {
                "instructions": "Rate patient cooperation.",
                "criteria": ["Uncooperative", "Neutral", "Cooperative"],
                "label": 2.0,
            },
        ],
        "_domain": "medical_triage",
        "_family_idx": 0,
        "_seeded": True,
    }

    def test_base_items_count(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        # 2 noul + 2 choice + 2 score + 1 shuffle (choice-1 has 3 keys) = 7
        assert len(items) == 7

    def test_base_items_types(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        types = [item["question"]["type"] for item in items]
        assert types.count("noul") == 2
        assert types.count("choice") == 3  # 2 base + 1 shuffle
        assert types.count("score") == 2

    def test_items_have_correct_source(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        assert all(item["source"] == "synthetic" for item in items)

    def test_items_have_group(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        assert all("group" in item for item in items)
        groups = {item["group"] for item in items}
        assert len(groups) == 1

    def test_items_have_unique_ids(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        ids = [item["id"] for item in items]
        assert len(ids) == len(set(ids))

    def test_counterfactual_variants_added(self):
        from data.synthetic import family_to_typed_questions
        variants = {
            "counterfactual": {
                "state": "Patient has BP 120/80 and no chest pain.",
                "changed_fact": "BP normalized",
                "noul_labels": [
                    {"cognitive_type": "entailment", "label": False, "flipped": True},
                    {"cognitive_type": "causal", "label": False, "flipped": True},
                ],
                "choice_label": "urgent",
                "score_label": 2.0,
            },
            "paraphrase": None,
            "negation": None,
        }
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, variants, "medical_triage", 0
        )
        cf_items = [i for i in items if "-cf-" in i["id"]]
        # 2 cf-noul + 1 cf-choice-0 + 1 cf-score-0 = 4
        assert len(cf_items) == 4

    def test_invalid_choice_label_dropped(self):
        from data.synthetic import family_to_typed_questions
        bad_family = dict(self.SAMPLE_FAMILY)
        bad_family["choice_questions"] = [
            {"instructions": "Q?", "criteria": {"a": "A", "b": "B"}, "label": "nonexistent"},
        ]
        items = family_to_typed_questions(bad_family, {}, "medical_triage", 0)
        choice_items = [i for i in items if i["question"]["type"] == "choice"]
        # The valid one (cardiology) from SAMPLE_FAMILY is gone since we replaced
        # Only the score + noul items should remain
        assert len(choice_items) == 0

    def test_negation_variants_added(self):
        from data.synthetic import family_to_typed_questions
        variants = {
            "counterfactual": None,
            "paraphrase": None,
            "negation": {
                "negated_questions": [
                    {
                        "cognitive_type": "entailment",
                        "original_instructions": "Is BP elevated?",
                        "negated_instructions": "Is BP normal?",
                        "original_label": True,
                        "negated_label": False,
                    },
                ],
            },
        }
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, variants, "medical_triage", 0
        )
        neg_items = [i for i in items if "-neg-" in i["id"]]
        assert len(neg_items) == 1
        assert neg_items[0]["label"] is False

    def test_score_labels_converted_to_0based(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        score_items = [i for i in items if i["question"]["type"] == "score"]
        labels = {i["label"] for i in score_items}
        assert 3.0 in labels  # 4.0 - 1.0 = 3.0
        assert 1.0 in labels  # 2.0 - 1.0 = 1.0
        assert all(l >= 0.0 for l in labels)

    def test_roundtrip_to_typed_question(self):
        from data.synthetic import family_to_typed_questions
        items = family_to_typed_questions(
            self.SAMPLE_FAMILY, {}, "medical_triage", 0
        )
        for item_dict in items:
            tq = TypedQuestion(**item_dict)
            assert tq.source == "synthetic"
            rt = tq.to_dict()
            assert rt["id"] == item_dict["id"]


# --- Label extraction tests ---


class TestLabelExtraction:
    def test_noul_probs(self):
        probs, conf = _extract_probs({"noul": 0.85}, "noul")
        assert probs == {"true": 0.85, "false": pytest.approx(0.15)}
        assert conf == 0.85

    def test_choice_probs(self):
        answer = {"choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}}
        probs, conf = _extract_probs(answer, "choice")
        assert probs == {"a": 0.9, "b": 0.1}
        assert conf == 0.9

    def test_score_probs(self):
        answer = {
            "score": 3.0,
            "confidence": 0.7,
            "probabilities": {"low": 0.1, "med": 0.7, "high": 0.2},
        }
        probs, conf = _extract_probs(answer, "score")
        assert probs["med"] == 0.7
        assert conf == 0.7

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            _extract_probs({}, "bounding_box")


class TestConfidenceBucketing:
    def test_high(self):
        assert _bucket_confidence(0.95) == "high"

    def test_medium(self):
        assert _bucket_confidence(0.75) == "medium"

    def test_uncertain(self):
        assert _bucket_confidence(0.45) == "uncertain"

    def test_boundary_medium_high(self):
        assert _bucket_confidence(0.9) == "high"

    def test_exact_1_0(self):
        assert _bucket_confidence(1.0) == "high"

    def test_boundary_uncertain_medium(self):
        assert _bucket_confidence(0.6) == "medium"


# --- Validation tests ---


class TestValidationResult:
    def test_dataclass(self):
        vr = ValidationResult(correct=True, confidence=0.9, reasoning="looks good")
        assert vr.correct is True
        assert vr.confidence == 0.9


# --- JSON utility ---


class TestDedup:
    def test_identical_states_deduped(self):
        families = [
            {"state": "The library is north of the park.", "_family_idx": 0, "_domain": "test"},
            {"state": "The library is north of the park.", "_family_idx": 1, "_domain": "test"},
            {"state": "A completely different scene with a river.", "_family_idx": 2, "_domain": "test"},
        ]
        kept, dropped = dedup_families(families, threshold=0.7)
        assert len(kept) == 2
        assert len(dropped) == 1
        assert dropped[0] == 1

    def test_unique_states_kept(self):
        families = [
            {"state": "The museum has a large marble statue in the center.", "_family_idx": 0},
            {"state": "A chemistry student answers a question about boiling points.", "_family_idx": 1},
            {"state": "User posted a heated comment about politics.", "_family_idx": 2},
        ]
        kept, dropped = dedup_families(families, threshold=0.7)
        assert len(kept) == 3
        assert len(dropped) == 0

    def test_near_duplicate_detected(self):
        families = [
            {"state": "The building has three floors. The cafe is on the ground floor, the office is on the second floor, and the gym is on the top floor.", "_family_idx": 0},
            {"state": "The building has three floors. The cafe is on the ground floor, the office is on the second floor, and the pool is on the top floor.", "_family_idx": 1},
        ]
        kept, dropped = dedup_families(families, threshold=0.7)
        # These are very similar — only "gym" vs "pool" differs
        assert len(kept) == 1
        assert len(dropped) == 1

    def test_empty_input(self):
        kept, dropped = dedup_families([], threshold=0.7)
        assert kept == []
        assert dropped == []

    def test_threshold_controls_sensitivity(self):
        families = [
            {"state": "A red ball is on the table near the window.", "_family_idx": 0},
            {"state": "A blue ball is on the table near the window.", "_family_idx": 1},
        ]
        # Low threshold — more aggressive dedup
        kept_low, _ = dedup_families(families, threshold=0.5)
        # High threshold — more permissive
        kept_high, _ = dedup_families(families, threshold=0.95)
        assert len(kept_high) >= len(kept_low)


class TestMinHash:
    def test_identical_texts_high_similarity(self):
        lsh = LSHIndex(num_perm=128, bands=16)
        mh1 = lsh.make_minhash("The quick brown fox jumps over the lazy dog")
        mh2 = lsh.make_minhash("The quick brown fox jumps over the lazy dog")
        assert mh1.jaccard(mh2) == 1.0

    def test_different_texts_low_similarity(self):
        lsh = LSHIndex(num_perm=128, bands=16)
        mh1 = lsh.make_minhash("The quick brown fox jumps over the lazy dog")
        mh2 = lsh.make_minhash("A completely unrelated sentence about quantum physics")
        assert mh1.jaccard(mh2) < 0.3

    def test_char_ngrams(self):
        ngrams = char_ngrams("hello world", n=3)
        assert "hel" in ngrams
        assert "llo" in ngrams
        assert "wor" in ngrams


class TestJsonCompact:
    def test_compact_dict(self):
        result = json_compact({"a": 1, "b": "hello"})
        assert result == '{"a": 1, "b": "hello"}'

    def test_compact_list(self):
        result = json_compact([1, 2, 3])
        assert result == "[1, 2, 3]"
