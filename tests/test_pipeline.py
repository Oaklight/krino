"""Validate all benchmark loaders produce correctly formatted TypedQuestion objects.

Uses mock data for _load_hf_parquet so tests run offline (no network, no pyarrow).
Each loader is tested independently for:
- Correct question type (choice / noul / score)
- Label type matches question type (str / bool / float)
- Criteria structure matches question type (dict / optional / list)
- Label is a valid key in criteria for choice questions
- ID, source, split fields are well-formed
- No duplicate IDs within a loader
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from data import pipeline
from data.format import TypedQuestion

# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------

_BANKING77_LABELS = pipeline.BANKING77_LABELS


def _mock_banking77(dataset, config, split):
    return [{"text": f"msg {i}", "label": i % len(_BANKING77_LABELS)} for i in range(20)]


def _mock_sst2(dataset, config, split):
    return [{"sentence": f"sent {i}", "label": i % 2} for i in range(20)]


def _mock_agnews(dataset, config, split):
    return [{"text": f"news {i}", "label": i % 4} for i in range(20)]


def _mock_mnli(dataset, config, split):
    return [
        {"premise": f"P {i}", "hypothesis": f"H {i}", "label": i % 3, "genre": "fiction"}
        for i in range(20)
    ]


def _mock_typed_decisions(dataset, config, split):
    import json

    items = []
    for i in range(5):
        q_def = {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}
        gold = {"q1": {"label": "a"}}
        items.append({
            "state": json.dumps({"context": f"ctx {i}"}),
            "questions": json.dumps({"q1": q_def}),
            "gold": json.dumps(gold),
            "workflow": f"wf-{i}",
        })
    return items


def _mock_stsb(dataset, config, split):
    return [
        {"sentence1": f"S1 {i}", "sentence2": f"S2 {i}", "label": (i % 6) * 1.0}
        for i in range(20)
    ]


def _mock_sst5(dataset, config, split):
    return [{"text": f"sent {i}", "label": i % 5} for i in range(20)]


def _mock_arc(dataset, config, split):
    return [
        {
            "question": f"Q {i}",
            "choices": {"text": ["opt A", "opt B", "opt C", "opt D"], "label": ["A", "B", "C", "D"]},
            "answerKey": ["A", "B", "C", "D"][i % 4],
        }
        for i in range(10)
    ]


def _mock_race(dataset, config, split):
    return [
        {
            "article": f"Article {i}",
            "question": f"Q {i}?",
            "options": ["opt A", "opt B", "opt C", "opt D"],
            "answer": ["A", "B", "C", "D"][i % 4],
        }
        for i in range(10)
    ]


def _mock_hellaswag(dataset, config, split):
    return [
        {
            "ctx": f"Context {i}",
            "activity_label": f"activity-{i % 3}",
            "endings": [f"end {j}" for j in range(4)],
            "label": str(i % 4),
        }
        for i in range(20)
    ]


def _mock_tabfact(dataset, config, split):
    return [
        {
            "table_text": f"col1|col2\nval1|val2",
            "table_caption": f"Table {i}",
            "statement": f"Statement {i}",
            "label": i % 2,
            "table_id": f"t-{i}",
        }
        for i in range(20)
    ]


def _mock_fever(dataset, config, split):
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    return [
        {
            "claim": f"Claim {i}",
            "label": labels[i % 3],
            "evidence": [["page", f"Evidence sentence {i}"]],
        }
        for i in range(20)
    ]


def _mock_yelp(dataset, config, split):
    return [{"text": f"Review {i}", "label": i % 5} for i in range(50)]


def _mock_swag(dataset, config, split):
    return [
        {
            "startphrase": f"The cat {i}",
            "ending0": "sat.",
            "ending1": "ran.",
            "ending2": "slept.",
            "ending3": "ate.",
            "label": i % 4,
            "video-id": f"vid-{i}",
        }
        for i in range(20)
    ]


def _mock_multirc(dataset, config, split):
    return [
        {
            "paragraph": f"Paragraph {i}",
            "question": f"Q {i}?",
            "answer": f"Answer {i}",
            "label": i % 2,
            "idx": {"paragraph": i // 3, "question": 0, "answer": 0},
        }
        for i in range(20)
    ]


def _mock_mednli(dataset, config, split):
    labels = ["entailment", "contradiction", "neutral"]
    return [
        {"sentence1": f"Clinical note {i}", "sentence2": f"Hypothesis {i}", "gold_label": labels[i % 3]}
        for i in range(20)
    ]


def _mock_contractnli(dataset, config, split):
    return [
        {"premise": f"Clause {i}", "hypothesis": f"Proposition {i}", "label": i % 3}
        for i in range(10)
    ]


def _mock_codesearchnet(dataset, config, split):
    return [
        {
            "func_documentation_string": f"Docstring for func {i}",
            "func_code_string": f"def func_{i}(): pass  # unique-{random.Random(i).randint(0,999999)}",
            "repository_name": f"repo-{i}",
        }
        for i in range(15)
    ]


# Dispatch table: (dataset, config) prefix → mock factory
def _mock_mmlu(dataset, config, split):
    return [{"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1, "subject": "math"}]


def _mock_winogrande(dataset, config, split):
    return [{"sentence": "The _ was heavy.", "option1": "bag", "option2": "feather", "answer": "1"}]


def _mock_piqa(dataset, config, split):
    return [{"goal": "Boil water", "sol1": "Use a kettle", "sol2": "Use a fork", "label": 0}]


def _mock_commonsenseqa(dataset, config, split):
    return [{"question": "Where do you keep food cold?", "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["fridge", "oven", "table", "bed", "car"]}, "answerKey": "A"}]


def _mock_logiqa(dataset, config, split):
    return [{"context": "All cats are animals.", "query": "Is a cat an animal?", "options": ["Yes", "No", "Maybe", "Unknown"], "correct_option": 0}]


def _mock_anli(dataset, config, split):
    return [{"premise": "The sun is bright.", "hypothesis": "It is daytime.", "label": 0}]


def _mock_boolq(dataset, config, split):
    return [{"passage": "Paris is the capital of France.", "question": "Is Paris the capital of France?", "answer": True}]


def _mock_quality(dataset, config, split):
    return [
        {"article": f"Long article about topic {i}. " * 50, "question": f"What about {i}?",
         "options": ["opt A", "opt B", "opt C", "opt D"], "gold_label": (i % 4) + 1, "set_unique_id": f"q{i}"}
        for i in range(10)
    ]


def _mock_hotpotqa(dataset, config, split):
    return [
        {"question": f"Who did X in Y?", "answer": ["yes", "no", "John"][i % 3],
         "context": {"title": [f"Title{j}" for j in range(3)], "sentences": [[f"Sent {j}."] for j in range(3)]},
         "type": "bridge"}
        for i in range(20)
    ]


def _mock_drop(dataset, config, split):
    return [
        {"passage": f"In 2020, team scored {10+i} points.", "question": f"How many points?",
         "answers_spans": {"spans": [str(10 + i)]}}
        for i in range(20)
    ]


_MOCK_DISPATCH: dict[str, callable] = {
    "legacy-datasets/banking77": _mock_banking77,
    "stanfordnlp/sst2": _mock_sst2,
    "fancyzhx/ag_news": _mock_agnews,
    "nyu-mll/multi_nli": _mock_mnli,
    "LocalLLaMA/typed-decisions": _mock_typed_decisions,
    "nyu-mll/glue": _mock_stsb,
    "SetFit/sst5": _mock_sst5,
    "allenai/ai2_arc": _mock_arc,
    "ehovy/race": _mock_race,
    "Rowan/hellaswag": _mock_hellaswag,
    "wenhu/tab_fact": _mock_tabfact,
    "copenlu/fever_gold_evidence": _mock_fever,
    "Yelp/yelp_review_full": _mock_yelp,
    "allenai/swag": _mock_swag,
    "aps/super_glue": _mock_multirc,
    "presencesw/mednli": _mock_mednli,
    "kiddothe2b/contract-nli": _mock_contractnli,
    "code-search-net/code_search_net": _mock_codesearchnet,
    "cais/mmlu": _mock_mmlu,
    "allenai/winogrande": _mock_winogrande,
    "ybisk/piqa": _mock_piqa,
    "tau/commonsense_qa": _mock_commonsenseqa,
    "lucasmccabe/logiqa": _mock_logiqa,
    "facebook/anli": _mock_anli,
    "google/boolq": _mock_boolq,
    "emozilla/quality": _mock_quality,
    "hotpotqa/hotpot_qa": _mock_hotpotqa,
    "ucinlp/drop": _mock_drop,
}


def _mock_load_hf_parquet(dataset: str, config: str, split: str):
    factory = _MOCK_DISPATCH.get(dataset)
    if factory is None:
        raise FileNotFoundError(f"No mock for dataset={dataset}")
    return factory(dataset, config, split)


# ---------------------------------------------------------------------------
# Expected question type per loader
# ---------------------------------------------------------------------------

EXPECTED_TYPES: dict[str, set[str]] = {
    "banking77": {"choice"},
    "sst2": {"noul"},
    "agnews": {"choice"},
    "mnli": {"noul"},
    "typed_decisions": {"choice", "noul", "score"},
    "stsb": {"score"},
    "sst5": {"score"},
    "arc": {"choice"},
    "race": {"choice"},
    "hellaswag": {"choice"},
    "tabfact": {"noul"},
    "fever": {"choice"},
    "yelp": {"score"},
    "swag": {"choice"},
    "multirc": {"noul"},
    "mednli": {"noul"},
    "contractnli": {"noul"},
    "codesearchnet": {"choice"},
    "mmlu": {"choice"},
    "winogrande": {"choice"},
    "piqa": {"choice"},
    "commonsenseqa": {"choice"},
    "logiqa": {"choice"},
    "anli": {"noul"},
    "boolq": {"noul"},
    "quality": {"choice"},
    "hotpotqa": {"noul"},
    "drop": {"noul", "score"},
    "synthetic": {"noul", "choice", "score"},
    "jevbench": {"noul", "choice", "score"},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_hf(tmp_path):
    # Write mock family + variant files so load_synthetic() can assemble
    synth_dir = tmp_path / "synthetic"
    synth_dir.mkdir()
    import json
    mock_family = {
        "state": "Patient presents with chest pain.",
        "noul_questions": [
            {"cognitive_type": "entailment", "instructions": "Is this urgent?", "label": True},
        ],
        "choice_questions": [
            {"instructions": "What is the triage level?",
             "criteria": {"immediate": "Life-threatening", "urgent": "Serious", "non_urgent": "Minor"},
             "label": "immediate"},
        ],
        "score_questions": [
            {"instructions": "Rate urgency.",
             "criteria": ["Low", "Medium", "High", "Critical"],
             "label": 4.0},
        ],
        "_domain": "medical_triage",
        "_family_idx": 0,
        "_seeded": False,
    }
    with open(synth_dir / "medical_triage_families.jsonl", "w") as f:
        f.write(json.dumps(mock_family) + "\n")
    with open(synth_dir / "medical_triage_variants.jsonl", "w") as f:
        f.write(json.dumps({}) + "\n")

    with patch.object(pipeline, "_load_hf_parquet", side_effect=_mock_load_hf_parquet), \
         patch.object(pipeline, "DATA_DIR", tmp_path):
        yield


@pytest.mark.parametrize("source", list(pipeline.LOADERS.keys()))
class TestLoaderFormat:
    """Validate TypedQuestion format for every registered loader."""

    def _load(self, source: str) -> list[TypedQuestion]:
        loader = pipeline.LOADERS[source]
        items = list(loader())
        assert len(items) > 0, f"{source}: loader produced zero items"
        return items

    def test_produces_items(self, source):
        items = self._load(source)
        assert len(items) > 0

    def test_question_type_valid(self, source):
        items = self._load(source)
        for item in items:
            assert item.question["type"] in ("choice", "noul", "score"), (
                f"{source}: unexpected question type {item.question['type']}"
            )

    def test_question_type_matches_expected(self, source):
        items = self._load(source)
        expected = EXPECTED_TYPES[source]
        observed = {item.question["type"] for item in items}
        assert observed <= expected, (
            f"{source}: unexpected types {observed - expected}"
        )

    def test_label_type_matches_question_type(self, source):
        items = self._load(source)
        for item in items:
            qt = item.question["type"]
            if qt == "noul":
                assert isinstance(item.label, bool), (
                    f"{source} [{item.id}]: noul label should be bool, got {type(item.label).__name__}"
                )
            elif qt == "choice":
                assert isinstance(item.label, str), (
                    f"{source} [{item.id}]: choice label should be str, got {type(item.label).__name__}"
                )
            elif qt == "score":
                assert isinstance(item.label, (int, float)), (
                    f"{source} [{item.id}]: score label should be numeric, got {type(item.label).__name__}"
                )

    def test_criteria_structure(self, source):
        items = self._load(source)
        for item in items:
            qt = item.question["type"]
            criteria = item.question.get("criteria")
            if qt == "choice":
                assert isinstance(criteria, dict), (
                    f"{source} [{item.id}]: choice criteria should be dict"
                )
                assert len(criteria) >= 2, (
                    f"{source} [{item.id}]: choice needs at least 2 options"
                )
            elif qt == "score":
                assert isinstance(criteria, list), (
                    f"{source} [{item.id}]: score criteria should be list"
                )
                assert len(criteria) >= 2, (
                    f"{source} [{item.id}]: score needs at least 2 levels"
                )

    def test_choice_label_in_criteria(self, source):
        items = self._load(source)
        for item in items:
            if item.question["type"] == "choice":
                criteria = item.question["criteria"]
                assert item.label in criteria, (
                    f"{source} [{item.id}]: label '{item.label}' not in criteria keys {list(criteria.keys())}"
                )

    def test_id_format(self, source):
        items = self._load(source)
        for item in items:
            assert item.id, f"{source}: empty ID"
            assert isinstance(item.id, str), f"{source}: ID should be str"

    def test_no_duplicate_ids(self, source):
        items = self._load(source)
        ids = [item.id for item in items]
        dupes = [x for x in ids if ids.count(x) > 1]
        assert not dupes, f"{source}: duplicate IDs found: {set(dupes)}"

    def test_source_field(self, source):
        items = self._load(source)
        for item in items:
            assert item.source == source, (
                f"{source} [{item.id}]: source field is '{item.source}', expected '{source}'"
            )

    def test_split_valid(self, source):
        items = self._load(source)
        for item in items:
            assert item.split in ("train", "test"), (
                f"{source} [{item.id}]: invalid split '{item.split}'"
            )

    def test_state_nonempty(self, source):
        items = self._load(source)
        for item in items:
            assert item.state, f"{source} [{item.id}]: empty state"

    def test_instructions_present(self, source):
        items = self._load(source)
        for item in items:
            assert item.question.get("instructions"), (
                f"{source} [{item.id}]: missing instructions"
            )

    def test_to_dict_roundtrip(self, source):
        items = self._load(source)
        for item in items[:3]:
            d = item.to_dict()
            reconstructed = TypedQuestion(**d)
            assert reconstructed.id == item.id
            assert reconstructed.source == item.source
            assert reconstructed.label == item.label
            assert reconstructed.question["type"] == item.question["type"]


class TestLoaderRegistry:
    """Verify the LOADERS dict is complete and consistent."""

    def test_all_expected_loaders_registered(self):
        expected = {
            "banking77", "sst2", "agnews", "mnli", "typed_decisions",
            "stsb", "sst5", "arc", "race", "hellaswag", "tabfact", "fever",
            "yelp", "swag", "multirc", "mednli", "contractnli", "codesearchnet",
            "mmlu", "winogrande", "piqa", "commonsenseqa", "logiqa", "anli", "boolq",
            "quality", "hotpotqa", "drop",
            "synthetic", "jevbench",
        }
        assert set(pipeline.LOADERS.keys()) == expected

    def test_loader_count(self):
        assert len(pipeline.LOADERS) == 30

    def test_load_all_works(self):
        items = pipeline.load_all()
        assert len(items) > 0
        sources = {item.source for item in items}
        assert len(sources) == 30

    def test_load_all_single_source(self):
        items = pipeline.load_all(["sst2"])
        assert all(item.source == "sst2" for item in items)

    def test_load_all_unknown_source(self):
        with pytest.raises(ValueError, match="Unknown source"):
            pipeline.load_all(["nonexistent"])


class TestBalanceCaps:
    """Verify balance.py DEFAULT_CAPS are consistent with registered loaders."""

    def test_caps_reference_valid_sources(self):
        from data.balance import DEFAULT_CAPS
        for source in DEFAULT_CAPS:
            assert source in pipeline.LOADERS, (
                f"DEFAULT_CAPS references unknown source '{source}'"
            )

    def test_caps_are_positive(self):
        from data.balance import DEFAULT_CAPS
        for source, cap in DEFAULT_CAPS.items():
            assert isinstance(cap, int) and cap > 0, (
                f"DEFAULT_CAPS['{source}'] should be positive int, got {cap}"
            )
