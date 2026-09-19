#!/usr/bin/env python3
"""Probe 3: Jev likelihood sensitivity under balanced semantic paraphrases."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from harness import Fixture, ProbeSpec, answer_probabilities, pearson, run_cli, simple_regression

PREDICTION = (
    "If option probabilities substantially inherit surface-form likelihood, semantically equivalent options with "
    "greater length, lower grammaticality, or lower base-rate plausibility will receive systematically different "
    "mass despite Latin-square balancing. Opaque-label classes distinguish label effects from description effects."
)
# Grammaticality and lexical-commonness are preregistered binary stimulus annotations,
# not measured language-model likelihoods. Measured reference probabilities must come
# from --reference-probabilities and are analyzed separately.
VARIANTS = [
    ("short_grammatical_common", "Approve the request.", 20, 1.0, 1.0),
    ("long_grammatical_common", "Approve the request after completing the ordinary review process.", 66, 1.0, 1.0),
    ("short_fragment_rare", "Request: approve.", 17, 0.0, 0.0),
    ("long_awkward_rare", "The request is what shall, following review completion, be approved by action.", 78, 0.0, 0.0),
]
LABEL_CLASSES = [("ascii", ["a", "b", "c", "d"]), ("digits", ["17", "42", "83", "96"]), ("unicode", ["雪", "月", "川", "石"])]


def build_fixtures(mode: str, seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    item_count = 3 if mode == "smoke" else 100
    fixtures: list[Fixture] = []
    for item in range(item_count):
        for label_class, labels in LABEL_CLASSES:
            shift = item % len(VARIANTS)
            assignment = [(labels[index], VARIANTS[(index + shift) % len(VARIANTS)]) for index in range(len(labels))]
            criteria = {label: variant[1] for label, variant in assignment}
            target_by_label = {label: variant[0] for label, variant in assignment}
            fixtures.append(
                Fixture(
                    f"label_class={label_class};latin_shift={shift}",
                    f"likelihood-{item:03d}-{label_class}",
                    PREDICTION,
                    {
                        "state": f"Case {item}: all listed actions have exactly the same consequence and utility.",
                        "model": "jev-latest",
                        "questions": {
                            f"q_{item:03d}": {
                                "type": "choice",
                                "instructions": "Choose among semantically equivalent actions; no action is preferred.",
                                "criteria": criteria,
                            }
                        },
                    },
                    {
                        "semantic_item": item,
                        "label_class": label_class,
                        "latin_shift": shift,
                        "variant_by_label": target_by_label,
                    },
                )
            )
    rng.shuffle(fixtures)
    return fixtures


def analyze(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    variant_values: dict[str, list[float]] = defaultdict(list)
    jev_by_id: dict[str, float] = {}
    reference: Mapping[str, Any] = {}
    for record in records:
        if "external_reference_probabilities" in record:
            candidate = record["external_reference_probabilities"]
            if isinstance(candidate, Mapping):
                reference = candidate
            continue
        probabilities = answer_probabilities(record)
        variant_by_label = record.get("metadata", {}).get("variant_by_label", {})
        for label, probability in probabilities.items():
            variant = variant_by_label.get(label)
            if variant:
                variant_values[str(variant)].append(probability)
                jev_by_id[f"{record.get('item_id')}:{variant}"] = probability
    summary = {
        variant: {"n": len(values), "mean_probability": sum(values) / len(values)}
        for variant, values in variant_values.items()
        if values
    }
    lengths: list[float] = []
    probabilities: list[float] = []
    grammar: list[float] = []
    plausibility: list[float] = []
    for name, _text, length, grammatical, plausible in VARIANTS:
        for probability in variant_values.get(name, []):
            lengths.append(float(length))
            grammar.append(grammatical)
            plausibility.append(plausible)
            probabilities.append(probability)
    paired_jev: list[float] = []
    paired_reference: list[float] = []
    for key, probability in jev_by_id.items():
        value = reference.get(key)
        if isinstance(value, (int, float)):
            paired_jev.append(probability)
            paired_reference.append(float(value))
    return {
        "variant_summary": summary,
        "surface_correlations": {
            "description_length": pearson(lengths, probabilities),
            "grammaticality": pearson(grammar, probabilities),
            "preregistered_lexical_commonness": pearson(plausibility, probabilities),
        },
        "length_regression": simple_regression(lengths, probabilities),
        "external_reference": {
            "paired_count": len(paired_jev),
            "correlation": pearson(paired_jev, paired_reference),
            "regression_reference_on_jev": simple_regression(paired_jev, paired_reference),
        },
        "annotation_guardrail": (
            "Grammaticality and lexical-commonness are preregistered stimulus annotations, not measured "
            "language-model probabilities. Architectural comparison requires --reference-probabilities."
        ),
        "reference_file_format": "JSON object: {'<item_id>:<variant_name>': probability}. No torch dependency.",
    }


SPEC = ProbeSpec("probe_3_likelihood_sensitivity", __doc__ or "", build_fixtures, analyze)

if __name__ == "__main__":
    raise SystemExit(run_cli(SPEC))
