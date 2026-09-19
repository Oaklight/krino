#!/usr/bin/env python3
"""Probe 5: matched output-entry and probability lexical signatures."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any, Mapping, Sequence

from harness import Fixture, ProbeSpec, answer_probabilities, run_cli

PREDICTION = (
    "Autoregressive constrained serialization may expose output-position or preceding-field-complexity effects and "
    "lexical probability heaping beyond mandatory 0.01 display rounding. Deterministic postprocessing may instead "
    "show a stable exact-sum correction rule tied to one position or maximum entry."
)


def build_fixtures(mode: str, seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    evidence_levels = (0.35, 0.50, 0.65) if mode == "smoke" else tuple(index / 100 for index in range(5, 96, 3))
    field_counts = (2, 3) if mode == "smoke" else (2, 5, 10, 20)
    repeats = 1 if mode == "smoke" else 4
    fixtures: list[Fixture] = []
    for block in range(repeats):
        block_fixtures: list[Fixture] = []
        for level_index, evidence in enumerate(evidence_levels):
            for preceding_complexity in (0, 3):
                for entries in field_counts:
                    questions: dict[str, Any] = {}
                    for field in range(preceding_complexity):
                        questions[f"prefix_{field}"] = {"type": "noul", "instructions": "The state contains evidence."}
                    for position in range(entries):
                        questions[f"target_{position:02d}"] = {
                            "type": "choice",
                            "instructions": "Estimate which hypothesis the smoothly interpolated evidence supports.",
                            "criteria": {"a": "hypothesis A", "b": "hypothesis B", "c": "neutral remainder"},
                        }
                    state = f"Evidence coordinate is {evidence:.2f}; values near 1 support A, near 0 support B, with smooth interpolation."
                    block_fixtures.append(
                        Fixture(
                            f"evidence={evidence:.2f};prefix={preceding_complexity};fields={entries};block={block}",
                            f"generation-{level_index:03d}-{preceding_complexity}-{entries}-{block}",
                            PREDICTION,
                            {"state": state, "model": "jev-latest", "questions": questions},
                            {
                                "evidence": evidence,
                                "preceding_field_complexity": preceding_complexity,
                                "number_of_target_fields": entries,
                                "block": block,
                            },
                        )
                    )
        rng.shuffle(block_fixtures)
        fixtures.extend(block_fixtures)
    return fixtures


def exact_sum_rule(probabilities: Mapping[str, float]) -> dict[str, Any]:
    """Analyze cent-grid sums and candidate residual-correction targets."""
    items = list(probabilities.items())
    cents = [(key, int((Decimal(str(value)) * 100).to_integral_value())) for key, value in items]
    total = sum(value for _key, value in cents)
    residual = 100 - total
    maximum_index = max(range(len(cents)), key=lambda index: cents[index][1]) if cents else None
    return {
        "cent_sum": total,
        "residual_cents": residual,
        "exact_sum": total == 100,
        "maximum_index": maximum_index,
        "last_index": len(cents) - 1 if cents else None,
        "candidate_correction_target": "none" if residual == 0 else "unknown_from_displayed_values",
    }


def analyze(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fingerprints: Counter[str] = Counter()
    sum_rules: list[dict[str, Any]] = []
    by_position: dict[int, list[float]] = defaultdict(list)
    by_complexity: dict[int, list[float]] = defaultdict(list)
    by_field_count: dict[int, list[float]] = defaultdict(list)
    by_evidence: dict[float, list[float]] = defaultdict(list)
    for record in records:
        parsed = record.get("parsed_response")
        answers = parsed.get("answers", {}) if isinstance(parsed, Mapping) else {}
        metadata = record.get("metadata", {})
        for question_id, answer in answers.items():
            probabilities = answer.get("probabilities", {}) if isinstance(answer, Mapping) else {}
            if not isinstance(probabilities, Mapping) or not probabilities:
                continue
            numeric = {str(key): float(value) for key, value in probabilities.items()}
            for value in numeric.values():
                fingerprints[f"{value:.2f}"] += 1
            sum_rules.append({"item_id": record.get("item_id"), "question_id": question_id, **exact_sum_rule(numeric)})
            if question_id.startswith("target_"):
                a_probability = numeric.get("a")
                if a_probability is not None:
                    target_position = int(question_id.removeprefix("target_"))
                    by_position[target_position].append(a_probability)
                    by_complexity[int(metadata.get("preceding_field_complexity", 0))].append(a_probability)
                    by_field_count[int(metadata.get("number_of_target_fields", 0))].append(a_probability)
                    by_evidence[float(metadata.get("evidence", 0.0))].append(a_probability)
    exact_count = sum(rule["exact_sum"] for rule in sum_rules)
    residual_counts = Counter(str(rule["residual_cents"]) for rule in sum_rules)
    maximum_last_count = sum(
        rule["maximum_index"] == rule["last_index"] for rule in sum_rules if rule["maximum_index"] is not None
    )
    heaping = {value: count for value, count in fingerprints.items() if value.endswith(("0", "5"))}
    return {
        "distributions_analyzed": len(sum_rules),
        "exact_cent_sum_count": exact_count,
        "non_exact_cent_sum_count": len(sum_rules) - exact_count,
        "residual_cent_histogram": dict(sorted(residual_counts.items())),
        "maximum_is_last_count": maximum_last_count,
        "sum_rules": sum_rules,
        "probability_lexical_fingerprint": dict(sorted(fingerprints.items())),
        "coarse_decimal_heaping_counts": heaping,
        "mean_a_probability_by_output_position": {str(key): sum(values) / len(values) for key, values in by_position.items()},
        "mean_a_probability_by_preceding_complexity": {str(key): sum(values) / len(values) for key, values in by_complexity.items()},
        "mean_a_probability_by_number_of_fields": {str(key): sum(values) / len(values) for key, values in by_field_count.items()},
        "smooth_evidence_curve": {str(key): sum(values) / len(values) for key, values in sorted(by_evidence.items())},
        "interpretation_guardrail": "All displayed values are already rounded to 0.01; only excess heaping or stable correction patterns are additional signatures.",
    }


SPEC = ProbeSpec("probe_5_generation_signatures", __doc__ or "", build_fixtures, analyze)

if __name__ == "__main__":
    raise SystemExit(run_cli(SPEC))
