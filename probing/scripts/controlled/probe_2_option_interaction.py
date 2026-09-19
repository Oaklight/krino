#!/usr/bin/env python3
"""Probe 2: interval-censored strict-IIA and duplicate-family tests."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from harness import (
    Fixture,
    ProbeSpec,
    answer_probabilities,
    classify_zero,
    log_odds_ratio_interval,
    run_cli,
    subtract_intervals,
)

PREDICTION = (
    "A strict independent-utility scorer preserves P(A)/P(B) when irrelevant, dominated, duplicate, or paraphrase "
    "options are added. A joint/listwise scorer may change interval-censored A:B log odds; duplicate assessment must "
    "also compare total duplicate-family mass so ordinary normalization is not mislabeled interaction."
)
BASE_ITEMS = [
    ("A commuter needs reliable transport in heavy rain and values low maintenance.", "covered rail", "open bicycle"),
    ("A team needs a durable archive with simple offline recovery.", "local snapshots", "temporary cache"),
    ("A hiker prioritizes potable water over comfort on a remote route.", "water filter", "travel pillow"),
    ("A buyer needs a quiet tool for apartment use at night.", "manual screwdriver", "impact driver"),
    ("A clinic prioritizes traceable records over decorative presentation.", "audit log", "color theme"),
]


def _conditions() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("pair", []),
        ("irrelevant", [("x", "a decorative sticker unrelated to the need")]),
        ("dominated", [("d", "an unreliable version of option B with higher cost")]),
        ("duplicate", [("a_dup", "__COPY_A__")]),
        ("paraphrase", [("a_para", "same practical choice as option A, reworded")]),
    ]


def build_fixtures(mode: str, seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    item_count = 3 if mode == "smoke" else 100
    positions = (0, 1, 2) if mode == "full" else (0, 2)
    fixtures: list[Fixture] = []
    for item_index in range(item_count):
        state, a_text, b_text = BASE_ITEMS[item_index % len(BASE_ITEMS)]
        for condition, additions in _conditions():
            active_positions = (0,) if condition == "pair" else positions
            for insertion_position in active_positions:
                options = [("a", a_text), ("b", b_text)]
                addition_keys: list[str] = []
                for key, description in additions:
                    if description == "__COPY_A__":
                        description = a_text
                    else:
                        description = description.replace("option A", a_text).replace("option B", b_text)
                    options.insert(min(insertion_position, len(options)), (key, description))
                    addition_keys.append(key)
                criteria = dict(options)
                fixture = Fixture(
                    f"{condition};position={insertion_position}",
                    f"iia-{item_index:03d}",
                    PREDICTION,
                    {
                        "state": state,
                        "model": "jev-latest",
                        "questions": {
                            f"q_{item_index:03d}": {
                                "type": "choice",
                                "instructions": "Which option better satisfies the stated priority?",
                                "criteria": criteria,
                            }
                        },
                    },
                    {
                        "semantic_item": item_index,
                        "condition": condition,
                        "insertion_position": insertion_position,
                        "option_order": list(criteria),
                        "duplicate_family": ["a", *addition_keys] if condition in {"duplicate", "paraphrase"} else ["a"],
                    },
                )
                fixtures.append(fixture)
    rng.shuffle(fixtures)
    return fixtures


def analyze(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        metadata = record.get("metadata", {})
        condition = str(metadata.get("condition"))
        position = metadata.get("insertion_position")
        grouped[int(metadata.get("semantic_item", -1))][f"{condition}:{position}"] = record

    comparisons: list[dict[str, Any]] = []
    for item, cells in sorted(grouped.items()):
        base = cells.get("pair:0")
        if not base:
            continue
        base_probs = answer_probabilities(base)
        if not {"a", "b"} <= base_probs.keys():
            continue
        base_interval = log_odds_ratio_interval(base_probs["a"], base_probs["b"])
        for key, record in cells.items():
            if key == "pair:0":
                continue
            probabilities = answer_probabilities(record)
            if not {"a", "b"} <= probabilities.keys():
                continue
            candidate_interval = log_odds_ratio_interval(probabilities["a"], probabilities["b"])
            change_interval = subtract_intervals(candidate_interval, base_interval)
            family = record.get("metadata", {}).get("duplicate_family", ["a"])
            family_mass = sum(probabilities.get(option, 0.0) for option in family)
            comparisons.append(
                {
                    "semantic_item": item,
                    "condition_position": key,
                    "base_log_ab_interval": base_interval,
                    "candidate_log_ab_interval": candidate_interval,
                    "change_interval": change_interval,
                    "change_class": classify_zero(change_interval),
                    "base_a_mass": base_probs["a"],
                    "candidate_duplicate_family_mass": family_mass,
                }
            )
    definite = sum(row["change_class"] != "contains_zero" for row in comparisons)
    return {
        "comparisons": comparisons,
        "definite_iia_changes": definite,
        "rounding_indeterminate_or_stable": len(comparisons) - definite,
        "interpretation_guardrail": (
            "Probability changes caused solely by adding another normalized alternative are not evidence of interaction. "
            "Strict IIA is assessed with A:B log odds; duplicate/paraphrase cases additionally report combined family mass."
        ),
    }


SPEC = ProbeSpec("probe_2_option_interaction", __doc__ or "", build_fixtures, analyze)

if __name__ == "__main__":
    raise SystemExit(run_cli(SPEC))
