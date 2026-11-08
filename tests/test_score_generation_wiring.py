"""Wiring tests for the score-generation accept loop (#259).

The level-0 cap is applied in the accept loop, not in candidate selection.
These tests drive the loop with a stub labeler, so moving the cap back to the
selection quota, or dropping the call site, fails them.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from data.score_generation import Candidate

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "generate_score_questions.py"

RUBRIC = {
    "instructions": "How constrained is the next step?",
    "criteria": ["a", "b", "c", "d", "e"],
    "evidence": "state text",
}


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "generate_score_questions_wiring", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidate(index: int, source: str = "source") -> Candidate:
    return Candidate(
        source=source,
        state=f"state {index}",
        split="train",
        group=None,
        rubric=RUBRIC,
    )


def _label(candidate: Candidate, level: int) -> dict:
    probabilities = {str(index): 0.1 for index in range(5)}
    probabilities[str(level)] = 0.6
    total = sum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()}
    expected = sum(index * value for index, value in enumerate(probabilities.values()))
    return {
        "candidate": candidate,
        "reason": None,
        "probabilities": probabilities,
        "expected": expected,
        "label": float(level),
        "api_score": None,
        "error": "",
    }


def test_the_cap_applies_to_accepted_rows_at_the_call_site():
    module = _load_script()
    candidates = [_candidate(index) for index in range(20)]

    def label(candidate):
        index = int(candidate.state.split()[1])
        # Nine of every ten rows sit at level 0; one in ten clears it.
        return _label(candidate, 0 if index % 10 != 9 else 1)

    result = module.accept_candidates(
        candidates,
        label,
        target=100,
        level0_cap=0.5,
        concurrency=4,
    )
    accepted = result["accepted"]
    level0 = sum(1 for row in accepted if row["label"] == 0.0)
    assert accepted, "the loop accepted nothing"
    assert level0 <= math.ceil(0.5 * len(accepted))
    assert result["source_stats"]["source"]["reject:level0_cap"] > 0
    assert len(accepted) < len(candidates)


def test_post_target_results_are_counted_separately():
    module = _load_script()
    candidates = [_candidate(index) for index in range(8)]

    result = module.accept_candidates(
        candidates,
        lambda candidate: _label(candidate, 1),
        target=3,
        level0_cap=0.5,
        concurrency=4,
    )
    assert len(result["accepted"]) == 3
    assert result["calls"] == 4
    assert result["post_target"] == 1
    assert (
        len(result["accepted"]) + len(result["rejected"]) + result["post_target"]
        == result["calls"]
    )
