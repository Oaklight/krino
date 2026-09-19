#!/usr/bin/env python3
"""Probe 4: factorial latency scaling and cue/reference-remapping topology."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from harness import Fixture, ProbeSpec, answer_probabilities, run_cli

PREDICTION = (
    "Latency coefficients for S, Q, K, L and interactions S×Q, S×K, Q×K characterize operational scaling. "
    "Late-vs-early cues and reference remapping may reveal positional sensitivity, but HTTP behavior cannot identify "
    "causal versus bidirectional attention because either architecture can compute the same mapping before readout."
)


def _latency_fixture(s: int, q: int, k: int, length: int, block: int) -> Fixture:
    questions = {}
    for question in range(q):
        criteria = {f"o{option}": (f"option {option} " + "d" * length) for option in range(k)}
        questions[f"q{question}"] = {"type": "choice", "instructions": "Choose option o0.", "criteria": criteria}
    return Fixture(
        f"factorial;S={s};Q={q};K={k};L={length};block={block}",
        f"factorial-{s}-{q}-{k}-{length}-{block}",
        PREDICTION,
        {"state": "s" * s, "model": "jev-latest", "questions": questions},
        {"kind": "factorial", "S": s, "Q": q, "K": k, "L": length, "block": block},
    )


def build_fixtures(mode: str, seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    levels = ((64, 512), (1, 3), (2, 4), (8, 64)) if mode == "smoke" else ((64, 1024, 4096), (1, 4, 12), (2, 8, 16), (8, 64, 256))
    blocks = 2 if mode == "smoke" else 5
    fixtures: list[Fixture] = []
    for block in range(blocks):
        block_fixtures = [_latency_fixture(s, q, k, length, block) for s in levels[0] for q in levels[1] for k in levels[2] for length in levels[3]]
        rng.shuffle(block_fixtures)
        fixtures.extend(block_fixtures)
    cue_items = 2 if mode == "smoke" else 100
    for item in range(cue_items):
        token_a, token_b = f"AX{item}", f"BZ{item}"
        for placement in ("early", "late"):
            cue = f"Mapping: {token_a}=approve; {token_b}=reject."
            usage = f"The referenced decision is {token_a}."
            state = f"{cue} {usage}" if placement == "early" else f"{usage} {cue}"
            for remapped in (False, True):
                criteria = {"approve": "approve", "reject": "reject"}
                if remapped:
                    criteria = {"reject": "approve", "approve": "reject"}
                fixtures.append(
                    Fixture(
                        f"cue={placement};remapped={remapped}",
                        f"cue-{item:03d}",
                        PREDICTION,
                        {"state": state, "model": "jev-latest", "questions": {"q": {"type": "choice", "instructions": "Resolve the code using the stated mapping.", "criteria": criteria}}},
                        {
                            "kind": "cue",
                            "placement": placement,
                            "remapped": remapped,
                            "semantic_item": item,
                            "correct_option": "reject" if remapped else "approve",
                        },
                    )
                )
    return fixtures


def _linear_fit(rows: list[list[float]], values: list[float]) -> dict[str, float] | None:
    if not rows:
        return None
    columns = len(rows[0])
    matrix = [[sum(row[i] * row[j] for row in rows) for j in range(columns)] for i in range(columns)]
    vector = [sum(row[i] * value for row, value in zip(rows, values)) for i in range(columns)]
    for pivot in range(columns):
        swap = max(range(pivot, columns), key=lambda row: abs(matrix[row][pivot]))
        matrix[pivot], matrix[swap] = matrix[swap], matrix[pivot]
        vector[pivot], vector[swap] = vector[swap], vector[pivot]
        if abs(matrix[pivot][pivot]) < 1e-12:
            return None
        divisor = matrix[pivot][pivot]
        matrix[pivot] = [value / divisor for value in matrix[pivot]]
        vector[pivot] /= divisor
        for row in range(columns):
            if row == pivot:
                continue
            factor = matrix[row][pivot]
            matrix[row] = [value - factor * base for value, base in zip(matrix[row], matrix[pivot])]
            vector[row] -= factor * vector[pivot]
    names = ["intercept", "S", "Q", "K", "L", "SxQ", "SxK", "QxK"]
    return dict(zip(names, vector))


def analyze(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[list[float]] = []
    times: list[float] = []
    cue: dict[str, list[float]] = {}
    for record in records:
        metadata = record.get("metadata", {})
        if metadata.get("kind") == "factorial" and isinstance(record.get("total_time_s"), (int, float)):
            s, q, k, length = (float(metadata[name]) for name in ("S", "Q", "K", "L"))
            rows.append([1.0, s, q, k, length, s * q, s * k, q * k])
            times.append(float(record["total_time_s"]))
        elif metadata.get("kind") == "cue":
            key = f"{metadata.get('placement')};remapped={metadata.get('remapped')}"
            probabilities = answer_probabilities(record)
            correct_option = str(metadata.get("correct_option"))
            if correct_option in probabilities:
                cue.setdefault(key, []).append(probabilities[correct_option])
    return {
        "latency_model_coefficients": _linear_fit(rows, times),
        "latency_features": ["S", "Q", "K", "L", "S×Q", "S×K", "Q×K"],
        "cue_condition_mean_probabilities": {key: sum(values) / len(values) for key, values in cue.items() if values},
        "non_identifiability": "Causal and bidirectional implementations can both consume the complete HTTP request before producing typed outputs; cue timing is not an architecture proof.",
    }


SPEC = ProbeSpec("probe_4_execution_topology", __doc__ or "", build_fixtures, analyze)

if __name__ == "__main__":
    raise SystemExit(run_cli(SPEC))
