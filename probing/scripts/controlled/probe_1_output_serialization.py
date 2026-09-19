#!/usr/bin/env python3
"""Probe 1: output identity preservation and serialization/latency scaling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from harness import Fixture, ProbeSpec, canonical_json_bytes, run_cli, simple_regression

PREDICTION = (
    "If visible JSON is generated token by token, latency should increase with visible output bytes after "
    "model-visible input bytes and returned-entry count are controlled. Flat scaling disfavors full visible JSON "
    "generation, but remains compatible with compact internal decoding followed by deterministic serialization."
)
FAMILIES = {
    "ascii": "x",
    "unicode": "雪",
    "combining": "é",
    "emoji": "🧪",
    "punctuation": "._-~",
}


def build_fixtures(mode: str, seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    lengths = (1, 8, 32) if mode == "smoke" else (1, 16, 64, 256)
    entry_counts = (2, 4) if mode == "smoke" else (2, 8, 32)
    repeats = 1 if mode == "smoke" else 6
    fixtures: list[Fixture] = []
    for block in range(repeats):
        cells: list[Fixture] = []
        for family, unit in FAMILIES.items():
            for length in lengths:
                for entry_count in entry_counts:
                    question_id = f"q_{family}_{unit * length}"
                    option_keys = [f"opt_{index:02d}_{unit * length}" for index in range(entry_count)]
                    questions = {
                        question_id: {
                            "type": "choice",
                            "instructions": "Choose the option explicitly marked preferred.",
                            "criteria": {
                                key: "preferred" if index == 0 else "not preferred"
                                for index, key in enumerate(option_keys)
                            },
                        }
                    }
                    payload = {
                        "state": "Matched control. " + ("P" * 1024),
                        "model": "jev-latest",
                        "questions": questions,
                    }
                    cells.append(
                        Fixture(
                            f"family={family};length={length};entries={entry_count};block={block}",
                            f"identity-{family}-{length}-{entry_count}-{block}",
                            PREDICTION,
                            payload,
                            {
                                "family": family,
                                "key_length_units": length,
                                "question_id": question_id,
                                "option_keys": option_keys,
                                "returned_entry_count": entry_count,
                                "block": block,
                            },
                        )
                    )
        target_request_bytes = max(len(canonical_json_bytes(cell.payload)) for cell in cells)
        padded_cells: list[Fixture] = []
        for cell in cells:
            missing = target_request_bytes - len(canonical_json_bytes(cell.payload))
            payload = dict(cell.payload)
            payload["state"] = str(payload["state"]) + ("P" * missing)
            matched_bytes = len(canonical_json_bytes(payload))
            metadata = {
                **cell.metadata,
                "input_padding_chars": 1024 + missing,
                "matched_request_bytes": matched_bytes,
            }
            padded_cells.append(Fixture(cell.condition_id, cell.item_id, cell.prediction, payload, metadata))
        if len({cell.metadata["matched_request_bytes"] for cell in padded_cells}) != 1:
            raise AssertionError("failed to construct byte-matched request controls")
        rng.shuffle(padded_cells)
        fixtures.extend(padded_cells)
    return fixtures


def _regressions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    xs_bytes: list[float] = []
    xs_tokens: list[float] = []
    ttfb: list[float] = []
    total: list[float] = []
    for record in records:
        byte_length = record.get("raw_response_byte_length")
        parsed = record.get("parsed_response")
        usage = parsed.get("usage", {}) if isinstance(parsed, Mapping) else {}
        output_tokens = usage.get("output_tokens") if isinstance(usage, Mapping) else None
        record_ttfb = record.get("ttfb_s")
        record_total = record.get("total_time_s")
        if not all(isinstance(value, (int, float)) for value in (byte_length, record_ttfb, record_total)):
            continue
        xs_bytes.append(float(byte_length))
        ttfb.append(float(record_ttfb))
        total.append(float(record_total))
        if isinstance(output_tokens, (int, float)):
            xs_tokens.append(float(output_tokens))
    result: dict[str, Any] = {
        "response_bytes_vs_ttfb": simple_regression(xs_bytes, ttfb),
        "response_bytes_vs_total": simple_regression(xs_bytes, total),
    }
    if len(xs_tokens) == len(ttfb):
        result["reported_output_tokens_vs_ttfb"] = simple_regression(xs_tokens, ttfb)
        result["reported_output_tokens_vs_total"] = simple_regression(xs_tokens, total)
    return result


def analyze(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identity_failures: list[str] = []
    by_entry_count: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    request_sizes: set[int] = set()
    for record in records:
        metadata = record.get("metadata", {})
        parsed = record.get("parsed_response")
        expected_q = metadata.get("question_id")
        answers = parsed.get("answers", {}) if isinstance(parsed, Mapping) else {}
        answer = answers.get(expected_q) if isinstance(answers, Mapping) else None
        probabilities = answer.get("probabilities", {}) if isinstance(answer, Mapping) else {}
        if expected_q not in answers or set(probabilities) != set(metadata.get("option_keys", [])):
            identity_failures.append(str(record.get("item_id")))
        entry_count = metadata.get("returned_entry_count")
        if isinstance(entry_count, int):
            by_entry_count[entry_count].append(record)
        matched_bytes = metadata.get("matched_request_bytes")
        if isinstance(matched_bytes, int):
            request_sizes.add(matched_bytes)
        total = record.get("total_time_s")
        if isinstance(total, (int, float)):
            by_family[str(metadata.get("family"))].append(float(total))
    return {
        "records": len(records),
        "identity_failures": identity_failures,
        "matched_request_byte_sizes": sorted(request_sizes),
        "overall_latency_scaling": _regressions(records),
        "within_entry_count_latency_scaling": {
            str(entry_count): _regressions(group)
            for entry_count, group in sorted(by_entry_count.items())
        },
        "mean_total_time_by_string_family": {
            family: sum(values) / len(values) for family, values in by_family.items() if values
        },
        "interpretation_guardrail": (
            "Within-entry-count response-size slopes control candidate-scoring count. Weak slopes can disfavor full "
            "visible JSON generation, but cannot exclude compact internal decoding followed by ordinary serialization."
        ),
    }


SPEC = ProbeSpec("probe_1_output_serialization", __doc__ or "", build_fixtures, analyze)

if __name__ == "__main__":
    raise SystemExit(run_cli(SPEC))
