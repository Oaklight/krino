"""Shared execution and recording utilities for controlled Jev probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPTS = SCRIPT_DIR.parent
if str(PARENT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS))

from jev_client import JevClient, RawResponse  # noqa: E402

PROBE_VERSION = "1.0"
DEFAULT_RESULTS = SCRIPT_DIR.parents[1] / "results" / "controlled"
SELECTED_HEADERS = {
    "request_id": ("x-typesafe-request-id", "x-request-id", "request-id", "x-amzn-requestid", "trace-id"),
    "upstream_time": (
        "x-envoy-upstream-service-time",
        "x-upstream-response-time",
        "x-upstream-time",
        "server-timing",
    ),
    "content_type": ("content-type",),
    "content_length": ("content-length",),
}


@dataclass(frozen=True)
class Fixture:
    """One preregistered API request."""

    condition_id: str
    item_id: str
    prediction: str
    payload: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProbeSpec:
    """Metadata and fixture builder for a probe."""

    probe: str
    description: str
    build_fixtures: Callable[[str, int], list[Fixture]]
    analyze: Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize non-secret data deterministically while preserving Unicode."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def probability_interval(probability: float) -> tuple[float, float]:
    """Return the interval represented by a probability rounded to two decimals."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return max(0.0, probability - 0.005), min(1.0, probability + 0.005)


def interval_relation(first: tuple[float, float], second: tuple[float, float]) -> str:
    """Classify a change as definite or hidden by rounding."""
    if first[1] < second[0]:
        return "definite_increase"
    if second[1] < first[0]:
        return "definite_decrease"
    return "rounding_indeterminate"


def log_odds_ratio_interval(
    a_probability: float,
    b_probability: float,
    epsilon: float = 1e-12,
) -> tuple[float, float]:
    """Bound log(P(A)/P(B)) for interval-censored displayed probabilities."""
    a_low, a_high = probability_interval(a_probability)
    b_low, b_high = probability_interval(b_probability)
    return math.log(max(a_low, epsilon) / max(b_high, epsilon)), math.log(max(a_high, epsilon) / max(b_low, epsilon))


def subtract_intervals(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    """Return the interval for left minus right."""
    return left[0] - right[1], left[1] - right[0]


def classify_zero(interval: tuple[float, float]) -> str:
    if interval[0] > 0.0:
        return "definitely_positive"
    if interval[1] < 0.0:
        return "definitely_negative"
    return "contains_zero"


def selected_headers(headers: Mapping[str, str]) -> dict[str, str | None]:
    """Select a stable, explicitly non-authorization response-header subset."""
    lowered = {key.lower(): value for key, value in headers.items()}
    return {
        output_name: next((lowered[name] for name in candidates if name in lowered), None)
        for output_name, candidates in SELECTED_HEADERS.items()
    }


def safe_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a request payload without changing experimentally relevant key order."""
    serialized = JevClient.encode_payload(payload)
    lowered = serialized.lower()
    if b"authorization" in lowered or b"bearer " in lowered:
        raise ValueError("request payload contains authorization-like material")
    copied = json.loads(serialized.decode("utf-8"))
    if not isinstance(copied, dict):
        raise ValueError("request payload must be a JSON object")
    return copied


def response_record(
    spec: ProbeSpec,
    fixture: Fixture,
    seed: int,
    response: RawResponse | None,
    error: str | None,
) -> dict[str, Any]:
    payload = safe_payload(fixture.payload)
    request_bytes = JevClient.encode_payload(payload)
    parsed: Any = None
    parse_error: str | None = None
    if response is not None and response.body:
        try:
            parsed = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "probe": spec.probe,
        "version": PROBE_VERSION,
        "preregistered_prediction": fixture.prediction,
        "seed": seed,
        "condition_id": fixture.condition_id,
        "item_id": fixture.item_id,
        "metadata": fixture.metadata,
        "request_payload": payload,
        "request_payload_sha256": sha256_bytes(request_bytes),
        "http_status": response.status if response else None,
        "parsed_response": parsed,
        "raw_response_sha256": sha256_bytes(response.body) if response else None,
        "raw_response_byte_length": len(response.body) if response else None,
        "response_headers": selected_headers(response.headers) if response else selected_headers({}),
        "ttfb_s": response.ttfb_s if response else None,
        "total_time_s": response.total_s if response else None,
        "error": error or parse_error,
    }


def execute(
    spec: ProbeSpec,
    fixtures: Sequence[Fixture],
    seed: int,
    output: Path,
    client: JevClient,
) -> list[dict[str, Any]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as stream:
        for fixture in fixtures:
            response: RawResponse | None = None
            error: str | None = None
            try:
                response = client.ask_raw(
                    fixture.payload["state"],
                    fixture.payload["questions"],
                    fixture.payload.get("model", "jev-latest"),
                )
                if response.status != 200:
                    error = f"HTTP {response.status}"
            except Exception as exc:  # Keep the experiment schedule running.
                error = f"{type(exc).__name__}: {exc}"
                if client.api_key:
                    error = error.replace(client.api_key, "[REDACTED]")
            record = response_record(spec, fixture, seed, response, error)
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_analysis(output: Path, analysis: Mapping[str, Any]) -> Path:
    analysis_path = output.with_suffix(".analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with analysis_path.open("w", encoding="utf-8") as stream:
        json.dump(analysis, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return analysis_path


def answer_probabilities(record: Mapping[str, Any]) -> dict[str, float]:
    """Extract the first answer's displayed probability mapping."""
    parsed = record.get("parsed_response")
    if not isinstance(parsed, Mapping):
        return {}
    answers = parsed.get("answers")
    if not isinstance(answers, Mapping) or not answers:
        return {}
    answer = next(iter(answers.values()))
    if not isinstance(answer, Mapping):
        return {}
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, Mapping):
        return {str(key): float(value) for key, value in probabilities.items()}
    noul = answer.get("noul")
    if isinstance(noul, (int, float)):
        return {"noul": float(noul)}
    probability = answer.get("probability")
    if isinstance(probability, (int, float)):
        return {"noul": float(probability)}
    return {}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Compute Pearson correlation without third-party numerical packages."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    if x_ss == 0.0 or y_ss == 0.0:
        return None
    return numerator / math.sqrt(x_ss * y_ss)


def simple_regression(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float] | None:
    """Fit y = intercept + slope*x by ordinary least squares."""
    correlation = pearson(xs, ys)
    if correlation is None:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    return {"intercept": y_mean - slope * x_mean, "slope": slope, "r": correlation, "r_squared": correlation**2}


def shuffled(rng: random.Random, values: Iterable[Any]) -> list[Any]:
    result = list(values)
    rng.shuffle(result)
    return result


def add_common_arguments(parser: argparse.ArgumentParser, spec: ProbeSpec) -> None:
    parser.description = spec.description
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS / f"{spec.probe}.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Build and count fixtures without network or output files")
    parser.add_argument("--confirm-live", action="store_true", help="Required acknowledgement before API calls")
    parser.add_argument("--analyze-only", type=Path, help="Analyze an existing JSONL file without API calls")
    parser.add_argument(
        "--reference-probabilities",
        type=Path,
        help="Optional JSON object mapping item/variant IDs to external reference probabilities",
    )


def _with_reference(records: list[dict[str, Any]], path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return records
    with path.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    if not isinstance(reference, Mapping):
        raise ValueError("reference probability file must contain a JSON object")
    return [*records, {"external_reference_probabilities": reference}]


def run_cli(spec: ProbeSpec, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser, spec)
    args = parser.parse_args(argv)
    if args.analyze_only:
        records = _with_reference(read_jsonl(args.analyze_only), args.reference_probabilities)
        analysis_path = write_analysis(args.output, spec.analyze(records))
        print(f"Analyzed {len(records)} records -> {analysis_path}")
        return 0

    fixtures = spec.build_fixtures(args.mode, args.seed)
    print(f"Probe {spec.probe}: mode={args.mode}, seed={args.seed}, estimated requests={len(fixtures)}")
    if args.dry_run:
        print("Dry run: no network calls and no output files written.")
        return 0
    if not args.confirm_live:
        parser.error("live execution requires --confirm-live (use --dry-run to inspect the schedule)")
    if "TYPESAFE_API_KEY" not in os.environ:
        parser.error("TYPESAFE_API_KEY is required for live execution")

    records = execute(spec, fixtures, args.seed, args.output, JevClient())
    analysis_path = write_analysis(args.output, spec.analyze(records))
    error_count = sum(record["error"] is not None for record in records)
    print(f"Wrote {len(records)} records ({error_count} errors) -> {args.output}")
    print(f"Wrote analysis -> {analysis_path}")
    return 0
