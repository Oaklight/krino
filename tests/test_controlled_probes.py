"""Unit tests for the controlled Jev probe suite; no test uses the network."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "probing" / "scripts" / "controlled"
SCRIPTS = ROOT / "probing" / "scripts"
sys.path.insert(0, str(CONTROLLED))
sys.path.insert(0, str(SCRIPTS))

from harness import (  # noqa: E402
    Fixture,
    ProbeSpec,
    interval_relation,
    log_odds_ratio_interval,
    probability_interval,
    response_record,
    selected_headers,
    subtract_intervals,
)
from jev_client import JevClient, RawResponse  # noqa: E402
from probe_1_output_serialization import build_fixtures as build_probe_1  # noqa: E402
from probe_2_option_interaction import build_fixtures as build_probe_2  # noqa: E402
from probe_3_likelihood_sensitivity import build_fixtures as build_probe_3  # noqa: E402
from probe_4_execution_topology import build_fixtures as build_probe_4  # noqa: E402
from probe_5_generation_signatures import build_fixtures as build_probe_5  # noqa: E402
from probe_5_generation_signatures import exact_sum_rule  # noqa: E402


class FakeTransport:
    """Capture request data and return a fixed response."""

    def __init__(self, response: RawResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> RawResponse:
        self.calls.append((url, dict(headers), body, timeout))
        return self.response


class ClientAndRecordTests(unittest.TestCase):
    def test_raw_transport_preserves_unicode_and_record_redacts_auth(self) -> None:
        body = json.dumps(
            {"answers": {"问_🧪_é": {"probabilities": {"雪": 0.51, "月": 0.49}}}},
            ensure_ascii=False,
        ).encode("utf-8")
        transport = FakeTransport(
            RawResponse(200, {"Content-Type": "application/json", "Authorization": "response-secret"}, body, 0.01, 0.02)
        )
        client = JevClient(api_key="top-secret", base_url="https://example.invalid", transport=transport)
        response = client.ask_raw(
            "状态 🧪",
            {"问_🧪_é": {"type": "choice", "instructions": "选择", "criteria": {"雪": "是", "月": "否"}}},
        )
        _url, headers, request_body, _timeout = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Bearer top-secret")
        self.assertIn("状态 🧪".encode(), request_body)
        self.assertNotIn(b"\\u", request_body)

        spec = ProbeSpec("test", "test", lambda _mode, _seed: [], lambda _records: {})
        fixture = Fixture(
            "unicode",
            "item",
            "prediction",
            {"state": "状态 🧪", "questions": {"问_🧪_é": {"criteria": {"雪": "是", "月": "否"}}}},
            {},
        )
        record = response_record(spec, fixture, 7, response, None)
        serialized = json.dumps(record, ensure_ascii=False).lower()
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("authorization", serialized)
        self.assertEqual(record["parsed_response"]["answers"]["问_🧪_é"]["probabilities"]["雪"], 0.51)
        self.assertEqual(record["response_headers"]["content_type"], "application/json")

    def test_unicode_identity_fixture_contains_distinct_exact_keys(self) -> None:
        fixtures = build_probe_1("smoke", 4)
        combining = next(fixture for fixture in fixtures if fixture.metadata["family"] == "combining")
        question_id = combining.metadata["question_id"]
        self.assertIn("é", question_id)
        self.assertIn(question_id, combining.payload["questions"])
        self.assertEqual(list(combining.payload["questions"][question_id]["criteria"]), combining.metadata["option_keys"])


class DeterminismTests(unittest.TestCase):
    def test_seeded_schedules_are_deterministic(self) -> None:
        for builder in (build_probe_1, build_probe_2, build_probe_3, build_probe_4, build_probe_5):
            first = [(fixture.condition_id, fixture.item_id) for fixture in builder("smoke", 123)]
            second = [(fixture.condition_id, fixture.item_id) for fixture in builder("smoke", 123)]
            different = [(fixture.condition_id, fixture.item_id) for fixture in builder("smoke", 124)]
            self.assertEqual(first, second)
            self.assertNotEqual(first, different)


class IntervalAnalysisTests(unittest.TestCase):
    def test_probability_rounding_intervals_clip_at_boundaries(self) -> None:
        self.assertEqual(probability_interval(0.0), (0.0, 0.005))
        self.assertEqual(probability_interval(1.0), (0.995, 1.0))
        low, high = probability_interval(0.42)
        self.assertAlmostEqual(low, 0.415)
        self.assertAlmostEqual(high, 0.425)
        self.assertEqual(interval_relation(probability_interval(0.50), probability_interval(0.50)), "rounding_indeterminate")
        self.assertEqual(interval_relation(probability_interval(0.50), probability_interval(0.52)), "definite_increase")
        with self.assertRaises(ValueError):
            probability_interval(1.1)

    def test_iia_log_odds_interval_distinguishes_definite_change(self) -> None:
        baseline = log_odds_ratio_interval(0.50, 0.50)
        unchanged = log_odds_ratio_interval(0.49, 0.49)
        shifted = log_odds_ratio_interval(0.70, 0.30)
        unchanged_delta = subtract_intervals(unchanged, baseline)
        shifted_delta = subtract_intervals(shifted, baseline)
        self.assertLessEqual(unchanged_delta[0], 0.0)
        self.assertGreaterEqual(unchanged_delta[1], 0.0)
        self.assertGreater(shifted_delta[0], 0.0)

    def test_exact_sum_analysis_uses_integer_cents(self) -> None:
        exact = exact_sum_rule({"a": 0.33, "b": 0.33, "c": 0.34})
        short = exact_sum_rule({"a": 0.33, "b": 0.33, "c": 0.33})
        self.assertTrue(exact["exact_sum"])
        self.assertEqual(exact["cent_sum"], 100)
        self.assertFalse(short["exact_sum"])
        self.assertEqual(short["residual_cents"], 1)


class DesignInvariantTests(unittest.TestCase):
    def test_probe_1_matches_request_bytes_and_crosses_entry_counts(self) -> None:
        fixtures = build_probe_1("smoke", 17)
        by_block: dict[int, list[Fixture]] = {}
        for fixture in fixtures:
            by_block.setdefault(fixture.metadata["block"], []).append(fixture)
        self.assertTrue(by_block)
        for cells in by_block.values():
            self.assertEqual(len({cell.metadata["matched_request_bytes"] for cell in cells}), 1)
            self.assertEqual({cell.metadata["returned_entry_count"] for cell in cells}, {2, 4})

    def test_probe_2_balances_added_option_positions(self) -> None:
        fixtures = build_probe_2("full", 17)
        positions: dict[tuple[int, str], set[int]] = {}
        for fixture in fixtures:
            condition = fixture.metadata["condition"]
            if condition == "pair":
                continue
            key = (fixture.metadata["semantic_item"], condition)
            positions.setdefault(key, set()).add(fixture.metadata["insertion_position"])
        self.assertTrue(positions)
        self.assertTrue(all(value == {0, 1, 2} for value in positions.values()))

    def test_probe_3_latin_square_balances_variants_by_label(self) -> None:
        fixtures = build_probe_3("full", 17)
        counts: dict[tuple[str, str, str], int] = {}
        for fixture in fixtures:
            label_class = fixture.metadata["label_class"]
            for label, variant in fixture.metadata["variant_by_label"].items():
                key = (label_class, label, variant)
                counts[key] = counts.get(key, 0) + 1
        grouped: dict[tuple[str, str], set[int]] = {}
        for (label_class, label, _variant), count in counts.items():
            grouped.setdefault((label_class, label), set()).add(count)
        self.assertTrue(grouped)
        self.assertTrue(all(len(values) == 1 for values in grouped.values()))

    def test_probe_4_full_factorial_has_five_replicates_per_cell(self) -> None:
        fixtures = build_probe_4("full", 17)
        counts: dict[tuple[int, int, int, int], int] = {}
        for fixture in fixtures:
            if fixture.metadata["kind"] != "factorial":
                continue
            cell = tuple(fixture.metadata[name] for name in ("S", "Q", "K", "L"))
            counts[cell] = counts.get(cell, 0) + 1
        self.assertEqual(len(counts), 3**4)
        self.assertEqual(set(counts.values()), {5})

    def test_typesafe_headers_are_captured(self) -> None:
        captured = selected_headers(
            {
                "x-typesafe-request-id": "req_test",
                "x-envoy-upstream-service-time": "64",
                "content-type": "application/json",
            }
        )
        self.assertEqual(captured["request_id"], "req_test")
        self.assertEqual(captured["upstream_time"], "64")


class CliTests(unittest.TestCase):
    def test_all_probe_dry_runs_need_no_key_and_write_nothing(self) -> None:
        scripts = sorted(CONTROLLED.glob("probe_[1-5]_*.py"))
        self.assertEqual(len(scripts), 5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "must-not-exist.jsonl"
            environment = os.environ.copy()
            environment.pop("TYPESAFE_API_KEY", None)
            for script in scripts:
                result = subprocess.run(
                    [sys.executable, str(script), "--mode", "smoke", "--seed", "9", "--output", str(output), "--dry-run"],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=f"{script.name}: {result.stderr}")
                self.assertIn("estimated requests=", result.stdout)
                self.assertIn("no network calls", result.stdout)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
