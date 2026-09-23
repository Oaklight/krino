"""Tests for scripts/transform_results.py."""

import json
from pathlib import Path

import pytest

# Import the transform module
import importlib.util

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "transform_results.py"

spec = importlib.util.spec_from_file_location("transform_results", SCRIPT_PATH)
transform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform)


class TestExtractBenchmarkResult:
    def test_choice_benchmark(self):
        raw = {
            "items": 200,
            "choice": {
                "accuracy": {"n": 200, "accuracy": 0.79},
                "ece": {"ece": 0.15, "n": 200, "n_bins": 10, "bins": []},
            },
            "latency": {"total_mean_s": 0.3},
        }
        result = transform.extract_benchmark_result(raw, "agnews")
        assert result["type"] == "choice"
        assert result["accuracy"] == 0.79
        assert result["ece"] == 0.15
        assert result["latency_ms"] == 300.0
        assert "bins" not in json.dumps(result)

    def test_noul_benchmark(self):
        raw = {
            "items": 200,
            "noul": {
                "accuracy": {"n": 200, "accuracy": 0.805, "threshold": 0.5},
                "ece": {"ece": 0.12, "n": 200, "n_bins": 10, "bins": []},
            },
        }
        result = transform.extract_benchmark_result(raw, "contractnli")
        assert result["type"] == "noul"
        assert result["accuracy"] == 0.805

    def test_score_benchmark(self):
        raw = {"items": 200, "score": {"n": 200, "mae": 0.489}}
        result = transform.extract_benchmark_result(raw, "sst5")
        assert result["type"] == "score"
        assert result["mae"] == 0.489
        assert "accuracy" not in result

    def test_mixed_benchmark(self):
        raw = {
            "items": 500,
            "noul": {"accuracy": {"n": 200, "accuracy": 0.8}, "ece": {"ece": 0.1}},
            "choice": {"accuracy": {"n": 200, "accuracy": 0.6}, "ece": {"ece": 0.2}},
            "score": {"n": 100, "mae": 0.3},
        }
        result = transform.extract_benchmark_result(raw, "typed_decisions")
        assert result["type"] == "mixed"
        assert "noul" in result["subtypes"]
        assert "choice" in result["subtypes"]
        assert "score" in result["subtypes"]


class TestGetAccuracyForAggregate:
    def test_choice(self):
        assert transform.get_accuracy_for_aggregate({"type": "choice", "accuracy": 0.8}) == 0.8

    def test_score_returns_none(self):
        assert transform.get_accuracy_for_aggregate({"type": "score", "mae": 0.5}) is None

    def test_mixed(self):
        result = {
            "type": "mixed",
            "subtypes": {
                "noul": {"accuracy": 0.8},
                "choice": {"accuracy": 0.6},
            },
        }
        assert transform.get_accuracy_for_aggregate(result) == pytest.approx(0.7)


class TestDetermineMethod:
    def test_logit(self):
        assert transform.determine_method("logit_Qwen_Qwen3-0.6B") == "logit"

    def test_phaseA_logit(self):
        assert transform.determine_method("phaseA_logit_Qwen_Qwen3-0.6B") == "logit"

    def test_reranker(self):
        assert transform.determine_method("reranker_cross-encoder_ettin-reranker-150m-v1") == "reranker"

    def test_jev_api(self):
        assert transform.determine_method("jev_api") == "jev_api"

    def test_openjev(self):
        assert transform.determine_method("openjev_Qwen_Qwen3-0.6B") == "openjev"


class TestEndToEnd:
    def test_output_file_valid(self):
        output_path = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"
        if not output_path.exists():
            pytest.skip("eval_results.json not generated")

        data = json.loads(output_path.read_text())

        assert "eval_runs" in data
        assert "training_runs" in data
        assert "jev_comparison" in data
        assert "benchmarks" in data
        assert len(data["eval_runs"]) > 0
        assert len(data["training_runs"]) > 0
        assert data["generated_at"] is not None

    def test_jev_run_exists(self):
        output_path = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"
        if not output_path.exists():
            pytest.skip("eval_results.json not generated")

        data = json.loads(output_path.read_text())
        jev_runs = [r for r in data["eval_runs"] if r["method"] == "jev_api"]
        assert len(jev_runs) == 1
        assert jev_runs[0]["model"]["name"] == "Jev (TypeSafe API)"
        assert jev_runs[0]["n_benchmarks"] == 18

    def test_no_ece_bins_in_output(self):
        output_path = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"
        if not output_path.exists():
            pytest.skip("eval_results.json not generated")

        raw = output_path.read_text()
        assert '"bins"' not in raw

    def test_jev_comparison_populated(self):
        output_path = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"
        if not output_path.exists():
            pytest.skip("eval_results.json not generated")

        data = json.loads(output_path.read_text())
        assert len(data["jev_comparison"]) > 0
        for bm, comp in data["jev_comparison"].items():
            assert "jev" in comp
            assert comp["jev"] > 0
