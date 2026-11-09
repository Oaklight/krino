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


class TestProvenanceIsNotABenchmark:
    """A file with ``_provenance`` must transform exactly like one without it.

    That is the property that matters: the September files carry no provenance,
    and adding the block to new files must not change the benchmark set or the
    aggregate the dashboard reads.
    """

    STEM = "logit_Qwen_Qwen3-4B.json"

    def _payload(self):
        return {
            "agnews": {
                "items": 200,
                "choice": {
                    "accuracy": {"n": 200, "accuracy": 0.79},
                    "ece": {"ece": 0.15, "n": 200, "n_bins": 10, "bins": []},
                },
            },
            "sst2": {
                "items": 200,
                "noul": {
                    "accuracy": {"n": 200, "accuracy": 0.71},
                    "ece": {"ece": 0.10, "n": 200, "n_bins": 10, "bins": []},
                },
            },
        }

    def test_same_benchmarks_and_aggregate(self, tmp_path):
        plain_dir = tmp_path / "plain"
        provenanced_dir = tmp_path / "provenanced"
        plain_dir.mkdir()
        provenanced_dir.mkdir()
        plain = plain_dir / self.STEM
        plain.write_text(json.dumps(self._payload()))
        payload = {"_provenance": {"model_id": "Qwen/Qwen3-4B", "revision": "abc"}}
        payload.update(self._payload())
        provenanced = provenanced_dir / self.STEM
        provenanced.write_text(json.dumps(payload))

        a = transform.transform_eval_file(plain)
        b = transform.transform_eval_file(provenanced)

        assert set(a["results"]["by_source"]) == set(b["results"]["by_source"])
        assert a["results"]["by_source"] == b["results"]["by_source"]
        assert a["results"]["aggregate"] == b["results"]["aggregate"]
        assert a["n_benchmarks"] == b["n_benchmarks"] == 2

    def test_provenance_key_never_becomes_a_benchmark(self, tmp_path):
        payload = {"_provenance": {"model_id": "m"}, **self._payload()}
        path = tmp_path / self.STEM
        path.write_text(json.dumps(payload))
        result = transform.transform_eval_file(path)
        assert "_provenance" not in result["results"]["by_source"]
        assert "provenance" not in json.dumps(result["results"]["by_source"])


class TestDeriveReadout:
    def test_judgement(self):
        path = Path("jevbench_judgement_M.json")
        raw = {"_provenance": {"strategy": "judgement"}}
        assert transform.derive_readout(path, raw) == "judgement"

    def test_continuation(self):
        path = Path("jevbench_logit_M.json")
        raw = {"_provenance": {"strategy": "description"}}
        assert transform.derive_readout(path, raw) == "continuation"

    def test_native_reranker(self):
        path = Path("jevbench_reranker_native_M.json")
        raw = {"_provenance": {"strategy": "reranker_native"}}
        assert transform.derive_readout(path, raw) == "native-reranker"

    def test_raw_reranker_mode(self):
        assert transform.derive_readout(Path("raw-M.json"), {"mode": "raw_reranker"}) == "native-reranker"

    def test_native_decision(self):
        path = Path("h2o-M.json")
        raw = {"_provenance": {"strategy": "native_decision"}}
        assert transform.derive_readout(path, raw) == "native-decision"

    def test_native_decision_oneway(self):
        path = Path("h2o-M-oneway.json")
        raw = {"_provenance": {"strategy": "native_decision_oneway"}}
        assert transform.derive_readout(path, raw) == "native-decision-oneway"

    def test_trained_heads_by_suffix(self):
        assert transform.derive_readout(Path("krino-M-heads.json"), {}) == "trained-heads"

    def test_unlabelled_returns_none(self):
        assert transform.derive_readout(Path("mystery.json"), {}) is None


class TestBuildJevbenchRun:
    PAYLOAD = {
        "_provenance": {
            "model_id": "Qwen/Qwen3-4B",
            "strategy": "judgement",
            "variant": "instruct",
            "readout": 'per-option judgement, log P(" yes") against log P(" no") at "Answer:"',
        },
        "items": 231,
        "n": 231,
        "summary": {
            "overall": {"accuracy": 0.6190476190476191, "correct": 143, "total": 231},
            "by_type": {"choice": {"accuracy": 0.6402, "correct": 89, "total": 139}},
            "by_tier": {"easy": {"accuracy": 1.0, "correct": 48, "total": 48}},
        },
    }

    def test_provenance_supplies_model_id_readout_and_n(self, tmp_path):
        path = tmp_path / "jevbench_judgement_Qwen_Qwen3-4B.json"
        path.write_text(json.dumps(self.PAYLOAD))

        run = transform.build_jevbench_run(path)

        assert run["model_id"] == "Qwen/Qwen3-4B"
        assert run["model"]["name"] == "Qwen/Qwen3-4B (instruct)"
        assert run["readout"] == "judgement"
        assert run["readout_detail"] == self.PAYLOAD["_provenance"]["readout"]
        assert run["method"] == "zero-shot"
        assert run["n"] == 231
        assert run["total"] == 231
        assert run["accuracy"] == 0.619048

    def test_trained_heads_file_uses_its_own_accuracy(self, tmp_path):
        path = tmp_path / "krino-ettin-150m-heads.json"
        path.write_text(json.dumps({
            "overall": {"accuracy": 0.36796536796536794, "correct": 85, "total": 231},
            "by_tier": {},
            "by_type": {},
        }))

        run = transform.build_jevbench_run(path)

        assert run["readout"] == "trained-heads"
        assert run["method"] == "trained-heads"
        assert run["model"]["name"] == "Ettin-150m (trained)"
        assert run["n"] == 231
        assert run["accuracy"] == 0.367965

    def test_unlabelled_file_is_skipped(self, tmp_path):
        path = tmp_path / "mystery.json"
        path.write_text(json.dumps({"overall": {"accuracy": 0.5, "total": 231}}))
        assert transform.build_jevbench_run(path) is None


class TestJevbenchSourceFiles:
    def test_every_row_has_a_readout_and_a_source_file(self):
        runs = transform.transform_jevbench()
        assert runs
        for run in runs:
            assert run["readout"] in {
                "trained-heads", "continuation", "judgement", "native-reranker", "panel2",
                "native-decision", "native-decision-oneway",
            }
            assert run["source_file"]

    def test_logit_prov_judgement_and_continuation_are_read(self):
        names = {p.name for p in transform.jevbench_source_files()}
        assert "jevbench_judgement_Qwen_Qwen3-4B.json" in names
        assert "jevbench_logit_Qwen_Qwen3-4B.json" in names

    def test_native_reranker_is_not_double_counted(self):
        names = {p.name for p in transform.jevbench_source_files()}
        assert "jevbench_reranker_native_Qwen_Qwen3-Reranker-4B.json" not in names


class TestAddHeadVersion:
    def test_parenthesised_name(self):
        assert transform.add_head_version("Ettin-150m (trained)", "v1") == "Ettin-150m (trained, v1)"

    def test_bare_name(self):
        assert transform.add_head_version("Ettin-150m", "v1") == "Ettin-150m (v1)"


class TestTrainedHeadFields:
    PAYLOAD = {
        "head_version": "v1",
        "run_id": "oaklight/krino-ettin-150m-heads",
        "backbone": {"model_id": "cross-encoder/ettin-reranker-150m-v1"},
        "code_commit": None,
        "overall": {"accuracy": 0.368, "correct": 85, "total": 231},
        "by_tier": {},
        "by_type": {},
    }

    def test_fields_reach_the_row(self, tmp_path):
        path = tmp_path / "krino-ettin-150m-heads.json"
        path.write_text(json.dumps(self.PAYLOAD))

        run = transform.build_jevbench_run(path)

        assert run["head_version"] == "v1"
        assert run["run_id"] == "oaklight/krino-ettin-150m-heads"
        assert run["backbone_model_id"] == "cross-encoder/ettin-reranker-150m-v1"
        assert run["code_commit"] is None
        assert run["model"]["name"] == "Ettin-150m (trained, v1)"

    def test_non_trained_row_has_no_generation(self, tmp_path):
        path = tmp_path / "jevbench_judgement_M.json"
        path.write_text(json.dumps({
            "_provenance": {"model_id": "Qwen/Qwen3-4B", "strategy": "judgement"},
            "n": 231,
            "summary": {"overall": {"accuracy": 0.6, "correct": 138, "total": 231}},
        }))

        run = transform.build_jevbench_run(path)

        assert run["head_version"] is None
        assert run["run_id"] == "jevbench-jevbench_judgement_M"


class TestTrainedHeadRows:
    def test_trained_head_rows_carry_version_and_run(self):
        runs = transform.transform_jevbench()
        trained = [run for run in runs if run["readout"] == "trained-heads"]
        assert len(trained) == 6
        for run in trained:
            assert run["head_version"] == "v1"
            assert run["run_id"].startswith("oaklight/")

    def test_other_rows_have_no_generation(self):
        for run in transform.transform_jevbench():
            if run["readout"] != "trained-heads":
                assert run["head_version"] is None


class TestLanding:
    OUTPUT = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"

    def _data(self) -> dict:
        if not self.OUTPUT.exists():
            pytest.skip("eval_results.json not generated")
        return json.loads(self.OUTPUT.read_text())

    def test_block_recomputes_from_the_rows(self):
        data = self._data()
        expected = transform.build_landing(
            data["jevbench_runs"], data["eval_runs"], data["training_runs"]
        )
        assert data["landing"] == expected

    def test_key_figures(self):
        landing = self._data()["landing"]

        assert landing["judgement_vs_continuation"]["measured"] == 16
        assert landing["judgement_vs_continuation"]["wins"] == 15
        assert landing["judgement_vs_continuation"]["above_10_count"] == 10
        assert landing["judgement_vs_continuation"]["above_10_min_pp"] == 10.4
        assert landing["judgement_vs_continuation"]["above_10_max_pp"] == 18.6

        assert landing["post_trained_4b"]["base_pp"] == 53.7
        assert landing["post_trained_4b"]["min_pp"] == 61.9
        assert landing["post_trained_4b"]["max_pp"] == 64.9

        assert landing["resolution"]["jevbench_pp"] == 9.3
        assert landing["resolution"]["dev_card_min_pp"] == 5
        assert landing["resolution"]["dev_card_max_pp"] == 6

        assert landing["instrumented_rows"] == 57
        assert landing["banking77"]["trained_pp"] == 95.2
        assert landing["banking77"]["jev_pp"] == 75.0

    def test_source_list_is_one_deduplicated_union(self):
        landing = self._data()["landing"]
        sources = landing["source_files"]

        assert sources == sorted(set(sources))
        assert "web/public/results/eval_results.json" in sources
        assert "model/experiments/resolution_inputs.json" in sources
        assert landing["banking77"]["trained_source_file"] in sources
        assert landing["banking77"]["jev_source_file"] in sources
        for section in ("judgement_vs_continuation", "post_trained_4b", "resolution"):
            assert set(landing[section]["source_files"]) <= set(sources)


class TestThirdPartyCells:
    """H2O-Lightning-4B is a trained third-party model, not a readout cell."""

    CELLS = (("h2o-lightning-4b", "native_decision", 206), ("h2o-lightning-4b-oneway", "native_decision_oneway", 202))

    def test_cells_carry_strategy_accuracy_and_latency(self):
        for name, strategy, correct in self.CELLS:
            raw = json.loads((REPO_ROOT / "results" / "jevbench" / f"{name}.json").read_text())
            assert raw["_provenance"]["strategy"] == strategy
            assert raw["_provenance"]["model_id"] == "h2oai/h2o-lightning-4b"
            assert raw["summary"]["overall"]["correct"] == correct
            assert raw["summary"]["overall"]["total"] == 231
            assert raw["latency"]["n"] == 231
            assert raw["latency"]["mean_s"] > 0

    def test_rows_are_third_party(self):
        runs = {run["run_id"]: run for run in transform.transform_jevbench()}
        two_way = runs["jevbench-h2o-lightning-4b"]
        one_way = runs["jevbench-h2o-lightning-4b-oneway"]
        assert two_way["method"] == one_way["method"] == "third-party"
        assert two_way["readout"] == "native-decision"
        assert one_way["readout"] == "native-decision-oneway"
        assert two_way["latency_ms"] is not None
