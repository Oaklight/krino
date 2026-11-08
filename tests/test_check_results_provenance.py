"""Tests for scripts/check_results_provenance.py."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_results_provenance.py"

spec = importlib.util.spec_from_file_location("check_results_provenance", SCRIPT_PATH)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

JEVBENCH_SOURCE = "model/experiments/logit_prov/jevbench_judgement_Qwen_Qwen3-4B.json"
EVAL_SOURCE = "model/experiments/phaseA_logit_Qwen_Qwen2.5-1.5B.json"


def jevbench_row() -> dict:
    row = guard.transform.build_jevbench_run(REPO_ROOT / JEVBENCH_SOURCE)
    assert row is not None
    return row


class TestJevBenchGuard:
    def test_committed_row_passes(self):
        assert guard.check_results({"jevbench_runs": [jevbench_row()]}, {JEVBENCH_SOURCE}) == []

    def test_missing_source_file_fails(self):
        row = jevbench_row()
        row.pop("source_file")
        errors = guard.check_results({"jevbench_runs": [row]}, {JEVBENCH_SOURCE})
        assert any("no source_file" in error for error in errors)

    def test_untracked_source_file_fails(self):
        errors = guard.check_results({"jevbench_runs": [jevbench_row()]}, set())
        assert any("not a committed file" in error for error in errors)

    def test_absolute_source_file_fails(self):
        row = jevbench_row()
        row["source_file"] = "/etc/passwd"
        errors = guard.check_results({"jevbench_runs": [row]}, {"/etc/passwd"})
        assert any("not repository-relative" in error for error in errors)

    def test_published_accuracy_must_recompute(self):
        row = jevbench_row()
        row["accuracy"] = 0.999
        errors = guard.check_results({"jevbench_runs": [row]}, {JEVBENCH_SOURCE})
        assert any("accuracy" in error for error in errors)

    def test_published_readout_must_recompute(self):
        row = jevbench_row()
        row["readout"] = "trained-heads"
        errors = guard.check_results({"jevbench_runs": [row]}, {JEVBENCH_SOURCE})
        assert any("readout" in error for error in errors)

    def test_duplicate_run_id_fails(self):
        row = jevbench_row()
        errors = guard.check_results({"jevbench_runs": [row, dict(row)]}, {JEVBENCH_SOURCE})
        assert any("duplicate run_id" in error for error in errors)


class TestEvalGuard:
    def eval_row(self) -> dict:
        row = guard.transform.transform_eval_file(REPO_ROOT / EVAL_SOURCE)
        assert row is not None
        return row

    def test_committed_row_passes(self):
        assert guard.check_results({"eval_runs": [self.eval_row()]}, {EVAL_SOURCE}) == []

    def test_aggregate_must_recompute(self):
        row = self.eval_row()
        row["results"]["aggregate"]["accuracy"] = 0.999
        errors = guard.check_results({"eval_runs": [row]}, {EVAL_SOURCE})
        assert any(error.startswith(f"{row['run_id']}: results") for error in errors)

    def test_untracked_source_file_fails(self):
        errors = guard.check_results({"eval_runs": [self.eval_row()]}, set())
        assert any("not a committed file" in error for error in errors)
