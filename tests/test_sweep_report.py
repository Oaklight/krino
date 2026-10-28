"""Tests for the training-log sweep report.

The report is a standalone script, so these run it as a subprocess: that
exercises the real CLI and avoids depending on ``model.scripts`` being
importable as a package (it has no ``__init__.py``).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "model" / "scripts" / "sweep_report.py"

_LOG_TEMPLATE = """\
=== Multi-task training ===
Backbone: Qwen/Qwen3-Reranker-0.6B (causal)
Config: model/configs/r2_reranker0.6b.yaml
{extra}  epoch 1/5: train_loss=1.2512 eval_loss=1.2000 eval_acc={ep1} (787.5s)
    by_type: choice=0.3310(6864) | noul=0.6150(4723) | score=0.3560(1857)
  epoch 2/5: train_loss=1.6151 eval_loss=1.5079 eval_acc={ep2} (783.3s)
    by_type: choice=0.3250(6864) | noul=0.6160(4723) | score=0.2550(1857)
"""


def _write_log(tmp_path: Path, name: str, ep1: str, ep2: str, extra: str = "") -> Path:
    path = tmp_path / f"{name}.log"
    path.write_text(_LOG_TEMPLATE.format(ep1=ep1, ep2=ep2, extra=extra))
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestParsing:
    def test_markdown_table_has_epoch_columns(self, tmp_path):
        log = _write_log(tmp_path, "run_a", "0.4342", "0.4174")
        result = _run(str(log))
        assert result.returncode == 0, result.stderr
        assert "| ep1 | ep2 |" in result.stdout
        assert "43.42" in result.stdout
        assert "41.74" in result.stdout

    def test_readout_and_lora_from_banner(self, tmp_path):
        extra = "  Readout: weighted over 37 hidden states\n"
        log = _write_log(tmp_path, "run_a", "0.4342", "0.4174", extra=extra)
        result = _run(str(log))
        assert "weighted" in result.stdout

    def test_lora_scope_parsed(self, tmp_path):
        extra = (
            "  LoRA: 56 layers (all blocks, r=16, alpha=32, "
            "targets=q_proj,v_proj), 2,293,760 trainable params in backbone\n"
        )
        log = _write_log(tmp_path, "run_a", "0.4342", "0.4174", extra=extra)
        result = _run(str(log))
        assert "all blocks" in result.stdout

    def test_epoch_without_eval_is_tolerated(self, tmp_path):
        """eval_every > 1 prints train loss only; that must not crash."""
        log = tmp_path / "partial.log"
        log.write_text(
            "Backbone: Qwen/Qwen3-Reranker-0.6B (causal)\n"
            "  epoch 1/20: train_loss=1.3510 (787.3s)\n"
        )
        result = _run(str(log))
        assert result.returncode == 0, result.stderr
        # No eval checkpoint yet → no best column value.
        assert "ep1" in result.stdout

    def test_missing_file_is_an_error(self, tmp_path):
        result = _run(str(tmp_path / "nope.log"))
        assert result.returncode == 1
        assert "not a file" in result.stderr


class TestDeltas:
    def test_delta_against_frozen_last_baseline(self, tmp_path):
        baseline = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        variant = _write_log(tmp_path, "arch_06b_frozen_weighted", "0.4333", "0.4357")
        result = _run(str(baseline), str(variant))
        assert result.returncode == 0, result.stderr
        # Compared at ep2, the last epoch both have: 43.57 - 41.74 = +1.83.
        assert "+1.83 (ep2)" in result.stdout

    def test_delta_uses_the_last_common_epoch(self, tmp_path):
        """An unfinished run must be compared where the curves overlap.

        Comparing bests would put a still-climbing LoRA run (ep1 only) at
        48.48 against the frozen run's final 49.11 and read as a regression,
        when at the one epoch both ran it is 5.06 pp ahead.
        """
        baseline = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        partial = tmp_path / "arch_06b_loraAll_last.log"
        partial.write_text(
            "Backbone: Qwen/Qwen3-Reranker-0.6B (causal)\n"
            "  LoRA: 56 layers (all blocks, r=16, alpha=32, targets=q, v)\n"
            "  epoch 1/5: train_loss=1.1732 eval_loss=1.2155 eval_acc=0.4848 (5053.9s)\n"
        )
        result = _run(str(baseline), str(partial))
        assert result.returncode == 0, result.stderr
        assert "+5.06 (ep1)" in result.stdout

    def test_no_delta_across_different_epoch_budgets(self, tmp_path):
        """A 20-epoch run has a different LR schedule, so no delta is shown."""
        short = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        long_log = tmp_path / "r2_06b_r64.log"
        long_log.write_text(
            "Backbone: Qwen/Qwen3-Reranker-0.6B (causal)\n"
            "  epoch 1/20: train_loss=1.3510 eval_loss=1.2000 eval_acc=0.4342 (787.3s)\n"
            "  epoch 2/20: train_loss=1.3527 eval_loss=1.3535 eval_acc=0.4438 (787.3s)\n"
        )
        result = _run(str(short), str(long_log))
        assert result.returncode == 0, result.stderr
        assert "20 vs 5 ep" in result.stdout
        assert "+0.96 pp" not in result.stdout

    def test_baseline_row_has_no_delta(self, tmp_path):
        baseline = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        result = _run(str(baseline))
        row = [line for line in result.stdout.splitlines() if "arch_06b_frozen_last" in line]
        assert row and row[0].rstrip().endswith("— |")

    def test_qualified_variant_is_not_the_baseline(self, tmp_path):
        """`frozen_32k` adds a second knob, so `frozen_last` must anchor the deltas."""
        plain = _write_log(tmp_path, "arch_4b_frozen_last", "0.4908", "0.4750")
        variant = _write_log(tmp_path, "arch_4b_frozen_32k", "0.4917", "0.4749")
        result = _run(str(variant), str(plain))
        assert result.returncode == 0, result.stderr

        lines = result.stdout.splitlines()
        plain_row = next(l for l in lines if "arch_4b_frozen_last" in l)
        variant_row = next(l for l in lines if "arch_4b_frozen_32k" in l)
        assert plain_row.rstrip().endswith("— |")
        # At ep2, the only shared epoch: 47.49 - 47.50 = -0.01.
        assert "-0.01 (ep2)" in variant_row


class TestGrouping:
    def test_runs_grouped_by_backbone(self, tmp_path):
        small = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        big = tmp_path / "arch_4b_frozen_last.log"
        big.write_text(
            "Backbone: Qwen/Qwen3-Reranker-4B (causal)\n"
            "  epoch 1/5: train_loss=1.2512 eval_loss=1.2000 eval_acc=0.4908 (787.5s)\n"
        )
        result = _run(str(small), str(big))
        assert result.returncode == 0, result.stderr
        assert "### Qwen/Qwen3-Reranker-0.6B (1 runs)" in result.stdout
        assert "### Qwen/Qwen3-Reranker-4B (1 runs)" in result.stdout

    def test_frozen_runs_sort_before_lora(self, tmp_path):
        lora = _write_log(
            tmp_path,
            "arch_06b_loraAll_last",
            "0.4342",
            "0.4174",
            extra="  LoRA: 56 layers (all blocks, r=16, alpha=32, targets=q, v)\n",
        )
        frozen = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        result = _run(str(lora), str(frozen))
        lines = result.stdout.splitlines()
        frozen_at = next(i for i, l in enumerate(lines) if "arch_06b_frozen_last" in l)
        lora_at = next(i for i, l in enumerate(lines) if "arch_06b_loraAll_last" in l)
        assert frozen_at < lora_at


class TestOutputFormats:
    def test_by_type_table(self, tmp_path):
        log = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        result = _run("--by-type", str(log))
        assert result.returncode == 0, result.stderr
        assert "Per-type accuracy at each run's best epoch:" in result.stdout
        assert "| run | best ep | choice | noul | score |" in result.stdout

    def test_json_output(self, tmp_path):
        log = _write_log(tmp_path, "arch_06b_frozen_last", "0.4342", "0.4174")
        result = _run("--json", str(log))
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert len(payload) == 1
        run = payload[0]
        assert run["backbone"] == "Qwen/Qwen3-Reranker-0.6B"
        assert run["epochs"]["1"]["accuracy"] == 0.4342
        assert run["epochs"]["1"]["by_type"]["noul"] == 0.615
