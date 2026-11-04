"""The calibration comparison refuses to compare different slices."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "model" / "scripts" / "compare_calibrations.py"


def artifact(counts: dict[str, int], scales: dict[str, int], ece: dict[str, float]) -> dict:
    return {
        "checkpoint": "synthetic",
        "config": "synthetic.yaml",
        "seed": 42,
        "by_type": {
            q_type: {
                "n": n,
                "temperature": {"holdout": {"ece_after": ece.get(q_type, 0.01)}},
                **(
                    {
                        "score_metrics": {
                            "qwk_by_levels": {
                                level: {"n": size, "qwk": 0.5}
                                for level, size in scales.items()
                            }
                        }
                    }
                    if q_type == "score"
                    else {}
                ),
            }
            for q_type, n in counts.items()
        },
    }


def run(tmp_path: Path, *artifacts: dict) -> tuple[int, str]:
    paths = []
    for index, payload in enumerate(artifacts):
        path = tmp_path / f"cal_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(str(path))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


class TestCompareCalibrations:
    def test_identical_slices_compare(self, tmp_path):
        counts = {"noul": 100, "choice": 200, "score": 150}
        scales = {"4": 50, "5": 80, "6": 20}
        code, out = run(
            tmp_path,
            artifact(counts, scales, {"noul": 0.02}),
            artifact(counts, scales, {"noul": 0.03}),
        )
        assert code == 0, out
        assert "+0.0100" in out

    def test_repaired_against_unrepaired_is_refused(self, tmp_path):
        """The 6-level scale's n is the signature of the STS-B repair."""
        base = artifact(
            {"noul": 100, "choice": 200, "score": 150},
            {"4": 50, "5": 80, "6": 20},
            {"noul": 0.02},
        )
        repaired = artifact(
            {"noul": 100, "choice": 200, "score": 150},
            {"4": 50, "5": 80, "6": 20},
            {"noul": 0.02},
        )
        unrepaired = json.loads(json.dumps(repaired))
        unrepaired["by_type"]["score"]["score_metrics"]["qwk_by_levels"]["6"]["n"] = 239
        code, out = run(tmp_path, base, unrepaired)
        assert code == 1, out
        assert "NOT COMPARABLE" in out
        assert "score level 6" in out

    def test_type_count_change_is_refused(self, tmp_path):
        base = artifact({"noul": 100, "score": 150}, {"4": 50}, {})
        other = artifact({"noul": 101, "score": 150}, {"4": 50}, {})
        code, out = run(tmp_path, base, other)
        assert code == 1, out
        assert "noul" in out

    def test_force_prints_deltas_anyway(self, tmp_path):
        base = artifact({"noul": 100, "score": 150}, {"4": 50}, {"noul": 0.02})
        other = artifact({"noul": 101, "score": 150}, {"4": 50}, {"noul": 0.05})
        paths = []
        for index, payload in enumerate((base, other)):
            path = tmp_path / f"force_{index}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(str(path))
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--force", *paths],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "+0.0300" in proc.stdout

    def test_a_single_artifact_is_an_error(self, tmp_path):
        path = tmp_path / "one.json"
        path.write_text(json.dumps(artifact({"noul": 10}, {}, {})), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
