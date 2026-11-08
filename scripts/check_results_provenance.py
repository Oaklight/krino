#!/usr/bin/env python3
"""Fail when a dashboard row cannot be traced to a committed result file.

Reads ``web/public/results/eval_results.json``, produced by
``scripts/transform_results.py``, and checks every row:

* it names a ``source_file``;
* that file exists and is tracked by git;
* recomputing the row from the file yields the same accuracy, scored count,
  model id and readout that the dashboard publishes.

Exit status is non-zero when any row fails, so the deploy job stops before it
publishes a number no committed file yields. It recomputes each row with the
same ``build_jevbench_run`` and ``transform_eval_file`` that produced the row,
so it catches untraceable files and stale or drifted output, but not a bug
inside the transform itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transform_results as transform  # noqa: E402

REPO_ROOT = transform.REPO_ROOT
OUTPUT_PATH = transform.OUTPUT_PATH

JEVBENCH_FIELDS = (
    "model",
    "model_id",
    "head_version",
    "run_id",
    "method",
    "readout",
    "readout_detail",
    "accuracy",
    "correct",
    "n",
    "total",
    "by_type",
    "by_tier",
    "latency_ms",
)
EVAL_FIELDS = ("model", "method", "phase", "n_benchmarks", "results")


def _compare_fields(
    published: dict,
    expected: dict,
    fields: tuple[str, ...],
    run_id: str,
    source: str,
) -> list[str]:
    errors = []
    for field in fields:
        if published.get(field) != expected.get(field):
            errors.append(
                f"{run_id}: {field} is {published.get(field)!r} on the dashboard "
                f"but {expected.get(field)!r} in {source}"
            )
    return errors


def committed_files() -> set[str]:
    """Paths tracked by git, relative to the repository root."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {path for path in result.stdout.split("\0") if path}


def _source_error(run_id: str, source_file: str | None, tracked: set[str]) -> str | None:
    if not source_file:
        return f"{run_id}: no source_file recorded"
    if Path(source_file).is_absolute() or ".." in Path(source_file).parts:
        return f"{run_id}: source_file is not repository-relative: {source_file}"
    if source_file not in tracked:
        return f"{run_id}: source_file is not a committed file: {source_file}"
    if not (REPO_ROOT / source_file).is_file():
        return f"{run_id}: source_file is missing from the checkout: {source_file}"
    return None


def check_jevbench_run(run: dict, tracked: set[str]) -> list[str]:
    run_id = run.get("run_id", "<no run_id>")
    error = _source_error(run_id, run.get("source_file"), tracked)
    if error:
        return [error]

    expected = transform.build_jevbench_run(REPO_ROOT / run["source_file"])
    if expected is None:
        return [f"{run_id}: {run['source_file']} yields no row"]

    errors = _compare_fields(run, expected, JEVBENCH_FIELDS, run_id, run["source_file"])
    if run.get("readout") == "trained-heads":
        raw = json.loads((REPO_ROOT / run["source_file"]).read_text())
        for field in ("head_version", "run_id"):
            if not raw.get(field):
                errors.append(
                    f"{run_id}: trained-heads file has no {field}: {run['source_file']}"
                )
    return errors


def check_eval_run(run: dict, tracked: set[str]) -> list[str]:
    run_id = run.get("run_id", "<no run_id>")
    error = _source_error(run_id, run.get("source_file"), tracked)
    if error:
        return [error]

    expected = transform.transform_eval_file(REPO_ROOT / run["source_file"])
    if expected is None:
        return [f"{run_id}: {run['source_file']} yields no row"]

    return _compare_fields(run, expected, EVAL_FIELDS, run_id, run["source_file"])


def check_landing(data: dict, tracked: set[str]) -> list[str]:
    expected = transform.build_landing(
        data.get("jevbench_runs", []),
        data.get("eval_runs", []),
        data.get("training_runs", []),
    )
    errors: list[str] = []
    if data.get("landing") != expected:
        errors.append("landing: block does not match the committed files")
    sources: set[str] = set()
    for section in ("judgement_vs_continuation", "post_trained_4b", "resolution"):
        sources.update(expected[section].get("source_files", []))
    for field in ("trained_source_file", "jev_source_file"):
        value = expected["banking77"].get(field)
        if value:
            sources.add(value)
    for source in sorted(sources):
        if source not in tracked:
            errors.append(f"landing: source is not a committed file: {source}")
    return errors


def check_results(data: dict, tracked: set[str]) -> list[str]:
    errors: list[str] = []
    if "landing" in data:
        errors.extend(check_landing(data, tracked))
    seen: set[str] = set()
    for run in data.get("jevbench_runs", []):
        errors.extend(check_jevbench_run(run, tracked))
        run_id = run.get("run_id")
        if run_id in seen:
            errors.append(f"{run_id}: duplicate run_id")
        seen.add(run_id)
    for run in data.get("eval_runs", []):
        errors.extend(check_eval_run(run, tracked))
    return errors


def main() -> int:
    if not OUTPUT_PATH.is_file():
        print(f"FAIL: {OUTPUT_PATH} does not exist; run scripts/transform_results.py first")
        return 1

    data = json.loads(OUTPUT_PATH.read_text())
    tracked = committed_files()
    errors = check_results(data, tracked)

    if errors:
        print(f"FAIL: {len(errors)} dashboard row(s) do not trace to a committed file")
        for error in errors:
            print(f"  {error}")
        return 1

    print(
        f"OK: {len(data.get('jevbench_runs', []))} JevBench rows, "
        f"{len(data.get('eval_runs', []))} eval rows and the landing block "
        f"trace to committed files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
