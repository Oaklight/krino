#!/usr/bin/env python3
"""Generate ordinal (score) questions over existing states and label them with Jev.

Pilot for issue #156.  The script reads a canonical corpus, selects distinct
states that do not already carry a score question, assigns one ordinal rubric
per source, asks Jev for a distribution over the levels, and writes validated
rows.  Every row carries ``provenance = teacher`` and the distribution under
``teacher_probs["jev"]``; ``label`` is the argmax level and is for evaluation
only, because teacher records are not eligible for a hard-label loss.

Usage:
    python scripts/generate_score_questions.py \
        --corpus /path/krino-canonical-243366.jsonl \
        --target 2000 --max-candidates 3000 --concurrency 8 \
        --out /path/scoregen-pilot.jsonl --report /path/scoregen-pilot.report.json

    # selection only, no API calls:
    ... --dry-run --report /path/selection.json

Requires ``TYPESAFE_API_KEY`` in the environment (the Jev client reads it).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "probing" / "scripts"))

from data.score_generation import (  # noqa: E402
    Candidate,
    build_item,
    expected_level,
    iter_candidates,
    label_level,
    level0_allowed,
    load_rubric_config,
    normalise_state,
    order_candidates,
    select_candidates,
    validate_distribution,
)
from data.synthetic_templates import DOMAIN_TEMPLATES  # noqa: E402


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jevbench_module():
    path = REPO / "scripts" / "check_jevbench_leakage.py"
    spec = importlib.util.spec_from_file_location("jevbench_leakage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contamination_check(rows: list[dict], jevbench_root: Path) -> dict[str, Any]:
    """Apply the #235 rule to every generated row.

    Rule 1: id equals a public JevBench id.  Rule 2: normalised state equals a
    JevBench state.  Rule 3: the state shares at least three 8-word shingles
    with a JevBench state.  Returns the hit ids by reason.
    """
    from data.pipeline import render_jevbench_state

    module = load_jevbench_module()
    probes = module.load_probes(
        [jevbench_root / f"{tier}.jsonl" for tier in ("easy", "hard", "original")]
    )
    jev_ids = {str(probe["row"].get("id")) for probe in probes}
    jev_states: set[str] = set()
    jev_shingles: set[int] = set()
    for probe in probes:
        state = render_jevbench_state(probe["row"].get("state"))
        jev_states.add(module.normalise(state))
        jev_shingles |= module.shingle_hashes(state)

    hits: dict[str, list[str]] = {"id": [], "state": [], "shingles": []}
    for row in rows:
        item_id = str(row.get("id"))
        state = row.get("state", "")
        if item_id in jev_ids:
            hits["id"].append(item_id)
            continue
        if module.normalise(state) in jev_states:
            hits["state"].append(item_id)
            continue
        shared = len(module.shingle_hashes(state) & jev_shingles)
        if shared >= 3:
            hits["shingles"].append(item_id)
    hit_ids = sorted({item for values in hits.values() for item in values})
    return {
        "rule": "id match, normalised state match, or >=3 shared 8-word shingles",
        "jevbench_items": len(probes),
        "hits_by_reason": {reason: len(values) for reason, values in hits.items()},
        "hit_ids": hit_ids,
    }


def label_one(
    client: Any, candidate: Candidate, model: str, retries: int = 3
) -> dict[str, Any]:
    question = {
        "type": "score",
        "instructions": candidate.rubric["instructions"],
        "criteria": list(candidate.rubric["criteria"]),
    }
    last_error = ""
    for attempt in range(retries):
        try:
            response = client.ask(
                candidate.state, {candidate.id: question}, model=model
            )
            answers = response.get("answers") or {}
            answer = answers.get(candidate.id)
            if answer is None and len(answers) == 1:
                answer = next(iter(answers.values()))
            if answer is None:
                return {"candidate": candidate, "reason": "no_answer", "error": ""}
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict):
                return {"candidate": candidate, "reason": "no_distribution", "error": ""}
            probabilities = {str(key): value for key, value in probabilities.items()}
            reason = validate_distribution(probabilities, len(candidate.rubric["criteria"]))
            if reason is not None:
                return {
                    "candidate": candidate,
                    "reason": reason,
                    "probabilities": probabilities,
                    "api_score": answer.get("score"),
                    "error": "",
                }
            return {
                "candidate": candidate,
                "reason": None,
                "probabilities": probabilities,
                "expected": expected_level(probabilities),
                "label": label_level(probabilities),
                "api_score": answer.get("score"),
                "error": "",
            }
        except Exception as exc:  # network, auth, malformed response
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < retries:
                time.sleep(2**attempt + random.random())
    return {"candidate": candidate, "reason": "api_error", "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, default=REPO / "data" / "score_rubrics.json")
    parser.add_argument("--jevbench-root", type=Path, default=REPO / "data" / "benchmarks" / "jevbench")
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--max-candidates", type=int, default=3000)
    parser.add_argument("--floor", type=int, default=25)
    parser.add_argument("--cap", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--level0-cap",
        type=float,
        default=0.5,
        help="maximum share of a source's rows at level 0; 0 disables the cap",
    )
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run")

    started = time.time()
    items = list(read_jsonl(args.corpus))
    score_states = {
        normalise_state(item.get("state"))
        for item in items
        if (item.get("question") or {}).get("type") == "score"
    }
    rubrics, default_rubric = load_rubric_config(str(args.rubrics), DOMAIN_TEMPLATES)
    candidates = list(
        iter_candidates(items, score_states, rubrics, default_rubric)
    )
    selected = select_candidates(
        candidates,
        target=args.max_candidates,
        floor=args.floor,
        cap=args.cap,
        seed=args.seed,
    )
    source_order = sorted({candidate.source for candidate in selected})
    ordered = order_candidates(selected, source_order)

    if args.dry_run:
        report = {
            "dry_run": True,
            "corpus": str(args.corpus),
            "corpus_sha256": file_sha256(args.corpus),
            "records": len(items),
            "score_states": len(score_states),
            "candidate_states": len(candidates),
            "selected": len(ordered),
            "target": args.target,
            "max_candidates": args.max_candidates,
            "by_source": dict(
                sorted(Counter(candidate.source for candidate in ordered).items())
            ),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    from jev_client import JevClient

    client = JevClient(timeout=args.timeout)

    accepted: list[dict[str, Any]] = []
    accepted_meta: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    calls = 0
    source_stats: dict[str, Counter] = defaultdict(Counter)
    source_accepted: Counter = Counter()
    source_level0: Counter = Counter()
    by_source_levels: dict[str, Counter] = defaultdict(Counter)
    level_counts: Counter = Counter()
    argmax_disagreements = 0
    abs_diffs: list[float] = []
    estimated_tokens = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for start in range(0, len(ordered), args.concurrency):
            if len(accepted) >= args.target:
                break
            chunk = ordered[start : start + args.concurrency]
            results = list(pool.map(lambda candidate: label_one(client, candidate, args.model), chunk))
            calls += len(chunk)
            for result in results:
                candidate = result["candidate"]
                source_stats[candidate.source]["calls"] += 1
                question_text = json.dumps(candidate.rubric)
                estimated_tokens += math.ceil((len(candidate.state) + len(question_text)) / 4)
                if result["reason"] is not None:
                    source_stats[candidate.source][f"reject:{result['reason']}"] += 1
                    rejected.append(
                        {
                            "id": candidate.id,
                            "source": candidate.source,
                            "reason": result["reason"],
                            "error": result.get("error", ""),
                            "probabilities": result.get("probabilities"),
                        }
                    )
                    continue
                if len(accepted) >= args.target:
                    continue
                label = label_level(result["probabilities"])
                if label == 0.0 and not level0_allowed(
                    args.level0_cap,
                    source_accepted[candidate.source],
                    source_level0[candidate.source],
                ):
                    source_stats[candidate.source]["reject:level0_cap"] += 1
                    rejected.append(
                        {
                            "id": candidate.id,
                            "source": candidate.source,
                            "reason": "level0_cap",
                            "error": "",
                            "probabilities": result["probabilities"],
                        }
                    )
                    continue
                item = build_item(candidate, result["probabilities"])
                accepted.append(item)
                source_stats[candidate.source]["accepted"] += 1
                source_accepted[candidate.source] += 1
                if item["label"] == 0.0:
                    source_level0[candidate.source] += 1
                by_source_levels[candidate.source][str(int(item["label"]))] += 1
                level_counts[str(int(item["label"]))] += 1
                expected = result["expected"]
                abs_diffs.append(abs(expected - item["label"]))
                if int(round(expected)) != int(item["label"]):
                    argmax_disagreements += 1
                accepted_meta.append(
                    {
                        "id": item["id"],
                        "source": candidate.source,
                        "label": item["label"],
                        "expected": expected,
                        "api_score": result.get("api_score"),
                        "n_levels": len(item["question"]["criteria"]),
                    }
                )

    contamination = contamination_check(accepted, args.jevbench_root)
    drop_ids = set(contamination["hit_ids"])
    if drop_ids:
        accepted = [row for row in accepted if row["id"] not in drop_ids]
        accepted_meta = [row for row in accepted_meta if row["id"] not in drop_ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    api_score_diffs = [
        abs(float(meta["api_score"]) - meta["expected"])
        for meta in accepted_meta
        if meta.get("api_score") is not None
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": str(args.corpus),
        "corpus_sha256": file_sha256(args.corpus),
        "rubrics": str(args.rubrics),
        "seed": args.seed,
        "model": args.model,
        "target": args.target,
        "max_candidates": args.max_candidates,
        "candidate_states": len(candidates),
        "selected_candidates": len(ordered),
        "calls": calls,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(
            sorted(Counter(row["reason"] for row in rejected).items())
        ),
        "yield_candidates_per_accepted": round(calls / len(accepted), 3) if accepted else None,
        "by_source": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(source_stats.items())
        },
        "label_level_distribution": dict(sorted(level_counts.items())),
        "level0_cap_fraction": args.level0_cap,
        "by_source_levels": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(by_source_levels.items())
        },
        "argmax_vs_expected": {
            "rows": len(accepted_meta),
            "disagree": argmax_disagreements,
            "disagree_rate": round(argmax_disagreements / len(accepted_meta), 4)
            if accepted_meta
            else None,
            "mean_abs_difference": round(sum(abs_diffs) / len(abs_diffs), 4)
            if abs_diffs
            else None,
        },
        "api_score_vs_expected": {
            "rows": len(api_score_diffs),
            "mean_abs_difference": round(sum(api_score_diffs) / len(api_score_diffs), 4)
            if api_score_diffs
            else None,
        },
        "contamination": {
            **contamination,
            "dropped": len(drop_ids),
            "accepted_after_check": len(accepted),
        },
        "loss_usage": {
            "training_target": "teacher_probs.jev, via the soft-target (KL/distillation) loss",
            "hard_label": "not eligible: provenance=teacher; label is the argmax and is for evaluation only",
        },
        "licence_inheritance": (
            "Each generated row inherits the licence of the source state. The generated "
            "question text adds no new third-party terms; verdicts are in "
            "research/30-canonical-source-licences.md."
        ),
        "estimated_input_tokens": estimated_tokens,
        "estimated_cost_usd_at_0.042_per_million": round(estimated_tokens * 0.042 / 1_000_000, 5),
        "elapsed_seconds": round(time.time() - started, 1),
        "out": str(args.out),
        "out_sha256": file_sha256(args.out),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
