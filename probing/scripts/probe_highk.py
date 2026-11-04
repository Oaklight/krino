"""Black-box probe: does Jev's latency and accuracy change with option count K?

Banking77 supplies real 80-way choice items. For smaller K we keep the gold
option and sample K-1 distractors from the same 80, so every variant asks the
same underlying question with a different candidate set.

Usage:
    python probing/scripts/probe_highk.py \
        --items data/benchmarks/corpus_full.jsonl \
        --limit 30 --ks 2,5,20,80 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "probing" / "scripts"))

from jev_client import JevClient  # noqa: E402

from data.pipeline import load_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--ks", default="2,5,20,80")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    items = [
        item
        for item in load_jsonl(args.items)
        if item.question["type"] == "choice"
        and len(item.question["criteria"]) == max(ks)
    ][: args.limit]
    rng = random.Random(args.seed)
    client = JevClient()
    records = []
    try:
        for item in items:
            criteria = item.question["criteria"]
            gold = str(item.label)
            if gold not in criteria:
                continue
            others = [key for key in criteria if key != gold]
            for k in ks:
                keep = [gold] + rng.sample(others, k - 1)
                qid = f"{item.id}-K{k}"
                question = {
                    "type": "choice",
                    "instructions": item.question["instructions"],
                    "criteria": {key: criteria[key] for key in keep},
                }
                start = time.monotonic()
                try:
                    response = client.ask(item.state, {qid: question})
                    elapsed = time.monotonic() - start
                    answer = response.get("answers", {}).get(qid, {})
                    probs = answer.get("probabilities", {})
                    records.append(
                        {
                            "id": item.id,
                            "k": k,
                            "latency_s": elapsed,
                            "n_returned": len(probs),
                            "prob_sum": sum(probs.values()),
                            "gold_top": max(probs, key=probs.get) == gold if probs else None,
                            "confidence": answer.get("confidence"),
                        }
                    )
                except Exception as error:  # noqa: BLE001 - report and continue
                    elapsed = time.monotonic() - start
                    records.append(
                        {
                            "id": item.id,
                            "k": k,
                            "latency_s": elapsed,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    finally:
        client.close()

    by_k: dict[int, list[dict]] = {}
    for record in records:
        by_k.setdefault(record["k"], []).append(record)
    for k, rows in sorted(by_k.items()):
        ok = [r for r in rows if r.get("gold_top") is not None]
        hits = sum(1 for r in ok if r["gold_top"])
        lat = sorted(r["latency_s"] for r in rows)
        covered = sum(1 for r in rows if r.get("n_returned") == k)
        print(
            f"K={k:3d} n={len(rows):3d} gold_top={hits}/{len(ok)}"
            f"={hits / max(1, len(ok)):.3f} covered={covered}/{len(rows)}"
            f" lat_med={lat[len(lat) // 2]:.2f}s lat_max={lat[-1]:.2f}s"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        print(f"wrote {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
