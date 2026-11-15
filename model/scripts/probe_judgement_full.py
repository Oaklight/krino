"""Full per-option judgement over all 150 CLINC150 classes (#356).

The reference point of the blocking frontier, measured on all 200 items instead
of the 50-item subsample. Same instrument as ``probe_blocking.judgement_full``:
the judgement template of ``model/evaluation/judgement_prompt.py``, the logit
scorer with ``norm="mean"``, ``strategy="description"`` and ``batch_size=1``,
the score of ``yes`` for every option, the highest score wins, ties to the
smaller natural option index.

The 50-item cell in the blocking run scored 34/50. This run repeats the same 50
items first, so the first-50 count is a consistency check on the instrument,
then continues over the remaining 150.

Usage::

    python model/scripts/probe_judgement_full.py --snapshot <snapshot> \\
        --output results/jevbench/clinc_judgement_full_Qwen_Qwen3-Reranker-4B.json

Rows are checkpointed to ``<output>.partial.jsonl`` after every item, so an
interrupted run loses at most one item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation import clinc  # noqa: E402
from model.evaluation.judgement_prompt import NO, YES, render  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402

SEED = 342
ITEMS = 200
K = 150


def pick(scorer, state, instructions, texts) -> tuple[int, float]:
    """The highest log P(' yes'), ties to the smaller natural index."""
    best, best_score = None, None
    started = time.perf_counter()
    for index, text in enumerate(texts):
        scores = scorer.score_options(render(state, instructions, text), [YES, NO])
        score = float(scores[0])
        if best_score is None or score > best_score:
            best, best_score = index, score
    return best, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--limit", type=int, help="smoke-test cap on items")
    args = parser.parse_args()

    config = {
        "arm": "full judgement at K=150",
        "dataset": "clinc150",
        "repo": clinc.REPO,
        "config": clinc.CONFIG,
        "split": clinc.SPLIT,
        "revision": clinc.REVISION,
        "classes_in_scope": 150,
        "k": K,
        "items": ITEMS,
        "options": "all in-scope classes in the dataset's class order",
        "labels": [YES, NO],
        "norm": "mean",
        "strategy": "description",
        "batch_size": 1,
        "score": "log P(' yes') per option",
        "tie_rule": "smaller natural option index",
        "model": args.snapshot.name,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    print(f"config_hash {config_hash}", flush=True)
    print(json.dumps(config, sort_keys=True, ensure_ascii=False), flush=True)
    if args.print_config_hash:
        return 0

    model, tokenizer = load_causal_lm(str(args.snapshot), dtype=torch.bfloat16)
    device = str(next(model.parameters()).device)
    scorer = LogitScorer(model, tokenizer, norm="mean", strategy="description", batch_size=1)
    items, _ = clinc.load(ITEMS, SEED)
    if args.limit:
        items = items[: args.limit]

    partial = args.output.with_suffix(".partial.jsonl")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("", encoding="utf-8")

    rows = []
    for item in items:
        criteria = item.question["criteria"]
        keys = list(criteria)
        texts = [str(criteria[key]) for key in keys]
        gold = keys.index(item.label)
        instructions = item.question.get("instructions", "")
        chosen, seconds = pick(scorer, item.state, instructions, texts)
        row = {
            "id": item.id,
            "gold": gold,
            "pick": chosen,
            "correct": chosen == gold,
            "seconds": round(seconds, 3),
        }
        rows.append(row)
        with partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(items)} items", flush=True)

    first = rows[:50]
    summary = {
        "items": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "chance": 1 / K,
        "mean_latency_s": sum(row["seconds"] for row in rows) / len(rows),
        "first_50_correct": sum(row["correct"] for row in first),
        "first_50_items": len(first),
    }
    result = {"_provenance": {"config_hash": config_hash, "config": config}, **summary, "rows": rows}
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    partial.unlink()
    print(
        f"judgement full: {summary['correct']}/{summary['items']} = {summary['accuracy']:.3f}; "
        f"first 50 {summary['first_50_correct']}/{summary['first_50_items']}; "
        f"mean {summary['mean_latency_s']:.2f}s per item",
        flush=True,
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
