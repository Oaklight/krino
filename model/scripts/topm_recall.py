"""Top-m recall of the gold option under the raw shared-prefix scorer.

Diagnostic for the v4 head choice (manifest arm ``topm_recall_diag``).  For
every choice/score item with K >= 3 options on JevBench public, rank the
options by the raw yes/no score and record where the gold lands.  High
recall@2 means a base shortlist plus a pairwise rerank can reach the gold;
low recall@3 means the base ranking itself is the bottleneck and a listwise
set head goes first.

Inference only.  Writes one JSON with the summary and per-item ranks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.topm_recall import gold_rank, k_bucket, summarize  # noqa: E402
from model.scripts.eval_gate import (  # noqa: E402
    RawReranker,
    items_digest,
    load_jevbench,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    items = list(load_jevbench())
    digest = items_digest(items)
    scorer = RawReranker(args.model, args.device)
    scorer.shared_prefix = True
    scorer.warmup_prefix_buckets()

    rows = []
    skipped = defaultdict(int)
    start = time.perf_counter()
    for item in items:
        pred = scorer.predict_row(item)
        k = len(pred.probs)
        if pred.q_type not in ("choice", "score") or k < 3:
            skipped[f"{pred.q_type}_k{k}" if k < 3 else pred.q_type] += 1
            continue
        gold = pred.label_index
        if gold is None:
            skipped["unmapped_gold"] += 1
            continue
        rows.append(
            {
                "id": pred.item_id,
                "q_type": pred.q_type,
                "tier": pred.tier,
                "k": k,
                "gold": gold,
                "rank": gold_rank(pred.probs, gold),
                "probs": [round(p, 6) for p in pred.probs],
            }
        )
    elapsed = time.perf_counter() - start

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"tier:{row['tier']}"].append(row)
        groups[f"type:{row['q_type']}"].append(row)
        groups[f"k:{k_bucket(row['k'])}"].append(row)
        groups[f"k={row['k']}"].append(row)
    result = {
        "model": args.model,
        "data": "jevbench",
        "data_sha256_prefix": digest[:8],
        "n_items": len(items),
        "path": "raw shared_prefix, float32_two_token",
        "filter": "choice/score with K>=3",
        "skipped": dict(skipped),
        "elapsed_s": round(elapsed, 1),
        "overall": summarize(rows),
        "groups": {name: summarize(g) for name, g in sorted(groups.items())},
        "items": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": args.model, "digest": digest[:8], **result["overall"]}))


if __name__ == "__main__":
    main()
