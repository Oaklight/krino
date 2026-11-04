"""Pairwise flips between raw scoring paths (shared-prefix parity check).

Reads ``{po_bf16,po_f32,shared_f32}_<model>.pred.jsonl`` from a directory
(default: the committed predictions under ``results/gate/qwen35_shared``).
``po_bf16 -> po_f32`` changes only the readout precision, so its flip count is
the noise floor; ``po_f32 -> shared_f32`` changes only the cache path.
"""

import argparse
import json
from math import comb
from pathlib import Path

MODELS = ("qwen3_reranker_4b", "qwen35_4b_instruct", "qwen35_4b_base")
PAIRS = (("po_bf16", "po_f32"), ("po_f32", "shared_f32"), ("po_bf16", "shared_f32"))


def load(path: Path) -> dict:
    rows = (json.loads(line) for line in path.open())
    return {row["id"]: row for row in rows}


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p.  With b + c <= 1 it is 1.0 by construction."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=Path("results/gate/qwen35_shared/predictions")
    )
    args = parser.parse_args()
    for name in MODELS:
        runs = {
            kind: load(args.dir / f"{kind}_{name}.pred.jsonl")
            for kind in ("po_bf16", "po_f32", "shared_f32")
        }
        acc = {k: sum(r["correct"] for r in v.values()) / len(v) for k, v in runs.items()}
        ties = {
            k: sum(abs(r["max_prob"] - 0.5) < 1e-9 for r in v.values())
            for k, v in runs.items()
        }
        print(name, {k: round(a, 4) for k, a in acc.items()}, "exact-0.5", ties)
        for a, b in PAIRS:
            left, right = runs[a], runs[b]
            flips = sum(left[i]["pred"] != right[i]["pred"] for i in left)
            gain = sum(right[i]["correct"] and not left[i]["correct"] for i in left)
            loss = sum(left[i]["correct"] and not right[i]["correct"] for i in left)
            dp = sum(abs(left[i]["max_prob"] - right[i]["max_prob"]) for i in left)
            print(
                f"  {a:>8} -> {b:<10} flips {flips:2d} (+{gain}/-{loss}) "
                f"mcnemar p={mcnemar(gain, loss):.3f} mean|dp| {dp / len(left):.4f}"
            )


if __name__ == "__main__":
    main()
