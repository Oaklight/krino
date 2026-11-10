"""Measure panel 2 at K = 52 with the letter alphabet only (#356).

The blocking readout's ceiling is the probability that the gold option wins its
own block, and a block is a 52-option item. The 39% figure quoted earlier came
from K = 58 with six tier codes occupying six slots; that is not the same
construction. This measures the 52-option cell directly: the gold plus 51
distractors from the same intent space, labelled with `A`-`Z` and `a`-`z` only.

No blocking is implemented here. One prompt per item.

Usage::

    python model/scripts/calibrate_k52.py --snapshot <snapshot> \\
        --output results/jevbench/k52_calibration_Qwen_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval_jevbench_panel2 import score_prompt  # noqa: E402
from model.evaluation.code_tiers import item_seed, subsample  # noqa: E402
from model.evaluation.panel2 import (  # noqa: E402
    code_token_ids,
    codes_for,
    has_exact_tie,
    render_panel2,
)
from model.src.backbone import load_causal_lm  # noqa: E402
from probe_code_tiers import banking77_items, dev_items  # noqa: E402

CODE_FORM = "space"
SEED = 342
K = 52
BANKING77_ITEMS = 200


def run(model, tokenizer, device, items, source: str) -> list[dict]:
    codes = codes_for(K)
    code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
    if multi:
        raise SystemExit(f"codes {multi} are not single tokens")
    rows = []
    for item in items:
        criteria = item.question["criteria"]
        texts, gold_pos = subsample(criteria, item.label, K, item_seed(item.id, SEED))
        prompt = render_panel2(
            item.state, item.question.get("instructions", ""), texts, codes
        )
        values, chosen, elapsed, _ = score_prompt(
            model, tokenizer, prompt, codes, code_ids, device, list(range(K))
        )
        rows.append(
            {
                "source": source,
                "id": item.id,
                "gold_pos": gold_pos,
                "pred_pos": chosen,
                "pred_code": codes[chosen],
                "correct": chosen == gold_pos,
                "tie": has_exact_tie(values),
                "latency_s": elapsed,
                "prompt_chars": len(prompt),
            }
        )
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    return {
        "items": n,
        "correct": correct,
        "accuracy": correct / n,
        "chance": 1 / K,
        "mean_latency_s": statistics.mean(r["latency_s"] for r in rows),
        "ties": sum(1 for r in rows if r["tie"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--dev-path", type=Path, default=Path("data/benchmarks/dev/dev_800.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/jevbench/k52_calibration_Qwen_Qwen3-Reranker-4B.json"),
    )
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = {
        "arm": "K=52 letter-only calibration",
        "k": K,
        "alphabet": "A-Z, a-z",
        "code_form": CODE_FORM,
        "seed": SEED,
        "item_seed": "random.Random(f'{seed}:{item_id}')",
        "banking77_items": BANKING77_ITEMS,
        "distractors": "uniform from the same 80-intent criteria, excluding gold",
        "order": "original criteria order of the kept options",
        "tie_rule": "first maximum by natural option index",
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
    dev = dev_items(args.dev_path)
    instruments = {
        "banking77": banking77_items({q.id for q in dev}, BANKING77_ITEMS),
        "dev24": dev,
    }
    if args.limit:
        instruments = {k: v[: args.limit] for k, v in instruments.items()}
    result = {"_provenance": {"config_hash": config_hash, "config": config}}
    for name, items in instruments.items():
        rows = run(model, tokenizer, device, items, name)
        result[name] = {"summary": summarise(rows), "rows": rows}
        s = result[name]["summary"]
        print(
            f"{name}: {s['correct']}/{s['items']} = {s['accuracy']:.4f} "
            f"(chance {s['chance']:.4f}, mean {s['mean_latency_s']:.3f}s)",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
