"""Measure panel 2 single-pass at a fixed K (#356).

Every point on the step-4 cost-accuracy frontier has to be measured on the same
items. This runs the unblocked readout at one option count with the shipped
alphabet, so the calibration cells, the blocking cells and the judgement point
share one coordinate system.

`codes_for(K)` uses `A`-`Z`, `a`-`z`, then the single symbols, so K=52 is letters
only and K=80 carries the 28 shipped symbols, which are known to be dead for
gold options.

Two item sets, and they are not interchangeable:

* ``banking77`` — real customer messages over the 80 Banking77 intents, which
  saturates at K=80;
* ``clinc150`` — CLINC150 in-scope queries over its 150 intents (CC BY 3.0,
  dataset ``clinc/clinc_oos`` at a pinned revision), which carries K up to 150.

The dataset itself is never written into the repository; only the id and
revision are recorded and the results are reported.

Usage::

    python model/scripts/calibrate_panel2_k.py --snapshot <snapshot> --k 52 \\
        --dataset clinc150 --output results/jevbench/clinc_k52_...json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

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
MAIN_ITEMS = 200
CLINC_REPO = "clinc/clinc_oos"
CLINC_CONFIG = "plus"
CLINC_SPLIT = "test"
CLINC_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
CLINC_INSTRUCTIONS = (
    "Which intent category does this customer message belong to?"
)


def clinc_items(count: int) -> tuple[list, dict]:
    """In-scope CLINC150 test items over one 150-class criteria dict."""
    from datasets import load_dataset

    data = load_dataset(
        CLINC_REPO, CLINC_CONFIG, split=CLINC_SPLIT, revision=CLINC_REVISION
    )
    names = data.features["intent"].names
    in_scope = [name for name in names if name != "oos"]
    if len(in_scope) != 150:
        raise SystemExit(f"expected 150 in-scope classes, got {len(in_scope)}")
    criteria = {name: name.replace("_", " ") for name in in_scope}
    rows = [
        (index, row)
        for index, row in enumerate(data)
        if names[int(row["intent"])] != "oos"
    ]
    random.Random(SEED).shuffle(rows)
    items = []
    for index, row in rows[:count]:
        items.append(
            SimpleNamespace(
                id=f"clinc150-{CLINC_SPLIT}-{index:05d}",
                state=row["text"],
                label=names[int(row["intent"])],
                question={"criteria": criteria, "instructions": CLINC_INSTRUCTIONS},
            )
        )
    meta = {
        "dataset": "clinc150",
        "repo": CLINC_REPO,
        "config": CLINC_CONFIG,
        "split": CLINC_SPLIT,
        "revision": CLINC_REVISION,
        "license": "CC BY 3.0",
        "classes_in_scope": len(in_scope),
        "instructions": CLINC_INSTRUCTIONS,
        "item_selection": "in-scope rows shuffled with Random(342), first 200",
    }
    return items, meta


def run(model, tokenizer, device, items, source: str, k: int) -> list[dict]:
    codes = codes_for(k)
    code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
    if multi:
        raise SystemExit(f"codes {multi} are not single tokens")
    rows = []
    for item in items:
        criteria = item.question["criteria"]
        texts, gold_pos = subsample(criteria, item.label, k, item_seed(item.id, SEED))
        prompt = render_panel2(
            item.state, item.question.get("instructions", ""), texts, codes
        )
        values, chosen, elapsed, _ = score_prompt(
            model, tokenizer, prompt, codes, code_ids, device, list(range(k))
        )
        rows.append(
            {
                "source": source,
                "id": item.id,
                "gold_pos": gold_pos,
                "pred_pos": chosen,
                "pred_code": codes[chosen],
                "correct": chosen == gold_pos,
                "gold_code": codes[gold_pos],
                "tie": has_exact_tie(values),
                "latency_s": elapsed,
            }
        )
    return rows


def summarise(rows: list[dict], k: int) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    return {
        "k": k,
        "items": n,
        "correct": correct,
        "accuracy": correct / n,
        "chance": 1 / k,
        "mean_latency_s": statistics.mean(r["latency_s"] for r in rows),
        "ties": sum(1 for r in rows if r["tie"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dataset", choices=("banking77", "clinc150"), default="banking77")
    parser.add_argument(
        "--dev-path", type=Path, default=Path("data/benchmarks/dev/dev_800.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = {
        "arm": f"panel 2 single pass at K={args.k}",
        "k": args.k,
        "codes": "codes_for(k): A-Z, a-z, then single symbols",
        "code_form": CODE_FORM,
        "seed": SEED,
        "item_seed": "random.Random(f'{seed}:{item_id}')",
        "order": "original criteria order of the kept options",
        "tie_rule": "first maximum by natural option index",
        "model": args.snapshot.name,
    }
    if args.dataset == "banking77":
        # Kept byte-identical to the recorded banking77 cells in #356.
        config["banking77_items"] = MAIN_ITEMS
        config["distractors"] = (
            "uniform from the same 80-intent criteria, excluding gold"
        )
    else:
        _, meta = clinc_items(1)
        config.update(meta)
        config["items"] = MAIN_ITEMS
        config["distractors"] = "uniform from the same 150-intent space, excluding gold"
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
    if args.dataset == "banking77":
        dev = dev_items(args.dev_path)
        instruments = {
            "banking77": banking77_items({q.id for q in dev}, MAIN_ITEMS),
            "dev24": dev,
        }
    else:
        items, _ = clinc_items(MAIN_ITEMS)
        instruments = {"clinc150": items}
    if args.limit:
        instruments = {k: v[: args.limit] for k, v in instruments.items()}
    result = {"_provenance": {"config_hash": config_hash, "config": config}}
    for name, items in instruments.items():
        rows = run(model, tokenizer, device, items, name, args.k)
        summary = summarise(rows, args.k)
        result[name] = {
            "summary": summary,
            "letter_gold": {
                "items": sum(1 for r in rows if r["gold_code"] in codes_for(args.k)[:52]),
                "correct": sum(
                    1
                    for r in rows
                    if r["gold_code"] in codes_for(args.k)[:52] and r["correct"]
                ),
            },
            "rows": rows,
        }
        print(
            f"{name}: {summary['correct']}/{summary['items']} = "
            f"{summary['accuracy']:.4f} (chance {summary['chance']:.4f}, "
            f"mean {summary['mean_latency_s']:.3f}s)",
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
