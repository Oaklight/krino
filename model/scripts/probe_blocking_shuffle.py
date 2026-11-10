"""Blocking readout with a per-item shuffled block assignment (#356).

The declared sensitivity variant of the blocking design: the 150 class indices
are permuted per item by ``random.Random(f"{342}:{item_id}")`` and cut into
consecutive slices of the permuted order, so block membership no longer follows
the dataset's class order. Under the recorded consecutive assignment each
item's gold has a fixed local position, which confounds the block-size
comparison with which classes happen to sit in the live part of a block. A
per-item shuffle makes the gold's local position uniform across items by
construction, so the block-win rate is an average over positions.

A single global permutation would not remove that confound: it only replaces
one fixed class-to-position assignment with another. The per-item shuffle is
the control that does, and it is the variant run here. The declared seed
convention is the repository's ``item_seed``, ``random.Random(f"{seed}:{item_id}")``.

Everything else is the recorded design, imported unchanged from
``probe_blocking``: the same schemes A, B and C, the same code form and tie
rule, the same position bins. Full judgement is not run here; it is a separate
cell on all 200 items.

Usage::

    python model/scripts/probe_blocking_shuffle.py --snapshot <snapshot> \\
        --output results/jevbench/clinc_blocking_shuffle_Qwen_Qwen3-Reranker-4B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from probe_blocking import (  # noqa: E402
    BLOCK_SIZES,
    CODE_FORM,
    ITEMS,
    K,
    SEED,
    block_answer,
    final_pass,
    judgement_pick,
    position_bin,
)
from model.evaluation import clinc  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402


def shuffled_blocks(item_id: str, size: int) -> list[list[int]]:
    order = list(range(K))
    random.Random(f"{SEED}:{item_id}").shuffle(order)
    return [order[start : start + size] for start in range(0, K, size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--limit", type=int, help="smoke-test cap on items")
    args = parser.parse_args()

    config = {
        "arm": "blocking readout frontier, per-item shuffled assignment",
        "dataset": "clinc150",
        "repo": clinc.REPO,
        "config": clinc.CONFIG,
        "split": clinc.SPLIT,
        "revision": clinc.REVISION,
        "classes_in_scope": 150,
        "k": K,
        "items": ITEMS,
        "block_sizes": list(BLOCK_SIZES),
        "blocks": "per-item seeded shuffle of the 150 class indices, consecutive slices of the permuted order",
        "assignment_seed": "random.Random(f'{seed}:{item_id}')",
        "scheme_a": "block winners compared by the winner's raw code logit",
        "scheme_b": "one panel-2 prompt over the block winners",
        "scheme_c": "per-option judgement over the block winners",
        "code_form": CODE_FORM,
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
    scorer = LogitScorer(model, tokenizer, norm="mean", strategy="description", batch_size=1)
    items, _ = clinc.load(ITEMS, SEED)
    if args.limit:
        items = items[: args.limit]

    rows = []
    for item in items:
        criteria = item.question["criteria"]
        keys = list(criteria)
        texts = [str(criteria[key]) for key in keys]
        gold = keys.index(item.label)
        instructions = item.question.get("instructions", "")
        record = {"id": item.id, "gold": gold}
        for size in BLOCK_SIZES:
            blocks = shuffled_blocks(item.id, size)
            winners, logits, elapsed, in_block = [], [], 0.0, {}
            for indices in blocks:
                winner, logit, seconds = block_answer(
                    model, tokenizer, device, item.state, instructions, texts, indices
                )
                elapsed += seconds
                winners.append(winner)
                logits.append(logit)
                in_block[len(winners) - 1] = indices.index(gold) if gold in indices else None
            gold_block = next(i for i, ind in enumerate(blocks) if gold in ind)
            won_block = winners[gold_block] == gold
            scheme_a = winners[max(range(len(logits)), key=lambda i: (logits[i], -i))]
            scheme_b, seconds_b = final_pass(
                model, tokenizer, device, item.state, instructions, texts, winners
            )
            scheme_c, seconds_c = judgement_pick(
                scorer, item.state, instructions, texts, winners
            )
            record[f"b{size}"] = {
                "blocks": len(blocks),
                "order": [index for block in blocks for index in block],
                "winners": winners,
                "gold_block": gold_block,
                "gold_local": in_block[gold_block],
                "gold_local_bin": position_bin(in_block[gold_block], len(blocks[gold_block])),
                "won_block": won_block,
                "scheme_a": scheme_a,
                "scheme_a_correct": scheme_a == gold,
                "scheme_b": scheme_b,
                "scheme_b_correct": scheme_b == gold,
                "scheme_c": scheme_c,
                "scheme_c_correct": scheme_c == gold,
                "block_seconds": round(elapsed, 3),
                "final_seconds": round(seconds_b, 3),
                "judgement_seconds": round(seconds_c, 3),
            }
        rows.append(record)
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(items)} items", flush=True)

    summary: dict = {"items": len(rows), "by_block_size": {}}
    for size in BLOCK_SIZES:
        cell = [r[f"b{size}"] for r in rows]
        n = len(cell)
        early = [c for c in cell if c["gold_local_bin"] == "first"]
        late = [c for c in cell if c["gold_local_bin"] == "last"]
        summary["by_block_size"][str(size)] = {
            "blocks": cell[0]["blocks"],
            "won_block": sum(c["won_block"] for c in cell),
            "won_block_rate": sum(c["won_block"] for c in cell) / n,
            "won_block_first_20pct": (
                sum(c["won_block"] for c in early) / len(early) if early else None
            ),
            "won_block_last_20pct": (
                sum(c["won_block"] for c in late) / len(late) if late else None
            ),
            "scheme_a_correct": sum(c["scheme_a_correct"] for c in cell),
            "scheme_b_correct": sum(c["scheme_b_correct"] for c in cell),
            "scheme_c_correct": sum(c["scheme_c_correct"] for c in cell),
            "forwards_a": cell[0]["blocks"],
            "forwards_b": cell[0]["blocks"] + 1,
            "forwards_c": 2 * cell[0]["blocks"],
        }
    for name, cell in summary["by_block_size"].items():
        print(
            f"b={name}: won block {cell['won_block_rate']:.3f} "
            f"(first {cell['won_block_first_20pct']}, last {cell['won_block_last_20pct']}) "
            f"A {cell['scheme_a_correct']} B {cell['scheme_b_correct']} "
            f"C {cell['scheme_c_correct']} of {len(rows)}",
            flush=True,
        )

    result = {"_provenance": {"config_hash": config_hash, "config": config}, **summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
