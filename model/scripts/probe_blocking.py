"""Blocking readout frontier on CLINC150 (#356).

K = 150 over CLINC150's in-scope classes, with three block sizes and three ways
of combining blocks:

* **A** — the block winners compared by their raw code logit;
* **B** — a final panel-2 prompt over the block winners;
* **C** — a final per-option judgement over the block winners.

Plus the two reference points: the full judgement over all 150 options on a
50-item subsample, and the single-pass cells measured separately.

Because the model can ignore the tail of a long list, every cell also reports
the gold's **in-block position**: a block of 52 options is not 52 effective
candidates if the tail of the block is dead.

Usage::

    python model/scripts/probe_blocking.py --snapshot <snapshot> \\
        --output results/jevbench/clinc_blocking_Qwen_Qwen3-Reranker-4B.json
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

from eval_jevbench_panel2 import score_prompt  # noqa: E402
from model.evaluation import clinc  # noqa: E402
from model.evaluation.blocking import (  # noqa: E402
    blocks_of,
    gold_block as find_gold_block,
    position_bin,
    scheme_a_correct,
    scheme_b_correct,
)
from model.evaluation.judgement_prompt import NO, YES, render  # noqa: E402
from model.evaluation.panel2 import code_token_ids, codes_for, render_panel2  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.logit_scorer import LogitScorer  # noqa: E402

CODE_FORM = "space"
SEED = 342
ITEMS = 200
JUDGEMENT_ITEMS = 50
BLOCK_SIZES = (52, 32, 26)
K = 150


def block_answer(model, tokenizer, device, state, instructions, texts, indices):
    """One block: its argmax (global index) and that code's logit."""
    block_texts = [texts[i] for i in indices]
    codes = codes_for(len(block_texts))
    code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
    if multi:
        raise SystemExit(f"codes {multi} are not single tokens")
    prompt = render_panel2(state, instructions, block_texts, codes)
    values, chosen, elapsed, _ = score_prompt(
        model, tokenizer, prompt, codes, code_ids, device, list(range(len(block_texts)))
    )
    return indices[chosen], values[chosen], elapsed


def final_pass(model, tokenizer, device, state, instructions, texts, winners):
    """Scheme B: one panel-2 prompt over the block winners."""
    final_texts = [texts[i] for i in winners]
    codes = codes_for(len(final_texts))
    code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
    if multi:
        raise SystemExit(f"codes {multi} are not single tokens")
    prompt = render_panel2(state, instructions, final_texts, codes)
    _, chosen, elapsed, _ = score_prompt(
        model, tokenizer, prompt, codes, code_ids, device, list(range(len(final_texts)))
    )
    return winners[chosen], elapsed


def judgement_pick(scorer, state, instructions, texts, winners):
    """Scheme C: per-option judgement over the block winners."""
    best, best_score = None, None
    started = time.perf_counter()
    for index in winners:
        scores = scorer.score_options(render(state, instructions, texts[index]), [YES, NO])
        score = float(scores[0])
        if best_score is None or score > best_score:
            best, best_score = index, score
    return best, time.perf_counter() - started


def judgement_full(scorer, state, instructions, texts):
    """The reference point: judgement over every option."""
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
        "arm": "blocking readout frontier",
        "dataset": "clinc150",
        "repo": clinc.REPO,
        "config": clinc.CONFIG,
        "split": clinc.SPLIT,
        "revision": clinc.REVISION,
        "classes_in_scope": 150,
        "k": K,
        "items": ITEMS,
        "judgement_items": JUDGEMENT_ITEMS,
        "block_sizes": list(BLOCK_SIZES),
        "blocks": "consecutive slices of the class order",
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
        instructions = item.question["instructions"]
        record = {"id": item.id, "gold": gold, "gold_pos_fraction": gold / K}
        for size in BLOCK_SIZES:
            blocks = blocks_of(K, size)
            winners, logits, elapsed, in_block = [], [], 0.0, {}
            for indices in blocks:
                winner, logit, seconds = block_answer(
                    model, tokenizer, device, item.state, instructions, texts, indices
                )
                elapsed += seconds
                winners.append(winner)
                logits.append(logit)
                local = indices.index(gold) if gold in indices else None
                in_block[len(winners) - 1] = local
            gold_block = find_gold_block(blocks, gold)
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
                "winners": winners,
                "gold_block": gold_block,
                "gold_local": in_block[gold_block],
                "gold_local_bin": position_bin(in_block[gold_block], len(blocks[gold_block])),
                "won_block": won_block,
                "scheme_a": scheme_a,
                "scheme_a_correct": scheme_a_correct(gold, winners, logits),
                "scheme_b": scheme_b,
                "scheme_b_correct": scheme_b_correct(gold, winners, scheme_b),
                "scheme_c": scheme_c,
                "scheme_c_correct": scheme_c == gold,
                "block_seconds": round(elapsed, 3),
                "final_seconds": round(seconds_b, 3),
                "judgement_seconds": round(seconds_c, 3),
            }
        if len(rows) < JUDGEMENT_ITEMS:
            picked, seconds = judgement_full(scorer, item.state, instructions, texts)
            record["judgement_full"] = {
                "pick": picked,
                "correct": picked == gold,
                "seconds": round(seconds, 3),
            }
        rows.append(record)
        if (len(rows)) % 25 == 0:
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
    judged = [r for r in rows if "judgement_full" in r]
    summary["judgement_full"] = {
        "items": len(judged),
        "correct": sum(r["judgement_full"]["correct"] for r in judged),
        "forwards": K,
    }
    for name, cell in summary["by_block_size"].items():
        print(
            f"b={name}: won block {cell['won_block_rate']:.3f} "
            f"(first {cell['won_block_first_20pct']}, last {cell['won_block_last_20pct']}) "
            f"A {cell['scheme_a_correct']} B {cell['scheme_b_correct']} "
            f"C {cell['scheme_c_correct']} of {len(rows)}",
            flush=True,
        )
    print(
        f"judgement full: {summary['judgement_full']['correct']}/"
        f"{summary['judgement_full']['items']} at {K} forwards",
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
