"""Position probes for panel 2's long-list tail (#357).

Panel 2 at K=52 loses the last 40% of the option list on Banking77 (0.0% in the
last position bucket) and only part of it on CLINC150 (59.1%). These probes ask
why. Every cell holds the items fixed and changes one thing about the display.

* **position** — the selected options keep their texts and only their display
  order is permuted per item. Accuracy by the natural position and by the
  permuted position separates "the model prefers early positions" from "these
  items' golds happen to sit late".
* **ladder** — K = 7, 14, 28, 52 on the same items, identity plus two seeded
  permutations each (four at K=52), so the tail's shape is read with position
  controlled.
* **alphabet** — K = 80 on each dataset, letters on the first 52 display
  positions against letters on the last 52, with the gold always on a letter at
  the same rank among the letters. This separates the symbol tier from the
  position the symbols occupy.
* **text** — the 51 distractor texts are replaced by labels drawn from the same
  dataset (control) or from the other dataset, keeping the gold text, state,
  instructions and gold position. This tests whether distractor text, rather
  than position, produces the tail.

The model readout is the recorded one: panel-2 template, space code form,
``codes_for`` alphabet, tie to the first maximum by natural option index. The
dataset is never written into the repository.

Usage::

    python model/scripts/probe_position.py --snapshot <snapshot> \\
        --dataset banking77 --output results/jevbench/position_banking77.json
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

from eval_jevbench_panel2 import score_prompt  # noqa: E402
from model.evaluation import clinc  # noqa: E402
from model.evaluation.code_tiers import item_seed, subsample  # noqa: E402
from model.evaluation.panel2 import code_token_ids, codes_for, render_panel2  # noqa: E402
from model.evaluation.position import bucket, check_gate, permuted  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from probe_code_tiers import banking77_items, dev_items  # noqa: E402

CODE_FORM = "space"
SEED = 342
ITEMS = 200
MAIN_K = 52
LADDER = (7, 14, 28, 52)
PERMS = {7: 2, 14: 2, 28: 2, 52: 4}
ALPHABET_K = 80
TEXT_K = 52

LETTERS = codes_for(52)
SYMBOLS = codes_for(ALPHABET_K)[52:]
EARLY_CODES = LETTERS + SYMBOLS
LATE_CODES = SYMBOLS + LETTERS


def arrange(texts: list[str], gold: int, target: int) -> tuple[list[str], int]:
    """Move the gold to ``target`` and keep every other option's relative order."""
    others = [text for position, text in enumerate(texts) if position != gold]
    return others[:target] + [texts[gold]] + others[target:], target


def select_nested(criteria: dict, label: str, k: int, item_id: str) -> tuple[list[str], int]:
    """The nested ladder draw: smaller K is a prefix of the K=52 draw.

    The K=52 set is the recorded subsample, drawn exactly as ``subsample`` does
    (one ``rng.sample`` call on the same RNG). At K < 52 the gold keeps the
    first ``k - 1`` distractors of that draw in original criteria order, so the
    ladder changes the list length and nothing else: no independent redraw.
    """
    keys = list(criteria)
    gold_index = keys.index(label)
    others = [index for index in range(len(keys)) if index != gold_index]
    kept = sorted([gold_index, *item_seed(item_id, SEED).sample(others, MAIN_K - 1)])
    if k < MAIN_K:
        prefix = [index for index in kept if index != gold_index][: k - 1]
        kept = sorted([gold_index, *prefix])
    texts = [str(criteria[keys[index]]) for index in kept]
    return texts, kept.index(gold_index)


def score(model, tokenizer, device, state, instructions, texts, codes):
    code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
    if multi:
        raise SystemExit(f"codes {multi} are not single tokens")
    prompt = render_panel2(state, instructions, texts, codes)
    values, chosen, elapsed, _ = score_prompt(
        model, tokenizer, prompt, codes, code_ids, device, list(range(len(texts)))
    )
    return chosen, elapsed


def record(dataset, cell, item, k, natural, display_gold, pred, elapsed):
    return {
        "dataset": dataset,
        "cell": cell,
        "id": item.id,
        "k": k,
        "natural_pos": natural,
        "display_pos": display_gold,
        "natural_bucket": bucket(natural, k),
        "display_bucket": bucket(display_gold, k),
        "pred_pos": pred,
        "correct": pred == display_gold,
        "latency_s": elapsed,
    }


def run_ladder(model, tokenizer, device, dataset, items) -> list[dict]:
    rows = []
    for k in LADDER:
        codes = codes_for(k)
        for item in items:
            texts, gold = select_nested(
                item.question["criteria"], item.label, k, item.id
            )
            instructions = item.question.get("instructions", "")
            for index in range(PERMS[k] + 1):
                if index == 0:
                    display, display_gold, cell = texts, gold, f"natural_k{k}"
                else:
                    display, display_gold = permuted(texts, gold, item.id, index)
                    cell = f"perm_k{k}_{index}"
                chosen, elapsed = score(
                    model, tokenizer, device, item.state, instructions, display, codes
                )
                rows.append(
                    record(dataset, cell, item, k, gold, display_gold, chosen, elapsed)
                )
        print(f"  ladder k={k} done", flush=True)
    return rows


def run_alphabet(model, tokenizer, device, dataset, items) -> list[dict]:
    rows = []
    for item in items:
        texts, gold = subsample(
            item.question["criteria"], item.label, ALPHABET_K, item_seed(item.id, SEED)
        )
        instructions = item.question.get("instructions", "")
        letter_rank = gold % 52
        for cell, target, codes in (
            ("letters_early", letter_rank, EARLY_CODES),
            ("letters_late", 28 + letter_rank, LATE_CODES),
        ):
            display, display_gold = arrange(texts, gold, target)
            chosen, elapsed = score(
                model, tokenizer, device, item.state, instructions, display, codes
            )
            rows.append(
                record(dataset, cell, item, ALPHABET_K, gold, display_gold, chosen, elapsed)
            )
    print("  alphabet done", flush=True)
    return rows


def run_text(model, tokenizer, device, dataset, items, own_pool, cross_pool) -> list[dict]:
    rows = []
    for item in items:
        texts, gold = subsample(
            item.question["criteria"], item.label, TEXT_K, item_seed(item.id, SEED)
        )
        instructions = item.question.get("instructions", "")
        gold_text = texts[gold]
        own = [text for text in own_pool if text != gold_text]
        cross = [text for text in cross_pool if text != gold_text]
        for condition, pool in (("text_own", own), ("text_cross", cross)):
            drawn = random.Random(f"{SEED}:{item.id}:{condition}").sample(pool, TEXT_K - 1)
            display = drawn[:gold] + [gold_text] + drawn[gold:]
            chosen, elapsed = score(
                model, tokenizer, device, item.state, instructions, display, codes_for(TEXT_K)
            )
            rows.append(
                record(dataset, condition, item, TEXT_K, gold, gold, chosen, elapsed)
            )
    print("  text done", flush=True)
    return rows


def label_texts(items) -> list[str]:
    """The display texts of one item set's label space."""
    return list(items[0].question["criteria"].values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dataset", choices=("banking77", "clinc150"), required=True)
    parser.add_argument(
        "--dev-path", type=Path, default=Path("data/benchmarks/dev/dev_800.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--limit", type=int, help="smoke-test cap on items")
    args = parser.parse_args()

    config = {
        "arm": "panel-2 position probes",
        "dataset": args.dataset,
        "items": ITEMS,
        "ladder": list(LADDER),
        "ladder_nesting": "K=7/14/28 keep the gold plus the first k-1 distractors of the K=52 draw, in original criteria order; the K=52 cell is the recorded subsample",
        "perms": {str(k): PERMS[k] for k in LADDER},
        "alphabet_k": ALPHABET_K,
        "letters_early": "letters on display 0-51, the 28 symbols on 52-79",
        "letters_late": "the 28 symbols on display 0-27, letters on 28-79",
        "alphabet_gold": "a letter in both cells, at the same rank among the letters",
        "text_k": TEXT_K,
        "text_swap": "51 distractor texts drawn from the other dataset's labels",
        "text_control": "51 distractor texts drawn from the same dataset's labels",
        "code_form": CODE_FORM,
        "tie_rule": "first maximum by natural option index",
        "seed": SEED,
        "item_seed": "random.Random(f'{seed}:{item_id}')",
        "perm_seed": "random.Random(f'{seed}:{item_id}:perm{index}')",
        "text_seed": "random.Random(f'{seed}:{item_id}:{{condition}}')",
        "model": args.snapshot.name,
    }
    if args.dataset == "clinc150":
        _, meta = clinc.load(ITEMS, SEED)
        config.update(meta)
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
    dev = dev_items(args.dev_path) if args.dataset == "banking77" else []
    if args.dataset == "banking77":
        items = banking77_items({q.id for q in dev}, ITEMS)
    else:
        items, _ = clinc.load(ITEMS, SEED)
    if args.limit:
        items = items[: args.limit]

    if args.dataset == "banking77":
        own_pool = label_texts(items)
        cross_pool = label_texts(clinc.load(ITEMS, SEED)[0])
    else:
        own_pool = label_texts(items)
        cross_pool = label_texts(banking77_items({q.id for q in dev}, ITEMS))

    rows: list[dict] = []
    rows += run_ladder(model, tokenizer, device, args.dataset, items)
    if not args.limit:
        gate = check_gate(rows, args.dataset)
        print(f"  gate {gate['correct']}/{gate['items']} reproduced", flush=True)
        rows += run_alphabet(model, tokenizer, device, args.dataset, items)
        rows += run_text(model, tokenizer, device, args.dataset, items, own_pool, cross_pool)

    result = {"_provenance": {"config_hash": config_hash, "config": config}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
