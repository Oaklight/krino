"""Which code tiers anchor an answer (#350 follow-up, step 2).

Panel 2 names each option with a one-token code. The shipped alphabet is 52
letters plus 32 symbols, and the K=80 dev items show the symbol half is dead:
9/15 correct when the gold option carries a letter code, 0/9 when it carries a
symbol. This experiment asks whether a *tier* can extend the alphabet, holding
the items, the option order and the gold's position fixed and changing only
which tier labels the gold option.

Design: K = 58 = 52 letters (A-Z, a-z) + 6 tier codes. The six tier positions
always include the gold's position. Each item is scored twice per tier:

* ``gold_in_tier`` — the gold option carries a code from the tier under test;
* ``letter_gold`` — the codes at the gold position and the first position that
  carries a letter are swapped, so the gold carries a letter and every other
  label is unchanged.

The comparison is paired within item: the same prompt with two labels exchanged.
The design is in ``model/evaluation/code_tiers.py``; this script runs it.

Four tiers are run on every item:

* ``shipped_symbols`` — the first six symbols of the shipped alphabet
  (``! # $ % ^ *``, with ``@`` replaced because it is also in the clean-symbol
  tier), the positive control: it must reproduce the known failure;
* ``accented_latin`` — six characters sampled from the Latin-script letters;
* ``hangul`` — six sampled from the Hangul tier;
* ``clean_symbols`` — the six symbols that do not glue to the separator.

Two instruments run the same design: ``banking77``, the main set of real
customer messages with 57 distractors drawn from the same 80-intent space, and
``dev24``, the 24 real K=80 development items, subsampled to 58 options, as an
external-validity replica.

Usage::

    python model/scripts/probe_code_tiers.py --snapshot <snapshot> \\
        --output results/jevbench/code_tiers_Qwen_Qwen3-Reranker-4B.json
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
from data.pipeline import load_banking77  # noqa: E402
from model.evaluation.code_space import sample_tier, survey, usable_chars  # noqa: E402
from model.evaluation.code_tiers import (  # noqa: E402
    K,
    TIER_SLOTS,
    evaluate_pairs,
    item_seed,
    paired_codes,
    subsample,
)
from model.evaluation.panel2 import (  # noqa: E402
    CODES,
    code_token_ids,
    glued_codes,
    has_exact_tie,
    render_panel2,
)
from model.src.backbone import load_causal_lm  # noqa: E402

CODE_FORM = "space"
SEED = 342
BANKING77_ITEMS = 200

# The shipped alphabet's first six symbols, with '@' replaced by '*' because
# '@' is also in the clean-symbol tier and the two cells must not share codes.
SHIPPED_SYMBOLS = ["!", "#", "$", "%", "^", "*"]


def banking77_items(exclude: set[str], count: int) -> list:
    """Real Banking77 items whose id is not in the excluded development set."""
    items = [q for q in load_banking77() if q.id not in exclude]
    items.sort(key=lambda q: q.id)
    random.Random(SEED).shuffle(items)
    return items[:count]


def dev_items(path: Path) -> list:
    """The K=80 development items, as the external-validity replica.

    An item qualifies only when its gold label is one of its own options. The
    development set also holds one 79-option abstention item whose correct label
    has been removed from the options on purpose; it has no gold to label, so it
    is not one of the 24 and is excluded rather than silently given a wrong
    gold.
    """
    from model.evaluation.dev_set import load_dev_set

    items = []
    for q in load_dev_set(path):
        criteria = q.question.get("criteria") or {}
        if len(criteria) >= K and q.label in criteria:
            items.append(q)
    items.sort(key=lambda q: q.id)
    return items


def run_instrument(model, tokenizer, device, items, tiers, source: str) -> dict:
    rows = []
    skipped = []
    for item in items:
        criteria = item.question["criteria"]
        texts, gold_pos = subsample(criteria, item.label, K, item_seed(item.id, SEED))
        instructions = item.question.get("instructions", "")
        for tier_name, tier_codes in tiers.items():
            pair = paired_codes(gold_pos, tier_codes)
            if not pair:
                # The two conditions would give the gold the same code; the
                # item carries no contrast and is not counted.
                skipped.append(
                    {
                        "id": item.id,
                        "tier_alphabet": tier_name,
                        "gold_pos": gold_pos,
                        "reason": "gold code unchanged between conditions",
                    }
                )
                continue
            for gold_in_tier in (True, False):
                codes = pair[gold_in_tier]
                code_ids, multi = code_token_ids(tokenizer, codes, CODE_FORM)
                if multi:
                    raise SystemExit(f"codes {multi} are not single tokens")
                prompt = render_panel2(item.state, instructions, texts, codes)
                values, chosen, elapsed, _ = score_prompt(
                    model, tokenizer, prompt, codes, code_ids, device, list(range(K))
                )
                rows.append(
                    {
                        "source": source,
                        "id": item.id,
                        "tier_alphabet": tier_name,
                        "gold_in_tier": gold_in_tier,
                        "gold_pos": gold_pos,
                        "gold_code": codes[gold_pos],
                        "pred_pos": chosen,
                        "pred_code": codes[chosen],
                        "correct": chosen == gold_pos,
                        "pred_is_tier_code": codes[chosen] in set(tier_codes),
                        "top_logit": max(values),
                        "second_logit": sorted(values)[-2],
                        "tie": has_exact_tie(values),
                        "n_options": K,
                        "latency_s": elapsed,
                    }
                )
    return {"rows": rows, "skipped": skipped}


def summarise(rows: list[dict], tiers: dict[str, list[str]]) -> dict:
    out = {}
    for tier_name, tier_codes in tiers.items():
        cell = [r for r in rows if r["tier_alphabet"] == tier_name]
        in_tier = {r["id"]: r["correct"] for r in cell if r["gold_in_tier"]}
        letter = {r["id"]: r["correct"] for r in cell if not r["gold_in_tier"]}
        ids = sorted(set(in_tier) & set(letter))
        result = evaluate_pairs([(in_tier[i], letter[i]) for i in ids])
        result["codes"] = tier_codes
        result["tier_code_emitted"] = {
            "gold_in_tier": sum(r["pred_is_tier_code"] for r in cell if r["gold_in_tier"]),
            "letter_gold": sum(
                r["pred_is_tier_code"] for r in cell if not r["gold_in_tier"]
            ),
        }
        result["ties"] = sum(1 for r in cell if r["tie"])
        out[tier_name] = result
    return out


def tier_spec(tokenizer) -> dict[str, list[str]]:
    rows = survey(tokenizer)
    usable = usable_chars(rows)
    clean_symbols = usable.get("ascii_symbol", [])
    if len(clean_symbols) > TIER_SLOTS:
        clean_symbols = sorted(clean_symbols)[:TIER_SLOTS]
    if len(clean_symbols) != TIER_SLOTS:
        raise SystemExit(f"expected {TIER_SLOTS} clean symbols, got {clean_symbols}")
    return {
        "shipped_symbols": SHIPPED_SYMBOLS,
        "accented_latin": sample_tier(
            rows, "other_letter", TIER_SLOTS, SEED, name_prefix="LATIN"
        ),
        "hangul": sample_tier(rows, "hangul", TIER_SLOTS, SEED),
        "clean_symbols": clean_symbols,
    }


def config_for(args, tiers: dict[str, list[str]]) -> dict:
    """The recipe. Unchanged from the pre-registration in #354."""
    return {
        "arm": "code-tier capacity, step 2",
        "code_form": CODE_FORM,
        "k": K,
        "alphabet": "A-Z, a-z",
        "tier_slots": TIER_SLOTS,
        "tiers": tiers,
        "seed": SEED,
        "tier_positions": "gold position plus the last five positions other than gold",
        "letter_gold_swap": "codes at the gold position and the first letter position",
        "item_seed": "random.Random(f'{seed}:{item_id}')",
        "banking77_items": BANKING77_ITEMS,
        "distractors": "uniform from the same 80-intent criteria, excluding gold",
        "tie_rule": "first maximum by natural option index",
        "gold_order": "original criteria order of the kept options",
        "model": args.snapshot.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--dev-path", type=Path, default=Path("data/benchmarks/dev/dev_800.jsonl")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="smoke-test cap on items per instrument"
    )
    args = parser.parse_args()

    model, tokenizer = load_causal_lm(str(args.snapshot), dtype=torch.bfloat16)
    tiers = tier_spec(tokenizer)
    config = config_for(args, tiers)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    config["config_hash"] = config_hash
    print(f"config_hash {config_hash}", flush=True)
    print(json.dumps(config, sort_keys=True, ensure_ascii=False), flush=True)
    if args.print_config_hash:
        return 0

    device = str(next(model.parameters()).device)
    dev = dev_items(args.dev_path)
    dev_ids = {q.id for q in dev}
    instruments = {
        "banking77": banking77_items(dev_ids, BANKING77_ITEMS),
        "dev24": dev,
    }
    if args.limit:
        instruments = {
            name: items[: args.limit] for name, items in instruments.items()
        }
    result = {
        "_provenance": {
            "config_hash": config_hash,
            "config": config,
            "glued_codes_shipped_84": glued_codes(tokenizer, list(CODES)),
        }
    }
    for name, items in instruments.items():
        print(f"{name}: {len(items)} items", flush=True)
        run = run_instrument(model, tokenizer, device, items, tiers, name)
        result[name] = {
            "items": len(items),
            "tiers": summarise(run["rows"], tiers),
            "skipped_pairs": run["skipped"],
            "rows": run["rows"],
        }
        for tier_name, cell in result[name]["tiers"].items():
            ci = cell["bootstrap_ci_pp"]
            print(
                f"  {name} {tier_name}: gold-in-tier {cell['gold_in_tier_correct']}"
                f"/{cell['items']} letter-gold {cell['letter_gold_correct']}"
                f"/{cell['items']} diff {cell['difference_pp']:+.1f} pp"
                f" p {cell['exact_p']:.2e} CI [{ci['low']:+.1f}, {ci['high']:+.1f}]"
                f" -> {cell['reading']}",
                flush=True,
            )
        if run["skipped"]:
            print(f"  {name} skipped pairs: {len(run['skipped'])}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
