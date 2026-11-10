"""Does a code's lexical content matter? (#350 follow-up, after step 1b).

Step 1b found 505 two-letter and 550 three-letter codes that survive the
panel-2 format, but the survivors are not neutral: 25 of the doubles are common
English words and the stable triples are dominated by words and abbreviations.
A code that is a word carries meaning into the option list, which the
single-character alphabet never did -- and in the opposite direction from the
symbol tier, which had no prior rather than a wrong one. `YES` as a code on a
noul item would write the answer into the label.

This runs the #354 paired design on four lexical cells, all at K = 58 with the
same items, option order, seeds and gold positions:

* ``double_listed_word`` -- six of the declared common-word doubles;
* ``double_unlisted`` -- six doubles outside the declared list;
* ``triple_listed_word`` -- THE AND FOR YOU ALL ONE, the obviously-wordy control;
* ``triple_unlisted`` -- six triples outside the declared list.

`YES` and `NO` are excluded from every cell: on a noul item their option texts
are literally ``no`` and ``yes``.

"Unlisted" means outside the declared list in this file, not meaningless. The
sample contains abbreviations (EU, USA, BMW, CSI, GNU) and one English word
outside the list (BOX), so the unlisted cell is contaminated toward the word
cell; that biases the lexical contrast toward the null, not away from it.

Usage::

    python model/scripts/probe_code_lexical.py --snapshot <snapshot> \\
        --output results/jevbench/code_lexical_Qwen_Qwen3-Reranker-4B.json
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

from model.evaluation.code_space import string_survey  # noqa: E402
from model.evaluation.panel2 import CODES, glued_codes  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from probe_code_tiers import (  # noqa: E402
    BANKING77_ITEMS,
    CODE_FORM,
    K,
    SEED,
    banking77_items,
    dev_items,
    run_instrument,
    summarise,
)

# Declared before the run; "listed word" means a member of these lists.
WORD2 = [
    "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN", "IS", "IT",
    "ME", "MY", "NO", "OF", "OH", "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE",
]
WORD3 = [
    "ALL", "AND", "ANY", "ARE", "BAD", "BIG", "BOY", "BUT", "CAN", "CAR", "DAY",
    "DID", "END", "EYE", "FAR", "FEW", "FOR", "GET", "GOD", "GOT", "HAS", "HER",
    "HIM", "HIS", "HOW", "ITS", "JOB", "LET", "MAN", "MAY", "NEW", "NOT", "NOW",
    "OLD", "ONE", "OUR", "OUT", "OWN", "PUT", "SAY", "SEE", "SHE", "TWO", "USE",
    "WAR", "WAY", "WHO", "WHY", "YES", "YOU",
]
TRIPLE_WORD_CELL = ["THE", "AND", "FOR", "YOU", "ALL", "ONE"]
# The noul option texts; a code equal to either writes the answer into a label.
EXCLUDED = {"YES", "NO"}
TIER_SLOTS = 6


def sample(pool: list[str], count: int = TIER_SLOTS) -> list[str]:
    return sorted(random.Random(SEED).sample(sorted(pool), count))


def lexical_spec(tokenizer) -> dict[str, list[str]]:
    doubles = string_survey(tokenizer, 2)["stable_codes"]
    triples = string_survey(tokenizer, 3)["stable_codes"]
    listed2 = [c for c in WORD2 if c in doubles and c not in EXCLUDED]
    unlisted2 = [c for c in doubles if c not in WORD2]
    unlisted3 = [c for c in triples if c not in WORD3]
    missing = [c for c in TRIPLE_WORD_CELL if c not in triples]
    if missing:
        raise SystemExit(f"declared triple cell not stable here: {missing}")
    return {
        "double_listed_word": sample(listed2),
        "double_unlisted": sample(unlisted2),
        "triple_listed_word": sorted(TRIPLE_WORD_CELL),
        "triple_unlisted": sample(unlisted3),
    }


def config_for(args, tiers: dict[str, list[str]]) -> dict:
    return {
        "arm": "code lexical content",
        "after": "step 1b census (#354)",
        "code_form": CODE_FORM,
        "k": K,
        "alphabet": "A-Z, a-z",
        "tier_slots": TIER_SLOTS,
        "tiers": tiers,
        "seed": SEED,
        "word2": WORD2,
        "word3": WORD3,
        "triple_word_cell": TRIPLE_WORD_CELL,
        "excluded_codes": sorted(EXCLUDED),
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
    parser.add_argument("--limit", type=int, help="smoke-test cap on items")
    args = parser.parse_args()

    model, tokenizer = load_causal_lm(str(args.snapshot), dtype=torch.bfloat16)
    tiers = lexical_spec(tokenizer)
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
