"""Measure the per-row CE and RPS that set ``--score-cramer-weight``.

The mix arm's two terms sit on different scales: CE is unbounded, RPS is at
most 1. The weight is therefore measured, not chosen. On the plan's
non-synthetic val score rows -- the population the primary endpoint reads --
one forward per row, this prints the mean per-row CE, the mean per-row RPS,
their ratio, and

    lambda = 0.25 * mean CE / mean RPS

which makes the RPS term ``0.25 *`` the CE term, i.e. 20% of the score-row
loss.

Per-level-count blocks are printed as well. The train split's level-count mix
need not match the val split's, and the ratio is K-dependent, so the by-K
lambdas are the check on whether one global lambda is defensible: if they
agree within about 2x, it is.

The resulting JSON is what ``--score-cramer-weight-source`` copies into the
training payload; the trainer checks the pinned weight against its lambda and
re-checks the file's internal arithmetic.

Usage:
    python -m model.scripts.measure_score_cramer --plan PLAN --output OUT
        [--weights WEIGHTS_OR_-] [--device cuda:0] [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.evaluation.panel2 import CODES, code_token_ids
from model.scripts.train_panel2_lora import build_rows, code_logits, prompt_of
from model.src.backbone import load_causal_lm
from model.training.losses import rps_loss

FACTOR = 0.25


def row_logits(model, tokenizer, row, code_ids, device):
    """The row's own K code logits, which is what the training loss sees.

    ``code_logits`` scores every code it is given. Passing the global map
    would score all of them and normalise CE and RPS over a distribution the
    model is never trained on, which inflates both and the ratio with them.
    """
    return code_logits(model, tokenizer, prompt_of(row),
                       {code: code_ids[code] for code in row.codes}, device)


def block(rows, factor: float = FACTOR) -> dict:
    """One population's means and the lambda its own ratio implies."""
    ce = sum(row["ce"] for row in rows) / len(rows)
    rps = sum(row["rps"] for row in rows) / len(rows)
    if rps <= 0:
        raise SystemExit(
            f"mean RPS is {rps}; the CE/RPS ratio and lambda are undefined"
        )
    lam = factor * ce / rps
    return {"n": len(rows), "mean_ce": ce, "mean_rps": rps,
            "ce_over_rps": ce / rps, "lambda": lam,
            "rps_term_fraction_at_block_lambda": lam * rps / (ce + lam * rps)}


def summarise(per_row, factor: float = FACTOR) -> dict:
    """Overall lambda, and every block's share under it.

    A block's own lambda makes its share factor/(1+factor) = 0.2 by
    construction, which is not a measurement. The number a reader needs is
    each block's share when the overall lambda is applied, so that is what
    every block carries beside its block-local value.
    """
    overall = block(per_row, factor)
    lam = overall["lambda"]

    def with_share(entry):
        share = lam * entry["mean_rps"] / (entry["mean_ce"]
                                           + lam * entry["mean_rps"])
        return {**entry, "rps_term_fraction_at_overall_lambda": share}

    return {"overall": with_share(overall),
            "by_k": {str(k): with_share(block(
                [row for row in per_row if row["k"] == k], factor))
                for k in sorted({row["k"] for row in per_row})},
            "by_gold": {str(g): with_share(block(
                [row for row in per_row if row["gold"] == g], factor))
                for g in sorted({row["gold"] for row in per_row})}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="the plan whose non-synthetic val score rows to "
                             "use: every val score row whose source is not "
                             "synthetic, which includes the typed_decisions "
                             "and apigen families, not only bench")
    parser.add_argument("--output", type=Path, required=True,
                        help="where to write the probe JSON")
    parser.add_argument("--weights", type=str, default="-",
                        help="state dict to load first, or - for the base "
                             "model (step 0)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N rows (a smoke)")
    args = parser.parse_args()

    data = args.plan.read_bytes()
    rows, field, masked = build_rows(data, plan_unpermuted_noul=True)
    selected = [row for row in rows
                if row.split == "val" and row.q_type == "score"
                and row.source != "synthetic"]
    if args.limit:
        selected = selected[: args.limit]
    print(f"plan {args.plan} split_field {field} masked {masked}; "
          f"{len(selected)} non-synthetic val score rows", flush=True)
    if not selected:
        raise SystemExit("no non-synthetic val score rows in this plan")

    model, tokenizer = load_causal_lm(
        "Qwen/Qwen3-Reranker-4B", device=args.device, dtype=torch.bfloat16,
        freeze=True, flash_attention=False,
    )
    if args.weights != "-":
        state = torch.load(args.weights, map_location="cpu",
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"loaded {args.weights}", flush=True)
    code_ids, multi = code_token_ids(tokenizer, CODES, "space")
    if len(code_ids) < 26 or multi:
        raise SystemExit(f"code ids unusable: {len(code_ids)} single-token, "
                         f"{multi[:5]}")
    model.eval()

    per_row = []
    with torch.inference_mode():
        for index, row in enumerate(selected):
            values = row_logits(model, tokenizer, row, code_ids, args.device)
            logits = torch.tensor(values, dtype=torch.float32,
                                  device=args.device)
            gold = int(row.gold)
            ce = float(F.cross_entropy(logits.unsqueeze(0),
                                       torch.tensor([gold],
                                                    device=args.device)))
            target = F.one_hot(torch.tensor(gold, device=args.device),
                               logits.numel()).float()
            rps = float(rps_loss(logits.unsqueeze(0), target.unsqueeze(0)))
            per_row.append({"row_id": row.row_id, "k": int(logits.numel()),
                            "gold": gold, "ce": ce, "rps": rps})
            if (index + 1) % 200 == 0:
                print(f"  {index + 1}/{len(selected)}", flush=True)

    summary = {"plan": str(args.plan), "weights": args.weights,
               "rows": len(per_row), "factor": FACTOR,
               **summarise(per_row), "per_row": per_row}
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "per_row"}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
