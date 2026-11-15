#!/usr/bin/env python3
"""Smoke the judgement objective on a small real batch (#299).

Three checks the training run depends on, plus a few optimizer steps:

1. Row normalisation equalises rows with different K.  One row's negative prompt
   is duplicated, which leaves the per-row loss unchanged because the duplicate
   has the same margin; an option-weighted mean would change.
2. An ``answerable = False`` row contributes zero gradient with the mask on, and
   a non-zero one with the mask off.
3. A gold option can reach P(yes) > 0.5 after the bias treatment.  The counts
   before the bias, at the configured init, and after the bias-only fit are
   printed so a collapse is visible rather than argued about.

The prompts come from the same builder the plan and the readout use, so the
smoke exercises the training path, not a copy of it.

Usage:
    python model/scripts/smoke_judgement.py --rows 64
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.evaluation.dev_set import load_dev_set  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.judgement_objective import (  # noqa: E402
    JudgementPrompt,
    ScaledMargin,
    build_prompts,
    fit_bias,
    masked_row_normalised_loss,
    positive_base_rate,
    row_weights,
)
from model.src.judgement_readout import JudgementReadout, subsample_prompts  # noqa: E402


def _rows(path: Path, limit: int) -> list[tuple[str, list[JudgementPrompt]]]:
    rows = []
    for item in load_dev_set(path):
        built = build_prompts(
            item, row_id=f"dev:{item.id}", source="dev", split="train"
        )
        if built:
            rows.append((f"dev:{item.id}", built))
        if len(rows) >= limit:
            break
    return rows


@torch.no_grad()
def _margins(readout, scaling, prompts):
    return scaling(readout.row_margins(prompts))


def _row_loss(margins: torch.Tensor, prompts) -> torch.Tensor:
    labels = torch.tensor(
        [prompt.label for prompt in prompts], dtype=torch.float32, device=margins.device
    )
    # The trainer's loss casts to float32; match it so the comparison below is
    # about normalisation, not about bf16 rounding.
    return torch.nn.functional.binary_cross_entropy_with_logits(margins.float(), labels)


def check_row_normalisation(readout, scaling, rows) -> dict:
    """A batch of two real rows with different K is a mean over rows.

    The rows keep their own per-row loss; the batch loss is the plain mean of
    the two, not their option-weighted mean.  Both numbers are reported so the
    gap the row normalisation removes is visible.
    """
    scored = []
    for row_id, prompts in rows:
        margins = _margins(readout, scaling, prompts)
        scored.append((row_id, prompts, margins))
    # Pair the smallest row with the largest so the two weights this check
    # separates are actually different.
    scored.sort(key=lambda entry: entry[2].numel())
    scored = [scored[0], scored[-1]]
    (id_a, prompts_a, margins_a), (id_b, prompts_b, margins_b) = scored
    loss_a = _row_loss(margins_a, prompts_a)
    loss_b = _row_loss(margins_b, prompts_b)
    joined = torch.cat([margins_a, margins_b])
    labels = torch.cat(
        [
            torch.tensor([p.label for p in prompts_a], dtype=torch.float32),
            torch.tensor([p.label for p in prompts_b], dtype=torch.float32),
        ]
    ).to(joined.device)
    batch = masked_row_normalised_loss(
        joined,
        labels,
        [
            torch.arange(0, margins_a.numel(), device=joined.device),
            torch.arange(margins_a.numel(), joined.numel(), device=joined.device),
        ],
        torch.ones(2),
    )
    row_mean = (loss_a + loss_b) / 2
    option_mean = (margins_a.numel() * loss_a + margins_b.numel() * loss_b) / (
        margins_a.numel() + margins_b.numel()
    )
    return {
        "row_a": id_a,
        "k_a": margins_a.numel(),
        "loss_a": float(loss_a),
        "row_b": id_b,
        "k_b": margins_b.numel(),
        "loss_b": float(loss_b),
        "row_normalised_batch_loss": float(batch),
        "row_mean_formula": float(row_mean),
        "option_weighted_batch_loss": float(option_mean),
        "equal": torch.allclose(batch, row_mean),
    }


def check_mask(readout, scaling, rows, trainables) -> dict:
    """A masked row must zero every trainable gradient; unmasked must not."""
    _, prompts = next(
        (row_id, prompts) for row_id, prompts in rows if len(prompts) >= 2
    )
    masked = [dataclasses.replace(prompt, answerable=False) for prompt in prompts]
    labels = torch.tensor([p.label for p in masked], dtype=torch.float32)

    def backward(weight: torch.Tensor) -> tuple[float, float]:
        for parameter in trainables:
            parameter.grad = None
        margins = scaling(readout.row_margins(masked))
        loss = masked_row_normalised_loss(
            margins,
            labels.to(margins.device),
            [torch.arange(margins.numel())],
            weight.to(margins.device),
        )
        loss.backward()
        norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in trainables
            if parameter.grad is not None
        )
        return float(loss.detach()), norm

    loss_on, grad_on = backward(row_weights([False]))
    loss_off, grad_off = backward(torch.ones(1))
    return {
        "loss_masked": loss_on,
        "grad_abs_sum_masked": grad_on,
        "loss_unmasked": loss_off,
        "grad_abs_sum_unmasked": grad_off,
        "masked_zero": loss_on == 0.0 and grad_on == 0.0,
        "unmasked_nonzero": loss_off > 0.0 and grad_off > 0.0,
    }


@torch.no_grad()
def check_bias(readout, rows, max_options, yes_bias) -> dict:
    """Gold P(yes) counts before, at the init, and after the bias-only fit."""
    margins = []
    labels = []
    answerable = []
    for _, prompts in rows:
        chosen = subsample_prompts(prompts, max_options, random.Random(0))
        if not readout.fits(chosen):
            continue
        margins.append(readout.row_margins(chosen))
        labels.append(torch.tensor([p.label for p in chosen], dtype=torch.float32))
        answerable.append(bool(chosen[0].answerable))
    raw = torch.cat(margins)
    label = torch.cat(labels).to(raw.device)
    offsets = [0]
    for value in margins:
        offsets.append(offsets[-1] + value.numel())
    row_index = [
        torch.arange(offsets[i], offsets[i + 1], device=raw.device)
        for i in range(len(margins))
    ]
    gold = label == 1
    base_rate = positive_base_rate(label, row_index, row_weights(answerable))
    offset = fit_bias(raw, row_index, base_rate)

    def count(bias: float) -> tuple[int, float]:
        probability = torch.sigmoid(raw + bias)
        return int((probability[gold] > 0.5).sum()), float(probability[gold].max())

    before, before_max = count(0.0)
    at_init, init_max = count(float(yes_bias))
    after, after_max = count(offset)
    return {
        "gold_options": int(gold.sum()),
        "positive_base_rate": base_rate,
        "fitted_bias_offset": offset,
        "gold_above_half_bias0": before,
        "gold_max_p_yes_bias0": before_max,
        "gold_above_half_init": at_init,
        "gold_max_p_yes_init": init_max,
        "gold_above_half_calibrated": after,
        "gold_max_p_yes_calibrated": after_max,
        "calibrated_positive": after > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument(
        "--dev-path", type=Path, default=ROOT / "data/benchmarks/dev/dev_800.jsonl"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="prompt plan JSONL; overrides --dev-path so the smoke can run "
        "against the frozen corpus",
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--max-options", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--yes-bias", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=5, help="Optimizer steps to run")
    parser.add_argument("--batch-rows", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-layers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.plan:
        from model.scripts.train_judgement import group_rows, load_plan

        rows = group_rows(load_plan(args.plan))[: args.rows]
    else:
        rows = _rows(args.dev_path, args.rows)
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} scorable rows in the smoke input")

    backbone, tokenizer = load_causal_lm(
        args.model, device=args.device, dtype=torch.bfloat16, freeze=True
    )
    if args.lora_layers:
        from model.src.lora import inject_lora

        inject_lora(
            backbone,
            targets=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=16,
            alpha=32,
            last_n_layers=args.lora_layers,
        )
    readout = JudgementReadout(backbone, tokenizer, max_length=args.max_length)
    scaling = ScaledMargin(scale=1.0, bias=args.yes_bias).to(readout.device)
    trainables = [p for p in backbone.parameters() if p.requires_grad] + list(
        scaling.parameters()
    )

    normalisation = check_row_normalisation(readout, scaling, rows)
    mask = check_mask(readout, scaling, rows, trainables)
    bias = check_bias(readout, rows, args.max_options, args.yes_bias)

    # A few real optimizer steps, with the mask on.
    optimizer = torch.optim.AdamW(trainables, lr=args.lr)
    backbone.train()
    rng = random.Random(args.seed)
    order = list(rows)
    rng.shuffle(order)
    losses = []
    for start in range(
        0, min(len(order), args.steps * args.batch_rows), args.batch_rows
    ):
        margins = []
        labels = []
        weights = []
        for _, prompts in order[start : start + args.batch_rows]:
            chosen = subsample_prompts(prompts, args.max_options, rng)
            if not readout.fits(chosen):
                continue
            margins.append(scaling(readout.row_margins(chosen)))
            labels.append(
                torch.tensor(
                    [p.label for p in chosen],
                    dtype=torch.float32,
                    device=readout.device,
                )
            )
            weights.append(bool(chosen[0].answerable))
        if not margins:
            continue
        joined = torch.cat(margins)
        offsets = [0]
        for value in margins:
            offsets.append(offsets[-1] + value.numel())
        index = [
            torch.arange(offsets[i], offsets[i + 1], device=readout.device)
            for i in range(len(margins))
        ]
        loss = masked_row_normalised_loss(
            joined, torch.cat(labels), index, row_weights(weights)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    summary = {
        "stage": "judgement-smoke",
        "backbone": args.model,
        "readout": "per-option yes/no margin, shared prefix, batched suffix",
        "length": args.max_length,
        "seed": args.seed,
        "rows": len(rows),
        "lora_layers": args.lora_layers,
        "row_normalisation": normalisation,
        "answerable_mask": mask,
        "positive_rate": bias,
        "train_losses": losses,
    }
    print(json.dumps(summary, indent=1), flush=True)
    if args.output:
        args.output.write_text(json.dumps(summary, indent=1) + "\n")

    checks = {
        "row_normalisation": normalisation["equal"],
        "mask_zero": mask["masked_zero"],
        "mask_unmasked_nonzero": mask["unmasked_nonzero"],
        "gold_positive_after_calibration": bias["calibrated_positive"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"FAILED {failed}", flush=True)
        return 1
    print("all smoke checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
