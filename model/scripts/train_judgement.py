#!/usr/bin/env python3
"""Train the judgement objective on a frozen prompt plan (#299).

One row contributes one prompt per option, with the shared ``State``/``Question``
prefix encoded once and the option suffixes scored in one batched forward.  The
loss is ``log P(" yes") - log P(" no")`` through a learned scale and bias,
averaged within a row and then across rows, so a 77-option row carries the same
weight as a 2-option row.  The scale/bias are the positive-base-rate treatment:
the bias starts positive and can be refitted on the train split after the main
fit.

``answerable = False`` rows are kept and masked, not dropped (#254).  The mask
zeroes the row's loss weight and removes it from the denominator; the row still
runs its forward, so the batch composition stays the one the plan defines.

The backbone is frozen; LoRA adapters on the last N blocks are the only
backbone parameters that move.  Every result names stage, backbone, readout,
length and seed, and the checkpoint records them.

Usage:
    python model/scripts/train_judgement.py \\
        --plan data/benchmarks/judgement_plan.jsonl \\
        --config model/configs/judgement_reranker4b.yaml \\
        --output model/experiments/judgement_lora_s42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from _vendor import yaml  # noqa: E402

from model.src.backbone import load_causal_lm  # noqa: E402
from model.src.judgement_objective import (  # noqa: E402
    JudgementPrompt,
    ScaledMargin,
    fit_bias,
    positive_base_rate,
    row_bce,
    row_weights,
)
from model.src.judgement_readout import (  # noqa: E402
    JudgementReadout,
    subsample_prompts,
)
from model.src.lora import inject_lora  # noqa: E402
from model.training.group_split import (  # noqa: E402
    load_cluster_manifest,
    split_items_by_group,
)

PROMPT_FIELDS = {field for field in JudgementPrompt.__dataclass_fields__}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _record_to_prompt(record: dict) -> JudgementPrompt:
    """Rebuild one prompt, deriving prefix/suffix for a pre-#299 plan line."""
    payload = {key: record[key] for key in record if key in PROMPT_FIELDS}
    if "prefix" not in payload or "suffix" not in payload:
        text = payload["text"]
        cut = text.find("Option: ")
        if payload.get("option_char_end") is None or cut < 0:
            payload["prefix"] = ""
            payload["suffix"] = text
        else:
            payload["prefix"] = text[:cut]
            payload["suffix"] = text[cut:]
    payload.setdefault("answerable", True)
    return JudgementPrompt(**payload)


def load_plan(path: Path) -> list[JudgementPrompt]:
    prompts = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prompts.append(_record_to_prompt(json.loads(line)))
    return prompts


def group_rows(
    prompts: list[JudgementPrompt],
) -> list[tuple[str, list[JudgementPrompt]]]:
    """Group prompts into rows, preserving first-seen row order."""
    order: list[str] = []
    grouped: dict[str, list[JudgementPrompt]] = {}
    for prompt in prompts:
        if prompt.row_id not in grouped:
            grouped[prompt.row_id] = []
            order.append(prompt.row_id)
        grouped[prompt.row_id].append(prompt)
    return [(row_id, grouped[row_id]) for row_id in order]


def split_rows(
    rows: list[tuple[str, list[JudgementPrompt]]],
    holdout_ids: set[str] | None,
    clusters: dict[str, str] | None,
    val_fraction: float,
    seed: int,
) -> tuple[list, list, list, dict]:
    """Internal train/val by whole groups; external eval rows stay separate.

    Held-out ids are removed first and never re-split.  The internal split is
    group-aware — near-duplicate cluster, exact normalized state, and the row's
    own passage/family key — so cluster mates never land on both sides.  Rows
    whose plan split is not ``train`` are the external diagnostic set and are
    not part of the internal split.
    """
    holdout = holdout_ids or set()
    pool, external = [], []
    for row_id, prompts in rows:
        keys = {row_id, row_id.rsplit(":", 1)[-1]}
        if keys & holdout:
            continue
        if prompts[0].split == "train":
            pool.append((row_id, prompts))
        else:
            external.append((row_id, prompts))
    train, val, report = split_items_by_group(
        pool,
        val_fraction,
        seed,
        id_of=lambda row: row[0],
        state_of=lambda row: row[1][0].state_key or row[1][0].prefix or row[1][0].text,
        group_of=lambda row: row[1][0].group,
        cluster_of=(
            lambda row: (
                clusters.get(f"id:{row[0]}")
                or clusters.get(f"state:{row[1][0].state_key}")
            )
        )
        if clusters
        else None,
        stratum_of=lambda row: (row[1][0].q_type, min(len(row[1]), 8)),
    )
    return train, val, external, report


def row_selection_sha256(rows: list[tuple[str, list[JudgementPrompt]]]) -> str:
    """Stable hash of the training rows: id, option count, positive index."""
    lines = []
    for row_id, prompts in rows:
        positives = [prompt.option_index for prompt in prompts if prompt.label == 1]
        lines.append(f"{row_id}\t{len(prompts)}\t{positives!r}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def trainable_rows(rows, readout: JudgementReadout):
    """Drop rows the readout cannot score, reporting why.

    A row over ``max_length`` is dropped, not truncated: a truncated state is a
    different question.  A tokenisation-drift error is not caught here; it
    propagates, because it means training and inference disagree.
    """
    kept = []
    dropped_long = 0
    for row_id, prompts in rows:
        if not readout.fits(prompts):
            dropped_long += 1
            continue
        kept.append((row_id, prompts))
    return kept, dropped_long


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--max-options",
        type=int,
        default=20,
        help="Negative-subsample a row above this option count (#299)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="LoRA learning rate")
    parser.add_argument(
        "--head-lr", type=float, default=1e-2, help="scale/bias learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scale-init", type=float, default=1.0)
    parser.add_argument(
        "--yes-bias",
        type=float,
        default=2.0,
        help="Initial yes-token bias; the positive rate is about 1/K (#299)",
    )
    parser.add_argument(
        "--calibrate-bias",
        action="store_true",
        help="After training, refit the bias so mean P(yes) equals the train positive rate",
    )
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-targets",
        type=lambda value: tuple(
            part.strip() for part in value.split(",") if part.strip()
        ),
        default=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--val-rows", type=int, default=256)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Internal train/val split, assigned by whole groups",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the group-aware internal split; keep 42 across seeds",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help="JSON id->cluster manifest merged into the group key",
    )
    parser.add_argument("--calibration-rows", type=int, default=512)
    parser.add_argument(
        "--holdout-ids",
        type=Path,
        default=None,
        help="JSON id list (or {'ids': [...]}) excluded before the split",
    )
    parser.add_argument("--allow-collapse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Per-phase timing breakdown"
    )
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--corpus-sha256", default=None)
    if config:
        parser.set_defaults(**_config_defaults(config))
    return parser


def _config_defaults(config: dict) -> dict:
    """Map a flat YAML config onto argparse defaults."""
    defaults = dict(config)
    if isinstance(defaults.get("lora_targets"), str):
        defaults["lora_targets"] = tuple(
            part.strip() for part in defaults["lora_targets"].split(",") if part.strip()
        )
    defaults.pop("sources", None)
    return defaults


def load_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    return yaml.load(Path(path).read_text())


def measure(
    readout: JudgementReadout,
    scaling: ScaledMargin,
    rows,
    max_options: int,
    limit: int,
    rng: random.Random,
    profile: bool = False,
) -> dict:
    """Mean loss and P(yes) statistics on up to ``limit`` rows."""
    selected = rows[:limit]
    if not selected:
        return {"rows": 0, "loss": None, "gold_positive_yes": 0, "gold_options": 0}
    total = 0.0
    gold_positive = 0
    gold_total = 0
    weights_total = 0.0
    with torch.no_grad():
        for _, prompts in selected:
            prompts = subsample_prompts(prompts, max_options, rng)
            if not readout.fits(prompts):
                continue
            raw = readout.row_margins(prompts)
            margin = scaling(raw)
            labels = torch.tensor(
                [p.label for p in prompts], dtype=torch.float32, device=margin.device
            )
            per_row = row_bce(margin, labels)
            weight = 0.0 if prompts[0].answerable is False else 1.0
            total += float(per_row) * weight
            weights_total += weight
            probability = ScaledMargin.probability(margin)
            gold = labels == 1
            gold_total += int(gold.sum())
            gold_positive += int((probability[gold] > 0.5).sum())
    return {
        "rows": len(selected),
        "loss": total / weights_total if weights_total else None,
        "gold_positive_yes": gold_positive,
        "gold_options": gold_total,
    }


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args()
    config = load_config(known.config)
    args = build_parser(config).parse_args()
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    args.output.mkdir(parents=True, exist_ok=True)

    holdout_ids = None
    if args.holdout_ids:
        payload = json.loads(Path(args.holdout_ids).read_text())
        holdout_ids = set(payload["ids"] if isinstance(payload, dict) else payload)

    prompts = load_plan(args.plan)
    rows = group_rows(prompts)
    clusters = load_cluster_manifest(args.clusters) if args.clusters else None
    train_rows, val_rows, eval_rows, split_report = split_rows(
        rows, holdout_ids, clusters, args.val_fraction, args.split_seed
    )
    if args.limit_rows:
        train_rows = train_rows[: args.limit_rows]
    if not train_rows:
        raise SystemExit("the plan has no train rows")
    print(f"split {json.dumps(split_report)}", flush=True)
    provenance = {
        "code_commit": args.code_commit or _git_commit(),
        "plan_sha256": args.corpus_sha256 or _sha256(args.plan),
        "train_selection_sha256": row_selection_sha256(train_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "eval_rows": len(eval_rows),
        "split_report": split_report,
        "train_prompts": sum(len(p) for _, p in train_rows),
        "masked_train_rows": sum(1 for _, p in train_rows if p[0].answerable is False),
        "config": {
            "stage": "judgement",
            "backbone": args.model,
            "readout": "per-option yes/no margin",
            "loss": "row-normalised BCE on log P(yes) - log P(no)",
            "max_length": args.max_length,
            "max_options": args.max_options,
            "seed": args.seed,
            "lora_layers": args.lora_layers,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": list(args.lora_targets),
            "yes_bias": args.yes_bias,
            "scale_init": args.scale_init,
            "calibrate_bias": args.calibrate_bias,
            "epochs": args.epochs,
            "batch_rows": args.batch_rows,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=1) + "\n"
    )
    print(f"provenance {json.dumps(provenance)}", flush=True)
    if args.dry_run:
        return 0

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    backbone, tokenizer = load_causal_lm(
        args.model, device=args.device, dtype=dtype, freeze=True, flash_attention=False
    )
    if args.lora_layers and args.lora_layers > 0:
        adapted = inject_lora(
            backbone,
            targets=args.lora_targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            last_n_layers=args.lora_layers,
        )
        print(f"adapted {adapted} linear layers", flush=True)
    else:
        adapted = 0
    readout = JudgementReadout(backbone, tokenizer, max_length=args.max_length)
    scaling = ScaledMargin(scale=args.scale_init, bias=args.yes_bias).to(readout.device)

    train_rows, dropped_long = trainable_rows(train_rows, readout)
    if dropped_long:
        print(f"dropped {dropped_long} train rows over max_length", flush=True)
    if not train_rows:
        raise SystemExit("no train row fits max_length")

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": list(scaling.parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    print(
        f"trainable backbone {sum(p.numel() for p in lora_params):,} "
        f"scaling {sum(p.numel() for p in scaling.parameters())}",
        flush=True,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        order = list(train_rows)
        rng = random.Random(args.seed + epoch)
        rng.shuffle(order)
        started = time.perf_counter()
        rows_done = prompts_done = masked = 0
        running = 0.0
        batches = 0
        profile_totals = {"forward_s": 0.0, "backward_s": 0.0, "step_s": 0.0}
        for start in range(0, len(order), args.batch_rows):
            chosen_rows = []
            for _, prompts in order[start : start + args.batch_rows]:
                chosen = subsample_prompts(prompts, args.max_options, rng)
                if readout.fits(chosen):
                    chosen_rows.append(chosen)
            if not chosen_rows:
                continue
            weights = row_weights([prompts[0].answerable for prompts in chosen_rows])
            if float(weights.sum()) == 0.0:
                # Every row is masked: no gradient, and a step would move the
                # parameters on weight decay alone.
                masked += len(chosen_rows)
                rows_done += len(chosen_rows)
                prompts_done += sum(len(prompts) for prompts in chosen_rows)
                continue
            denominator = weights.sum().clamp_min(1.0).to(readout.device)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for prompts, weight in zip(chosen_rows, weights):
                if args.profile:
                    torch.cuda.synchronize()
                    mark = time.perf_counter()
                raw = readout.row_margins(prompts)
                if args.profile:
                    torch.cuda.synchronize()
                    mid = time.perf_counter()
                margin = scaling(raw)
                labels = torch.tensor(
                    [p.label for p in prompts],
                    dtype=torch.float32,
                    device=margin.device,
                )
                # Mean within the row (row_bce), then one over the number of
                # unmasked rows (denominator).  Backward per row frees the
                # prefix graph before the next row's forward.
                term = row_bce(margin, labels) * weight.to(margin.device) / denominator
                batch_loss += float(term.detach())
                rows_done += 1
                prompts_done += margin.numel()
                if float(weight) == 0.0:
                    # Masked, not dropped: the forward above still ran, but the
                    # row adds no gradient.
                    masked += 1
                    continue
                term.backward()
                if args.profile:
                    torch.cuda.synchronize()
                    after_backward = time.perf_counter()
                    profile_totals["forward_s"] += mid - mark
                    profile_totals["backward_s"] += after_backward - mid
            if args.profile:
                torch.cuda.synchronize()
                step_mark = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                lora_params + list(scaling.parameters()), args.grad_clip
            )
            optimizer.step()
            if args.profile:
                torch.cuda.synchronize()
                profile_totals["step_s"] += time.perf_counter() - step_mark
            running += batch_loss
            batches += 1
            if batches % 20 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  epoch {epoch} rows {rows_done}/{len(order)} "
                    f"loss={running / batches:.4f} "
                    f"{prompts_done / elapsed:.1f} prompts/s "
                    f"{rows_done / elapsed:.2f} rows/s",
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "rows": rows_done,
            "prompts": prompts_done,
            "masked_rows": masked,
            "rows_per_s": rows_done / elapsed if elapsed else None,
            "prompts_per_s": prompts_done / elapsed if elapsed else None,
            "profile": profile_totals if args.profile else None,
            "scale": float(scaling.scale.detach()),
            "bias": float(scaling.bias.detach()),
        }
        eval_metrics = measure(
            readout,
            scaling,
            val_rows or eval_rows,
            args.max_options,
            args.val_rows,
            random.Random(args.seed),
        )
        metrics["eval"] = eval_metrics
        history.append(metrics)
        print(f"  eval {json.dumps(metrics)}", flush=True)

    calibration = None
    if args.calibrate_bias:
        sample = train_rows[: args.calibration_rows]
        raw_parts = []
        label_parts = []
        answerable = []
        with torch.no_grad():
            for _, prompts in sample:
                chosen = subsample_prompts(
                    prompts, args.max_options, random.Random(args.seed)
                )
                if not readout.fits(chosen):
                    continue
                raw_parts.append(scaling(readout.row_margins(chosen)))
                label_parts.append(
                    torch.tensor(
                        [p.label for p in chosen],
                        dtype=torch.float32,
                        device=readout.device,
                    )
                )
                answerable.append(bool(chosen[0].answerable))
        if raw_parts:
            offsets = [0]
            for margins in raw_parts:
                offsets.append(offsets[-1] + margins.numel())
            row_index = [
                torch.arange(offsets[i], offsets[i + 1], device=readout.device)
                for i in range(len(raw_parts))
            ]
            labels = torch.cat(label_parts)
            weights = row_weights(answerable)
            target = positive_base_rate(labels, row_index, weights)
            offset = fit_bias(torch.cat(raw_parts), row_index, target)
            with torch.no_grad():
                scaling.bias.add_(float(offset))
            calibration = {"bias_offset": float(offset), "target_mean_yes": target}
            print(f"calibrated {json.dumps(calibration)}", flush=True)

    final = measure(
        readout,
        scaling,
        val_rows or eval_rows or train_rows,
        args.max_options,
        args.val_rows,
        random.Random(args.seed),
    )
    if (
        final["gold_options"]
        and final["gold_positive_yes"] == 0
        and not args.allow_collapse
    ):
        raise SystemExit(
            "no gold option reaches P(yes) > 0.5; the run has collapsed and is "
            "not readable (pass --allow-collapse to keep an unreadable arm)"
        )

    state = {
        "lora": {
            name: value.detach().cpu()
            for name, value in backbone.state_dict().items()
            if "lora_" in name
        },
        "scaling": {
            "scale": float(scaling.scale.detach()),
            "bias": float(scaling.bias.detach()),
        },
        "config": provenance["config"],
        "provenance": provenance,
        "calibration": calibration,
        "final": final,
        "history": history,
    }
    torch.save(state, args.output / "judgement.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {args.output / 'judgement.pt'}", flush=True)
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"max_memory_allocated {peak:.1f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
