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
from model.src.shape_buckets import PREFIX_BUCKETS, bucket_length  # noqa: E402
from model.training import distributed  # noqa: E402
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


def draw_subset(rows, count: int, seed: int):
    """Draw ``count`` rows once, deterministically, keeping plan order.

    The draw is a seeded shuffle of the row indices; the kept indices are then
    sorted, so the epoch walks the subset in the same order it walks the full
    split.  A prefix or a stride would take whatever the plan happens to put
    first, which is a source-composition bias rather than a sample.
    """
    if count >= len(rows):
        return list(rows)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    keep = sorted(order[:count])
    return [rows[index] for index in keep]


def row_selection_sha256(rows: list[tuple[str, list[JudgementPrompt]]]) -> str:
    """Stable hash of the training rows: id, option count, positive index."""
    lines = []
    for row_id, prompts in rows:
        positives = [prompt.option_index for prompt in prompts if prompt.label == 1]
        lines.append(f"{row_id}\t{len(prompts)}\t{positives!r}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def trainable_rows(rows, readout: JudgementReadout, max_row_tokens: int | None = None):
    """Drop rows the readout cannot score, reporting why and their lengths.

    A row over ``max_length`` is dropped, not truncated: a truncated state is a
    different question.  A tokenisation-drift error is not caught here; it
    propagates, because it means training and inference disagree.

    ``max_row_tokens`` is a memory guard: a row longer than it is skipped and
    counted, so one pathological row cannot take a rank to the ceiling.  It is
    never below ``max_length`` in practice, which means it skips nothing unless
    the caller sets it deliberately.

    Returns the kept rows, the long-row count, the guard-skipped count, and
    ``row_id -> longest prompt tokens``.  The lengths come from the same
    tokenisation the filter needs, so they cost nothing extra and they are
    exact, which the chunk budget relies on.
    """
    kept = []
    lengths: dict[str, int] = {}
    dropped_long = 0
    skipped_over = 0
    limit = readout.max_length if max_row_tokens is None else max_row_tokens
    block_rows = 256
    for start in range(0, len(rows), block_rows):
        block = rows[start : start + block_rows]
        row_lengths = readout.lengths([prompts for _, prompts in block])
        for (row_id, prompts), length in zip(block, row_lengths):
            if length > readout.max_length:
                dropped_long += 1
                continue
            if length > limit:
                skipped_over += 1
                continue
            kept.append((row_id, prompts))
            lengths[row_id] = length
    return kept, dropped_long, skipped_over, lengths


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
    parser.add_argument(
        "--mem-guard-gib",
        type=float,
        default=88.0,
        help=(
            "stop the run cleanly when a rank's allocated memory passes this, "
            "instead of letting the allocator abort and the collective time out"
        ),
    )
    parser.add_argument(
        "--max-row-tokens",
        type=int,
        default=None,
        help=(
            "memory guard: train rows longer than this many prompt tokens are "
            "skipped and counted; default is max_length, which skips none"
        ),
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=8192,
        help=(
            "prefix-token budget of one forward/backward chunk inside an "
            "optimizer step; bounds graph memory so batch_rows stays a "
            "throughput knob"
        ),
    )
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
    parser.add_argument(
        "--subset-rows",
        type=int,
        default=None,
        help="draw this many train rows once with --subset-seed, in plan order",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="seed of the subset draw; defaults to --seed",
    )
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
        "--log-every", type=int, default=20, help="Optimizer steps between log lines"
    )
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Rank 0 writes {step, loss, rows} per optimizer step to steps.jsonl",
    )
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


def measure_sums(
    readout: JudgementReadout,
    scaling: ScaledMargin,
    rows,
    max_options: int,
    limit: int,
    rng: random.Random,
) -> dict:
    """Loss and P(yes) sums on up to ``limit`` rows, for cross-rank reduction."""
    selected = rows[:limit]
    loss_sum = 0.0
    unmasked_rows = 0
    rows_scored = 0
    gold_positive = 0
    gold_total = 0
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
            loss_sum += float(per_row) * weight
            unmasked_rows += int(weight)
            probability = ScaledMargin.probability(margin)
            gold = labels == 1
            gold_total += int(gold.sum())
            gold_positive += int((probability[gold] > 0.5).sum())
            rows_scored += 1
    return {
        "rows": rows_scored,
        "loss_sum": loss_sum,
        "unmasked_rows": unmasked_rows,
        "gold_positive_yes": gold_positive,
        "gold_options": gold_total,
    }


def reduce_sums(sums: dict, world_size: int, device) -> dict:
    """Sum the measure counters across ranks."""
    if world_size <= 1:
        return sums
    keys = ("rows", "loss_sum", "unmasked_rows", "gold_positive_yes", "gold_options")
    values = torch.tensor([float(sums[key]) for key in keys], device=device)
    torch.distributed.all_reduce(values)
    return {
        key: (float(value) if key == "loss_sum" else int(value))
        for key, value in zip(keys, values.tolist())
    }


def measure(
    readout: JudgementReadout,
    scaling: ScaledMargin,
    rows,
    max_options: int,
    limit: int,
    rng: random.Random,
    world_size: int = 1,
) -> dict:
    """Mean loss and P(yes) statistics over the global row set."""
    sums = reduce_sums(
        measure_sums(readout, scaling, rows, max_options, limit, rng),
        world_size,
        readout.device,
    )
    unmasked = sums["unmasked_rows"]
    return {
        "rows": sums["rows"],
        "loss": sums["loss_sum"] / unmasked if unmasked else None,
        "gold_positive_yes": sums["gold_positive_yes"],
        "gold_options": sums["gold_options"],
    }


def reduce_step(loss: float, rows: int, device, world_size: int) -> tuple[float, int]:
    """Mean of the per-rank step losses, and the global row count."""
    if world_size <= 1:
        return loss, rows
    values = torch.tensor([loss, 1.0, float(rows)], device=device)
    torch.distributed.all_reduce(values)
    return float(values[0] / values[1]), int(values[2])


def step_chunks(rows, token_budget: int, lengths: dict[str, int] | None = None):
    """Index ranges of rows whose prefixes together fit ``token_budget``.

    A step's graph memory is set by the padded prefix tokens of one forward,
    not by the optimizer batch, so a chunk budget lets batch_rows stay a
    throughput knob.  Each row costs its *padded* prefix bucket, not its raw
    length: the readout pads every row of a bucket group to the same bucket, so
    a raw-length budget underestimates memory by up to two bucket steps.  A row
    whose own bucket exceeds the budget forms its own chunk.

    ``lengths`` maps ``row_id`` to the measured prompt-token length, from
    :func:`trainable_rows`; without it the length is estimated at four
    characters per token, which is wrong by more than 2x on non-Latin text.
    """
    chunks: list[tuple[int, int]] = []
    start = 0
    total = 0
    for index, prompts in enumerate(rows):
        measured = lengths.get(prompts[0].row_id) if lengths else None
        tokens = (
            measured
            if measured is not None
            else max(len(prompt.prefix) for prompt in prompts) // 4
        )
        cost = bucket_length(tokens, PREFIX_BUCKETS) + 16
        if index > start and total + cost > token_budget:
            chunks.append((start, index))
            start, total = index, 0
        total += cost
    if start < len(rows):
        chunks.append((start, len(rows)))
    return chunks


def _batch_shard(rows, rank: int, world_size: int, batch_rows: int):
    """This rank's contiguous slice of every global batch.

    The global batch is ``batch_rows * world_size`` rows of one identically
    shuffled order.  Rank ``r`` takes positions ``[r*batch_rows, (r+1)*batch_rows)``
    of each block, so the union of the ranks' steps is exactly the single-GPU
    batch of the same size and the loss curves are comparable.
    """
    if world_size <= 1 or not rows:
        return list(rows)
    width = batch_rows * world_size
    out = []
    for start in range(0, len(rows), width):
        block = list(rows[start : start + width])
        while len(block) < width:
            block.append(block[-1])
        out.extend(block[rank * batch_rows : (rank + 1) * batch_rows])
    return out


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args()
    config = load_config(known.config)
    args = build_parser(config).parse_args()
    rank, world_size = distributed.init_process_group()
    if args.torch_seed is not None:
        # Every replica starts from the same weights; only the data shard
        # differs.  Seeding per rank would train different adapters.
        torch.manual_seed(args.torch_seed)
    try:
        return _run(args, rank, world_size)
    finally:
        distributed.shutdown()


def _run(args, rank: int, world_size: int) -> int:
    main_process = rank == 0
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
    if args.subset_rows:
        subset_seed = args.subset_seed if args.subset_seed is not None else args.seed
        train_rows = draw_subset(train_rows, args.subset_rows, subset_seed)
        if main_process:
            print(f"subset rows {len(train_rows)} seed {subset_seed}", flush=True)
    if not train_rows:
        raise SystemExit("the plan has no train rows")
    if main_process:
        print(f"split {json.dumps(split_report)}", flush=True)
    provenance = {
        "code_commit": args.code_commit or _git_commit(),
        "plan_sha256": _sha256(args.plan),
        "corpus_sha256": args.corpus_sha256,
        "train_selection_sha256": row_selection_sha256(train_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "eval_rows": len(eval_rows),
        "subset_rows": args.subset_rows,
        "subset_seed": (
            args.subset_seed if args.subset_seed is not None else args.seed
        ),
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
            "world_size": world_size,
            "lr": args.lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
        },
    }
    if main_process:
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
        if main_process:
            print(f"adapted {adapted} linear layers", flush=True)
    readout = JudgementReadout(backbone, tokenizer, max_length=args.max_length)
    scaling = ScaledMargin(scale=args.scale_init, bias=args.yes_bias).to(readout.device)

    train_rows, dropped_long, skipped_over, row_token_lengths = trainable_rows(
        train_rows, readout, args.max_row_tokens
    )
    if dropped_long and main_process:
        print(f"dropped {dropped_long} train rows over max_length", flush=True)
    if skipped_over and main_process:
        print(
            f"skipped {skipped_over} train rows over max_row_tokens "
            f"({args.max_row_tokens})",
            flush=True,
        )
    if not train_rows:
        raise SystemExit("no train row fits max_length")
    rows_per_rank = _batch_shard(train_rows, rank, world_size, args.batch_rows)
    print(
        f"rank {rank}/{world_size}: {len(rows_per_rank)} rows per epoch",
        flush=True,
    )

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    opt_params = lora_params + list(scaling.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": list(scaling.parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    if main_process:
        print(
            f"trainable backbone {sum(p.numel() for p in lora_params):,} "
            f"scaling {sum(p.numel() for p in scaling.parameters())}",
            flush=True,
        )

    step_log = (
        (args.output / "steps.jsonl").open("w", encoding="utf-8")
        if main_process and args.log_steps
        else None
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        order = list(train_rows)
        # One identical global order on every rank; then each rank takes its
        # slice of every global batch.
        rng = random.Random(args.seed + epoch)
        rng.shuffle(order)
        order = _batch_shard(order, rank, world_size, args.batch_rows)
        started = time.perf_counter()
        rows_done = prompts_done = masked = 0
        running = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_rows):
            chosen_rows = []
            for _, prompts in order[start : start + args.batch_rows]:
                chosen = subsample_prompts(prompts, args.max_options, rng)
                if readout.fits(chosen):
                    chosen_rows.append(chosen)
            if not chosen_rows:
                # A shard can pad to an empty batch only if every row was
                # filtered; keep the collective cadence anyway.
                weights = torch.zeros(0)
            else:
                weights = row_weights(
                    [prompts[0].answerable for prompts in chosen_rows]
                )
            all_masked = len(chosen_rows) == 0 or float(weights.sum()) == 0.0
            if all_masked:
                masked += len(chosen_rows)
                rows_done += len(chosen_rows)
                prompts_done += sum(len(prompts) for prompts in chosen_rows)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            if not all_masked:
                denominator = weights.sum().clamp_min(1.0).to(readout.device)
                # One prefix forward and one suffix forward cover the whole
                # step, and one backward covers the step's rows.  The per-row
                # path launched a forward and a backward per row and left the
                # card idle between them (#299 throughput).
                for start, stop in step_chunks(
                    chosen_rows, args.chunk_tokens, row_token_lengths
                ):
                    chunk = chosen_rows[start:stop]
                    chunk_weights = weights[start:stop]
                    chunk_terms = []
                    for prompts, weight, raw in zip(
                        chunk, chunk_weights, readout.step_margins(chunk)
                    ):
                        margin = scaling(raw)
                        labels = torch.tensor(
                            [p.label for p in prompts],
                            dtype=torch.float32,
                            device=margin.device,
                        )
                        # Mean within the row (row_bce), then one over the
                        # step's unmasked rows (denominator).
                        term = (
                            row_bce(margin, labels) * weight.to(margin.device) / denominator
                        )
                        batch_loss += float(term.detach())
                        rows_done += 1
                        prompts_done += margin.numel()
                        if float(weight) == 0.0:
                            # Masked, not dropped: the forward above still ran,
                            # but the row adds no gradient.
                            masked += 1
                            continue
                        chunk_terms.append(term)
                    if chunk_terms:
                        # One backward per chunk bounds the graph memory by the
                        # chunk, not the optimizer batch, so rows per step stay
                        # a throughput knob.
                        torch.stack(chunk_terms).sum().backward()
            # Data-parallel: every replica must reach the collective, so this
            # runs even when a rank contributed no gradient.
            distributed.average_gradients(opt_params, world_size)
            torch.nn.utils.clip_grad_norm_(opt_params, args.grad_clip)
            optimizer.step()
            global_loss, global_rows = reduce_step(
                batch_loss, len(chosen_rows), readout.device, world_size
            )
            running += global_loss
            batches += 1
            if step_log is not None:
                step_log.write(
                    json.dumps(
                        {"step": batches, "loss": global_loss, "rows": global_rows}
                    )
                    + "\n"
                )
                step_log.flush()
            if batches % args.log_every == 0:
                allocated = (
                    torch.cuda.memory_allocated() / 2**30
                    if torch.cuda.is_available()
                    else 0.0
                )
                reserved = (
                    torch.cuda.memory_reserved() / 2**30
                    if torch.cuda.is_available()
                    else 0.0
                )
                peak_reserved = (
                    torch.cuda.max_memory_reserved() / 2**30
                    if torch.cuda.is_available()
                    else 0.0
                )
                if main_process:
                    elapsed = time.perf_counter() - started
                    print(
                        f"  epoch {epoch} step {batches} "
                        f"loss={running / batches:.4f} "
                        f"{global_rows * batches / elapsed:.2f} rows/s "
                        f"mem={allocated:.1f} res={reserved:.1f} "
                        f"maxres={peak_reserved:.1f} GiB",
                        flush=True,
                    )
                elif torch.cuda.is_available():
                    print(
                        f"  rank {rank} step {batches} mem={allocated:.1f} "
                        f"res={reserved:.1f} maxres={peak_reserved:.1f} GiB",
                        flush=True,
                    )
                if allocated > args.mem_guard_gib:
                    print(
                        f"rank {rank} hit the memory guard: {allocated:.1f} GiB "
                        f"> {args.mem_guard_gib} GiB at step {batches}; stopping "
                        "before the allocator does",
                        flush=True,
                    )
                    raise SystemExit(3)
        elapsed = time.perf_counter() - started
        expected_steps = distributed.steps_for(len(rows_per_rank), args.batch_rows)
        if batches != expected_steps:
            raise RuntimeError(
                f"rank {rank} took {batches} steps, expected {expected_steps}; "
                "the collective sequence is not aligned"
            )
        counts = torch.tensor(
            [rows_done, prompts_done, masked],
            dtype=torch.float64,
            device=readout.device,
        )
        if world_size > 1:
            torch.distributed.all_reduce(counts)
        global_counts = [int(value) for value in counts.tolist()]
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(batches, 1),
            "rows": global_counts[0],
            "trainable_rows_per_rank": len(rows_per_rank),
            "prompts": global_counts[1],
            "masked_rows": global_counts[2],
            "skipped_over_tokens": skipped_over,
            "rows_per_s": global_counts[0] / elapsed if elapsed else None,
            "world_size": world_size,
            "scale": float(scaling.scale.detach()),
            "bias": float(scaling.bias.detach()),
        }
        eval_pool = val_rows or eval_rows
        if world_size > 1:
            eval_pool = eval_pool[rank::world_size]
            eval_limit = max(1, args.val_rows // world_size)
        else:
            eval_limit = args.val_rows
        metrics["eval"] = measure(
            readout,
            scaling,
            eval_pool,
            args.max_options,
            eval_limit,
            random.Random(args.seed),
            world_size,
        )
        history.append(metrics)
        if main_process:
            print(f"  eval {json.dumps(metrics)}", flush=True)

    calibration = None
    if args.calibrate_bias:
        offset_value = torch.zeros((), device=readout.device)
        if main_process:
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
                offset_value.fill_(float(offset))
                calibration = {
                    "bias_offset": float(offset),
                    "target_mean_yes": target,
                }
        if world_size > 1:
            torch.distributed.broadcast(offset_value, src=0)
        if args.calibrate_bias:
            # Every rank applies the broadcast offset; rank 0 is the only one
            # with a calibration dict, so only it prints.
            with torch.no_grad():
                scaling.bias.add_(float(offset_value))
            if main_process:
                print(f"calibrated {json.dumps(calibration)}", flush=True)

    final_pool = val_rows or eval_rows or train_rows
    if world_size > 1:
        final_pool = final_pool[rank::world_size]
        final_limit = max(1, args.val_rows // world_size)
    else:
        final_limit = args.val_rows
    final = measure(
        readout,
        scaling,
        final_pool,
        args.max_options,
        final_limit,
        random.Random(args.seed),
        world_size,
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

    if main_process:
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
    if step_log is not None:
        step_log.close()
    distributed.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
