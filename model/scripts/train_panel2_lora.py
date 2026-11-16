"""Panel-2 vocabulary readout with LoRA: train, evaluate, and dump logits.

One forward per item: state, question and every option carry one single-token
code in one sequence, and the answer is the argmax over the K code logits at
the final position. Training is a listwise cross-entropy over those K logits
with the gold code as target, which is the same quantity the readout scores.

Rows come from the v3 plan and are split by each record's ``internal_split``;
the splitter is not re-run (#379).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from model.evaluation.panel2 import (
    CODES,
    choose_option,
    code_token_ids,
    has_exact_tie,
    option_lines,
    render_panel2,
)
from model.src.backbone import load_causal_lm
from model.src.lora import inject_lora

POLARITY_MARK = "\nQuestion: "
OPTION_MARK = "\nDoes this option correctly answer the question?"


def read_plan(path: Path, expected: str | None, label: str):
    """One rank's read outcome: bytes, digest, or an error string.

    This function never raises. A rank that exits on its own failure before
    the collective below turns a detected mismatch into a hang on every other
    rank, which is worse than the failure it detects. Every per-rank problem
    -- an unreadable file, a pin too short to identify one -- is returned and
    carried through the gather so that all ranks fail together.

    The bytes returned are the bytes hashed *and* the bytes the caller
    parses: hashing a path and then reopening it would pair one file's digest
    with another file's rows.
    """
    if expected is not None:
        wanted = expected.strip()
        if len(wanted) < 16:
            return None, None, (
                f"--{label}-sha256 {expected!r} is shorter than 16 hex "
                "characters; refusing before reading the file"
            )
    try:
        data = path.read_bytes()
    except OSError as error:
        return None, None, f"{path}: cannot read the plan: {error}"
    return data, hashlib.sha256(data).hexdigest(), None


def agree_plan(data, digest, error, world: int, rank: int):
    """The outcome every rank agrees on, or one error for all of them.

    The gather is the first thing that can fail collectively, and every rank
    reaches it whatever its own read returned. The whole (digest, error) pair
    is gathered rather than a reduced value, so the message names the
    divergent digest instead of only reporting inequality.
    """
    if world == 1:
        return data, digest, error
    gathered: list = [None] * world
    dist.all_gather_object(gathered, (digest, error))
    distinct = sorted({tuple(item) for item in gathered}, key=str)
    if len(distinct) != 1:
        return None, None, f"ranks disagree on the plan digest: {distinct}"
    return data, digest, error


def load_plan_rows(path: Path, expected: str | None, label: str,
                   world: int = 1, rank: int = 0,
                   **reader_options) -> tuple[list[Row], str, int, str]:
    """Read, agree, check and parse one plan, returning rows and digest.

    Order matters: nothing that can raise on one rank alone may run before
    the gather. The pin comparison runs after it, on a digest every rank now
    shares, so a mismatch raises on all ranks at once; the parse runs after
    that, on the bytes that were hashed.
    """
    data, digest, error = read_plan(path, expected, label)
    data, digest, error = agree_plan(data, digest, error, world, rank)
    if error is not None:
        raise SystemExit(error)
    wanted = expected.strip().lower() if expected is not None else None
    if wanted is not None and not digest.startswith(wanted):
        raise SystemExit(
            f"{path}: sha256 {digest} does not match the expected {wanted}; "
            "refusing to read a plan that is not the one this run pinned"
        )
    rows, split_field, masked = build_rows(data, **reader_options)
    return rows, split_field, masked, digest


def parse_prompt_text(text: str) -> tuple[str, str]:
    """State and instructions from a rendered judgement prompt."""
    if not text.startswith("State: "):
        raise ValueError(f"not a judgement prompt: {text[:40]!r}")
    head, sep, rest = text[len("State: "):].rpartition(POLARITY_MARK)
    if not sep:
        raise ValueError(f"prompt has no {POLARITY_MARK!r} marker: {text[:40]!r}")
    instructions = rest.split("\nAnswer:", 1)[0].rstrip("\n")
    return head, instructions


def option_of(suffix: str) -> str:
    assert suffix.startswith("Option: "), suffix[:20]
    body = suffix[len("Option: "):]
    return body.split(OPTION_MARK, 1)[0]


class Row:
    __slots__ = ("row_id", "split", "q_type", "state", "instructions", "texts",
                 "gold", "codes", "source")

    def __init__(self, row_id, split, q_type, state, instructions, texts, gold,
                 codes, source):
        self.row_id = row_id
        self.split = split
        self.q_type = q_type
        self.state = state
        self.instructions = instructions
        self.texts = texts
        self.gold = gold
        self.codes = codes
        self.source = source


def source_class(row_id: str) -> str:
    name = row_id.split(":", 1)[-1]
    return name.split("-", 1)[0] if "-" in name else name


def step_plan(n_rows: int, world: int, batch_rows: int) -> tuple[int, int]:
    """Steps per epoch and rows per rank, identical on every rank.

    Ranks must take the same number of steps. A rank that finishes one step
    early enters the epoch-end statistics all-reduce while its peers are still
    in the per-step gradient all-reduces, so the two pair with each other: the
    statistics come back as gradient sums and the collectives eventually time
    out. ``n_rows`` is the plan's whole train count, never the calling rank's
    shard, so every rank derives the same number from the same input.
    """
    shard_rows = math.ceil(n_rows / world) if world > 1 else n_rows
    return math.ceil(shard_rows / batch_rows), shard_rows


def check_shared_non_negative(value: float, what: str, rank: int, world: int,
                              device: str, step: int | None = None) -> float:
    """A cross-entropy mean cannot be negative; share the verdict, then exit.

    A negative CE is not a bad model, it is a pairing bug: when ranks are one
    collective out of step, an all-reduce returns another collective's
    payload -- a gradient sum, say -- and that number gets printed as if it
    were a loss. One run logged a mean loss of 0.0027 and a frozen CE of
    -0.0018 this way before the NCCL watchdog killed it.

    The verdict is reduced before anyone raises: a rank that exits on a local
    check leaves the others blocked in the next collective, which is the
    hang this file keeps having to avoid.
    """
    failed = torch.tensor(
        [0.0 if math.isfinite(value) and value >= 0.0 else 1.0], device=device)
    if world > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if failed.item() > 0:
        where = f"rank {rank}" + (f" step {step}" if step is not None else "")
        bad_here = not (math.isfinite(value) and value >= 0.0)
        if bad_here:
            detail = f"{what} is {value!r}, which cannot be a cross-entropy mean"
        else:
            # This rank's value is fine; naming it as the offender would send
            # whoever reads the abort to the wrong rank.
            detail = (f"{what} is healthy here ({value:.4f}); the failure is "
                      "on another rank")
        raise SystemExit(
            f"{where}: {detail}; {int(failed.item())} rank(s) failed the "
            "non-negative check"
        )
    return value


def shard_and_steps(train_rows: list, rank: int, world: int,
                    batch_rows: int) -> tuple[list, int]:
    """One rank's rows and the step count every rank shares.

    The step count comes from the plan's total, not from this rank's shard:
    that is the property that keeps the collectives aligned, and the only
    place it is decided.
    """
    if world > 1 and len(train_rows) < world:
        # Fewer rows than ranks leaves some shards empty; an empty shard makes
        # batch_indices divide by zero while the other ranks wait in the next
        # collective. Both inputs are identical on every rank, so every rank
        # refuses here, in the same way, and none is left blocking.
        raise SystemExit(
            f"{len(train_rows)} train rows cannot be spread over {world} ranks; "
            "no rank would have a full shard"
        )
    shard = train_rows[rank :: world] if world > 1 else train_rows
    steps, _ = step_plan(len(train_rows), world, batch_rows)
    return shard, steps


def batch_indices(step: int, batch_rows: int, shard_rows: int) -> list[int]:
    """Shard positions for one step; a short shard wraps its tail.

    When the plan's train count is not divisible by the world size some
    shards hold one row fewer. Wrapping repeats a few early rows on those
    ranks instead of shortening their epoch, which would desynchronise the
    collectives.
    """
    start = (step - 1) * batch_rows
    return [index % shard_rows for index in range(start, start + batch_rows)]


def reduce_gradients(params, world: int, bucket_bytes: int = 64 << 20) -> int:
    """All-reduce gradients in flat buckets; returns the collective count.

    Two shapes are wrong at 4B parameters. Concatenating every gradient first
    needs an 8 GB cat plus an 8 GB destination per step, and one all-reduce
    per parameter is hundreds of latency-bound calls per step. Bucketing keeps
    one flat buffer per group of parameters of about ``bucket_bytes`` and one
    collective per bucket.

    Parameters are grouped by dtype as well as by size: a flat buffer cannot
    mix bf16 and fp32, and the LoRA factors are fp32 while the weights are
    bf16.
    """
    if world <= 1:
        return 0
    by_dtype: dict = {}
    for param in params:
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        by_dtype.setdefault(param.grad.dtype, []).append(param)

    collectives = 0
    for dtype, group in by_dtype.items():
        bucket: list = []
        size = 0
        for param in group:
            width = param.grad.numel() * param.grad.element_size()
            if bucket and size + width > bucket_bytes:
                _reduce_bucket(bucket, world)
                collectives += 1
                bucket, size = [], 0
            bucket.append(param)
            size += width
        if bucket:
            _reduce_bucket(bucket, world)
            collectives += 1
    return collectives


def _reduce_bucket(bucket, world: int) -> None:
    grads = [param.grad for param in bucket]
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset: offset + count].view_as(grad))
        offset += count


def split_by_token_budget(rows, token_counts, budget: int) -> list:
    """Chunks of rows whose summed prompt tokens stay at or under ``budget``.

    The memory model was measured at 8 rows of 486-token prefixes; a batch of
    rows whose prefixes are longer would exceed it. Splitting the step by the
    tokens it actually carries keeps the per-step peak bounded without
    changing the rows per step, which would change the optimisation.
    """
    if not budget:
        return [rows] if rows else []
    chunks, current, total = [], [], 0
    for row in rows:
        width = token_counts[id(row)]
        if current and total + width > budget:
            chunks.append(current)
            current, total = [], 0
        current.append(row)
        total += width
    if current:
        chunks.append(current)
    return chunks


def cap_rows(rows, token_counts, cap: int) -> tuple[list, int, int]:
    """Rows at or under ``cap`` tokens, how many were skipped, the longest skip.

    A row is one forward, so a single row above the memory the device can hold
    kills the step whatever the batch budget says. The cap is a backstop and
    it is counted rather than silent: the caller reports the number and the
    longest skipped length.
    """
    kept, skipped, longest = [], 0, 0
    for row in rows:
        width = token_counts[id(row)]
        if width > cap:
            skipped += 1
            longest = max(longest, width)
        else:
            kept.append(row)
    return kept, skipped, longest


def build_rows(data: bytes, noul_gold: str = "position",
               plan_unpermuted_noul: bool = False) -> tuple[list[Row], str, int]:
    """Rows from plan bytes, the field that supplied their split, and masks.

    ``data`` is the buffer that was hashed, not a path: the reader must not be
    able to see bytes that the digest check did not cover.

    ``internal_split`` is the v3 field; the v2 plans predate it and carry only
    ``split``. The reader records which one it used, requires every record to
    agree, and raises rather than counting a row without a split as train.

    ``noul_gold`` says how a criteria-bearing noul row names its gold side.
    ``position`` assumes option 0 carries the true wording and is refused unless
    the caller also asserts ``plan_unpermuted_noul``, because a permuted plan
    would invert every noul gold silently.  ``role`` reads the option's polarity
    marker and survives permutation; it is for a plan built through
    ``build_prompts``, which writes that field.

    Rows the plan marks ``answerable: false`` carry no gold and are excluded
    from the returned rows and counted, the way the judgement path masked them
    rather than dropping the file.
    """
    if noul_gold not in ("role", "position"):
        raise ValueError(f"noul_gold must be 'role' or 'position', not {noul_gold!r}")
    if noul_gold == "position" and not plan_unpermuted_noul:
        raise ValueError(
            "noul_gold='position' assumes the plan left criteria-bearing noul rows "
            "unpermuted; pass plan_unpermuted_noul=True to state that contract"
        )
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    split_field: str | None = None
    masked = 0
    # BytesIO keeps the one copy of the buffer that was hashed; the reader
    # never opens the path, so it cannot see a file the digest did not cover.
    for line in io.BufferedReader(io.BytesIO(data)):
        if not line.strip():
            continue
        record = json.loads(line)
        row_id = record["row_id"]
        field = ("internal_split" if "internal_split" in record
                 else "split" if "split" in record else None)
        if field is None:
            raise ValueError(f"record {row_id!r} carries neither split field")
        if split_field is None:
            split_field = field
        elif field != split_field:
            raise ValueError(
                f"plan mixes {split_field!r} and {field!r} split fields"
            )
        if row_id not in grouped:
            grouped[row_id] = []
            order.append(row_id)
        grouped[row_id].append(record)

    rows: list[Row] = []
    for row_id in order:
        records = sorted(grouped[row_id], key=lambda r: r["option_index"])
        first = records[0]
        if first.get("answerable") is False:
            # No gold to train or score against; the plan keeps these rows so
            # the file matches the corpus, not so they can be learnt.
            masked += 1
            continue
        q_type = first["q_type"]
        if len(records) == 1 and q_type == "noul":
            state, instructions = parse_prompt_text(first["suffix"])
            texts, gold = ["no", "yes"], 1 if first["label"] == 1 else 0
        else:
            state, instructions = parse_prompt_text(first["prefix"])
            if q_type == "noul":
                if len(records) == 1:
                    texts, gold = ["no", "yes"], 1 if first["label"] == 1 else 0
                else:
                    gold_record = [r for r in records if r["label"] == 1]
                    if len(gold_record) != 1:
                        raise ValueError(
                            f"noul row {row_id!r} has {len(gold_record)} gold options"
                        )
                    gold_record = gold_record[0]
                    if noul_gold == "role":
                        role = (gold_record.get("polarity")
                                or gold_record.get("option_role"))
                        if role not in ("true", "false"):
                            raise ValueError(
                                f"noul row {row_id!r} has no polarity marker on "
                                "its gold option; use --noul-gold position only for "
                                "an unpermuted plan"
                            )
                        texts, gold = ["no", "yes"], 1 if role == "true" else 0
                    else:
                        texts = ["no", "yes"]
                        gold = 1 if gold_record["option_index"] == 0 else 0
            else:
                texts = [option_of(r["suffix"]) for r in records]
                golds = [i for i, r in enumerate(records) if r["label"] == 1]
                if len(golds) != 1:
                    raise ValueError(
                        f"row {row_id!r} has {len(golds)} options labelled 1"
                    )
                gold = golds[0]
        codes = list(CODES[: len(texts)])
        split = first.get("internal_split") or first.get("split") or "train"
        rows.append(Row(row_id, split, q_type, state,
                        instructions, texts, gold, codes, source_class(row_id)))
    return rows, split_field, masked


def prompt_of(row: Row) -> str:
    return render_panel2(row.state, row.instructions, row.texts, row.codes)


def code_tensor(model, tokenizer, prompt, codes, code_ids, device):
    """The K code logits at the final position, differentiable."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded)
    last = output.logits[0, -1].float()
    return torch.stack([last[code_ids[code]] for code in codes])


def code_logits(model, tokenizer, prompt: str, code_ids, device) -> list[float]:
    codes = list(code_ids)
    with torch.inference_mode():
        values = code_tensor(model, tokenizer, prompt, codes, code_ids, device)
    return [float(v) for v in values]


def evaluate(model, tokenizer, rows, code_ids, device, limit=None):
    """Accuracy and calibration bins per row, natural-order index chosen.

    The bins are keyed by question type and option count and hold
    ``[rows, correct]`` per confidence decile. They are counts, so shards add:
    the report sums them across ranks and computes ECE once. Calibration is
    recorded here because a full fine-tune leaves no adapter behind, and a
    number computed after the fact would need the weights.
    """
    model.eval()
    counts = {}
    per_item = []
    bins = {}
    with torch.inference_mode():
        for row in (rows[:limit] if limit else rows):
            values = code_logits(model, tokenizer, prompt_of(row),
                                 {c: code_ids[c] for c in row.codes}, device)
            chosen = choose_option(values, list(range(len(values))))
            ok = chosen == row.gold
            confidence = float(torch.softmax(
                torch.tensor(values, dtype=torch.float32), dim=0).max())
            decile = min(9, int(confidence * 10))
            cell = bins.setdefault(f"{row.q_type}|{len(values)}", {})
            slot = cell.setdefault(str(decile), [0, 0])
            slot[0] += 1
            slot[1] += int(ok)
            for key in ("all", f"type:{row.q_type}", f"source:{row.source}"):
                cell = counts.setdefault(key, [0, 0])
                cell[0] += int(ok)
                cell[1] += 1
            per_item.append([row.row_id, int(ok), row.q_type, row.source])
    model.train()
    return counts, per_item, bins


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--plan-sha256",
        default=None,
        help=(
            "abort unless the plan hashes to this value (prefix of at least 16 "
            "hex characters); the computed digest is recorded either way"
        ),
    )
    parser.add_argument(
        "--eval-plan",
        type=Path,
        default=None,
        help="score this plan's val rows instead of the training plan's",
    )
    parser.add_argument(
        "--eval-plan-sha256",
        default=None,
        help=(
            "abort unless the eval plan hashes to this value (prefix of at "
            "least 16 hex characters); the computed digest is recorded either way"
        ),
    )
    parser.add_argument(
        "--noul-gold",
        choices=["role", "position"],
        default="position",
        help=(
            "how a criteria-bearing noul row names its gold side: role reads the "
            "option's polarity marker and survives permutation; position assumes "
            "option 0 is the true wording and is valid only for an unpermuted plan"
        ),
    )
    parser.add_argument(
        "--plan-unpermuted-noul",
        action="store_true",
        help=(
            "assert that the plan leaves criteria-bearing noul rows in their "
            "original true/false order; required with --noul-gold position"
        ),
    )
    parser.add_argument(
        "--stop-on-loss-rise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="compute device; defaults to cuda:<local_rank>",
    )
    parser.add_argument(
        "--full-ft",
        action="store_true",
        help="train every parameter with no LoRA adapter",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=0,
        help=(
            "split each step's rows into chunks whose summed prompt tokens stay "
            "under this budget; 0 keeps the row batch whole"
        ),
    )
    parser.add_argument(
        "--grad-bucket-mb",
        type=int,
        default=64,
        help="size of the flat buffer per gradient all-reduce",
    )
    parser.add_argument(
        "--max-row-tokens",
        type=int,
        default=None,
        help=(
            "skip a train row whose prompt exceeds this many tokens; rows are "
            "one forward each, so the longest row sets the peak. Defaults to "
            "4096 under --full-ft and 0 (no cap) otherwise"
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "recompute activations in backward; needed for full fine-tuning, "
            "where one long row alone exceeds the device memory budget"
        ),
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.0,
        help="linear warmup over this fraction of the first epoch's steps",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-layers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--frozen-loss-sample", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.max_row_tokens is None:
        # A cap is only a full-fine-tuning backstop: it would silently change
        # the LoRA ladder's row set, which is already on record.
        args.max_row_tokens = 4096 if args.full_ft else 0

    if args.world > 1:
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.rank)))
    device = args.device or f"cuda:{local_rank}"
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
    torch.manual_seed(42)

    # Pin the inputs before reading them: a swapped plan must stop the run at
    # startup, not train a distribution nobody can name. The bytes hashed are
    # the bytes parsed, and every rank must agree on the digest before any of
    # them trains.
    rows, split_field, masked, plan_sha = load_plan_rows(
        args.plan, args.plan_sha256, "plan",
        world=args.world, rank=args.rank,
        noul_gold=args.noul_gold, plan_unpermuted_noul=args.plan_unpermuted_noul,
    )
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    eval_plan_sha = None
    if args.eval_plan is not None:
        eval_rows, eval_split_field, _, eval_plan_sha = load_plan_rows(
            args.eval_plan, args.eval_plan_sha256, "eval-plan",
            world=args.world, rank=args.rank,
            noul_gold=args.noul_gold,
            plan_unpermuted_noul=args.plan_unpermuted_noul,
        )
        val_rows = [r for r in eval_rows if r.split == "val"]
        print(f"rank {args.rank}: eval plan val rows {len(val_rows)} "
              f"split_field {eval_split_field} sha {eval_plan_sha[:16]}", flush=True)
    print(f"rank {args.rank}: rows {len(rows)} train {len(train_rows)} "
          f"val {len(val_rows)} masked {masked} split_field {split_field} "
          f"plan_sha {plan_sha[:16]}", flush=True)

    model, tokenizer = load_causal_lm(
        "Qwen/Qwen3-Reranker-4B", device=device, dtype=torch.bfloat16,
        freeze=not args.full_ft, flash_attention=False,
    )
    if not args.full_ft:
        inject_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                    rank=args.lora_rank, alpha=args.lora_alpha,
                    last_n_layers=args.lora_layers)
    if args.gradient_checkpointing:
        # Rows are processed one at a time and their graphs are freed at
        # backward, so peak memory is one row's stored activations. With full
        # fine-tuning that is more than the device holds at these prompt
        # lengths; recomputing them in backward trades about a third of the
        # step time for the difference.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    code_ids, multi = code_token_ids(tokenizer, CODES, "space")
    if len(code_ids) < 26 or multi:
        raise SystemExit(f"code ids unusable: {len(code_ids)} single-token, {multi[:5]}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"rank {args.rank}: trainable {sum(p.numel() for p in trainable)}", flush=True)

    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def loss_of(row):
        values = code_tensor(model, tokenizer, prompt_of(row), row.codes,
                             code_ids, device)
        return F.cross_entropy(values.unsqueeze(0),
                               torch.tensor([row.gold], device=device))

    # Frozen reference loss for the stop rule, on a fixed sample.
    frozen_mean = None
    if not args.eval_only:
        sample = train_rows[: args.frozen_loss_sample]
        frozen_losses = []
        for row in sample:
            values = code_logits(model, tokenizer, prompt_of(row),
                                 {c: code_ids[c] for c in row.codes}, device)
            logits = torch.tensor(values, dtype=torch.float32, device=device)
            frozen_losses.append(float(F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([row.gold], device=device))))
        frozen_mean = check_shared_non_negative(
            sum(frozen_losses) / len(frozen_losses), "frozen sample CE",
            args.rank, args.world, device)
        print(f"rank {args.rank}: frozen sample CE {frozen_mean:.4f}", flush=True)

    history = []
    reduce_seconds = 0.0
    skipped_over_cap = 0
    token_counts: dict = {}
    stopped = False
    if not args.eval_only:
        if args.max_row_tokens:
            # The cap is a backstop, and it is counted: a row longer than the
            # cap is its own forward and would set the peak by itself. The
            # whole train set is measured on every rank so the retained count,
            # and therefore the step count, is identical everywhere.
            for row in train_rows:
                token_counts[id(row)] = len(tokenizer(
                    prompt_of(row), add_special_tokens=False)["input_ids"])
            train_rows, skipped_over_cap, longest = cap_rows(
                train_rows, token_counts, args.max_row_tokens)
            print(f"rank {args.rank}: max_row_tokens {args.max_row_tokens} "
                  f"skips {skipped_over_cap} of {len(train_rows) + skipped_over_cap} "
                  f"train rows (longest skipped {longest} tokens)", flush=True)
            if not train_rows:
                raise SystemExit("every train row is over --max-row-tokens")
        shard, steps = shard_and_steps(train_rows, args.rank, args.world,
                                       args.batch_rows)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        print(f"rank {args.rank}: {len(shard)} rows, {steps} steps per epoch, "
              f"{args.epochs} epoch(s)", flush=True)
        if args.max_batch_tokens:
            for row in shard:
                if id(row) not in token_counts:
                    token_counts[id(row)] = len(tokenizer(
                        prompt_of(row), add_special_tokens=False)["input_ids"])
            print(f"rank {args.rank}: token budget {args.max_batch_tokens} per "
                  f"step, longest row {max(token_counts.values(), default=0)}",
                  flush=True)
        warmup_steps = int(args.warmup_fraction * steps) if args.warmup_fraction else 0
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            for step in range(1, steps + 1):
                batch = [shard[index]
                         for index in batch_indices(step, args.batch_rows, len(shard))]
                if epoch == 1 and warmup_steps and step <= warmup_steps:
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * step / warmup_steps
                optimizer.zero_grad(set_to_none=True)
                losses = []
                chunks = split_by_token_budget(batch, token_counts,
                                               args.max_batch_tokens)
                for chunk in chunks:
                    for row in chunk:
                        loss = loss_of(row) / len(batch)
                        loss.backward()
                        losses.append(float(loss))
                if args.world > 1:
                    reduce_started = time.perf_counter()
                    reduce_gradients(trainable, args.world,
                                     args.grad_bucket_mb << 20)
                    reduce_seconds += time.perf_counter() - reduce_started
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                mean = check_shared_non_negative(
                    sum(losses), "mean loss", args.rank, args.world, device,
                    step=step)
                history.append(mean)
                if step % args.log_every == 0 or step == steps:
                    print(f"rank {args.rank}: epoch {epoch} step {step}/{steps} "
                          f"loss {mean:.4f} elapsed {time.time() - started:.0f}s",
                          flush=True)
            epoch_mean = sum(history[-steps:]) / steps
            if args.world > 1:
                # The stop decision must be identical on every rank, or the ones
                # that keep going hang on the next collective.
                stats = torch.tensor([epoch_mean, frozen_mean], device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                stats = stats / args.world
                epoch_mean, frozen_mean = float(stats[0]), float(stats[1])
            # Both values are already group-wide, so the check is local: the
            # reduce above returned the same numbers on every rank, and a
            # negative one means it paired with another collective.
            epoch_mean = check_shared_non_negative(
                epoch_mean, "epoch mean loss", args.rank, 1, device)
            frozen_mean = check_shared_non_negative(
                frozen_mean, "frozen sample CE", args.rank, 1, device)
            print(f"rank {args.rank}: epoch {epoch} mean loss {epoch_mean:.4f} "
                  f"frozen {frozen_mean:.4f}"
                  + (f" collective {reduce_seconds / (steps * epoch):.3f}s/step"
                     if args.world > 1 else ""), flush=True)
            if epoch_mean > frozen_mean and args.stop_on_loss_rise:
                print(f"rank {args.rank}: stop rule fired (loss above frozen)",
                      flush=True)
                stopped = True
                break

    # Validation on the rank's stride of the val split.
    val_source = "val"
    val_shard = val_rows[args.rank :: args.world] if args.world > 1 else val_rows
    if not val_shard:
        # A plan without an internal split (the v2 authoritative plan) has no
        # val rows; an eval-only smoke then uses the head of the train rows.
        # The payload says so, so the number cannot be read as an estimate.
        val_source = "train_head"
        val_shard = train_rows[args.rank :: args.world]
    if args.val_limit:
        val_shard = val_shard[: args.val_limit]
    counts, per_item, ece_bins = evaluate(model, tokenizer, val_shard,
                                          code_ids, device)
    payload = {"rank": args.rank, "lr": args.lr, "val_counts": counts,
               "val_items": per_item, "ece_bins": ece_bins,
               "val_source": val_source,
               "stopped": stopped,
               "stop_reason": "epoch_mean_loss_above_frozen" if stopped else None,
               "plan": str(args.plan), "plan_sha256": plan_sha,
               "plan_sha256_expected": args.plan_sha256,
               "eval_plan": str(args.eval_plan) if args.eval_plan else None,
               "eval_plan_sha256": eval_plan_sha,
               "eval_plan_sha256_expected": args.eval_plan_sha256,
               "full_ft": args.full_ft, "grad_bucket_mb": args.grad_bucket_mb,
               "max_batch_tokens": args.max_batch_tokens,
               "warmup_fraction": args.warmup_fraction,
               "gradient_checkpointing": args.gradient_checkpointing,
               "max_row_tokens": args.max_row_tokens,
               "skipped_over_cap_rows": skipped_over_cap,
               "lora": None if args.full_ft else {
                   "rank": args.lora_rank, "alpha": args.lora_alpha,
                   "layers": args.lora_layers,
                   "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
               "split_field": split_field, "noul_gold": args.noul_gold,
               "plan_unpermuted_noul": args.plan_unpermuted_noul,
               "masked_rows": masked, "epochs": args.epochs,
               "epoch_mean_loss": (sum(history) / len(history)) if history else None,
               "collective_seconds_per_step": (
                   reduce_seconds / (steps * args.epochs) if history else None),
               "max_memory_gib": (torch.cuda.max_memory_allocated() / 2**30
                                  if device.startswith("cuda") else None),
               "frozen_sample_ce": frozen_mean}

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"shard_{args.rank}.json"
    if stopped:
        # A fired stop rule ends the point, but the numbers it produced are
        # still the evidence for stopping: write them before leaving.
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        print(f"rank {args.rank}: wrote {path} (stopped)", flush=True)
        if args.world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return 3

    if args.rank == 0:
        jev = evaluate_jevbench(model, tokenizer, code_ids, device)
        payload["jevbench"] = jev

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"rank {args.rank}: wrote {path}", flush=True)
    if args.rank == 0:
        torch.save({"lora": {k: v.detach().cpu() for k, v in model.state_dict().items()
                             if "lora" in k}}, args.output / "lora.pt")
    if args.world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def evaluate_jevbench(model, tokenizer, code_ids, device):
    """The 231 public items, natural and reversed option order, per-item logits."""
    from data.pipeline import load_jevbench
    from model.evaluation.panel2 import options_for

    tiers = {}
    items = []
    for item in load_jevbench():
        prepared = options_for(item)
        if prepared is None:
            continue
        texts, gold = prepared
        items.append((item, texts, gold))
    out = []
    with torch.inference_mode():
        for item, texts, gold in items:
            for order in ("natural", "reversed"):
                if order == "natural":
                    shown = list(texts)
                    display_to_option = list(range(len(texts)))
                else:
                    shown = list(reversed(texts))
                    display_to_option = list(range(len(texts)))[::-1]
                codes = list(CODES[: len(shown)])
                prompt = render_panel2(item.state, item.question.get("instructions", ""),
                                       shown, codes)
                values = code_logits(model, tokenizer, prompt,
                                     {c: code_ids[c] for c in codes}, device)
                by_option = [0.0] * len(shown)
                for position, option in enumerate(display_to_option):
                    by_option[option] = values[position]
                chosen = choose_option(values, display_to_option)
                tier = item.id.split("-", 1)[0]
                cell = tiers.setdefault(tier, [0, 0])
                cell[0] += int(chosen == gold and order == "natural")
                if order == "natural":
                    cell[1] += 1
                out.append({"id": item.id, "tier": tier,
                            "q_type": item.question.get("type", "noul"),
                            "k": len(shown), "gold": gold, "order": order,
                            "values_by_option": by_option, "chosen": chosen,
                            "correct": chosen == gold,
                            "tie": has_exact_tie(values)})
    return {"items": out, "by_tier_natural": tiers}


if __name__ == "__main__":
    raise SystemExit(main())
