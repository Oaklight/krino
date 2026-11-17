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
from model.training.losses import rps_loss

POLARITY_MARK = "\nQuestion: "
OPTION_MARK = "\nDoes this option correctly answer the question?"

# The population the calibration bins and every ECE cell are defined on:
# every val row whose source is not synthetic. That is wider than "benchmark":
# it includes typed_decisions and apigen rows, and it is the 9,923-row
# comparator the LoRA arms' ECE was scored on. The name states the rule, and
# the payload repeats the rule verbatim so no reader has to guess the coverage.
BIN_POPULATION = "non_synthetic"
BIN_POPULATION_RULE = 'source != "synthetic"'


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    """The digest of a file, read in blocks so an 8 GB checkpoint is not held.

    A full fine-tune's weights are the arm: without their digest recorded in
    the payload, a later scoring pass cannot say which weights produced the
    cells it is attached to.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()



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

    A trainable parameter with no gradient gets zeros, so a rank that skipped
    it contributes nothing to the sum and the mean stays right. Under full
    fine-tuning every parameter takes a gradient in every step that uses it,
    so the flat path and the per-parameter path agree; the zero row only
    appears for a parameter that did not participate, and there its
    contribution is zero either way.
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


CHUNK_LENGTH_RATIO = 1.25


def deal_rows(rows, token_counts, world: int) -> list:
    """Deal a step's rows to ranks by token count, longest first.

    The step trains the same rows whatever the placement; only which rank
    holds them changes. Longest-first onto the least-loaded rank leaves the
    per-rank token sums within one row of each other, which is the skew the
    collective otherwise waits out. The order is total (token count, then row
    id), so every rank derives the same partition without communicating it.
    """
    if world <= 1:
        return [list(rows)]
    ordered = sorted(rows, key=lambda row: (-token_counts[id(row)], row.row_id))
    loads = [[] for _ in range(world)]
    totals = [0] * world
    for row in ordered:
        target = min(range(world), key=lambda rank: (totals[rank], rank))
        loads[target].append(row)
        totals[target] += token_counts[id(row)]
    return loads


def chunk_by_length(rows, token_counts, budget: int,
                    ratio: float = CHUNK_LENGTH_RATIO,
                    max_rows: int = 0, decisions=None) -> list:
    """Chunks of similar-length rows, each within ``budget`` prompt tokens.

    A forward pads every row to the chunk's longest, so a 30-token row in a
    chunk with a 3,000-token row pays 3,000 tokens of compute. Admitting a row
    only while it is at least ``1 / ratio`` of the chunk's longest keeps that
    waste near the ratio instead of near the plan's length spread. Row order
    does not enter the loss.

    ``max_rows`` caps the rows in a chunk independently of the token budget,
    which is the knob a cost fit needs: it fixes the forwards per rank while
    the rows' tokens stay where they were. ``ratio`` must exceed 1 to split at
    all; at 1.0 the test can never fire, so 1.0 leaves only the budget and the
    row cap to decide the chunks. When ``decisions`` is a list, the constraint
    behind every split is appended to it, so a run can report which knob
    actually bound rather than which was configured.
    """
    ordered = sorted(rows, key=lambda row: (-token_counts[id(row)], row.row_id))
    chunks, current, total, longest = [], [], 0, 0
    for row in ordered:
        width = token_counts[id(row)]
        over_budget = total + width > budget
        too_short = ratio > 1.0 and width * ratio < longest
        too_many = max_rows and len(current) >= max_rows
        if current and (over_budget or too_short or too_many):
            if decisions is not None:
                if over_budget:
                    decisions.append("chunk_tokens")
                if too_short:
                    decisions.append("ratio")
                if too_many:
                    decisions.append("chunk_rows")
            chunks.append(current)
            current, total, longest = [], 0, 0
        current.append(row)
        total += width
        longest = max(longest, width)
    if current:
        chunks.append(current)
    return chunks


CHUNK_DECISIONS = ("chunk_tokens", "ratio", "chunk_rows")


def chunk_decision_label(decisions) -> str:
    """The constraint that bound this step, as one of the closed enum.

    A step can hit more than one at different splits; the label is the one
    that fired most often, ties broken in ``CHUNK_DECISIONS`` order. With no
    split at all the budget never bound, which is its own label.
    """
    if not decisions:
        return "budget_non_binding"
    return max(CHUNK_DECISIONS,
               key=lambda name: (decisions.count(name),
                                 -CHUNK_DECISIONS.index(name)))


def chunk_cost(chunk, token_counts) -> tuple[int, int]:
    """Real and padded prompt tokens one chunk's forward carries."""
    real = sum(token_counts[id(row)] for row in chunk)
    return real, len(chunk) * max(token_counts[id(row)] for row in chunk)


BUCKET_WIDTHS = (512, 1024, 2048, 4096)
# Steady-state cost model for balancing: one padded forward costs an intercept
# plus a term per padded token plus a quadratic term. The coefficients are a
# least-squares fit to 199 measured padded forwards of the a20k twenty-step
# bucket run at 8 rows per rank (four forwards above 1.5x their bucket mean
# excluded; rms residual 0.026 s over forwards from 0.45 s to 5.4 s). A fit
# with no quadratic term is biased by width: -0.21 s at 2048 and +0.13 s at
# 4096, and its intercept goes negative, which is the model saying the form is
# wrong. The quadratic is the attention and checkpoint-recompute cost that
# grows with width; it is why a 4096 forward costs 9.8x a 512 one, not 8x.
BUCKET_CALL_SECONDS = 0.0219
BUCKET_TOKEN_SECONDS = 0.0001134
BUCKET_TOKEN_SQUARED_SECONDS = 1.358e-9


def bucket_width(width: int, buckets=BUCKET_WIDTHS):
    """The smallest fixed bucket that holds a row of ``width`` tokens."""
    for size in buckets:
        if width <= size:
            return size
    return None


def bucket_rows(rows, token_counts, buckets=BUCKET_WIDTHS) -> list:
    """Rows grouped by bucket, in fixed width order.

    The widths are fixed rather than derived from the plan: a shape is warm
    only if it has been seen, so a stable set of widths is what makes every
    step after the first pay warm costs. Returns ``(width, rows)`` pairs for
    the non-empty buckets.
    """
    grouped = {size: [] for size in buckets}
    for row in rows:
        size = bucket_width(token_counts[id(row)], buckets)
        if size is None:
            raise ValueError(
                f"row {row.row_id} needs {token_counts[id(row)]} tokens, above "
                f"the largest bucket {max(buckets)}")
        grouped[size].append(row)
    return [(size, grouped[size]) for size in buckets if grouped[size]]


def bucket_cost(groups, batch_rows: int) -> float:
    """Steady-state seconds for a rank holding these bucket groups.

    Each forward is padded to ``batch_rows`` rows at the bucket's width, so a
    bucket with fewer than a full batch still pays a full batch's tokens. That
    is deliberate: the batch dimension is part of the shape, and a shape is
    warm only if its size does not vary. The quadratic term is per forward,
    not on the rank's summed padded tokens: two warm forwards cost twice one,
    and squaring the sum would charge a rank for forwards it does not make.
    """
    total = 0.0
    for width, rows in groups:
        forwards = -(-len(rows) // batch_rows)
        padded = batch_rows * width
        total += forwards * (BUCKET_CALL_SECONDS
                             + padded * BUCKET_TOKEN_SECONDS
                             + padded * padded * BUCKET_TOKEN_SQUARED_SECONDS)
    return total


def deal_rows_by_bucket_cost(rows, token_counts, world: int, batch_rows: int,
                             buckets=BUCKET_WIDTHS) -> list:
    """Deal rows to ranks to level the steady-state bucket cost.

    Longest first, each row to the rank whose cost rises least; opening a
    bucket on a rank costs its call term, so the deal also avoids spreading a
    step's rows over more buckets than it must. Deterministic: ties break on
    rank index and rows are ordered by (tokens, row id).
    """
    if world <= 1:
        return [list(rows)]
    loads = [[] for _ in range(world)]
    ordered = sorted(rows, key=lambda row: (-token_counts[id(row)], row.row_id))
    for row in ordered:
        best_rank, best_cost = 0, None
        for rank in range(world):
            candidate = loads[rank] + [row]
            cost = bucket_cost(bucket_rows(candidate, token_counts, buckets),
                               batch_rows)
            if best_cost is None or cost < best_cost:
                best_rank, best_cost = rank, cost
        loads[best_rank].append(row)
    return loads


def chunk_inputs(tokenizer, rows, device, width=None, pad_rows=None):
    """Padded ids, attention mask and per-row position ids for one chunk.

    Rows are separate batch elements, so no row can attend to another; the
    batch dimension, not a mask, is what separates them. The rows are padded
    to the chunk's longest and the mask excludes the pads. Position ids are
    built from the mask rather than from the padding's side, so a row's first
    token is position 0 whether the padding sits before or after it.

    ``width`` and ``pad_rows`` fix the two shape dimensions: the ids are padded
    to that width and filled out with masked dummy rows to that batch size, so
    a bucket's forward always presents the same shape and stays warm.
    """
    prompts = [prompt_of(row) for row in rows]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention = encoded["attention_mask"]
    target_width = max(input_ids.shape[1], width or 0)
    target_rows = max(input_ids.shape[0], pad_rows or 0)
    if (target_width, target_rows) != tuple(input_ids.shape):
        ids = torch.zeros((target_rows, target_width), dtype=input_ids.dtype,
                          device=input_ids.device)
        mask = torch.zeros((target_rows, target_width), dtype=attention.dtype,
                           device=attention.device)
        ids[:input_ids.shape[0], :input_ids.shape[1]] = input_ids
        mask[:attention.shape[0], :attention.shape[1]] = attention
        input_ids, attention = ids, mask
    input_ids = input_ids.to(device)
    attention = attention.to(device)
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    return input_ids, attention, positions


def chunk_code_logits(model, tokenizer, rows, code_ids, device, encoded=None):
    """One vector of K code logits per row, from one forward over the chunk."""
    if encoded is None:
        encoded = chunk_inputs(tokenizer, rows, device)
    input_ids, attention, positions = encoded
    logits = model(input_ids=input_ids, attention_mask=attention,
                   position_ids=positions).logits
    if attention.shape[0] != len(rows):
        # A bucket pads the batch with dummy rows to fix the shape; only the
        # real rows' last positions are read.
        attention = attention[:len(rows)]
    last = attention.sum(-1) - 1
    picked = logits[torch.arange(len(rows), device=device), last].float()
    return [torch.stack([picked[index][code_ids[code]] for code in row.codes])
            for index, row in enumerate(rows)]


def step_union(step, train_rows, world, batch_rows) -> list:
    """The step's rows: the ranks' strided batches, wrapped per shard.

    Building from ``batch_indices`` per shard keeps the wrapped last step the
    same rows the strided loop would train; a global slice wraps differently.
    """
    union = []
    for other in range(world):
        other_shard = train_rows[other::world] if world > 1 else train_rows
        union.extend(other_shard[index] for index in
                     batch_indices(step, batch_rows, len(other_shard)))
    return union


def balanced_step_batch(step, train_rows, rank, world, batch_rows,
                        token_counts):
    """The rows this rank trains at ``step`` under the balanced rule."""
    return deal_rows(step_union(step, train_rows, world, batch_rows),
                     token_counts, world)[rank]


def chunk_loss(chunk, values, row_loss_of, nominal_rows: int):
    """A chunk's contribution to the step: its summed row losses, scaled once.

    The scale is the step's nominal rows per rank, not the chunk's size: a
    chunk of two rows and a chunk of five carry their own row counts, so the
    step's gradient does not depend on how the rows were grouped. This is the
    property the gradient-equality test pins, and it is why the divisor is an
    argument rather than ``len(chunk)``.
    """
    total = None
    for row, value in zip(chunk, values):
        term = row_loss_of(row, value)
        total = term if total is None else total + term
    return total / nominal_rows


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


def encode_prompt(tokenizer, prompt, device):
    """One prompt's ids, moved to the device; the step's tokenisation cost."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


def code_tensor(model, tokenizer, prompt, codes, code_ids, device,
                encoded=None):
    """The K code logits at the final position, differentiable.

    ``encoded`` lets a caller that has already timed the tokenisation pass the
    ids in, so a phase profile can separate the two.
    """
    if encoded is None:
        encoded = encode_prompt(tokenizer, prompt, device)
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

    Only non-synthetic rows enter the bins. The eval plan's val split is
    41,232 scored rows, 31,309 of them synthetic (the plan's 41,871 val rows
    carry 639 rows with no gold; 41,232 = 31,309 + 9,923), and the LoRA arms'
    ECE comes from a later scoring pass over the same 9,923 non-synthetic
    rows; binning every row here would put the two families on different
    populations, and the admission gate compares them. Accuracy still counts
    every row.

    The fourth return value is the bins' composition, counted at the same
    place the bin slot is incremented. A composition recomputed from the bin
    totals afterwards would be a second spelling of one sum and could not
    disagree with it; this one is the evidence that the filter admitted the
    families it was meant to.
    """
    model.eval()
    counts = {}
    per_item = []
    bins = {}
    bin_sources = {"by_source": {}, "by_type": {}}
    with torch.inference_mode():
        for row in (rows[:limit] if limit else rows):
            values = code_logits(model, tokenizer, prompt_of(row),
                                 {c: code_ids[c] for c in row.codes}, device)
            chosen = choose_option(values, list(range(len(values))))
            ok = chosen == row.gold
            confidence = float(torch.softmax(
                torch.tensor(values, dtype=torch.float32), dim=0).max())
            decile = min(9, int(confidence * 10))
            if row.source != "synthetic":
                cell = bins.setdefault(f"{row.q_type}|{len(values)}", {})
                slot = cell.setdefault(str(decile), [0, 0])
                slot[0] += 1
                slot[1] += int(ok)
                bin_sources["by_source"][row.source] = (
                    bin_sources["by_source"].get(row.source, 0) + 1)
                bin_sources["by_type"][row.q_type] = (
                    bin_sources["by_type"].get(row.q_type, 0) + 1)
            for key in ("all", f"type:{row.q_type}", f"source:{row.source}"):
                cell = counts.setdefault(key, [0, 0])
                cell[0] += int(ok)
                cell[1] += 1
            per_item.append([row.row_id, int(ok), row.q_type, row.source])
    model.train()
    return counts, per_item, bins, bin_sources


def target_loss(values, gold: int, mode: str, tau: float = 1.0):
    """One row's training loss over its K code logits.

    ``onehot`` is the shipped listwise cross-entropy: every level but the gold
    is treated as equally wrong. ``soft`` gives the adjacent level
    ``exp(-1)`` of the gold's weight and two levels away ``exp(-2)``, so a
    near miss costs less; it cannot reach an error further than that.
    ``expected`` penalises the distance between the gold level and the
    distribution's mean level linearly, so a far miss costs more than a near
    one. The decode stays argmax in every mode: the target is a training
    change, not a decode change.
    """
    if mode == "onehot":
        return F.cross_entropy(values.unsqueeze(0),
                               torch.tensor([gold], device=values.device))
    index = torch.arange(len(values), dtype=torch.float32, device=values.device)
    if mode == "expected":
        probabilities = torch.softmax(values, dim=0)
        return (probabilities * index).sum().sub(float(gold)).abs()
    if mode == "soft":
        target = torch.exp(-(index - float(gold)).abs() / tau)
        target = target / target.sum()
        return -(target * torch.log_softmax(values, dim=0)).sum()
    raise ValueError(f"unknown score target {mode!r}")


def score_loss(values, gold: int, loss: str, cramer_weight: float | None,
               target_mode: str = "onehot", tau: float = 1.0):
    """One score row's loss under ``--score-loss``, so the arms differ only there.

    ``ce`` is target_loss unchanged, which is what every earlier arm ran.
    ``rps`` is the ranked probability score against the gold's one-hot
    distribution: it sums the squared CDF differences over the level
    thresholds, i.e. the squared Cramer distance on an ordered categorical,
    normalised by K - 1. ``mix`` is ``ce + cramer_weight * rps``.

    The two terms have different scales -- CE is unbounded, RPS is at most 1
    -- so a mix weight is a weight change as much as a shape change. The
    caller passes a measured lambda; the trainer refuses to guess one.
    """
    if loss == "ce":
        return target_loss(values, gold, target_mode, tau)
    target = F.one_hot(torch.tensor(gold, device=values.device),
                       values.numel()).float()
    rps = rps_loss(values.unsqueeze(0), target.unsqueeze(0))
    if loss == "rps":
        return rps
    return target_loss(values, gold, target_mode, tau) + cramer_weight * rps


def reweighted_mean(block, composition):
    """A per-type mean block reweighted to one pinned composition.

    ``block`` is ``{type: [sum, count]}``. Both the frozen reference and the
    epoch mean are reweighted to the same ``composition``, taken from the
    plan's train rows, so the comparison does not depend on which rows a shard
    or a 64-row sample happened to contain. Returns the reweighted mean (None
    if a weighted type is absent) and the missing types.
    """
    total = 0.0
    missing = []
    for kind, fraction in composition.items():
        entry = block.get(kind)
        if entry and entry[1]:
            total += fraction * (entry[0] / entry[1])
        elif fraction:
            missing.append(kind)
    return (None if missing else total), missing


def per_type_means(block):
    return {kind: {"rows": int(round(entry[1])),
                   "mean": (entry[0] / entry[1]) if entry[1] else None}
            for kind, entry in block.items()}


def add_mask_tokens(type_tokens, chunk, encoded):
    """Add each row's real token count, from the mask the model was given.

    The mask is the honest measure: it excludes the pads chunk_inputs adds and
    it follows a wrapped step's repeated rows, which a count keyed by row id
    could not. Reads the same tensor the forward used, so no row is tokenised
    twice.
    """
    _, mask, _ = encoded
    for row, count in zip(chunk, mask.sum(dim=1).tolist()):
        type_tokens[row.q_type] = type_tokens.get(row.q_type, 0) + int(count)


def step_mix_text(step_rows, step_tokens, type_order):
    """The step's own mix, per type: rows trained on and token share.

    The order is the union of the plan's types and the step's. A step row's
    type is drawn from the plan today, so the union is defensive for the
    record's totals; naming it here keeps the promise in one place, and the
    share is this rank's, which is what the label on the line says.
    """
    total = sum(step_tokens.values())
    order = sorted(set(type_order) | set(step_rows) | set(step_tokens))
    shares = {kind: (100.0 * step_tokens.get(kind, 0) / total if total else 0.0)
              for kind in order}
    return " ".join(f"{kind} {step_rows.get(kind, 0)}r {shares[kind]:.1f}%"
                    for kind in order)


def abort_if_any_failed(failed, what: str, world: int) -> None:
    """Share a 0/1 failure flag, then abort every rank together.

    The expensive work a checkpoint and the final payload do is local -- rank
    0's JevBench pass and the 8.8 GB write -- so an exception there would
    leave the other ranks in the next gradient all-reduce or at the final
    barrier with no traceback: a silent stall instead of a clean abort. The
    reduce is the one collective every rank reaches, so the raise happens
    after it, on all of them.
    """
    if world > 1:
        dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if failed.item() > 0:
        raise SystemExit(
            f"{what}: {int(failed.item())} rank(s) failed; aborting the group "
            "through the collective instead of leaving the others waiting in "
            "the next all-reduce"
        )


def synchronise(device: str) -> None:
    """Wait for queued device work.

    A phase timer without this measures kernel launches, not the work: CUDA
    returns from a call as soon as it is queued.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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
        "--balance-rows",
        action="store_true",
        help=(
            "deal each step's rows across ranks by prompt length instead of "
            "each rank taking its own stride. The step trains the same rows; "
            "the change is which rank holds them, so the busiest rank costs "
            "closer to the mean and the collective waits less"
        ),
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=0,
        help=(
            "forward rows together in chunks of similar length, up to this many "
            "prompt tokens and within --chunk-ratio of the chunk's longest row; "
            "0 with --chunk-rows 0 keeps one row per forward. Padding to the "
            "longest row is masked out of the loss and counted in the payload"
        ),
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=0,
        help=(
            "cap the rows in one forward independently of --chunk-tokens; the "
            "knob a cost fit uses to fix forwards per rank at a fixed token load"
        ),
    )
    parser.add_argument(
        "--chunk-ratio",
        type=float,
        default=CHUNK_LENGTH_RATIO,
        help=(
            "a row joins a chunk only while it is at least 1/ratio of the "
            "chunk's longest; 1.0 disables the ratio split and leaves the token "
            "budget and --chunk-rows to decide the chunks"
        ),
    )
    parser.add_argument(
        "--bucket-tokens",
        default="",
        help=(
            "comma-separated fixed widths, e.g. 512,1024,2048,4096: rows are "
            "grouped into the smallest width that holds them, padded up to it "
            "and forwarded together with the batch padded to --batch-rows, so "
            "the shapes are fixed and stay warm after the first appearance"
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
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help=(
            "on rank 0, save the full final state dict to weights.pt and record "
            "its sha256 and size in the payload; a full fine-tune leaves no "
            "adapter, so without this the arm cannot be re-scored or shipped"
        ),
    )
    parser.add_argument(
        "--load-weights",
        type=Path,
        default=None,
        help=(
            "load a full state dict saved by --save-weights before evaluating; "
            "use with --eval-only to score a saved arm"
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help=(
            "score the non-synthetic val rows and the 231 items every N steps "
            "and write "
            "checkpoint_<step>_shard_<rank>.json; with --save-weights the "
            "weights at that step are saved too, so a longer run produces a "
            "curve and its best point can be kept rather than the last"
        ),
    )
    parser.add_argument(
        "--score-target",
        choices=["onehot", "soft", "expected"],
        default="onehot",
        help=(
            "training target for score rows only: onehot is the shipped "
            "listwise cross-entropy, soft weights adjacent levels by distance, "
            "expected penalises the distance between the gold and the mean "
            "level; choice and noul rows always keep onehot, and the decode "
            "stays argmax"
        ),
    )
    parser.add_argument(
        "--score-target-tau",
        type=float,
        default=1.0,
        help="decay of the soft target, in levels; pre-registered at 1.0",
    )
    parser.add_argument(
        "--score-loss",
        choices=["ce", "mix", "rps"],
        default="ce",
        help=(
            "loss on the score rows: ce is the shipped target dispatch "
            "(--score-target), rps is the ranked probability score alone, "
            "mix is ce + cramer_weight * rps; noul and choice rows keep "
            "one-hot CE in every mode, so only the score head moves"
        ),
    )
    parser.add_argument(
        "--score-cramer-weight",
        type=float,
        default=None,
        help=(
            "lambda of the rps term under --score-loss mix. Measure it with "
            "model/scripts/measure_score_cramer.py on the pinned plan's "
            "non-synthetic val score rows at step 0 and pin the value it "
            "prints. lambda = 0.25 * the measured CE/RPS ratio, so the rps "
            "term is 0.25 * the CE term: 20% of the score-row loss. Required "
            "for mix, must be finite and greater than zero, and refused for "
            "ce and rps"
        ),
    )
    parser.add_argument(
        "--score-cramer-weight-source",
        type=Path,
        default=None,
        help=(
            "the lambda probe JSON that measured --score-cramer-weight; its "
            "digest and the measured mean CE, mean RPS, ratio and row count "
            "are copied into the payload, and the pinned weight is checked "
            "against it. Without this the payload says the weight was passed "
            "by hand, so no artifact claims a measurement it does not have"
        ),
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help=(
            "global step the loaded weights already completed; the loop "
            "continues at start_step + 1 and the warmup is read against the "
            "global step, so a resumed leg does not warm up again. Requires "
            "--load-weights"
        ),
    )
    parser.add_argument(
        "--phase-profile",
        type=int,
        default=0,
        help=(
            "time tokenise, forward, backward, all-reduce, clip and optimizer "
            "for the first N steps and write phase_profile_rank<N>.json; "
            "the phases are synchronised, so the numbers include device work"
        ),
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
    args.bucket_list = tuple(
        int(value) for value in args.bucket_tokens.split(",") if value.strip()
    ) if args.bucket_tokens else ()
    if args.bucket_list and (args.chunk_tokens or args.chunk_rows):
        raise SystemExit(
            "--bucket-tokens sets the grouping and padding; drop "
            "--chunk-tokens and --chunk-rows when it is set"
        )
    if args.bucket_list and args.balance_rows:
        # The bucket deal is itself a balanced deal, and it runs first; the
        # payload would then report balanced_by_length for a deal it did not
        # run. Two balancers need one of them named.
        raise SystemExit(
            "--bucket-tokens deals the rows by bucket cost; drop "
            "--balance-rows when it is set"
        )
    if args.bucket_list and any(
            later <= earlier for earlier, later
            in zip(args.bucket_list, args.bucket_list[1:])):
        raise SystemExit("--bucket-tokens must be strictly increasing")
    if args.start_step and args.load_weights is None:
        # A resumed leg that starts its schedule at step 1 but claims to be at
        # step 543 would warm up again and overwrite the step accounting.
        raise SystemExit(
            "--start-step needs --load-weights: a resumed leg must load the "
            "weights it continues from"
        )
    if args.start_step and args.epochs > 1:
        # --start-step names steps of one pass. With a second pass it cannot
        # say whether the later epochs start at 1 or at start_step + 1, so the
        # combination is refused rather than recorded as one of the readings.
        raise SystemExit(
            "--start-step with --epochs > 1 is ambiguous: it can only skip "
            "steps in the first epoch; resume a single-epoch leg instead"
        )
    if args.score_loss == "mix" and args.score_cramer_weight is None:
        # CE is unbounded and RPS is at most 1, so an unmeasured weight would
        # rescale score supervision and the arm would be a weight change
        # wearing a loss-shape label.
        raise SystemExit(
            "--score-loss mix needs --score-cramer-weight, measured as "
            "0.25 * mean CE / mean RPS on the score rows; it is not guessed"
        )
    if args.score_loss != "mix" and args.score_cramer_weight is not None:
        raise SystemExit(
            f"--score-cramer-weight has no effect under --score-loss "
            f"{args.score_loss}; set it only with mix"
        )
    if (args.score_loss == "mix" and args.score_cramer_weight is not None
            and not (math.isfinite(args.score_cramer_weight)
                     and args.score_cramer_weight > 0)):
        # A zero weight is a ce arm wearing a mix label, and a negative one
        # rewards the inverse of the ranked score.
        raise SystemExit(
            "--score-cramer-weight must be finite and greater than zero under "
            "mix; a zero or negative weight is not a mix arm"
        )
    if args.score_loss != "ce" and args.score_target != "onehot":
        # RPS takes the gold's one-hot distribution. A soft or expected target
        # alongside it is a third axis nobody has pre-registered, so refuse
        # rather than silently drop one of the two flags.
        raise SystemExit(
            f"--score-loss {args.score_loss} uses the gold's one-hot target; "
            f"--score-target {args.score_target} was not pre-registered with it"
        )
    cramer_provenance = None
    if args.score_loss == "mix":
        if args.score_cramer_weight_source is None:
            # Allowed, but the payload must not imply a measurement that no
            # artifact backs.
            cramer_provenance = {
                "measured": False,
                "reason": "no --score-cramer-weight-source: the weight was "
                          "passed by hand and no probe artifact supports it",
            }
        else:
            if not args.score_cramer_weight_source.exists():
                raise SystemExit(
                    f"--score-cramer-weight-source does not exist: "
                    f"{args.score_cramer_weight_source}"
                )
            # One read: the digest and the numbers must describe the same
            # bytes, or `measured: true` pairs one version's hash with
            # another version's means.
            probe_bytes = args.score_cramer_weight_source.read_bytes()
            probe = json.loads(probe_bytes)
            overall = probe.get("overall") or {}
            for key in ("mean_ce", "mean_rps", "ce_over_rps", "lambda", "n"):
                if key not in overall:
                    raise SystemExit(
                        f"--score-cramer-weight-source has no overall.{key}; "
                        "it is not a lambda probe output"
                    )
            factor = probe.get("factor")
            if factor is None:
                raise SystemExit(
                    "--score-cramer-weight-source records no factor; it is "
                    "what relates its ratio to its lambda"
                )
            if not (isinstance(factor, (int, float)) and 0 < factor):
                raise SystemExit(
                    f"--score-cramer-weight-source has factor {factor!r}; "
                    "it must be finite and greater than zero"
                )
            if overall["mean_rps"] <= 0:
                raise SystemExit(
                    "the probe's mean RPS is "
                    f"{overall['mean_rps']}; the CE/RPS ratio and lambda are "
                    "undefined, so the weight was not measured"
                )

            def agrees(left, right):
                return abs(left - right) <= 1e-6 * max(
                    1.0, abs(left), abs(right))

            # The probe's own arithmetic, so a file whose means, ratio and
            # lambda do not relate cannot be stamped measured.
            if not agrees(overall["ce_over_rps"],
                          overall["mean_ce"] / overall["mean_rps"]):
                raise SystemExit(
                    "--score-cramer-weight-source is internally inconsistent: "
                    f"ce_over_rps {overall['ce_over_rps']} is not "
                    f"mean_ce/mean_rps {overall['mean_ce'] / overall['mean_rps']}"
                )
            if not agrees(overall["lambda"], factor * overall["ce_over_rps"]):
                raise SystemExit(
                    "--score-cramer-weight-source is internally inconsistent: "
                    f"lambda {overall['lambda']} is not factor {factor} * "
                    f"ce_over_rps {overall['ce_over_rps']}"
                )
            if not agrees(overall["lambda"], args.score_cramer_weight):
                raise SystemExit(
                    f"--score-cramer-weight {args.score_cramer_weight} is not "
                    f"the measured lambda {overall['lambda']}; the pinned "
                    "number must come from the probe"
                )
            cramer_provenance = {
                "measured": True,
                "source": str(args.score_cramer_weight_source),
                "source_sha256": hashlib.sha256(probe_bytes).hexdigest(),
                "plan": probe.get("plan"),
                "weights": probe.get("weights"),
                "rows": overall["n"],
                "factor": factor,
                "mean_ce": overall["mean_ce"],
                "mean_rps": overall["mean_rps"],
                "ce_over_rps": overall["ce_over_rps"],
                "lambda": overall["lambda"],
                "by_k": probe.get("by_k"),
                "by_gold": probe.get("by_gold"),
                "share_note": "the RPS term's share of the score-row loss at "
                              "the overall lambda is lambda*mean_rps/"
                              "(mean_ce+lambda*mean_rps), reported per block "
                              "as rps_term_fraction_at_overall_lambda. It is "
                              "heterogeneous by gold position, a property of "
                              "mixing CE with a bounded proper scoring rule; "
                              "each block's value at its own lambda is 0.2 by "
                              "construction and is not a measurement",
            }
    elif args.score_cramer_weight_source is not None:
        raise SystemExit(
            "--score-cramer-weight-source has no effect under --score-loss "
            f"{args.score_loss}; set it only with mix"
        )
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
    if cramer_provenance and cramer_provenance.get("measured") and val_rows:
        # The probe and the trainer must mean the same population, or the
        # measured lambda was scaled on a different row set than the primary
        # endpoint reads. Checked after the eval plan, which is the val split
        # the trainer scores.
        expected_rows = sum(1 for row in val_rows
                            if row.q_type == "score"
                            and row.source != "synthetic")
        if cramer_provenance["rows"] != expected_rows:
            raise SystemExit(
                f"the probe measured {cramer_provenance['rows']} rows but the "
                f"eval plan has {expected_rows} non-synthetic val score rows"
            )
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

    weights_loaded_sha = None
    if args.load_weights is not None:
        # Strict load: a partial match would score a model nobody saved.
        # Only weights are restored. AdamW starts cold, so a resumed leg is
        # not a continuation of the first leg's optimizer state; the payload
        # records optimizer_state: fresh rather than claiming a continuation.
        state = torch.load(args.load_weights, map_location=device,
                           weights_only=True)
        model.load_state_dict(state, strict=True)
        weights_loaded_sha = file_sha256(args.load_weights)
        print(f"rank {args.rank}: loaded weights {args.load_weights} "
              f"sha256 {weights_loaded_sha[:16]}", flush=True)


    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    prewarm_seconds = 0.0
    prewarm_shapes: list = []
    # Prompt tokens by row id, filled lazily: the prewarm needs one to pick a
    # row that fits each width, and the batched paths need all of them.
    token_counts: dict = {}
    if not args.eval_only:
        if (args.max_row_tokens or args.balance_rows or args.chunk_tokens
                or args.chunk_rows or args.bucket_list):
            # The cap is a backstop, and it is counted: a row longer than the
            # cap is its own forward and would set the peak by itself. The
            # whole train set is measured on every rank so the retained count,
            # and therefore the step count, is identical everywhere. The
            # length-balanced rule and the chunker need the same counts even
            # when no cap is set.
            for row in train_rows:
                token_counts[id(row)] = len(tokenizer(
                    prompt_of(row), add_special_tokens=False)["input_ids"])
        if args.max_row_tokens:
            train_rows, skipped_over_cap, longest = cap_rows(
                train_rows, token_counts, args.max_row_tokens)
            print(f"rank {args.rank}: max_row_tokens {args.max_row_tokens} "
                  f"skips {skipped_over_cap} of {len(train_rows) + skipped_over_cap} "
                  f"train rows (longest skipped {longest} tokens)", flush=True)
            if not train_rows:
                raise SystemExit("every train row is over --max-row-tokens")
        if args.bucket_list:
            # bucket_rows raises inside the step loop, which would take the
            # group down by timeout one rank at a time. The longest retained
            # row is known here, so the refusal happens before the loop.
            longest_kept = max(token_counts.values(), default=0)
            if longest_kept > max(args.bucket_list):
                raise SystemExit(
                    f"a train row of {longest_kept} tokens is above the "
                    f"largest bucket {max(args.bucket_list)}; raise "
                    f"--bucket-tokens or lower --max-row-tokens")
    if args.bucket_list and not args.eval_only:
        # A shape is cheap only after it has been seen, and that first cost is
        # per process, not per step. One forward and backward per bucket at the
        # exact shape the step will use pays it here, where it can be timed and
        # recorded, instead of inside step 1 where it would look like per-step
        # cost. The gradients are dropped before any optimizer step.
        #
        # chunk_inputs treats the width as a floor, not a request, so a warm
        # row longer than a narrow bucket would present the row's own length
        # and leave that width cold while the record said it was warmed. Take a
        # row that fits each width and check the tensor that comes back.
        prewarm_shapes = []
        synchronise(device)
        started = time.perf_counter()
        for width in args.bucket_list:
            warm_row, warm_count = None, None
            for row in train_rows:
                if token_counts[id(row)] <= width:
                    warm_row, warm_count = row, token_counts[id(row)]
                    break
            if warm_row is None:
                # No row falls in this bucket, so no step presents it and
                # there is no shape to warm.
                continue
            encoded = chunk_inputs(tokenizer, [warm_row], device, width=width,
                                   pad_rows=args.batch_rows)
            shape = tuple(encoded[0].shape)
            if shape != (args.batch_rows, width):
                raise SystemExit(
                    f"prewarm asked for {(args.batch_rows, width)} and was "
                    f"given {shape}; the width is a floor, so a row of "
                    f"{warm_count} tokens is not what presents it")
            prewarm_shapes.append(list(shape))
            values = chunk_code_logits(model, tokenizer, [warm_row], code_ids,
                                       device, encoded=encoded)
            target_loss(values[0], warm_row.gold, "onehot").backward()
            optimizer.zero_grad(set_to_none=True)
        synchronise(device)
        prewarm_seconds = time.perf_counter() - started
        print(f"rank {args.rank}: prewarmed buckets {list(args.bucket_list)} "
              f"in {prewarm_seconds:.2f}s", flush=True)

    def row_loss(row, values):
        # Only score rows take an alternative target or loss; choice and noul
        # rows keep the shipped one-hot cross-entropy whatever the flags say.
        if row.q_type == "score":
            return score_loss(values, row.gold, args.score_loss,
                              args.score_cramer_weight, args.score_target,
                              args.score_target_tau)
        return target_loss(values, row.gold, "onehot")

    # The token count each row presented, cached from the mask of the encode
    # that fed its forward. Filled in loss_of because the plain path encodes
    # per row there, and re-encoding to count would be the second pass this
    # head exists to remove.
    encoded_tokens: dict = {}

    def loss_of(row, profile=None):
        prompt = prompt_of(row)
        if profile is None:
            encoded = encode_prompt(tokenizer, prompt, device)
            values = code_tensor(model, tokenizer, prompt, row.codes,
                                 code_ids, device, encoded=encoded)
        else:
            synchronise(device)
            started = time.perf_counter()
            encoded = encode_prompt(tokenizer, prompt, device)
            synchronise(device)
            profile["tokenise"] += time.perf_counter() - started
            started = time.perf_counter()
            values = code_tensor(model, tokenizer, prompt, row.codes,
                                 code_ids, device, encoded=encoded)
            synchronise(device)
            profile["forward"] += time.perf_counter() - started
        encoded_tokens[id(row)] = int(encoded["attention_mask"].sum())
        return row_loss(row, values)

    # Frozen reference loss for the stop rule, on a fixed sample. It is the
    # training loss, not plain cross-entropy: under --score-loss rps the epoch
    # mean is in RPS units, and a CE reference would make the stop rule fire
    # on the scale change alone.
    frozen_mean = None
    frozen_block = {}
    frozen_sample_ids_sha = None
    frozen_sample_rows = None
    stop_rule = None
    stop_rule_missing = []
    stop_reference = None
    stop_composition = None
    composition_counts = {}
    total_train = None
    if not args.eval_only:
        sample = train_rows[: args.frozen_loss_sample]
        frozen_losses = []
        for row in sample:
            value = float(loss_of(row))
            frozen_losses.append(value)
            entry = frozen_block.setdefault(row.q_type, [0.0, 0])
            entry[0] += value
            entry[1] += 1
        # The sample's identity, so the three arms' references can be shown
        # comparable rather than assumed to be: same plan, same order, same
        # row ids in the same sequence.
        frozen_sample_ids_sha = hashlib.sha256(
            "\n".join(row.row_id for row in sample).encode()).hexdigest()
        frozen_sample_rows = len(sample)
        frozen_mean = check_shared_non_negative(
            sum(frozen_losses) / len(frozen_losses), "frozen sample loss",
            args.rank, args.world, device)
        print(f"rank {args.rank}: frozen sample loss {frozen_mean:.4f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)

    # The rank's stride of the val split, fixed before training so a periodic
    # checkpoint scores the same rows the final pass does.
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

    def evaluate_checkpoint(step):
        """Score one in-training checkpoint and record it with its weights.

        A long run is not trusted to its last step: the same lr that helped at
        303 LoRA steps destroyed a run at more, so the curve and the per-step
        weights are what let the best point be kept instead of the last.

        Everything this function does is bracketed by a 0/1 flag and one
        all-reduce (abort_if_any_failed), which every rank reaches
        unconditionally whatever its own work did. The evaluation itself is
        inside that bracket because it is the allocation-heaviest call here
        and it runs on rows the training cap never filtered; rank 0's JevBench
        pass and the 8.8 GB write are inside it for the same reason. The raise
        happens after the reduce, so a failure on one rank surfaces as a group
        abort and its peers no longer wait in the next gradient all-reduce.
        """
        failed = torch.tensor([0.0], device=device)
        try:
            counts, per_item, bins, bin_sources = evaluate(
                model, tokenizer, val_shard, code_ids, device)
            record = {"step": step, "rank": args.rank, "val_counts": counts,
                      "val_items": per_item, "ece_bins": bins,
                      "ece_bins_population": BIN_POPULATION,
                      "ece_bins_population_rule": BIN_POPULATION_RULE,
                      "ece_bins_sources": bin_sources,
                      "val_source": val_source,
                      "weights_path": None, "weights_sha256": None,
                      "weights_bytes": None}
            if args.rank == 0:
                record["jevbench"] = evaluate_jevbench(model, tokenizer,
                                                       code_ids, device)
                if args.save_weights:
                    path = args.output / f"weights_step{step}.pt"
                    torch.save({key: value.detach().cpu()
                                for key, value in model.state_dict().items()},
                               path)
                    record["weights_path"] = str(path)
                    record["weights_sha256"] = file_sha256(path)
                    record["weights_bytes"] = path.stat().st_size
            out = args.output / f"checkpoint_{step}_shard_{args.rank}.json"
            out.write_text(json.dumps(record))
            print(f"rank {args.rank}: checkpoint step {step} "
                  f"non-synthetic {counts.get('all')}", flush=True)
        except Exception as error:  # noqa: BLE001 - reported through the group
            failed.fill_(1.0)
            print(f"rank {args.rank}: checkpoint step {step} failed: {error!r}",
                  flush=True)
        abort_if_any_failed(failed, f"checkpoint step {step}", args.world)

    history = []
    phase_records = []
    reduce_seconds = 0.0
    skipped_over_cap = 0
    # token_counts is declared before the prewarm, which reads it first.
    stopped = False
    # Batching accounting: the tokens a step's forwards really carried, the
    # tokens they paid for after padding, and, on this rank, which rows it was
    # dealt. They are per-step lists so a reader can see the distribution, not
    # only a total.
    real_tokens: list = []
    padded_cost: list = []
    forwards_per_step: list = []
    chunk_sizes: list = []
    rows_per_step: list = []
    chunk_decisions: list = []
    # The mix each step trained on, per rank: the stop rule's composition is
    # the plan's, so this is the step's own, summed over every bucket the
    # step's rows were spread across.
    composition_per_step: list = []
    bucket_groups: list = []
    # Per-forward actuals for the bucketed path, so the deal's cost model can
    # be checked against what each bucket really cost on this rank.
    bucket_actuals: list = []
    assignment_ids: list = []
    if not args.eval_only:
        shard, steps = shard_and_steps(train_rows, args.rank, args.world,
                                       args.batch_rows)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        if args.start_step >= steps:
            raise SystemExit(
                f"--start-step {args.start_step} leaves no steps to run of "
                f"{steps}; the loaded weights already cover this leg")
        print(f"rank {args.rank}: {len(shard)} rows, steps "
              f"{args.start_step + 1}..{steps} of {steps}, "
              f"{args.epochs} epoch(s)", flush=True)
        # The pinned composition both reference blocks are reweighted to. It
        # comes from the plan's train rows, not from a shard or the frozen
        # sample, so the stop comparison does not depend on which rows either
        # side happened to contain.
        composition_counts = {}
        for row in train_rows:
            composition_counts[row.q_type] = composition_counts.get(
                row.q_type, 0) + 1
        total_train = sum(composition_counts.values())
        stop_composition = {kind: count / total_train
                            for kind, count in composition_counts.items()}
        # The order the step's own composition is logged in: every type the
        # plan has, so a step that carries none of one shows it as zero
        # rather than by omission.
        type_order = sorted(composition_counts)
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
            # A resumed run skips the steps the loaded weights already cover.
            # --start-step with more than one epoch is refused at argument
            # time, so this offset is nonzero only in a single-pass leg.
            first = args.start_step
            epoch_block = {}
            started = time.time()
            for step in range(first + 1, steps + 1):
                profile = None
                if args.phase_profile and step <= args.phase_profile:
                    profile = {key: 0.0 for key in
                               ("tokenise", "forward", "backward", "allreduce",
                                "clip", "optimizer")}
                    synchronise(device)
                    step_started = time.perf_counter()
                if args.bucket_list:
                    batch = deal_rows_by_bucket_cost(
                        step_union(step, train_rows, args.world,
                                   args.batch_rows),
                        token_counts, args.world, args.batch_rows,
                        args.bucket_list)[args.rank]
                    assignment_ids.extend(row.row_id for row in batch)
                elif args.balance_rows:
                    batch = balanced_step_batch(step, train_rows, args.rank,
                                                args.world, args.batch_rows,
                                                token_counts)
                    assignment_ids.extend(row.row_id for row in batch)
                else:
                    batch = [shard[index]
                             for index in batch_indices(step, args.batch_rows,
                                                        len(shard))]
                if epoch == 1 and warmup_steps and step <= warmup_steps:
                    for group in optimizer.param_groups:
                        group["lr"] = args.lr * step / warmup_steps
                # The step's own mix: rows counted here, tokens added where
                # each forward's mask is available, so the count is the tensor
                # the model saw rather than a second tokenisation.
                step_counts = {kind: 0 for kind in type_order}
                step_tokens = {kind: 0 for kind in type_order}
                for row in batch:
                    step_counts[row.q_type] = step_counts.get(
                        row.q_type, 0) + 1
                optimizer.zero_grad(set_to_none=True)
                losses = []
                chunked = bool(args.chunk_tokens or args.chunk_rows
                               or args.bucket_list)
                # The divisor is the step's nominal rows per rank, so a balanced
                # step is the mean over all 64 rows rather than the mean of the
                # ranks' means: the deal gives ranks unequal row counts on a
                # step, and averaging per-rank means would weight a rank with
                # more rows as much as one with fewer.
                divisor = (args.batch_rows
                           if (args.balance_rows or args.bucket_list)
                           else len(batch))
                step_loss = None
                widths = None
                if args.bucket_list:
                    # Fixed-width sub-batches, each padded to the bucket width
                    # and to a full batch, so every forward presents a shape
                    # that was warmed at startup.
                    chunks, widths = [], []
                    bucket_groups.append(
                        {str(width): len(rows) for width, rows in
                         bucket_rows(batch, token_counts, args.bucket_list)})
                    for width, group_rows in bucket_rows(batch, token_counts,
                                                         args.bucket_list):
                        for start in range(0, len(group_rows),
                                           args.batch_rows):
                            chunks.append(group_rows[start:start + args.batch_rows])
                            widths.append(width)
                elif chunked:
                    decisions = []
                    chunks = chunk_by_length(batch, token_counts,
                                             args.chunk_tokens,
                                             args.chunk_ratio,
                                             args.chunk_rows,
                                             decisions)
                    chunk_decisions.append(chunk_decision_label(decisions))
                else:
                    chunks = split_by_token_budget(batch, token_counts,
                                                   args.max_batch_tokens)
                if chunked:
                    real_this_step = padded_this_step = 0
                    forwards_this_step = len(chunks)
                    bucket_actuals_this_step = []
                    chunk_sizes.append([len(chunk) for chunk in chunks])
                    for index, chunk in enumerate(chunks):
                        real, padded = chunk_cost(chunk, token_counts)
                        if widths:
                            # A bucketed forward pays for the shape it
                            # presents, which is read from the tensor after
                            # encoding rather than recomputed from the
                            # arguments: the record must describe the batch
                            # that ran, not the one the flags intended.
                            padded = None
                        real_this_step += real
                        if padded is not None:
                            padded_this_step += padded
                        # One encode per forward serves both features: the
                        # bucket's width and padded batch for the shapes, and
                        # the same mask for the step's composition. Counting
                        # from a second tokenisation would be the double pass
                        # this branch and the mix log were split to avoid.
                        if profile is None:
                            encoded = chunk_inputs(
                                tokenizer, chunk, device,
                                width=widths[index] if widths else None,
                                pad_rows=args.batch_rows if widths else None)
                            if widths:
                                presented = tuple(int(size) for size
                                                  in encoded[0].shape)
                                padded_this_step += presented[0] * presented[1]
                            values = chunk_code_logits(model, tokenizer, chunk,
                                                       code_ids, device,
                                                       encoded=encoded)
                        else:
                            synchronise(device)
                            started_phase = time.perf_counter()
                            encoded = chunk_inputs(
                                tokenizer, chunk, device,
                                width=widths[index] if widths else None,
                                pad_rows=args.batch_rows if widths else None)
                            synchronise(device)
                            profile["tokenise"] += (time.perf_counter()
                                                    - started_phase)
                            if widths:
                                presented = tuple(int(size) for size
                                                  in encoded[0].shape)
                                padded_this_step += presented[0] * presented[1]
                            started_phase = time.perf_counter()
                            values = chunk_code_logits(model, tokenizer, chunk,
                                                       code_ids, device,
                                                       encoded=encoded)
                            synchronise(device)
                            forward_elapsed = (time.perf_counter()
                                               - started_phase)
                            profile["forward"] += forward_elapsed
                            if widths:
                                bucket_actuals_this_step.append(
                                    {"step": step, "width": widths[index],
                                     "rows": len(chunk),
                                     "shape": list(presented),
                                     "padded_tokens": (presented[0]
                                                       * presented[1]),
                                     "forward": forward_elapsed,
                                     "backward": None})
                        # The step's mix is accumulated across every bucket
                        # this step presented, from each forward's own mask.
                        add_mask_tokens(step_tokens, chunk, encoded)
                        for row, row_values in zip(chunk, values):
                            term = row_loss(row, row_values)
                            entry = epoch_block.setdefault(row.q_type,
                                                           [0.0, 0])
                            entry[0] += float(term.detach())
                            entry[1] += 1
                        # One backward per chunk: its summed row losses over
                        # the fixed divisor, so moving a row between chunks or
                        # between ranks cannot change the step's gradient.
                        loss = chunk_loss(chunk, values, row_loss, divisor)
                        step_loss = (float(loss.detach()) if step_loss is None
                                     else step_loss + float(loss.detach()))
                        if profile is None:
                            loss.backward()
                        else:
                            synchronise(device)
                            started_phase = time.perf_counter()
                            loss.backward()
                            synchronise(device)
                            backward_elapsed = (time.perf_counter()
                                                - started_phase)
                            profile["backward"] += backward_elapsed
                            if widths:
                                bucket_actuals_this_step[-1]["backward"] = (
                                    backward_elapsed)
                    real_tokens.append(real_this_step)
                    padded_cost.append(padded_this_step)
                    forwards_per_step.append(forwards_this_step)
                    bucket_actuals.extend(bucket_actuals_this_step)
                else:
                    for chunk in chunks:
                        for row in chunk:
                            term = loss_of(row, profile)
                            step_tokens[row.q_type] = step_tokens.get(
                                row.q_type, 0) + encoded_tokens[id(row)]
                            loss = term / divisor
                            step_loss = (float(loss.detach())
                                         if step_loss is None
                                         else step_loss + float(loss.detach()))
                            entry = epoch_block.setdefault(row.q_type, [0.0, 0])
                            entry[0] += float(term.detach())
                            entry[1] += 1
                            if profile is None:
                                loss.backward()
                            else:
                                synchronise(device)
                                started_phase = time.perf_counter()
                                loss.backward()
                                synchronise(device)
                                profile["backward"] += (time.perf_counter()
                                                        - started_phase)
                # The recorded step loss is the loss that was backpropagated,
                # on the same divisor, so the log and the stop rule cannot
                # disagree with what was optimised.
                losses.append(step_loss)
                rows_per_step.append(len(batch))
                arrival = None
                if args.world > 1:
                    if profile is not None:
                        synchronise(device)
                        arrival = time.perf_counter()
                    reduce_started = time.perf_counter()
                    reduce_gradients(trainable, args.world,
                                     args.grad_bucket_mb << 20)
                    if profile is not None:
                        synchronise(device)
                    reduce_elapsed = time.perf_counter() - reduce_started
                    reduce_seconds += reduce_elapsed
                    if profile is not None:
                        profile["allreduce"] += reduce_elapsed
                        # Each rank's arrival at the collective, gathered once
                        # the collective has finished so the gather itself
                        # cannot affect the timestamps. The spread is the skew
                        # the reduce waits out; the transfer floor is fixed.
                        gathered = [torch.zeros(1, device=device)
                                    for _ in range(args.world)]
                        dist.all_gather(
                            gathered,
                            torch.tensor([arrival], device=device))
                        profile["arrivals"] = [float(item.item())
                                               for item in gathered]
                        profile["arrival_spread"] = (max(profile["arrivals"])
                                                     - min(profile["arrivals"]))
                if profile is not None:
                    synchronise(device)
                    started_phase = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if profile is not None:
                    synchronise(device)
                    profile["clip"] += time.perf_counter() - started_phase
                    started_phase = time.perf_counter()
                optimizer.step()
                if profile is not None:
                    synchronise(device)
                    profile["optimizer"] += time.perf_counter() - started_phase
                    # arrivals is a list of timestamps and its spread is
                    # derived from it, not a phase: summing every value would
                    # count the clock reading as elapsed time.
                    measured = sum(value for key, value in profile.items()
                                   if key not in ("arrivals",
                                                  "arrival_spread"))
                    profile["unaccounted"] = (time.perf_counter()
                                              - step_started) - measured
                    profile["step"] = step
                    phase_records.append(profile)
                composition_per_step.append({
                    "epoch": epoch,
                    "step": step,
                    "rows": step_counts,
                    "tokens": step_tokens,
                    "tokens_total": sum(step_tokens.values()),
                })
                mean = check_shared_non_negative(
                    sum(losses) / max(1, len(losses)), "mean loss", args.rank,
                    args.world, device, step=step)
                history.append(mean)
                if step % args.log_every == 0 or step == steps:
                    print(f"rank {args.rank}: epoch {epoch} step {step}/{steps} "
                          f"loss {mean:.4f} elapsed {time.time() - started:.0f}s "
                          f"types_this_rank "
                          f"{step_mix_text(step_counts, step_tokens, type_order)}",
                          flush=True)
                if (args.eval_every and step % args.eval_every == 0
                        and step != steps):
                    evaluate_checkpoint(step)
            ran_steps = steps - first
            epoch_mean = sum(history[-ran_steps:]) / ran_steps
            # Both references are per-type blocks reweighted to the pinned
            # composition. Under a non-CE score loss the two types sit on
            # different scales, so a mixed mean on one side and a different mix
            # on the other is not a comparison. The reduce carries sums and
            # counts, not means, so the global ratio can be formed afterwards;
            # every rank then computes the same decision.
            kinds = sorted(set(stop_composition) | set(epoch_block)
                           | set(frozen_block))
            flat = []
            for kind in kinds:
                for block in (frozen_block, epoch_block):
                    entry = block.get(kind, [0.0, 0])
                    flat.extend([entry[0], float(entry[1])])
            flat = torch.tensor(flat, device=device)
            if args.world > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            values = flat.tolist()
            reduced_frozen, reduced_epoch = {}, {}
            for index, kind in enumerate(kinds):
                # The frozen sample is the same rows on every rank, so its
                # sums divide by the world size; the epoch block is a
                # partition, so its sums are already the global totals.
                reduced_frozen[kind] = [values[4 * index] / args.world,
                                        values[4 * index + 1] / args.world]
                reduced_epoch[kind] = [values[4 * index + 2],
                                       values[4 * index + 3]]
            frozen_weighted, frozen_missing = reweighted_mean(
                reduced_frozen, stop_composition)
            epoch_weighted, epoch_missing = reweighted_mean(
                reduced_epoch, stop_composition)
            stop_rule_missing = sorted(set(frozen_missing) | set(epoch_missing))
            mixed_frozen = (sum(entry[0] for entry in reduced_frozen.values())
                            / max(1.0, sum(entry[1]
                                           for entry in reduced_frozen.values())))
            mixed_epoch = (sum(entry[0] for entry in reduced_epoch.values())
                           / max(1.0, sum(entry[1]
                                          for entry in reduced_epoch.values())))
            stop_reference = {
                "frozen": {"per_type": per_type_means(reduced_frozen),
                           "reweighted": frozen_weighted,
                           "mixed": mixed_frozen},
                "epoch": {"per_type": per_type_means(reduced_epoch),
                          "reweighted": epoch_weighted,
                          "mixed": mixed_epoch},
                "reweighted_difference": (
                    None if frozen_weighted is None or epoch_weighted is None
                    else epoch_weighted - frozen_weighted),
                "mixed_difference": mixed_epoch - mixed_frozen,
                "note": "the stop decision uses the pair reweighted to "
                        "stop_composition; the mixed pair is recorded for "
                        "continuity and is not comparable across arms",
            }
            if stop_rule_missing:
                # A rule that cannot be evaluated must not read as "no stop".
                stop_rule = "insufficient_sample_composition"
            elif not args.stop_on_loss_rise:
                stop_rule = "disabled_by_flag"
            elif epoch_weighted > frozen_weighted:
                stop_rule = "fired"
            else:
                stop_rule = "armed_not_fired"
            if args.world > 1:
                # The printed epoch mean must be group-wide, as before.
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
                frozen_mean, "frozen sample loss", args.rank, 1, device)
            epoch_text = ("undefined" if epoch_weighted is None
                          else f"{epoch_weighted:.4f}")
            frozen_text = ("undefined" if frozen_weighted is None
                           else f"{frozen_weighted:.4f}")
            print(f"rank {args.rank}: epoch {epoch} mean loss {epoch_mean:.4f} "
                  f"frozen {frozen_mean:.4f} reweighted {epoch_text} vs "
                  f"{frozen_text} stop_rule {stop_rule}"
                  + (f" collective {reduce_seconds / (ran_steps * epoch):.3f}s/step"
                     if args.world > 1 else ""), flush=True)
            if chunk_decisions:
                counts = ", ".join(f"{label} {chunk_decisions.count(label)}"
                                   for label in sorted(set(chunk_decisions)))
                print(f"rank {args.rank}: chunk decisions over "
                      f"{len(chunk_decisions)} steps: {counts}", flush=True)
            if stop_rule == "fired":
                print(f"rank {args.rank}: stop rule fired (reweighted epoch "
                      f"{epoch_text} above frozen {frozen_text})", flush=True)
                stopped = True
                break

    if args.phase_profile and phase_records:
        phases = [key for key in phase_records[0]
                  if key not in ("step", "unaccounted", "arrivals",
                                 "arrival_spread")]
        recorded = len(phase_records)
        totals = {key: sum(record[key] for record in phase_records)
                  for key in phases}
        unaccounted = sum(record["unaccounted"] for record in phase_records)
        profile_payload = {
            "rank_this_process": args.rank,
            "steps_this_rank": recorded,
            "first_step": phase_records[0]["step"],
            "includes_startup": phase_records[0]["step"] == 1,
            "totals_seconds_this_rank": totals,
            "mean_seconds_this_rank": {key: value / recorded
                                       for key, value in totals.items()},
            "mean_measured_phases_seconds": sum(totals.values()) / recorded,
            "mean_unaccounted_seconds": unaccounted / recorded,
            # The run's per-type totals on this rank. The per-step list stays
            # in the shard payload; this is the compact form a reader can use
            # if the payload write never happens.
            "composition_totals_this_rank": {
                kind: {"rows": sum(entry["rows"].get(kind, 0)
                                   for entry in composition_per_step),
                       "tokens": sum(entry["tokens"].get(kind, 0)
                                     for entry in composition_per_step)}
                for kind in sorted({kind for entry in composition_per_step
                                    for kind in entry["rows"]})},
            "mean_profiled_seconds": (sum(totals.values())
                                      + unaccounted) / recorded,
            "profiled_seconds_excludes": (
                "the per-step loss all-reduce and stop check, logging, "
                "checkpoint evaluation and weight writes, and every step "
                "after the first --phase-profile; the batch gather and "
                "zero_grad are inside the window"),
            "unaccounted_covers": "batch gather, warmup lr update, zero_grad, "
                                  "chunk split, per-row loss bookkeeping, and "
                                  "the per-row mask sums on any path that does "
                                  "not time its own tokenisation",
            "skipped_rows_this_rank": skipped_over_cap,
            # The per-step records and the chunk accounting, so a cost fit can
            # take both regressors from the same artifact as the timings.
            "steps": phase_records,
            "chunking_this_rank": ({
                "tokens": args.chunk_tokens,
                "rows": args.chunk_rows,
                "length_ratio": args.chunk_ratio,
                "real_tokens_per_step": real_tokens,
                "forwards_per_step": forwards_per_step,
                "chunk_sizes": chunk_sizes,
                "rows_per_step": rows_per_step,
                "chunk_decision_per_step": chunk_decisions,
                "chunk_decision_counts": {
                    label: chunk_decisions.count(label)
                    for label in sorted(set(chunk_decisions))},
            } if (args.chunk_tokens or args.chunk_rows
                  or args.bucket_list) else {"tokens": 0}),
            "padded_cost_this_rank": (padded_cost
                                      if (args.chunk_tokens or args.chunk_rows
                                          or args.bucket_list)
                                      else None),
            # The bucketing measurements live here as well as in the shard
            # payload: this file is written before the epoch-end evaluation,
            # so a measurement run stopped after the profiles exists does not
            # lose the per-bucket costs it was run for.
            "bucketing_this_rank": ({
                "widths": list(args.bucket_list),
                "prewarm_seconds_this_rank": prewarm_seconds,
                "prewarm_shapes_this_rank": prewarm_shapes,
                "actuals_per_forward": bucket_actuals,
                "cost_model": {
                    "call_seconds": BUCKET_CALL_SECONDS,
                    "token_seconds": BUCKET_TOKEN_SECONDS,
                    "token_squared_seconds": BUCKET_TOKEN_SQUARED_SECONDS,
                    "fitted_on": ("199 padded forwards of the a20k twenty-step "
                                  "bucket run, 8 rows per rank, four slow "
                                  "forwards excluded, rms 0.026 s"),
                },
            } if args.bucket_list else {"widths": []}),
        }
        # One file per rank: every rank runs this block, and a shared name let
        # them overwrite each other, leaving a torn file that parses as neither.
        profile_path = args.output / f"phase_profile_rank{args.rank}.json"
        profile_payload["path_this_rank"] = profile_path.name
        failed = torch.tensor([0.0], device=device)
        try:
            profile_path.write_text(json.dumps(profile_payload))
        except Exception as error:  # noqa: BLE001 - reported through the group
            failed.fill_(1.0)
            print(f"rank {args.rank}: phase profile write failed: {error!r}",
                  flush=True)
        abort_if_any_failed(failed, "phase profile write", args.world)
        print(f"rank {args.rank}: phase profile "
              f"{json.dumps(profile_payload['mean_seconds_this_rank'])}", flush=True)

    # Validation on the same stride the periodic checkpoints scored.
    counts, per_item, ece_bins, bin_sources = evaluate(
        model, tokenizer, val_shard, code_ids, device)
    # Two counts that can disagree, because one comes from the rows handed to
    # evaluate and the other from the rows it binned. If the filter drops or
    # admits the wrong family, the payload carries both numbers and a mismatch.
    ece_bins_rows = sum(slot[0] for cell in ece_bins.values()
                        for slot in cell.values())
    val_rows_non_synthetic = sum(1 for row in val_shard
                                 if row.source != "synthetic")
    payload = {"rank": args.rank, "lr": args.lr, "val_counts": counts,
               "val_items": per_item, "ece_bins": ece_bins,
               "ece_bins_population": BIN_POPULATION,
               "ece_bins_population_rule": BIN_POPULATION_RULE,
               "ece_bins_sources": bin_sources,
               "val_source": val_source,
               "stopped": stopped,
               "stop_reason": "reweighted_epoch_above_frozen" if stopped else None,
               "stop_rule": (stop_rule if stop_rule is not None
                             else ("not_run_eval_only" if args.eval_only
                                   else None)),
               "stop_rule_missing_types": stop_rule_missing,
               "stop_composition": (
                   {"fractions": stop_composition,
                    "counts": composition_counts,
                    "n_train_rows": total_train,
                    "source": "the plan's train rows after the row-token cap, "
                              "counted on every rank; both references are "
                              "reweighted to it"}
                   if stop_composition else None),
               "stop_reference": stop_reference,
               "plan": str(args.plan), "plan_sha256": plan_sha,
               "plan_sha256_expected": args.plan_sha256,
               "eval_plan": str(args.eval_plan) if args.eval_plan else None,
               "eval_plan_sha256": eval_plan_sha,
               "eval_plan_sha256_expected": args.eval_plan_sha256,
               "full_ft": args.full_ft, "grad_bucket_mb": args.grad_bucket_mb,
               "row_assignment": {
                   "rule": "balanced_by_length" if args.balance_rows else "rank_stride",
                   "rows_per_step": args.world * args.batch_rows,
                   "digest_this_rank": (hashlib.sha256(
                       "\n".join(assignment_ids).encode()).hexdigest()
                       if args.balance_rows else None),
               },
               "step_losses": history,
               # The mix each step actually trained on, per rank, summed
               # over every bucket the step's rows were spread across. The
               # stop rule's composition is the plan's, so this is the only
               # record of what a step really carried.
               "composition_per_step": composition_per_step,
               "bucketing": ({
                   "widths": list(args.bucket_list),
                   "prewarm_seconds_this_rank": prewarm_seconds,
                   "prewarm_shapes_this_rank": prewarm_shapes,
                   "groups_per_step": bucket_groups,
                   # One record per padded forward on this rank, with the
                   # width and row count it actually presented and its
                   # forward and backward seconds. Empty unless
                   # --phase-profile is set: the timings carry a device
                   # synchronise, which a timed run must not pay silently.
                   "actuals_per_forward": bucket_actuals,
                   "cost_model": {
                       "call_seconds": BUCKET_CALL_SECONDS,
                       "token_seconds": BUCKET_TOKEN_SECONDS,
                       "token_squared_seconds": BUCKET_TOKEN_SQUARED_SECONDS,
                       "fitted_on": ("199 padded forwards of the a20k "
                                     "twenty-step bucket run, 8 rows per rank, "
                                     "four slow forwards excluded, rms 0.026 s"),
                   },
               } if args.bucket_list else {"widths": []}),
               "eval_every": args.eval_every,
               "phase_profile_steps": args.phase_profile,
               "ece_bins_rows_this_rank": ece_bins_rows,
               "val_rows_non_synthetic_this_rank": val_rows_non_synthetic,
               "optimizer_state": "fresh: the trainer never saves or restores "
                                  "Adam moments, so a resumed leg's optimizer "
                                  "starts cold",
               "token_counts_key": "id(row), object identity, stable while "
                                   "this run holds the rows",
               "score_target": args.score_target,
               "score_target_tau": args.score_target_tau,
               "score_loss": args.score_loss,
               "score_cramer_weight": args.score_cramer_weight,
               "score_cramer_weight_provenance": cramer_provenance,
               "start_step": args.start_step,
               "planned_step_range": ([args.start_step + 1, steps]
                                      if not args.eval_only else None),
               "steps_run": len(history),
               "frozen_sample_measured_on": (
                   "loaded weights" if args.load_weights is not None
                   else "base weights"),
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
                   reduce_seconds / ((steps - args.start_step) * args.epochs)
                   if history else None),
               "max_memory_gib": (torch.cuda.max_memory_allocated() / 2**30
                                  if device.startswith("cuda") else None),
               "frozen_sample_loss": frozen_mean,
               "frozen_sample_rows": frozen_sample_rows,
               "frozen_sample_row_ids_sha256": frozen_sample_ids_sha}

    args.output.mkdir(parents=True, exist_ok=True)
    payload["weights_path"] = None
    payload["weights_sha256"] = None
    payload["weights_bytes"] = None
    payload["weights_loaded_sha256"] = weights_loaded_sha
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

    # The last rank-0 work before the barrier is bracketed like a checkpoint:
    # a disk-full or a failed JevBench read must reach every rank as an abort,
    # not leave the peers waiting for the launcher to notice.
    failed = torch.tensor([0.0], device=device)
    if args.rank == 0:
        try:
            payload["jevbench"] = evaluate_jevbench(model, tokenizer,
                                                    code_ids, device)
            if args.save_weights:
                # The full fine-tune's arm is its weights: without this file
                # the cells cannot be re-scored on a later metric or shipped.
                # Written before the payload so the digest it records covers
                # these bytes.
                weights_path = args.output / "weights.pt"
                torch.save({key: value.detach().cpu()
                            for key, value in model.state_dict().items()},
                           weights_path)
                payload["weights_path"] = str(weights_path)
                payload["weights_sha256"] = file_sha256(weights_path)
                payload["weights_bytes"] = weights_path.stat().st_size
                print(f"rank 0: saved weights {weights_path} "
                      f"sha256 {payload['weights_sha256'][:16]} "
                      f"bytes {payload['weights_bytes']}", flush=True)
        except Exception as error:
            failed.fill_(1.0)
            print(f"rank 0: final evaluation or weight write failed: {error}",
                  flush=True)
    abort_if_any_failed(failed, "final evaluation or weight write", args.world)

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
