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
            "score benchmark val and the 231 items every N steps and write "
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
            "lambda of the rps term under --score-loss mix. Not guessed: it "
            "is 0.25 * mean per-row CE / mean per-row RPS measured on the "
            "pinned plan's score rows at step 0, then pinned in the arm. "
            "Required for mix and refused for ce and rps, so no run records "
            "a weight the loss ignored"
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
            "for the first N steps and write phase_profile.json; the phases "
            "are synchronised, so the numbers include device work"
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
    if args.score_loss != "ce" and args.score_target != "onehot":
        # RPS takes the gold's one-hot distribution. A soft or expected target
        # alongside it is a third axis nobody has pre-registered, so refuse
        # rather than silently drop one of the two flags.
        raise SystemExit(
            f"--score-loss {args.score_loss} uses the gold's one-hot target; "
            f"--score-target {args.score_target} was not pre-registered with it"
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
        state = torch.load(args.load_weights, map_location=device)
        model.load_state_dict(state, strict=True)
        weights_loaded_sha = file_sha256(args.load_weights)
        print(f"rank {args.rank}: loaded weights {args.load_weights} "
              f"sha256 {weights_loaded_sha[:16]}", flush=True)


    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    def loss_of(row, profile=None):
        prompt = prompt_of(row)
        if profile is None:
            values = code_tensor(model, tokenizer, prompt, row.codes,
                                 code_ids, device)
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
        # Only score rows take an alternative target or loss; choice and noul
        # rows keep the shipped one-hot cross-entropy whatever the flags say.
        if row.q_type == "score":
            return score_loss(values, row.gold, args.score_loss,
                              args.score_cramer_weight, args.score_target,
                              args.score_target_tau)
        return target_loss(values, row.gold, "onehot")

    # Frozen reference loss for the stop rule, on a fixed sample. It is the
    # training loss, not plain cross-entropy: under --score-loss rps the epoch
    # mean is in RPS units, and a CE reference would make the stop rule fire
    # on the scale change alone.
    frozen_mean = None
    if not args.eval_only:
        sample = train_rows[: args.frozen_loss_sample]
        frozen_losses = []
        for row in sample:
            frozen_losses.append(float(loss_of(row)))
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
                  f"benchmark {counts.get('all')}", flush=True)
        except Exception as error:  # noqa: BLE001 - reported through the group
            failed.fill_(1.0)
            print(f"rank {args.rank}: checkpoint step {step} failed: {error!r}",
                  flush=True)
        abort_if_any_failed(failed, f"checkpoint step {step}", args.world)

    history = []
    phase_records = []
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
        if args.start_step >= steps:
            raise SystemExit(
                f"--start-step {args.start_step} leaves no steps to run of "
                f"{steps}; the loaded weights already cover this leg")
        print(f"rank {args.rank}: {len(shard)} rows, steps "
              f"{args.start_step + 1}..{steps} of {steps}, "
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
            # A resumed run skips the steps the loaded weights already cover.
            # --start-step with more than one epoch is refused at argument
            # time, so this offset is nonzero only in a single-pass leg.
            first = args.start_step
            started = time.time()
            for step in range(first + 1, steps + 1):
                profile = None
                if args.phase_profile and step <= args.phase_profile:
                    profile = {key: 0.0 for key in
                               ("tokenise", "forward", "backward", "allreduce",
                                "clip", "optimizer")}
                    synchronise(device)
                    step_started = time.perf_counter()
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
                        loss = loss_of(row, profile) / len(batch)
                        if profile is None:
                            loss.backward()
                        else:
                            synchronise(device)
                            started_phase = time.perf_counter()
                            loss.backward()
                            synchronise(device)
                            profile["backward"] += (time.perf_counter()
                                                    - started_phase)
                        losses.append(float(loss))
                if args.world > 1:
                    if profile is not None:
                        synchronise(device)
                    reduce_started = time.perf_counter()
                    reduce_gradients(trainable, args.world,
                                     args.grad_bucket_mb << 20)
                    if profile is not None:
                        synchronise(device)
                    reduce_elapsed = time.perf_counter() - reduce_started
                    reduce_seconds += reduce_elapsed
                    if profile is not None:
                        profile["allreduce"] += reduce_elapsed
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
                    measured = sum(profile.values())
                    profile["unaccounted"] = (time.perf_counter()
                                              - step_started) - measured
                    profile["step"] = step
                    phase_records.append(profile)
                mean = check_shared_non_negative(
                    sum(losses), "mean loss", args.rank, args.world, device,
                    step=step)
                history.append(mean)
                if step % args.log_every == 0 or step == steps:
                    print(f"rank {args.rank}: epoch {epoch} step {step}/{steps} "
                          f"loss {mean:.4f} elapsed {time.time() - started:.0f}s",
                          flush=True)
                if (args.eval_every and step % args.eval_every == 0
                        and step != steps):
                    evaluate_checkpoint(step)
            ran_steps = steps - first
            epoch_mean = sum(history[-ran_steps:]) / ran_steps
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
                frozen_mean, "frozen sample loss", args.rank, 1, device)
            print(f"rank {args.rank}: epoch {epoch} mean loss {epoch_mean:.4f} "
                  f"frozen {frozen_mean:.4f}"
                  + (f" collective {reduce_seconds / (ran_steps * epoch):.3f}s/step"
                     if args.world > 1 else ""), flush=True)
            if epoch_mean > frozen_mean and args.stop_on_loss_rise:
                print(f"rank {args.rank}: stop rule fired (loss above frozen)",
                      flush=True)
                stopped = True
                break

    if args.phase_profile and phase_records:
        phases = [key for key in phase_records[0] if key not in ("step", "unaccounted")]
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
            "mean_profiled_seconds": (sum(totals.values())
                                      + unaccounted) / recorded,
            "profiled_seconds_excludes": (
                "the per-step loss all-reduce and stop check, logging, "
                "checkpoint evaluation and weight writes, and every step "
                "after the first --phase-profile; the batch gather and "
                "zero_grad are inside the window"),
            "unaccounted_covers": "batch gather, warmup lr update, zero_grad, "
                                  "chunk split, per-row loss bookkeeping",
            "skipped_rows_this_rank": skipped_over_cap,
        }
        (args.output / "phase_profile.json").write_text(
            json.dumps(profile_payload))
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
               "stop_reason": "epoch_mean_loss_above_frozen" if stopped else None,
               "plan": str(args.plan), "plan_sha256": plan_sha,
               "plan_sha256_expected": args.plan_sha256,
               "eval_plan": str(args.eval_plan) if args.eval_plan else None,
               "eval_plan_sha256": eval_plan_sha,
               "eval_plan_sha256_expected": args.eval_plan_sha256,
               "full_ft": args.full_ft, "grad_bucket_mb": args.grad_bucket_mb,
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
               "start_step": args.start_step,
               "step_range": ([args.start_step + 1, steps]
                              if args.start_step and not args.eval_only
                              else None),
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
               "frozen_sample_ce": frozen_mean}

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
