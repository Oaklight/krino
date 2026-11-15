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


def build_rows(path: Path, noul_gold: str = "position",
               plan_unpermuted_noul: bool = False) -> tuple[list[Row], str, int]:
    """Rows from a plan, the field that supplied their split, and masks.

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
    with path.open(encoding="utf-8") as handle:
        for line in handle:
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
    """Accuracy per row, with the natural-order option index chosen."""
    model.eval()
    counts = {}
    per_item = []
    with torch.inference_mode():
        for row in (rows[:limit] if limit else rows):
            values = code_logits(model, tokenizer, prompt_of(row),
                                 {c: code_ids[c] for c in row.codes}, device)
            chosen = choose_option(values, list(range(len(values))))
            ok = chosen == row.gold
            for key in ("all", f"type:{row.q_type}", f"source:{row.source}"):
                cell = counts.setdefault(key, [0, 0])
                cell[0] += int(ok)
                cell[1] += 1
            per_item.append([row.row_id, int(ok), row.q_type, row.source])
    model.train()
    return counts, per_item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--eval-plan",
        type=Path,
        default=None,
        help="score this plan's val rows instead of the training plan's",
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

    if args.world > 1:
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.rank)))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(42)

    rows, split_field, masked = build_rows(
        args.plan, args.noul_gold, args.plan_unpermuted_noul
    )
    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    eval_plan_sha = None
    if args.eval_plan is not None:
        eval_rows, eval_split_field, _ = build_rows(
            args.eval_plan, args.noul_gold, args.plan_unpermuted_noul
        )
        val_rows = [r for r in eval_rows if r.split == "val"]
        eval_plan_sha = hashlib.sha256(args.eval_plan.read_bytes()).hexdigest()
        print(f"rank {args.rank}: eval plan val rows {len(val_rows)} "
              f"split_field {eval_split_field} sha {eval_plan_sha[:16]}", flush=True)
    print(f"rank {args.rank}: rows {len(rows)} train {len(train_rows)} "
          f"val {len(val_rows)} masked {masked} split_field {split_field} "
          f"plan_sha {plan_sha[:16]}", flush=True)

    model, tokenizer = load_causal_lm(
        "Qwen/Qwen3-Reranker-4B", device=device, dtype=torch.bfloat16,
        freeze=True, flash_attention=False,
    )
    inject_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                rank=args.lora_rank, alpha=args.lora_alpha,
                last_n_layers=args.lora_layers)
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
        frozen_mean = sum(frozen_losses) / len(frozen_losses)
        print(f"rank {args.rank}: frozen sample CE {frozen_mean:.4f}", flush=True)

    history = []
    stopped = False
    if not args.eval_only:
        shard = train_rows[args.rank :: args.world] if args.world > 1 else train_rows
        steps = math.ceil(len(shard) / args.batch_rows)
        if args.max_steps:
            steps = min(steps, args.max_steps)
        print(f"rank {args.rank}: {len(shard)} rows, {steps} steps per epoch, "
              f"{args.epochs} epoch(s)", flush=True)
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            for step in range(1, steps + 1):
                batch = shard[(step - 1) * args.batch_rows: step * args.batch_rows]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for row in batch:
                    loss = loss_of(row) / len(batch)
                    loss.backward()
                    losses.append(float(loss))
                if args.world > 1:
                    for parameter in trainable:
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                            parameter.grad /= args.world
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                mean = sum(losses)
                if not math.isfinite(mean):
                    print(f"rank {args.rank}: NaN at step {step}", flush=True)
                    raise SystemExit(2)
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
            print(f"rank {args.rank}: epoch {epoch} mean loss {epoch_mean:.4f} "
                  f"frozen {frozen_mean:.4f}", flush=True)
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
    counts, per_item = evaluate(model, tokenizer, val_shard, code_ids, device)
    payload = {"rank": args.rank, "lr": args.lr, "val_counts": counts,
               "val_items": per_item, "val_source": val_source,
               "stopped": stopped,
               "stop_reason": "epoch_mean_loss_above_frozen" if stopped else None,
               "plan": str(args.plan), "plan_sha256": plan_sha,
               "eval_plan": str(args.eval_plan) if args.eval_plan else None,
               "eval_plan_sha256": eval_plan_sha,
               "split_field": split_field, "noul_gold": args.noul_gold,
               "plan_unpermuted_noul": args.plan_unpermuted_noul,
               "masked_rows": masked, "epochs": args.epochs,
               "epoch_mean_loss": (sum(history) / len(history)) if history else None,
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
