#!/usr/bin/env python3
"""Panel 2 under a content-addressed option order (canonical order).

One forward per item with the options sorted by ``sha256(text.strip())`` and
coded A, B, C in that sequence, so the prompt depends on the option multiset
and not on the order the caller supplied.  The same harness scores the shipped
order, so the two arms share code and implementation.

Modes:
    --check-premise   render every item from its natural and its reversed
                      option list through the canonical order and assert the
                      two prompts are byte-identical; no model is loaded
    (default)         score one arm
    --compare         paired comparison of two arms on identical items

Usage:
    python model/scripts/eval_jevbench_canonical.py --check-premise
    python model/scripts/eval_jevbench_canonical.py --model Qwen/Qwen3-Reranker-4B \
        --option-order canonical --output results/jevbench/canonical_Qwen_Qwen3-Reranker-4B.json
    python model/scripts/eval_jevbench_canonical.py --compare arm_a.json arm_b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_jevbench
from model.evaluation.jevbench import summarize, tier_from_id
from model.evaluation.panel2 import (
    CODES,
    PANEL2_TEMPLATE,
    choose_option,
    code_token_ids,
    codes_for,
    has_exact_tie,
    options_for,
    position_diagnostics,
    render_panel2,
)
from model.evaluation.panel2_canonical import canonical_order
from model.evaluation.run_provenance import build_run_provenance
from model.src.backbone import load_causal_lm

STORED_PANEL2 = ROOT / "results/jevbench/panel2_Qwen_Qwen3-Reranker-4B.json"
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
TIE_RULE = "first maximum by natural option index"


def data_digest(items: list[Any]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(
            json.dumps(
                [item.id, item.question.get("type", "noul"), str(item.label)],
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def canonical_config(
    args: argparse.Namespace, code_ids: dict[str, int], used_codes: list[str]
) -> dict[str, Any]:
    """The declared configuration; hashed before the run and recorded with it."""
    return {
        "panel": "2",
        "dataset": "jevbench",
        "option_order": args.option_order,
        "option_order_rule": (
            "ascending sha256(option text stripped of surrounding whitespace)"
            if args.option_order == "canonical"
            else "the order the item supplies"
        ),
        "prompt_template": PANEL2_TEMPLATE,
        "tie_rule": TIE_RULE,
        "code_scheme": "A-Z, a-z, then single-character symbols, one code per option",
        "code_token_form": args.code_form,
        "code_token_ids": {
            code: code_ids[code] for code in used_codes if code in code_ids
        },
        "noul_rendering": "two options: A. no / B. yes",
        "choice_rendering": "criteria descriptions in item order",
        "score_rendering": "'Level {i}: {description}' (canonical order applied for this measurement only)",
        "orders": ["natural"],
        "tokenizer_add_special_tokens": False,
    }


def print_config_hash(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    items = list(load_jevbench())
    prepared = [options_for(item) for item in items]
    prepared = [entry for entry in prepared if entry is not None]
    max_options = max(len(entry[0]) for entry in prepared) if prepared else 0
    used_codes = list(CODES[:max_options])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)
    config = canonical_config(args, code_ids, used_codes)
    payload = {
        "model": args.model,
        "config": config,
        "config_sha256": config_digest(config),
        "data_sha256": data_digest(items),
        "items": len(items),
        "multi_token_codes": multi,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


def check_premise() -> int:
    """The whole claim's premise: natural and reversed inputs render identically.

    No model is loaded.  For every JevBench item the options are put in
    canonical order starting from the natural list and from the reversed list;
    the two rendered prompt strings must be byte-identical, and the two
    canonical permutations must describe the same text sequence.
    """
    items = list(load_jevbench())
    mismatches: list[str] = []
    duplicate_texts: list[str] = []
    permutation_digest = hashlib.sha256()
    checked = 0
    for item in items:
        prepared = options_for(item)
        if prepared is None:
            continue
        texts, _gold = prepared
        forward = canonical_order(texts)
        backward = canonical_order(list(reversed(texts)))
        forward_texts = [texts[index] for index in forward]
        reversed_texts = list(reversed(texts))
        backward_texts = [reversed_texts[index] for index in backward]
        prompt_forward = render_panel2(
            item.state,
            item.question.get("instructions", ""),
            forward_texts,
            codes_for(len(texts)),
        )
        prompt_backward = render_panel2(
            item.state,
            item.question.get("instructions", ""),
            backward_texts,
            codes_for(len(texts)),
        )
        if prompt_forward != prompt_backward:
            mismatches.append(item.id)
        if len({option_identity(text) for text in texts}) != len(texts):
            duplicate_texts.append(item.id)
        permutation_digest.update(
            json.dumps([item.id, forward]).encode("utf-8")
        )
        checked += 1
    payload = {
        "items": len(items),
        "checked": checked,
        "mismatches": mismatches,
        "items_with_duplicate_option_texts": duplicate_texts,
        "permutation_sha256": permutation_digest.hexdigest(),
        "premise_holds": not mismatches,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not mismatches else 1


def option_identity(text: str) -> str:
    return text.strip()


def score_one(
    model: Any,
    tokenizer: Any,
    item: Any,
    texts: list[str],
    gold: int,
    code_ids: dict[str, int],
    device: str,
    option_order: str,
) -> dict[str, Any]:
    order = canonical_order(texts) if option_order == "canonical" else list(range(len(texts)))
    display_texts = [texts[index] for index in order]
    codes = codes_for(len(texts))
    prompt = render_panel2(
        item.state, item.question.get("instructions", ""), display_texts, codes
    )
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].float()
    elapsed = time.perf_counter() - started
    values = [float(logits[code_ids[code]]) for code in codes]
    chosen_natural = choose_option(values, order)
    display_position = order.index(chosen_natural)
    tier = tier_from_id(item.id)
    if tier not in ("easy", "hard", "original"):
        tier = "dev"
    return {
        "id": item.id,
        "tier": tier,
        "type": item.question.get("type", "noul"),
        "option_order": option_order,
        "n_options": len(texts),
        "canonical_order": order,
        "gold_natural": gold,
        "pred_natural": chosen_natural,
        "gold_display": order.index(gold),
        "pred_display": display_position,
        "correct": chosen_natural == gold,
        "score_values": values,
        "tie": has_exact_tie(values),
        "latency_s": elapsed,
    }


def score_arm(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    items = list(load_jevbench())
    print(f"Loaded {len(items)} JevBench items", flush=True)

    snapshot_path = args.snapshot or Path(snapshot_download(args.model))
    model, tokenizer = load_causal_lm(
        str(snapshot_path), device=args.device, dtype=DTYPE_MAP[args.dtype]
    )
    device = str(next(model.parameters()).device)

    prepared = [options_for(item) for item in items]
    prepared = [entry for entry in prepared if entry is not None]
    max_options = max(len(entry[0]) for entry in prepared) if prepared else 0
    used_codes = list(CODES[:max_options])
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)

    rows: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        prepared_item = options_for(item)
        if prepared_item is None:
            unscorable.append({"id": item.id, "reason": "gold_not_in_options"})
            continue
        texts, gold = prepared_item
        missing = [code for code in codes_for(len(texts)) if code not in code_ids]
        if missing:
            unscorable.append(
                {"id": item.id, "reason": "multi_token_code", "codes": missing}
            )
            continue
        rows.append(
            score_one(
                model,
                tokenizer,
                item,
                texts,
                gold,
                code_ids,
                device,
                args.option_order,
            )
        )
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(items)} items scored", flush=True)

    by_id = {item.id: item for item in items}
    with_items = [{**row, "_item": by_id[row["id"]]} for row in rows]
    summary = summarize(with_items, failed=len(unscorable))
    overall = summary["overall"]
    tied = [row for row in rows if row["tie"]]
    config = canonical_config(args, code_ids, used_codes)
    validity = {
        "single_token_codes": len(multi) == 0
        and not any(entry["reason"] == "multi_token_code" for entry in unscorable),
        "multi_token_codes": multi,
        "unscorable_items": len(unscorable),
        "tie_items": len(tied),
        "tie_items_correct": sum(1 for row in tied if row["correct"]),
        "valid": len(multi) == 0,
    }
    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot_path,
        dtype=args.dtype,
        device=device,
        strategy="panel2_canonical_order",
        norm="none",
        root=ROOT,
        code_commit=args.code_commit,
        extra={
            "readout": "one forward; argmax over the K code token ids at the "
                       "final position after 'Answer:'",
            "option_order": args.option_order,
            "option_order_rule": config["option_order_rule"],
            "prompt_template": config["prompt_template"],
            "tie_rule": TIE_RULE,
            "code_token_form": args.code_form,
            "dataset": "jevbench",
            "config_sha256": config_digest(config),
            "data_sha256": data_digest(items),
            "items": len(items),
        },
    )
    payload = {
        "_provenance": provenance,
        "config": config,
        "validity": validity,
        "unscorable": unscorable,
        "metrics": {
            "overall": overall,
            "by_tier": summary["by_tier"],
            "by_type": summary["by_type"],
            "position": position_diagnostics(
                [
                    {
                        "pred_pos": row["pred_display"],
                        "gold_pos": row["gold_display"],
                        "correct": row["correct"],
                    }
                    for row in rows
                ]
            ),
            "latency": {
                "mean_s": statistics.mean(row["latency_s"] for row in rows)
                if rows
                else 0.0
            },
        },
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"written {args.output}: {overall['correct']}/{overall['total']} "
        f"= {overall['accuracy']:.4f}, valid={validity['valid']}",
        flush=True,
    )
    return 0


def compare(args: argparse.Namespace) -> int:
    from model.scripts.paired_mcnemar import mcnemar

    with open(args.compare[0], encoding="utf-8") as handle:
        first = json.load(handle)
    with open(args.compare[1], encoding="utf-8") as handle:
        second = json.load(handle)
    first_rows = {row["id"]: row for row in first["predictions"]}
    second_rows = {row["id"]: row for row in second["predictions"]}
    shared = sorted(set(first_rows) & set(second_rows))
    first_correct = {key: bool(first_rows[key]["correct"]) for key in shared}
    second_correct = {key: bool(second_rows[key]["correct"]) for key in shared}
    wins, losses, p = mcnemar(second_correct, first_correct, shared)
    changed = sum(
        1
        for key in shared
        if first_rows[key]["pred_natural"] != second_rows[key]["pred_natural"]
    )
    first_label = first["config"]["option_order"]
    second_label = second["config"]["option_order"]
    payload = {
        "first": {"path": str(args.compare[0]), "option_order": first_label},
        "second": {"path": str(args.compare[1]), "option_order": second_label},
        "n": len(shared),
        "first_correct": sum(first_correct.values()),
        "second_correct": sum(second_correct.values()),
        "second_only_correct": wins,
        "first_only_correct": losses,
        "mcnemar_p": p,
        "answer_changed_fraction": changed / len(shared) if shared else 0.0,
        "answer_changed_items": changed,
    }
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        baseline_correct = {
            row["id"]: bool(row["correct"]) for row in baseline["predictions"]
        }
        keys = sorted(set(baseline_correct) & set(second_correct))
        b_wins, b_losses, b_p = mcnemar(
            {key: second_correct[key] for key in keys},
            {key: baseline_correct[key] for key in keys},
            keys,
        )
        payload["stored_baseline"] = {
            "path": str(args.baseline),
            "n": len(keys),
            "second_only_correct": b_wins,
            "stored_only_correct": b_losses,
            "mcnemar_p": b_p,
        }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument(
        "--option-order", choices=["shipped", "canonical"], default="canonical"
    )
    parser.add_argument("--code-form", choices=["space", "bare"], default="space")
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-premise", action="store_true")
    parser.add_argument("--compare", type=Path, nargs=2, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.check_premise:
        return check_premise()
    if args.compare:
        return compare(args)
    if args.print_config_hash:
        if not args.model:
            parser.error("--print-config-hash needs --model")
        return print_config_hash(args)
    if not args.model or args.output is None:
        parser.error("scoring needs --model and --output")
    return score_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
