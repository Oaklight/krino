#!/usr/bin/env python3
"""Panel 2 with three option-to-code assignments, natural and reversed (#339).

Three arms and only three, declared in :mod:`model.evaluation.panel2_arms`:
the shipped positional codes, content-addressed codes from a SHA-256 of the
option text, and the mean of three seeded random assignments.  Each arm is
scored on JevBench public 231 in the natural and the reversed display order,
so the order-reversal cost can be read off directly.

The arms differ only in which code names which option.  The prompt template,
the option texts, the code form and the read position are identical to the
shipped panel-2 cell; only the letters change.

Reported per arm and checkpoint: natural accuracy, reversed accuracy, the
signed and absolute reversal difference, the exact paired McNemar on the
items that changed correctness, the fraction of items whose chosen option
changes under reversal, the per-item latency, and the position diagnostic.

Usage:
    python model/scripts/eval_panel2_order_arms.py \
        --model Qwen/Qwen3-Reranker-4B \
        --output results/jevbench/panel2_order_arms_Qwen_Qwen3-Reranker-4B.json
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

from data.pipeline import load_jevbench  # noqa: E402
from model.evaluation.jevbench import tier_from_id  # noqa: E402
from model.evaluation.panel2 import (  # noqa: E402
    CODES,
    PANEL2_TEMPLATE,
    code_token_ids,
    options_for,
    position_diagnostics,
    render_panel2,
)
from model.evaluation.panel2_arms import (  # noqa: E402
    ARMS,
    HASHED,
    MEAN3,
    SEEDS,
    display_codes,
    mixture_prediction,
    natural_codes,
    option_index_for_display,
    probabilities_by_option,
)
from model.evaluation.run_provenance import build_run_provenance  # noqa: E402
from model.src.backbone import load_causal_lm  # noqa: E402
from model.scripts.paired_mcnemar import mcnemar  # noqa: E402

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

HASH_DEFINITION = (
    "sha256(text.strip().encode('utf-8')).digest(); options sorted by the "
    "32-byte digest ascending; the i-th option in that order takes CODES[i]"
)


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


def score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    codes: list[str],
    code_ids: dict[str, int],
    device: str,
) -> tuple[list[float], float]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].float()
    elapsed = time.perf_counter() - started
    return [float(logits[code_ids[code]]) for code in codes], elapsed


def score_item(
    model: Any,
    tokenizer: Any,
    item: Any,
    code_ids: dict[str, int],
    device: str,
) -> list[dict[str, Any]]:
    """Every arm × display-order cell for one item."""
    texts, gold = options_for(item)
    width = len(texts)
    instructions = item.question.get("instructions", "")
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for reversed_order in (False, True):
            display = list(reversed(texts)) if reversed_order else list(texts)
            mappings = display_codes(arm, texts, reversed_order)
            to_option = option_index_for_display(width, reversed_order)
            per_forward: list[list[float]] = []
            per_forward_logits: list[list[float]] = []
            elapsed = 0.0
            for mapping in mappings:
                prompt = render_panel2(item.state, instructions, display, mapping)
                values, took = score_prompt(model, tokenizer, prompt, mapping, code_ids, device)
                per_forward.append(probabilities_by_option(values, to_option))
                per_forward_logits.append([round(value, 6) for value in values])
                elapsed += took
            chosen = mixture_prediction(per_forward)
            rows.append(
                {
                    "id": item.id,
                    "tier": tier_from_id(item.id),
                    "type": item.question.get("type", "noul"),
                    "arm": arm,
                    "order": "reversed" if reversed_order else "natural",
                    "n_options": width,
                    "gold_pos": gold,
                    "pred_pos": chosen,
                    "correct": chosen == gold,
                    "per_forward_pred": [
                        max(range(width), key=distribution.__getitem__)
                        for distribution in per_forward
                    ],
                    "code_logits": per_forward_logits,
                    "latency_s": elapsed,
                }
            )
    return rows


def _breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = buckets.setdefault(row[key], {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += int(bool(row["correct"]))
    return {
        name: {**entry, "accuracy": entry["correct"] / entry["total"]}
        for name, entry in sorted(buckets.items())
    }


def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Natural and reversed accuracy, the reversal cost, and the position shape."""
    natural = [row for row in rows if row["order"] == "natural"]
    reversed_order = [row for row in rows if row["order"] == "reversed"]
    by_id_reversed = {row["id"]: row for row in reversed_order}
    shared = [row["id"] for row in natural if row["id"] in by_id_reversed]
    correct_natural = {row["id"]: bool(row["correct"]) for row in natural}
    correct_reversed = {item_id: bool(by_id_reversed[item_id]["correct"]) for item_id in shared}
    wins, losses, p_value = mcnemar(correct_reversed, correct_natural, shared)
    natural_pred = {row["id"]: row["pred_pos"] for row in natural}
    changed = sum(
        1 for item_id in shared if by_id_reversed[item_id]["pred_pos"] != natural_pred[item_id]
    )
    accuracy_natural = sum(correct_natural[item_id] for item_id in shared) / len(shared)
    accuracy_reversed = sum(correct_reversed[item_id] for item_id in shared) / len(shared)
    return {
        "n": len(shared),
        "accuracy_natural": accuracy_natural,
        "accuracy_reversed": accuracy_reversed,
        "reversal_delta_pp": (accuracy_reversed - accuracy_natural) * 100.0,
        "reversal_abs_pp": abs(accuracy_reversed - accuracy_natural) * 100.0,
        "reversal_mcnemar": {
            "reversed_only_correct": wins,
            "natural_only_correct": losses,
            "exact_p": p_value,
        },
        "changed_fraction": changed / len(shared),
        "changed_items": changed,
        "latency_s": {
            "mean": statistics.mean(row["latency_s"] for row in natural),
            "median": statistics.median(row["latency_s"] for row in natural),
        },
        "forwards_per_item": len(natural[0]["per_forward_pred"]),
        "position_natural": position_diagnostics(natural),
        "by_tier": _breakdown(natural, "tier"),
        "by_type": _breakdown(natural, "type"),
    }


def build_config(code_form: str, max_items: int | None) -> dict[str, Any]:
    """The run config that is hashed and recorded before the run.

    Deliberately model-independent: the two Qwen3 tokenizers assign the same
    ids to every code, and the model identity belongs in the provenance, so
    one config hash covers both checkpoints.
    """
    return {
        "panel": "2",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "hash_definition": HASH_DEFINITION,
        "mean3_mixture": "mean of per-option probabilities over the three "
        "assignments, then argmax",
        "prompt_template": PANEL2_TEMPLATE,
        "code_scheme": "A..Z, one code per option",
        "code_token_form": code_form,
        "orders": ["natural", "reversed"],
        "max_items": max_items,
        "tokenizer_add_special_tokens": False,
    }


def main() -> int:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-form", choices=["space", "bare"], default="space")
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--print-config-hash", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    prepared = [(item, options_for(item)) for item in items]
    prepared = [(item, pair) for item, pair in prepared if pair is not None]
    config = build_config(args.code_form, args.max_items)

    if args.print_config_hash:
        print(
            json.dumps(
                {
                    "config": config,
                    "config_sha256": config_digest(config),
                    "data_sha256": data_digest([item for item, _ in prepared]),
                    "items": len(prepared),
                },
                indent=2,
            )
        )
        return 0
    if args.output is None:
        parser.error("scoring needs --output")

    print(f"Loaded {len(items)} JevBench items, {len(prepared)} scorable", flush=True)

    snapshot_path = args.snapshot or Path(snapshot_download(args.model))
    print(f"Loading {args.model} from {snapshot_path}", flush=True)
    model, tokenizer = load_causal_lm(
        str(snapshot_path), device=args.device, dtype=DTYPE_MAP[args.dtype]
    )
    device = str(next(model.parameters()).device)

    max_options = max(len(pair[0]) for _, pair in prepared)
    used_codes = list(CODES[:max_options])
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)

    # The reversal invariance is a structural claim for the two content arms;
    # check it here rather than trusting the construction.
    invariance = {}
    for arm in (HASHED, MEAN3):
        ok = True
        for _, (texts, _) in prepared:
            for natural, displayed in zip(natural_codes(arm, texts), display_codes(arm, texts, True)):
                to_option = option_index_for_display(len(texts), True)
                by_option = [None] * len(texts)
                for position, option in enumerate(to_option):
                    by_option[option] = displayed[position]
                ok = ok and by_option == natural
        invariance[arm] = ok

    rows: list[dict[str, Any]] = []
    for index, (item, _) in enumerate(prepared):
        rows.extend(score_item(model, tokenizer, item, code_ids, device))
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(prepared)} items scored", flush=True)

    metrics: dict[str, Any] = {}
    for arm in ARMS:
        metrics[arm] = arm_metrics([row for row in rows if row["arm"] == arm])

    validity = {
        "single_token_codes": len(multi) == 0,
        "multi_token_codes": multi,
        "unscorable_items": len(items) - len(prepared),
        "reversal_invariant_by_construction": invariance,
        "valid": len(multi) == 0 and invariance[HASHED] and invariance[MEAN3],
    }
    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot_path,
        dtype=args.dtype,
        device=device,
        strategy="panel2_order_arms",
        norm="none",
        root=ROOT,
        code_commit=args.code_commit,
        extra={
            "readout": "one forward per assignment; argmax over the K code "
                       "token ids at the final position after 'Answer:'",
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "hash_definition": HASH_DEFINITION,
            "config_sha256": config_digest(config),
            "data_sha256": data_digest([item for item, _ in prepared]),
            "items": len(prepared),
        },
    )

    payload = {
        "_provenance": provenance,
        "config": config,
        "validity": validity,
        "metrics": metrics,
        "predictions": [
            {
                "id": row["id"],
                "tier": row["tier"],
                "type": row["type"],
                "arm": row["arm"],
                "order": row["order"],
                "n_options": row["n_options"],
                "gold_pos": row["gold_pos"],
                "pred_pos": row["pred_pos"],
                "correct": row["correct"],
                "per_forward_pred": row["per_forward_pred"],
                "latency_s": row["latency_s"],
            }
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"valid={validity['valid']} invariance={invariance}")
    for arm in ARMS:
        entry = metrics[arm]
        print(
            f"{arm:<10} natural {entry['accuracy_natural']:.4f} "
            f"reversed {entry['accuracy_reversed']:.4f} "
            f"delta {entry['reversal_delta_pp']:+.2f} pp "
            f"changed {entry['changed_fraction']:.3f} "
            f"W/L {entry['reversal_mcnemar']['reversed_only_correct']}/"
            f"{entry['reversal_mcnemar']['natural_only_correct']} "
            f"p {entry['reversal_mcnemar']['exact_p']:.4g} "
            f"forwards {entry['forwards_per_item']}"
        )
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
