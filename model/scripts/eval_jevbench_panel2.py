#!/usr/bin/env python3
"""Panel 2 zero-shot: one-forward vocabulary-code readout on JevBench public.

The context and every option go into one sequence, each option carries a
single-token code, and the answer is the argmax of the next-token distribution
at the final position restricted to the K code ids.  One forward per item
instead of one per option, every option visible, and the score comes from the
model's own vocabulary.

All four checkpoints use the same template, the same code scheme and the same
option construction.  That identity is the experiment (#326).

Modes:
    --check-codes        verify every code is one token in all four tokenizers
    --print-config-hash  print the config and data hashes a run will record
    (default)            score one checkpoint in natural and reversed option order
    --summarize          build the cross-model matrix and the reranker's McNemar

``--prompt-order`` selects one of the declared templates in
``model.evaluation.panel2.ORDER_TEMPLATES`` (#336).  ``shipped`` is the template
every panel-2 cell before #336 used; ``options_first`` moves the question block
after the options and changes nothing else, so the pair is an order-only
contrast; ``sandwich`` repeats the question after the options, so it differs in
content and token count as well as position and is a repetition treatment.

The prompt template lives in ``model.evaluation.panel2`` and is built only by
``render_panel2``.  A noul item is rendered as a two-option question,
``A. no`` / ``B. yes``, so the 231-item total stays comparable to the 150/231
judgement baseline.

``--dataset dev`` runs the same readout on the internal 828-row development
set.  The 16 rows whose gold is withheld cannot be scored by any readout; they
are reported as ``unscorable`` with reason ``gold_not_in_options`` and are left
out of the accuracy, as the judgement dev cell leaves them out.

Usage:
    python model/scripts/eval_jevbench_panel2.py --check-codes \\
        --models Qwen/Qwen3-Reranker-4B,Qwen/Qwen3-4B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-4B-Base

    python model/scripts/eval_jevbench_panel2.py --model Qwen/Qwen3-4B \\
        --code-form space --output results/jevbench/panel2_Qwen_Qwen3-4B.json

    python model/scripts/eval_jevbench_panel2.py --summarize results/jevbench/panel2_*.json \\
        --output results/jevbench/panel2_summary.json
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
from model.evaluation.dev_set import load_dev_set
from model.evaluation.jevbench import summarize, tier_from_id
from model.evaluation.panel2 import (
    CODES,
    ORDER_TEMPLATES,
    choose_option,
    code_token_ids,
    codes_for,
    has_exact_tie,
    option_index_for_display,
    options_for,
    order_effect,
    pick_code_form,
    position_diagnostics,
    render_panel2,
    tie_report,
)
from model.evaluation.run_provenance import build_run_provenance
from model.src.backbone import load_causal_lm

BASELINE_JUDGEMENT = (
    ROOT / "model/experiments/logit_prov/jevbench_judgement_Qwen_Qwen3-Reranker-4B.json"
)
COMPARISON_MODEL = "Qwen/Qwen3-Reranker-4B"
DIAGNOSTIC_SAMPLE = 10

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def data_digest(items: list[Any]) -> str:
    """Hash of the identities and labels this run scores."""
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


def load_items(args: argparse.Namespace) -> list[Any]:
    if args.dataset == "dev":
        items = list(load_dev_set(args.dev_path))
    else:
        items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    return items


def readout_codes(items: list[Any]) -> list[str]:
    """The codes the scored items need, in order."""
    prepared = [options_for(item) for item in items]
    prepared = [entry for entry in prepared if entry is not None]
    max_options = max(len(entry[0]) for entry in prepared) if prepared else 0
    return list(CODES[:max_options])


def panel2_config(
    args: argparse.Namespace, code_ids: dict[str, int], used_codes: list[str]
) -> dict[str, Any]:
    """The declared configuration; hashed before a run and recorded with it."""
    return {
        "panel": "2",
        "dataset": args.dataset,
        "prompt_order": args.prompt_order,
        "prompt_template": ORDER_TEMPLATES[args.prompt_order],
        "tie_rule": "first maximum by natural option index",
        "code_scheme": "A-Z, a-z, then single-character symbols, one code per option",
        "code_token_form": args.code_form,
        "code_token_ids": {
            code: code_ids[code] for code in used_codes if code in code_ids
        },
        "noul_rendering": "two options: A. no / B. yes",
        "choice_rendering": "criteria descriptions in item order",
        "score_rendering": "'Level {i}: {description}'",
        "orders": ["natural", "reversed"],
        "diagnostic_sample_items": args.diagnostic_sample,
        "max_items": args.max_items,
        "tokenizer_add_special_tokens": False,
    }


def print_config_hash(args: argparse.Namespace) -> int:
    """The config and data hashes a run of these arguments will record.

    Loads the tokenizer and the items only -- no weights -- so the hash can be
    posted on the issue before the model run starts.
    """
    from transformers import AutoTokenizer

    items = load_items(args)
    used_codes = readout_codes(items)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)
    config = panel2_config(args, code_ids, used_codes)
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


def check_codes(args: argparse.Namespace) -> int:
    """Verify every code against every tokenizer before any model is loaded."""
    from transformers import AutoTokenizer

    codes = list(CODES[: args.max_codes])
    tokenizers = {
        model_id: AutoTokenizer.from_pretrained(model_id)
        for model_id in args.models
    }
    form = pick_code_form(list(tokenizers.values()), codes)
    report: dict[str, Any] = {
        "codes": codes,
        "chosen_form": form,
        "models": {},
    }
    for model_id, tokenizer in tokenizers.items():
        ids, multi = code_token_ids(tokenizer, codes, form)
        bare_ids, bare_multi = code_token_ids(tokenizer, codes, "bare")
        report["models"][model_id] = {
            "code_token_ids": ids,
            "multi_token": multi,
            "bare_token_ids": bare_ids,
            "bare_multi_token": bare_multi,
        }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    return 0


def score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    codes: list[str],
    code_ids: dict[str, int],
    device: str,
    display_to_option: list[int],
) -> tuple[list[float], int, float, Any, int]:
    """One forward; the K code logits and the natural option they choose.

    The choice goes through :func:`model.evaluation.panel2.choose_option`, so
    an exact tie resolves to the same natural option in either display order
    (#347).  The prompt token count is returned for the order comparison (#336).
    """
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    n_tokens = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].float()
    elapsed = time.perf_counter() - started
    values = [float(logits[code_ids[code]]) for code in codes]
    chosen = choose_option(values, display_to_option)
    return values, chosen, elapsed, logits, n_tokens


def prediction_row(
    item: Any,
    order: str,
    texts: list[str],
    gold: int,
    values: list[float],
    prediction: int,
    elapsed: float,
) -> dict[str, Any]:
    """One saved prediction, in the contract the summary mode reads."""
    tier = tier_from_id(item.id)
    if tier not in ("easy", "hard", "original"):
        # The dev set's ids are dataset/split hashes, not JevBench tiers.
        tier = "dev"
    return {
        "id": item.id,
        "tier": tier,
        "type": item.question.get("type", "noul"),
        "order": order,
        "n_options": len(texts),
        "gold_pos": gold,
        "pred_pos": prediction,
        "pred_code": codes_for(len(texts))[prediction],
        "correct": prediction == gold,
        "code_logits": values,
        "latency_s": elapsed,
    }


def score_model(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    items = load_items(args)
    print(f"Loaded {len(items)} {args.dataset} items", flush=True)

    snapshot_path = args.snapshot or Path(snapshot_download(args.model))
    print(f"Loading {args.model} from {snapshot_path}", flush=True)
    model, tokenizer = load_causal_lm(
        str(snapshot_path), device=args.device, dtype=DTYPE_MAP[args.dtype]
    )
    device = str(next(model.parameters()).device)

    used_codes = readout_codes(items)
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)

    natural: list[dict[str, Any]] = []
    reversed_order: list[dict[str, Any]] = []
    ties_natural: list[bool] = []
    ties_reversed: list[bool] = []
    natural_tokens: list[int] = []
    reversed_tokens: list[int] = []
    unscorable: list[dict[str, Any]] = []
    diagnostic_sample: list[dict[str, Any]] = []
    options_by_id: dict[str, int] = {}
    step = max(1, len(items) // max(1, args.diagnostic_sample))
    sample_ids = {item.id for item in items[::step][: args.diagnostic_sample]}

    for index, item in enumerate(items):
        prepared = options_for(item)
        if prepared is None:
            unscorable.append({"id": item.id, "reason": "gold_not_in_options"})
            continue
        texts, gold = prepared
        codes = codes_for(len(texts))
        options_by_id[item.id] = len(texts)
        missing = [code for code in codes if code not in code_ids]
        if missing:
            unscorable.append(
                {
                    "id": item.id,
                    "reason": "multi_token_code",
                    "codes": missing,
                }
            )
            continue

        prompt = render_panel2(
            item.state,
            item.question.get("instructions", ""),
            texts,
            codes,
            order=args.prompt_order,
        )
        display_to_option = option_index_for_display(len(texts), False)
        values, chosen, elapsed, logits, n_tokens = score_prompt(
            model, tokenizer, prompt, codes, code_ids, device, display_to_option
        )
        prediction = display_to_option.index(chosen)
        ties_natural.append(has_exact_tie(values))
        natural_tokens.append(n_tokens)
        natural.append(
            prediction_row(item, "natural", texts, gold, values, prediction, elapsed)
        )

        if item.id in sample_ids:
            probabilities = torch.softmax(logits, dim=-1)
            top = torch.topk(probabilities, 5)
            diagnostic_sample.append(
                {
                    "id": item.id,
                    "gold_code": codes[gold],
                    "predicted_code": codes[prediction],
                    "top5": [
                        {
                            "token_id": int(token_id),
                            "token": tokenizer.decode([int(token_id)]),
                            "probability": float(probability),
                        }
                        for probability, token_id in zip(top.values, top.indices)
                    ],
                }
            )

        reversed_texts = list(reversed(texts))
        reversed_gold = len(texts) - 1 - gold
        reversed_prompt = render_panel2(
            item.state,
            item.question.get("instructions", ""),
            reversed_texts,
            codes,
            order=args.prompt_order,
        )
        reversed_display_to_option = option_index_for_display(len(texts), True)
        (
            reversed_values,
            reversed_chosen,
            reversed_elapsed,
            _,
            reversed_n_tokens,
        ) = score_prompt(
            model,
            tokenizer,
            reversed_prompt,
            codes,
            code_ids,
            device,
            reversed_display_to_option,
        )
        ties_reversed.append(has_exact_tie(reversed_values))
        reversed_tokens.append(reversed_n_tokens)
        reversed_order.append(
            prediction_row(
                item,
                "reversed",
                reversed_texts,
                reversed_gold,
                reversed_values,
                reversed_display_to_option.index(reversed_chosen),
                reversed_elapsed,
            )
        )

        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(items)} items scored", flush=True)

    def with_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {item.id: item for item in items}
        return [{**row, "_item": by_id[row["id"]]} for row in rows]

    natural_summary = summarize(with_items(natural), failed=len(unscorable))
    reversed_summary = summarize(with_items(reversed_order), failed=len(unscorable))
    natural_positions = position_diagnostics(natural)
    reversed_positions = position_diagnostics(reversed_order)
    order = order_effect(natural, reversed_order, options_by_id)

    latency_natural = [row["latency_s"] for row in natural] or [0.0]
    latency_reversed = [row["latency_s"] for row in reversed_order] or [0.0]
    latency = {
        "natural": {
            "mean_s": statistics.mean(latency_natural),
            "median_s": statistics.median(latency_natural),
        },
        "reversed": {
            "mean_s": statistics.mean(latency_reversed),
            "median_s": statistics.median(latency_reversed),
        },
    }
    prompt_tokens = {
        "natural": {
            "mean": statistics.mean(natural_tokens) if natural_tokens else 0.0,
            "median": statistics.median(natural_tokens) if natural_tokens else 0.0,
        },
        "reversed": {
            "mean": statistics.mean(reversed_tokens) if reversed_tokens else 0.0,
            "median": statistics.median(reversed_tokens) if reversed_tokens else 0.0,
        },
    }

    config = panel2_config(args, code_ids, used_codes)
    code_failures = [
        entry for entry in unscorable if entry["reason"] == "multi_token_code"
    ]
    withheld_gold = [
        entry for entry in unscorable if entry["reason"] == "gold_not_in_options"
    ]
    validity = {
        "single_token_codes": len(multi) == 0 and not code_failures,
        "multi_token_codes": multi,
        "unscorable_items": len(unscorable),
        "unscorable_code_failures": len(code_failures),
        "unscorable_withheld_gold": len(withheld_gold),
        "position_diagnostic": True,
        "position_dominated": natural_positions["position_dominated"],
        "valid": False,
        "invalid_reason": None,
    }
    validity["valid"] = validity["single_token_codes"] and validity["position_diagnostic"]
    validity["tie_items"] = {
        "definition": "an item whose two largest code logits are exactly equal",
        "cells": {
            "positional": {
                "natural": tie_report(natural, [[flag] for flag in ties_natural]),
                "reversed": tie_report(reversed_order, [[flag] for flag in ties_reversed]),
            }
        },
    }
    validity["tie_items_correct"] = {
        "positional": {
            "natural": validity["tie_items"]["cells"]["positional"]["natural"]["correct"],
            "reversed": validity["tie_items"]["cells"]["positional"]["reversed"]["correct"],
        }
    }
    if not validity["single_token_codes"]:
        validity["invalid_reason"] = (
            "at least one code used by an item is not a single token in this "
            "tokenizer; see multi_token_codes and unscorable"
        )
    if validity["position_dominated"]:
        validity["note"] = (
            "position-dominated: raw and bias-excluded accuracy differ by more "
            "than 10 pp; report the bias-excluded figure beside the raw one"
        )

    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot_path,
        dtype=args.dtype,
        device=device,
        strategy="panel2_vocab_code",
        norm="none",
        root=ROOT,
        code_commit=args.code_commit,
        extra={
            "readout": "one forward; the K code logits at the final position "
                       "after 'Answer:'; first maximum by natural option index, "
                       "ties included",
            "prompt_template": ORDER_TEMPLATES[args.prompt_order],
            "prompt_order": args.prompt_order,
            "code_token_form": args.code_form,
            "dataset": args.dataset,
            "dev_path": (
                str(args.dev_path.relative_to(ROOT))
                if args.dataset == "dev" and args.dev_path.is_relative_to(ROOT)
                else None
            ),
            "config_sha256": config_digest(config),
            "data_sha256": data_digest(items),
            "items": len(items),
            "elapsed_s": round(sum(latency_natural) + sum(latency_reversed), 1),
        },
    )

    payload = {
        "_provenance": provenance,
        "config": config,
        "validity": validity,
        "unscorable": unscorable,
        "metrics": {
            "natural": {
                "summary": natural_summary,
                "position": natural_positions,
            },
            "reversed": {
                "summary": reversed_summary,
                "position": reversed_positions,
            },
            "order_effect": order,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
        },
        "diagnostic_sample": diagnostic_sample,
        "predictions_natural": natural,
        "predictions_reversed": reversed_order,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    overall = natural_summary["overall"]
    print(
        f"written {args.output}: {overall['correct']}/{overall['total']} "
        f"= {overall['accuracy']:.4f}, valid={validity['valid']}",
        flush=True,
    )
    return 0


def summarize_panel2(args: argparse.Namespace) -> int:
    from model.scripts.paired_mcnemar import mcnemar

    runs: dict[str, dict[str, Any]] = {}
    for path in args.summarize:
        payload = json.loads(path.read_text())
        model_id = payload["_provenance"]["model_id"]
        runs[model_id] = {"path": path, "payload": payload}

    models: dict[str, Any] = {}
    for model_id, run in runs.items():
        payload = run["payload"]
        natural = payload["metrics"]["natural"]
        models[model_id] = {
            "valid": payload["validity"]["valid"],
            "validity": payload["validity"],
            "overall": natural["summary"]["overall"],
            "by_tier": natural["summary"]["by_tier"],
            "by_type": natural["summary"]["by_type"],
            "by_tier_type": natural["summary"]["by_tier_type"],
            "position": natural["position"],
            "order_effect": payload["metrics"]["order_effect"],
            "latency": payload["metrics"]["latency"],
            "prompt_tokens": payload["metrics"].get("prompt_tokens"),
            "reversed_overall": payload["metrics"]["reversed"]["summary"]["overall"],
        }

    baseline = json.loads(args.baseline.read_text())
    baseline_correct = {
        row["id"]: bool(row["correct"]) for row in baseline["predictions"]
    }
    comparisons: dict[str, Any] = {}
    reranker = runs.get(COMPARISON_MODEL)
    if reranker is not None and reranker["payload"]["validity"]["valid"]:
        panel2_correct = {
            row["id"]: bool(row["correct"])
            for row in reranker["payload"]["predictions_natural"]
        }
        shared = sorted(set(panel2_correct) & set(baseline_correct))
        wins, losses, p = mcnemar(panel2_correct, baseline_correct, shared)
        comparisons[COMPARISON_MODEL] = {
            "baseline": "judgement readout, same 231 items",
            "baseline_accuracy": sum(baseline_correct[i] for i in shared) / len(shared),
            "panel2_accuracy": sum(panel2_correct[i] for i in shared) / len(shared),
            "n": len(shared),
            "panel2_only_correct": wins,
            "baseline_only_correct": losses,
            "mcnemar_p": p,
        }

    payload = {
        "_provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_commit": runs[next(iter(runs))]["payload"]["_provenance"]["code_commit"]
            if runs
            else "unresolved",
            "files": {
                model_id: {
                    "path": str(run["path"]),
                    "sha256": hashlib.sha256(run["path"].read_bytes()).hexdigest(),
                }
                for model_id, run in runs.items()
            },
            "baseline": {
                "path": str(args.baseline),
                "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            },
        },
        "models": models,
        "comparisons": comparisons,
        "validity_gate": {
            model_id: {
                "valid": entry["validity"]["valid"],
                "reason": entry["validity"].get("invalid_reason"),
            }
            for model_id, entry in models.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"written {args.output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-codes", action="store_true")
    parser.add_argument("--models", default="", help="comma-separated ids for --check-codes")
    parser.add_argument("--max-codes", type=int, default=6)
    parser.add_argument("--model", default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-form", choices=["space", "bare"], default="space")
    parser.add_argument(
        "--prompt-order",
        choices=list(ORDER_TEMPLATES),
        default="shipped",
        help=(
            "shipped: as in #326; options_first: question after the options, "
            "otherwise identical; sandwich: question repeated after the "
            "options, a repetition treatment rather than an order point"
        ),
    )
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dataset",
        choices=["jevbench", "dev"],
        default="jevbench",
        help="jevbench: the 231-item public set; dev: the internal 828-row set",
    )
    parser.add_argument(
        "--dev-path",
        type=Path,
        default=ROOT / "data/benchmarks/dev/dev_800.jsonl",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--diagnostic-sample", type=int, default=DIAGNOSTIC_SAMPLE)
    parser.add_argument("--summarize", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--print-config-hash",
        action="store_true",
        help="print the config and data hashes this argument set will record",
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE_JUDGEMENT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.check_codes:
        args.models = [model.strip() for model in args.models.split(",") if model.strip()]
        if not args.models:
            parser.error("--check-codes needs --models")
        return check_codes(args)
    if args.summarize:
        if args.output is None:
            parser.error("--summarize needs --output")
        return summarize_panel2(args)
    if args.print_config_hash:
        if not args.model:
            parser.error("--print-config-hash needs --model")
        return print_config_hash(args)
    if not args.model or args.output is None:
        parser.error("scoring needs --model and --output")
    return score_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
