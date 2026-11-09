#!/usr/bin/env python3
"""Attention-mass readout (issue #341): mass over option spans at the answer.

One eager forward per prompt.  At the final position, the attention of the
last four blocks, averaged over heads and blocks, gives one distribution over
the prompt tokens; summing it over each option line's token span gives a
K-vector.  The script scores three readouts on the same forward:

``panel2_eager``
    the shipped vocabulary-code readout, recomputed under eager attention, so
    the comparison with the mass readout uses one implementation;
``mass``
    the span mass alone, argmaxed with the same natural-index tie rule;
``combined``
    the single pre-registered combination, ``log p_code + log mass``.

Read layer, head pooling, span rule and combination were fixed in #341 before
the run; there are no sweep flags.

Usage:
    python model/scripts/eval_jevbench_mass.py --model Qwen/Qwen3-Reranker-4B \
        --code-form space --output results/jevbench/mass_Qwen_Qwen3-Reranker-4B.json
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
    option_index_for_display,
    options_for,
    order_effect,
    position_diagnostics,
    render_panel2,
)
from model.evaluation.panel2_mass import (
    COMBINATION_LAMBDA,
    MASS_FLOOR,
    combine_scores,
    length_guard,
    mean_chance,
    option_spans,
    span_masses,
)
from model.evaluation.run_provenance import build_run_provenance
from model.src.backbone import load_causal_lm

STORED_PANEL2 = ROOT / "results/jevbench/panel2_Qwen_Qwen3-Reranker-4B.json"
COMPARISON_MODEL = "Qwen/Qwen3-Reranker-4B"
#: The four blocks whose attention is averaged; the last four, the same
#: aggregate #337 declared as ``mean_last_n=4``.  A single final layer is
#: dominated by an attention sink: on the alignment smoke it put about 7 % of
#: the head-mean mass on the option spans and about 64 % on the preamble.
READ_LAYERS = (-4, -3, -2, -1)
HEAD_POOLING = "mean over all heads"
LAYER_POOLING = "mean over the last four blocks"
TIE_RULE = "first maximum by natural option index"
PROMPT_ORDER = "shipped"

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


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


def mass_config(
    args: argparse.Namespace, code_ids: dict[str, int], used_codes: list[str], n_layers: int
) -> dict[str, Any]:
    """The declared configuration; hashed before the run and recorded with it."""
    return {
        "panel": "2",
        "dataset": "jevbench",
        "prompt_order": PROMPT_ORDER,
        "prompt_template": PANEL2_TEMPLATE,
        "tie_rule": TIE_RULE,
        "attn_implementation": "eager",
        "read_layers": {
            "relative": list(READ_LAYERS),
            "zero_based_blocks": [n_layers + offset for offset in READ_LAYERS],
        },
        "layer_pooling": LAYER_POOLING,
        "head_pooling": HEAD_POOLING,
        "query_position": "final prompt token",
        "span_rule": "tokens fully inside each option line '{code}. {text}'",
        "mass_aggregation": "sum of head-pooled attention over the span",
        "combination": "log p_code + lambda * log(max(mass, floor))",
        "combination_lambda": COMBINATION_LAMBDA,
        "mass_floor": MASS_FLOOR,
        "num_hidden_layers": n_layers,
        "code_scheme": "A-Z, a-z, then single-character symbols, one code per option",
        "code_token_form": args.code_form,
        "code_token_ids": {
            code: code_ids[code] for code in used_codes if code in code_ids
        },
        "noul_rendering": "two options: A. no / B. yes",
        "choice_rendering": "criteria descriptions in item order",
        "score_rendering": "'Level {i}: {description}'",
        "orders": ["natural", "reversed"],
        "tokenizer_add_special_tokens": False,
    }


def print_config_hash(args: argparse.Namespace) -> int:
    from transformers import AutoConfig, AutoTokenizer

    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    prepared = [options_for(item) for item in items]
    prepared = [entry for entry in prepared if entry is not None]
    max_options = max(len(entry[0]) for entry in prepared) if prepared else 0
    used_codes = list(CODES[:max_options])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)
    n_layers = int(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    config = mass_config(args, code_ids, used_codes, n_layers)
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


def attach_attention_hooks(model: Any) -> tuple[dict[str, Any], list[Any]]:
    """Capture the declared blocks' attention weights and nothing else.

    The top-level forward runs with ``output_attentions=False`` so no other
    layer materialises its matrix; each pre-hook turns the flag on for one
    declared block and its forward hook keeps that block's weights.
    """
    captured: dict[str, Any] = {}
    handles: list[Any] = []
    n_layers = len(model.model.layers)
    for offset in READ_LAYERS:
        block = n_layers + offset
        target = model.model.layers[offset].self_attn

        def force_weights(module: Any, args: Any, kwargs: dict[str, Any]) -> Any:
            kwargs["output_attentions"] = True
            return args, kwargs

        def keep_weights(
            module: Any, args: Any, output: Any, block: int = block
        ) -> None:
            captured[block] = output[1]

        handles.append(
            target.register_forward_pre_hook(force_weights, with_kwargs=True)
        )
        handles.append(target.register_forward_hook(keep_weights))
    return captured, handles


def check_attention(captured: dict[str, Any], n_tokens: int, n_layers: int) -> Any:
    """One head-mean attention row per declared block, averaged over blocks."""
    rows = []
    for offset in READ_LAYERS:
        block = n_layers + offset
        attention = captured.pop(block, None)
        if attention is None:
            raise RuntimeError(f"block {block} returned no attention weights")
        expected = (1, int(attention.shape[1]), n_tokens, n_tokens)
        if tuple(attention.shape) != expected:
            raise RuntimeError(
                f"attention shape {tuple(attention.shape)} != expected {expected}"
            )
        row = attention[0, :, -1, :].float().mean(0)
        # bf16 attention rounds per element; the head-mean row sums to 1 within
        # about one part in a thousand, while a mis-sliced row is far outside.
        if not torch.allclose(row.sum(), torch.ones((), device=row.device), atol=2e-3):
            raise RuntimeError(
                f"block {block} attention row sums to {float(row.sum()):.6f}, not one"
            )
        rows.append(row)
    return torch.stack(rows).mean(0)


def score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    texts: list[str],
    codes: list[str],
    code_ids: dict[str, int],
    device: str,
    captured: dict[str, Any],
) -> dict[str, Any]:
    """One eager forward; code logits, span masses and the residual."""
    alignment = option_spans(tokenizer, prompt, texts, codes)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    n_tokens = int(encoded["input_ids"].shape[1])
    if n_tokens != alignment["n_tokens"]:
        raise RuntimeError(
            f"forward has {n_tokens} tokens, alignment has {alignment['n_tokens']}"
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**encoded)
        row = check_attention(captured, n_tokens, len(model.model.layers))
    elapsed = time.perf_counter() - started
    values = [float(output.logits[0, -1, code_ids[code]]) for code in codes]
    masses = span_masses(row.tolist(), alignment["spans"])
    return {
        "values": values,
        "masses": masses["masses"],
        "span_sum": masses["span_sum"],
        "preamble_mass": masses["preamble_mass"],
        "residual": masses["residual"],
        "straddling": len(alignment["straddling_tokens"]),
        "n_tokens": n_tokens,
        "elapsed": elapsed,
    }


def prediction_row(
    item: Any,
    order: str,
    texts: list[str],
    gold: int,
    values: list[float],
    prediction: int,
    elapsed: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    tier = tier_from_id(item.id)
    if tier not in ("easy", "hard", "original"):
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
        "score_values": values,
        "span_sum": diagnostics["span_sum"],
        "preamble_mass": diagnostics["preamble_mass"],
        "residual": diagnostics["residual"],
        "straddling_tokens": diagnostics["straddling"],
        "latency_s": elapsed,
    }


def cell_summary(rows: list[dict[str, Any]], items: list[Any], failed: int) -> dict[str, Any]:
    by_id = {item.id: item for item in items}
    with_items = [{**row, "_item": by_id[row["id"]]} for row in rows]
    summary = summarize(with_items, failed=failed)
    tied = [row for row in rows if has_exact_tie(row["score_values"])]
    overall = summary["overall"]
    return {
        "summary": summary,
        "position": position_diagnostics(rows),
        "tie_items": len(tied),
        "tie_items_correct": sum(1 for row in tied if row["correct"]),
        "accuracy": overall["accuracy"],
        "correct": overall["correct"],
        "total": overall["total"],
    }


def score_model(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    items = list(load_jevbench())
    if args.max_items:
        items = items[: args.max_items]
    print(f"Loaded {len(items)} JevBench items", flush=True)

    snapshot_path = args.snapshot or Path(snapshot_download(args.model))
    model, tokenizer = load_causal_lm(
        str(snapshot_path), device=args.device, dtype=DTYPE_MAP[args.dtype]
    )
    model.set_attn_implementation("eager")
    if model.config._attn_implementation != "eager":
        raise RuntimeError(
            f"attention implementation is {model.config._attn_implementation}, not eager"
        )
    device = str(next(model.parameters()).device)
    captured, handles = attach_attention_hooks(model)

    prepared = [options_for(item) for item in items]
    prepared = [entry for entry in prepared if entry is not None]
    max_options = max(len(entry[0]) for entry in prepared) if prepared else 0
    used_codes = list(CODES[:max_options])
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)

    cells = {name: {"natural": [], "reversed": []} for name in ("panel2_eager", "mass", "combined")}
    unscorable: list[dict[str, Any]] = []
    options_by_id: dict[str, int] = {}
    residuals: list[float] = []
    span_sums: list[float] = []
    preambles: list[float] = []
    straddling_total = 0
    length_entries: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        prepared_item = options_for(item)
        if prepared_item is None:
            unscorable.append({"id": item.id, "reason": "gold_not_in_options"})
            continue
        texts, gold = prepared_item
        codes = codes_for(len(texts))
        options_by_id[item.id] = len(texts)
        missing = [code for code in codes if code not in code_ids]
        if missing:
            unscorable.append(
                {"id": item.id, "reason": "multi_token_code", "codes": missing}
            )
            continue

        prompt = render_panel2(
            item.state,
            item.question.get("instructions", ""),
            texts,
            codes,
        )
        result = score_prompt(
            model, tokenizer, prompt, texts, codes, code_ids, device, captured
        )
        residuals.append(result["residual"])
        span_sums.append(result["span_sum"])
        preambles.append(result["preamble_mass"])
        straddling_total += result["straddling"]

        display_to_option = option_index_for_display(len(texts), False)
        panel2_choice = display_to_option.index(
            choose_option(result["values"], display_to_option)
        )
        mass_natural = choose_option(result["masses"], display_to_option)
        mass_choice = display_to_option.index(mass_natural)
        if item.question.get("type", "noul") in ("choice", "score"):
            length_entries.append(
                {
                    "texts": texts,
                    "n_options": len(texts),
                    "gold": gold,
                    "chosen": mass_natural,
                }
            )
        combined_values = combine_scores(result["values"], result["masses"])
        combined_choice = display_to_option.index(
            choose_option(combined_values, display_to_option)
        )
        for name, values, choice in (
            ("panel2_eager", result["values"], panel2_choice),
            ("mass", result["masses"], mass_choice),
            ("combined", combined_values, combined_choice),
        ):
            cells[name]["natural"].append(
                prediction_row(
                    item, "natural", texts, gold, values, choice,
                    result["elapsed"], result,
                )
            )

        reversed_texts = list(reversed(texts))
        reversed_gold = len(texts) - 1 - gold
        reversed_prompt = render_panel2(
            item.state,
            item.question.get("instructions", ""),
            reversed_texts,
            codes,
        )
        reversed_result = score_prompt(
            model,
            tokenizer,
            reversed_prompt,
            reversed_texts,
            codes,
            code_ids,
            device,
            captured,
        )
        reversed_display = option_index_for_display(len(texts), True)
        reversed_panel2 = reversed_display.index(
            choose_option(reversed_result["values"], reversed_display)
        )
        reversed_mass = reversed_display.index(
            choose_option(reversed_result["masses"], reversed_display)
        )
        reversed_combined_values = combine_scores(
            reversed_result["values"], reversed_result["masses"]
        )
        reversed_combined = reversed_display.index(
            choose_option(reversed_combined_values, reversed_display)
        )
        for name, values, choice in (
            ("panel2_eager", reversed_result["values"], reversed_panel2),
            ("mass", reversed_result["masses"], reversed_mass),
            ("combined", reversed_combined_values, reversed_combined),
        ):
            cells[name]["reversed"].append(
                prediction_row(
                    item, "reversed", reversed_texts, reversed_gold, values,
                    choice, reversed_result["elapsed"], reversed_result,
                )
            )

        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(items)} items scored", flush=True)

    metrics: dict[str, Any] = {}
    for name in cells:
        natural_rows = cells[name]["natural"]
        reversed_rows = cells[name]["reversed"]
        metrics[name] = {
            "natural": cell_summary(natural_rows, items, len(unscorable)),
            "reversed": cell_summary(reversed_rows, items, len(unscorable)),
            "order_effect": order_effect(natural_rows, reversed_rows, options_by_id),
            "latency": {
                "mean_s": statistics.mean(row["latency_s"] for row in natural_rows)
                if natural_rows
                else 0.0,
            },
        }

    from model.scripts.paired_mcnemar import mcnemar

    def correctness(name: str, order: str) -> dict[str, bool]:
        return {
            row["id"]: bool(row["correct"]) for row in cells[name][order]
        }

    comparisons: dict[str, Any] = {}
    for other in ("mass", "combined"):
        for order in ("natural", "reversed"):
            base = correctness("panel2_eager", order)
            current = correctness(other, order)
            shared = sorted(set(base) & set(current))
            wins, losses, p = mcnemar(current, base, shared)
            comparisons[f"{other}_vs_panel2_eager_{order}"] = {
                "n": len(shared),
                "wins": wins,
                "losses": losses,
                "mcnemar_p": p,
            }

    stored_note = None
    if STORED_PANEL2.exists() and not args.max_items:
        stored = json.loads(STORED_PANEL2.read_text())
        stored_correct = {
            row["id"]: bool(row["correct"]) for row in stored["predictions_natural"]
        }
        current = correctness("panel2_eager", "natural")
        shared = sorted(set(stored_correct) & set(current))
        wins, losses, p = mcnemar(current, stored_correct, shared)
        stored_note = {
            "stored": str(STORED_PANEL2.relative_to(ROOT)),
            "n": len(shared),
            "eager_only_correct": wins,
            "stored_only_correct": losses,
            "mcnemar_p": p,
        }

    chance = mean_chance(options_by_id)
    config = mass_config(
        args, code_ids, used_codes, int(model.config.num_hidden_layers)
    )
    code_failures = [
        entry for entry in unscorable if entry["reason"] == "multi_token_code"
    ]
    withheld_gold = [
        entry for entry in unscorable if entry["reason"] == "gold_not_in_options"
    ]
    residual_stats = {
        "mean": statistics.mean(residuals) if residuals else 0.0,
        "median": statistics.median(residuals) if residuals else 0.0,
        "max": max(residuals) if residuals else 0.0,
        "min": min(residuals) if residuals else 0.0,
        "span_sum_mean": statistics.mean(span_sums) if span_sums else 0.0,
        "span_sum_min": min(span_sums) if span_sums else 0.0,
        "preamble_mean": statistics.mean(preambles) if preambles else 0.0,
    }
    alignment_ok = (
        residual_stats["min"] >= -1e-4
        and residual_stats["span_sum_min"] > 0.0
    )
    validity = {
        "single_token_codes": len(multi) == 0 and not code_failures,
        "multi_token_codes": multi,
        "unscorable_items": len(unscorable),
        "unscorable_code_failures": len(code_failures),
        "unscorable_withheld_gold": len(withheld_gold),
        "span_alignment": alignment_ok,
        "straddling_tokens_total": straddling_total,
        "residual": residual_stats,
        "length_guard": length_guard(length_entries),
        "attn_implementation": "eager",
        "valid": False,
    }
    validity["valid"] = validity["single_token_codes"] and alignment_ok

    provenance = build_run_provenance(
        model_id=args.model,
        snapshot_path=snapshot_path,
        dtype=args.dtype,
        device=device,
        strategy="panel2_attention_mass",
        norm="none",
        root=ROOT,
        code_commit=args.code_commit,
        extra={
            "readout": "eager forward; last-four-block attention at the final "
                       "position, mean over heads and blocks; mass summed over "
                       "each option line's tokens",
            "prompt_template": PANEL2_TEMPLATE,
            "prompt_order": PROMPT_ORDER,
            "attn_implementation": "eager",
            "read_layers": [
                int(model.config.num_hidden_layers) + offset
                for offset in READ_LAYERS
            ],
            "layer_pooling": LAYER_POOLING,
            "head_pooling": HEAD_POOLING,
            "combination": "log p_code + lambda * log mass",
            "combination_lambda": COMBINATION_LAMBDA,
            "mass_floor": MASS_FLOOR,
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
        "chance_level": chance,
        "metrics": metrics,
        "comparisons": comparisons,
        "panel2_eager_vs_stored": stored_note,
        "predictions": {name: cells[name] for name in cells},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    for handle in handles:
        handle.remove()
    overall = metrics["panel2_eager"]["natural"]["summary"]["overall"]
    mass_overall = metrics["mass"]["natural"]["summary"]["overall"]
    combined_overall = metrics["combined"]["natural"]["summary"]["overall"]
    print(
        f"written {args.output}: panel2_eager {overall['correct']}/{overall['total']}, "
        f"mass {mass_overall['correct']}/{mass_overall['total']}, "
        f"combined {combined_overall['correct']}/{combined_overall['total']}, "
        f"chance {chance:.4f}, valid={validity['valid']}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--code-form", choices=["space", "bare"], default="space")
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--print-config-hash",
        action="store_true",
        help="print the config and data hashes this argument set will record",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.print_config_hash:
        if not args.model:
            parser.error("--print-config-hash needs --model")
        return print_config_hash(args)
    if not args.model or args.output is None:
        parser.error("scoring needs --model and --output")
    return score_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
