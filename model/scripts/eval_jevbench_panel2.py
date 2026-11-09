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

``--read-config`` selects where the code logits are read (#337).  ``last`` is
the shipped read and returns the model's own next-token logits.  ``layer=K``
applies the final norm and LM head to hidden state K; on the verified
transformers 5.21 layout, index 0 is the embedding output, index k is block k's
output, and the last entry is the final normed state.  ``mean_last_n=4``
averages the last four raw block outputs and excludes the final normed entry.
A middle-layer read applies the norm and head to a representation that norm was
never trained for: that is the logit-lens caveat, and it applies to every
configuration except ``last``.  ``all`` reads every declared configuration from
one forward; the five are then paired by construction, and the reported latency
splits into the shared forward and the per-configuration norm-and-head cost.
The first forward of a run checks the layout and fails loudly if the library's
hidden-state semantics change.

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
    parse_read_spec,
    pick_code_form,
    position_diagnostics,
    read_specs,
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
    args: argparse.Namespace,
    code_ids: dict[str, int],
    used_codes: list[str],
    n_layers: int,
    specs: dict[str, tuple[str, int] | tuple[str]],
) -> dict[str, Any]:
    """The declared configuration; hashed before a run and recorded with it."""
    return {
        "panel": "2",
        "dataset": args.dataset,
        "prompt_order": args.prompt_order,
        "prompt_template": ORDER_TEMPLATES[args.prompt_order],
        "tie_rule": "first maximum by natural option index",
        "read_config": args.read_config,
        "read_configs": {name: list(spec) for name, spec in specs.items()},
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
        "diagnostic_sample_items": args.diagnostic_sample,
        "max_items": args.max_items,
        "tokenizer_add_special_tokens": False,
    }


def print_config_hash(args: argparse.Namespace) -> int:
    """The config and data hashes a run of these arguments will record.

    Loads the tokenizer and the items only -- no weights -- so the hash can be
    posted on the issue before the model run starts.  The layer count comes
    from the model config, not from the weights.
    """
    from transformers import AutoConfig, AutoTokenizer

    items = load_items(args)
    used_codes = readout_codes(items)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    code_ids, multi = code_token_ids(tokenizer, used_codes, args.code_form)
    n_layers = int(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    if args.read_config == "all":
        specs = read_specs(n_layers)
    else:
        specs = {args.read_config: parse_read_spec(args.read_config, n_layers)}
    config = panel2_config(args, code_ids, used_codes, n_layers, specs)
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


def check_read_layout(model: Any, output: Any) -> None:
    """Fail loudly if the hidden-state layout is not the one the reads assume.

    Verified on transformers 5.21: the tuple has ``n_layers + 1`` entries,
    index 0 is the embedding output, index k is the output of block k for
    ``1 <= k <= n_layers - 1``, and the last entry is the final normed state
    rather than block ``n_layers``'s raw output.  The last entry must be the
    tensor the LM head reads; if a library change breaks that, a middle-layer
    read would silently mean a different layer.
    """
    hidden_states = output.hidden_states
    n_layers = int(model.config.num_hidden_layers)
    if len(hidden_states) != n_layers + 1:
        raise RuntimeError(
            f"hidden-state tuple has {len(hidden_states)} entries for "
            f"{n_layers} blocks; the read layout no longer matches read_specs"
        )
    if not torch.equal(model.lm_head(hidden_states[-1]), output.logits):
        raise RuntimeError(
            "the last hidden state is not the tensor the LM head reads; "
            "hidden-state semantics changed"
        )


def read_logits(
    model: Any,
    hidden_states: tuple[Any, ...],
    spec: tuple[str, int] | tuple[str],
    last_logits: Any,
) -> Any:
    """Logits from one declared read of the final position's hidden states.

    ``last`` returns the model's own next-token logits unchanged.  A
    middle-layer read applies the final norm and the LM head to a
    representation that norm was never trained for: the logit-lens caveat.
    ``mean_last_n`` averages raw block outputs and excludes the final normed
    entry, so the average is over the last n blocks the tuple exposes.
    """
    kind = spec[0]
    if kind == "last":
        return last_logits
    if kind == "layer":
        hidden = hidden_states[spec[1]][0, -1]
    elif kind == "mean_last_n":
        hidden = torch.stack(
            [state[0, -1] for state in hidden_states[-(spec[1] + 1):-1]]
        ).mean(0)
    else:
        raise ValueError(f"unknown read spec {spec!r}")
    return model.lm_head(model.model.norm(hidden)).float()


def score_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    codes: list[str],
    code_ids: dict[str, int],
    device: str,
    display_to_option: list[int],
    specs: dict[str, tuple[str, int] | tuple[str]],
    verify_layout: bool = False,
) -> tuple[dict[str, dict[str, Any]], Any, float, int]:
    """One forward; one code read per declared configuration.

    The forward is shared, so the five configurations see identical hidden
    states and the comparison is paired by construction.  ``read_s`` is the
    per-configuration norm-and-head cost on top of the shared forward.
    ``verify_layout`` checks the hidden-state semantics on the first item.
    Every configuration chooses through
    :func:`model.evaluation.panel2.choose_option`, so an exact tie resolves to
    the same natural option in either display order (#347).
    """
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    n_tokens = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(device) for key, value in encoded.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True)
    forward_elapsed = time.perf_counter() - started
    if verify_layout:
        check_read_layout(model, output)
    last_logits = output.logits[0, -1].float()

    reads: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for name, spec in specs.items():
            read_started = time.perf_counter()
            logits = read_logits(model, output.hidden_states, spec, last_logits)
            values = [float(logits[code_ids[code]]) for code in codes]
            chosen = choose_option(values, display_to_option)
            reads[name] = {
                "values": values,
                "chosen": chosen,
                "read_s": time.perf_counter() - read_started,
            }
    return reads, last_logits, forward_elapsed, n_tokens


def prediction_row(
    item: Any,
    order: str,
    texts: list[str],
    gold: int,
    values: list[float],
    prediction: int,
    elapsed: float,
    read: str = "last",
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
        "read": read,
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

    n_layers = int(model.config.num_hidden_layers)
    if args.read_config == "all":
        specs = read_specs(n_layers)
    else:
        specs = {args.read_config: parse_read_spec(args.read_config, n_layers)}
    print(f"Read configurations: {list(specs)}", flush=True)

    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"natural": [], "reversed": []} for name in specs
    }
    latencies: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: {"natural": [], "reversed": []} for name in specs
    }
    ties_natural: dict[str, list[bool]] = {name: [] for name in specs}
    ties_reversed: dict[str, list[bool]] = {name: [] for name in specs}
    natural_tokens: list[int] = []
    reversed_tokens: list[int] = []
    unscorable: list[dict[str, Any]] = []
    diagnostic_sample: list[dict[str, Any]] = []
    options_by_id: dict[str, int] = {}
    layout_checked = False
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
        reads, logits, forward_elapsed, n_tokens = score_prompt(
            model,
            tokenizer,
            prompt,
            codes,
            code_ids,
            device,
            display_to_option,
            specs,
            verify_layout=not layout_checked,
        )
        layout_checked = True
        natural_tokens.append(n_tokens)
        for name, read in reads.items():
            ties_natural[name].append(has_exact_tie(read["values"]))
            latencies[name]["natural"].append((forward_elapsed, read["read_s"]))
            predictions[name]["natural"].append(
                prediction_row(
                    item,
                    "natural",
                    texts,
                    gold,
                    read["values"],
                    display_to_option.index(read["chosen"]),
                    forward_elapsed + read["read_s"],
                    read=name,
                )
            )

        if item.id in sample_ids:
            # The diagnostic is the model's own next-token distribution, so it
            # stays comparable across read configurations.
            model_values = [float(logits[code_ids[code]]) for code in codes]
            model_prediction = display_to_option.index(
                choose_option(model_values, display_to_option)
            )
            probabilities = torch.softmax(logits, dim=-1)
            top = torch.topk(probabilities, 5)
            diagnostic_sample.append(
                {
                    "id": item.id,
                    "read": "model_next_token",
                    "gold_code": codes[gold],
                    "predicted_code": codes[model_prediction],
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
        reversed_reads, _, reversed_forward, reversed_n_tokens = score_prompt(
            model,
            tokenizer,
            reversed_prompt,
            codes,
            code_ids,
            device,
            reversed_display_to_option,
            specs,
        )
        reversed_tokens.append(reversed_n_tokens)
        for name, read in reversed_reads.items():
            ties_reversed[name].append(has_exact_tie(read["values"]))
            latencies[name]["reversed"].append((reversed_forward, read["read_s"]))
            predictions[name]["reversed"].append(
                prediction_row(
                    item,
                    "reversed",
                    reversed_texts,
                    reversed_gold,
                    read["values"],
                    reversed_display_to_option.index(read["chosen"]),
                    reversed_forward + read["read_s"],
                    read=name,
                )
            )

        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(items)} items scored", flush=True)

    def with_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id = {item.id: item for item in items}
        return [{**row, "_item": by_id[row["id"]]} for row in rows]

    def latency_stats(entries: list[tuple[float, float]]) -> dict[str, float]:
        if not entries:
            return {
                "mean_total_s": 0.0,
                "median_total_s": 0.0,
                "mean_forward_s": 0.0,
                "mean_read_s": 0.0,
            }
        totals = [forward + read for forward, read in entries]
        return {
            "mean_total_s": statistics.mean(totals),
            "median_total_s": statistics.median(totals),
            "mean_forward_s": statistics.mean(entry[0] for entry in entries),
            "mean_read_s": statistics.mean(entry[1] for entry in entries),
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

    read_metrics: dict[str, dict[str, Any]] = {}
    for name in specs:
        natural_rows = predictions[name]["natural"]
        reversed_rows = predictions[name]["reversed"]
        read_metrics[name] = {
            "natural": {
                "summary": summarize(with_items(natural_rows), failed=len(unscorable)),
                "position": position_diagnostics(natural_rows),
            },
            "reversed": {
                "summary": summarize(with_items(reversed_rows), failed=len(unscorable)),
                "position": position_diagnostics(reversed_rows),
            },
            "order_effect": order_effect(natural_rows, reversed_rows, options_by_id),
            "latency": {
                "natural": latency_stats(latencies[name]["natural"]),
                "reversed": latency_stats(latencies[name]["reversed"]),
            },
        }

    gate_name = "last" if "last" in specs else next(iter(specs))
    natural_positions = read_metrics[gate_name]["natural"]["position"]

    config = panel2_config(args, code_ids, used_codes, n_layers, specs)
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
            name: {
                "natural": tie_report(
                    predictions[name]["natural"],
                    [[flag] for flag in ties_natural[name]],
                ),
                "reversed": tie_report(
                    predictions[name]["reversed"],
                    [[flag] for flag in ties_reversed[name]],
                ),
            }
            for name in specs
        },
    }
    validity["tie_items_correct"] = {
        name: {
            "natural": validity["tie_items"]["cells"][name]["natural"]["correct"],
            "reversed": validity["tie_items"]["cells"][name]["reversed"]["correct"],
        }
        for name in specs
    }
    if args.read_config == "all":
        validity["position_dominated_by_read"] = {
            name: read_metrics[name]["natural"]["position"]["position_dominated"]
            for name in specs
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
            "read_configs": {name: list(spec) for name, spec in specs.items()},
            "elapsed_s": round(
                sum(
                    sum(forward + read for forward, read in entries)
                    for name in specs
                    for entries in (
                        latencies[name]["natural"],
                        latencies[name]["reversed"],
                    )
                ),
                1,
            ),
        },
    )

    if args.read_config == "all":
        metrics = {"reads": read_metrics, "prompt_tokens": prompt_tokens}
    else:
        only = next(iter(specs))
        metrics = {
            "natural": read_metrics[only]["natural"],
            "reversed": read_metrics[only]["reversed"],
            "order_effect": read_metrics[only]["order_effect"],
            "latency": read_metrics[only]["latency"],
            "prompt_tokens": prompt_tokens,
        }

    payload = {
        "_provenance": provenance,
        "config": config,
        "validity": validity,
        "unscorable": unscorable,
        "metrics": metrics,
        "diagnostic_sample": diagnostic_sample,
    }
    if args.read_config == "all":
        payload["predictions_by_read"] = {
            name: {
                "natural": predictions[name]["natural"],
                "reversed": predictions[name]["reversed"],
            }
            for name in specs
        }
    else:
        only = next(iter(specs))
        payload["predictions_natural"] = predictions[only]["natural"]
        payload["predictions_reversed"] = predictions[only]["reversed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    overall = read_metrics[gate_name]["natural"]["summary"]["overall"]
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
    parser.add_argument(
        "--read-config",
        default="last",
        help=(
            "last, layer=K, mean_last_n=4, or all (#337). The declared middle "
            "layers depend on the model's block count, so the name is "
            "validated against read_specs() after the model config is read"
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
