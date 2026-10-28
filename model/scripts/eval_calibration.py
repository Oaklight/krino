#!/usr/bin/env python3
"""Calibration and robustness report for a trained decision model.

Loads a checkpoint with the same config and architecture arguments as
``model/scripts/train_multitask.py``, then reports, per question type:

* n, accuracy, ECE over top-1 confidence, and Brier score;
* for score items, MAE on the expected level, quadratic-weighted kappa and RPS;
* a temperature fitted on one half of the items and applied to the other half,
  with ECE/Brier before and after;
* option-order sensitivity: the fraction of answers that flip, the mean
  total-variation distance between the label -> probability maps, and
  accuracy after reordering the options;
* accuracy and ECE bucketed by option count.

Jev allows at most 255 options per Choice question
(https://docs.typesafe.ai/primitives/choice), so ``--max-options`` defaults to
255 and every item above the guard is counted, not silently dropped.

Usage:
    python model/scripts/eval_calibration.py \
        --config model/configs/r2_reranker0.6b.yaml \
        --load-heads model/checkpoints/arch_06b_loraAll_weighted/best_heads.pt \
        --lora-layers -1 --lora-rank 16 --device cuda \
        --json model/results/calibration_06b.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import data.pipeline as pipeline_mod
import data.sources as sources_mod
from data.sources import load_source_items
from model.evaluation.calibration import (
    brier_score,
    fit_temperature,
    multiclass_brier,
    quadratic_weighted_kappa,
    softmax,
    top1_calibration,
)
from model.evaluation.order_sensitivity import (
    make_permutation,
    option_count_bucket,
    remap_probe_probs,
    total_variation,
)
from model.src.gpu_config import GPUConfig, estimate_model_billions, print_gpu_info
from model.src.model_factory import build_decision_model
from model.training.losses import rps_from_probs

TYPES = ("noul", "choice", "score")
# Jev's Choice primitive accepts at most 255 options.
JEV_MAX_OPTIONS = 255


# ---------------------------------------------------------------------------
# Item-level prediction
# ---------------------------------------------------------------------------


def _as_bool(label: Any) -> bool:
    """Normalise a noul label, which may be a bool or a yes/no string."""
    if isinstance(label, str):
        return label.strip().lower() in ("true", "yes", "1")
    return bool(label)


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _option_count(item: Any) -> int:
    """Number of options (or score levels) the forward pass will score."""
    question = item.question
    q_type = question.get("type", "noul")
    if q_type == "choice":
        return len(question.get("criteria") or {})
    if q_type == "score":
        return len(question.get("criteria") or [])
    augmented = question.get("augmented_options")
    teacher_probs = getattr(item, "teacher_probs", None)
    # Mirrors _eval_item: the extra option list is used only when the item
    # carries more than two teacher probabilities.
    if augmented and teacher_probs and len(teacher_probs) > 2:
        return len(augmented)
    return 2


@torch.no_grad()
def _predict_original(model: Any, item: Any) -> dict[str, Any] | None:
    """Score one item and return its logits, labels and gold index.

    This follows the same branches as ``_eval_item`` in
    ``model/training/supervised.py`` so the accuracy here matches the training
    evaluation, but it keeps the raw logits and probabilities that calibration
    needs.  ``DecisionModel.predict`` rounds probabilities to two decimals,
    which is fine for a demo and useless for ECE.

    Args:
        model: Loaded ``DecisionModel`` in eval mode.
        item: A ``TypedQuestion`` from the eval split.

    Returns:
        A record with ``logits``, ``labels``, ``label_idx``, ``option_count``
        and type-specific fields, or ``None`` when the item cannot be scored.
    """
    question = item.question
    q_type = question.get("type", "noul")
    state = item.state
    instructions = question.get("instructions", "")

    if q_type == "noul":
        positive = _as_bool(item.label)
        augmented = question.get("augmented_options")
        teacher_probs = getattr(item, "teacher_probs", None)
        if augmented and teacher_probs and len(teacher_probs) > 2:
            keys = list(augmented.keys())
            option_texts = [augmented[key] for key in keys]
            logits = model.forward_choice(state, instructions, option_texts)[0]
            logits = logits.float().cpu().tolist()
            target_key = "true" if positive else "false"
            if target_key not in keys:
                return None
            # Brier needs P(positive).  With an augmented option list the
            # positive class is the "true" (or "yes") option.
            positive_key = next(
                (key for key in keys if key.strip().lower() in ("true", "yes")),
                target_key,
            )
            return {
                "type": "noul",
                "logits": logits,
                "labels": keys,
                "label_idx": keys.index(target_key),
                "gold_label": target_key,
                "option_count": len(keys),
                "positive_index": keys.index(positive_key),
                "positive": positive,
                "_item": item,
            }
        logit = float(model.forward_noul(state, instructions).reshape(-1)[0].item())
        return {
            "type": "noul",
            "logits": [0.0, logit],
            "labels": ["false", "true"],
            "label_idx": 1 if positive else 0,
            "gold_label": "true" if positive else "false",
            "option_count": 2,
            "positive_index": 1,
            "positive": positive,
            "_item": item,
        }

    if q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        option_texts = [value or key for key, value in criteria.items()]
        logits = model.forward_choice(state, instructions, option_texts)[0]
        logits = logits.float().cpu().tolist()
        if item.label not in keys:
            return None
        return {
            "type": "choice",
            "logits": logits,
            "labels": keys,
            "label_idx": keys.index(item.label),
            "gold_label": item.label,
            "option_count": len(keys),
            "_item": item,
        }

    if q_type == "score":
        criteria = question["criteria"]
        n_levels = len(criteria)
        logits = model.forward_score(state, instructions, criteria)[0]
        logits = logits.float().cpu().tolist()
        label_f = float(item.label)
        gold_idx = max(0, min(n_levels - 1, int(round(label_f))))
        return {
            "type": "score",
            "logits": logits,
            "labels": list(range(n_levels)),
            "label_idx": gold_idx,
            "gold_label": gold_idx,
            "option_count": n_levels,
            "n_levels": n_levels,
            "score_label": label_f,
            "_item": item,
        }

    return None


@torch.no_grad()
def _predict_probe(
    model: Any, item: Any, seed: int, probe: str
) -> tuple[list[float], list[Any]]:
    """Re-score one item with its options reordered.

    For choice items ``probe="order"`` shuffles the ``(key, description)``
    pairs, so every option keeps its text and only its position changes.
    ``probe="text"`` permutes the effective option texts among fixed key
    positions.  A model that consumes only the option texts sees the same
    thing in both cases; the different seeds just take two independent
    permutation samples.

    Score items are ordinal.  ``probe="order"`` reverses the level list;
    ``probe="text"`` permutes the level descriptions.  In both cases the
    returned source labels let the caller mirror the output back onto the
    original level indices.

    Args:
        model: Loaded ``DecisionModel`` in eval mode.
        item: A ``TypedQuestion`` from the eval split.
        seed: Run-level seed.
        probe: ``"order"`` or ``"text"``.

    Returns:
        ``(logits, source_labels)`` where ``source_labels[i]`` is the original
        label of the option placed at probe position ``i``.
    """
    question = item.question
    q_type = question.get("type")
    state = item.state
    instructions = question.get("instructions", "")

    if q_type == "choice":
        criteria = question["criteria"]
        keys = list(criteria.keys())
        texts = [value or key for key, value in criteria.items()]
        if probe == "order":
            permutation = make_permutation(len(keys), seed, item.id, "order")
        else:
            permutation = make_permutation(len(keys), seed, item.id, "text")
        probe_texts = [texts[index] for index in permutation]
        source_labels = [keys[index] for index in permutation]
        logits = model.forward_choice(state, instructions, probe_texts)[0]
        return logits.float().cpu().tolist(), source_labels

    if q_type == "score":
        levels = question["criteria"]
        n_levels = len(levels)
        if probe == "order":
            permutation = list(reversed(range(n_levels)))
        else:
            permutation = make_permutation(n_levels, seed, item.id, "text")
        probe_texts = [levels[index] for index in permutation]
        source_labels = list(permutation)
        logits = model.forward_score(state, instructions, probe_texts)[0]
        return logits.float().cpu().tolist(), source_labels

    raise ValueError(f"Order probes are not defined for type {q_type!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _probs(record: dict[str, Any], temperature: float = 1.0) -> list[float]:
    return softmax([value / temperature for value in record["logits"]])


def _type_metrics(
    records: Sequence[dict[str, Any]], temperature: float = 1.0
) -> dict[str, Any]:
    """Accuracy, top-1 ECE and Brier at one temperature."""
    if not records:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0}
    probabilities = [_probs(record, temperature) for record in records]
    labels = [record["label_idx"] for record in records]
    ece = top1_calibration(probabilities, labels)["ece"]
    if records[0]["type"] == "noul":
        positive_probs = [
            probability[record["positive_index"]]
            for probability, record in zip(probabilities, records)
        ]
        brier = brier_score(positive_probs, [record["positive"] for record in records])[
            "brier"
        ]
    else:
        brier = multiclass_brier(probabilities, labels)["brier"]
    correct = sum(
        _argmax(probability) == label
        for probability, label in zip(probabilities, labels)
    )
    return {
        "n": len(records),
        "accuracy": correct / len(records),
        "ece": ece,
        "brier": brier,
    }


def _score_target(label: float, n_levels: int) -> list[float]:
    """One-hot target for an integer label, linear interpolation otherwise."""
    label = max(0.0, min(float(n_levels - 1), float(label)))
    if label == int(label):
        target = [0.0] * n_levels
        target[int(label)] = 1.0
        return target
    low = int(label)
    high = min(low + 1, n_levels - 1)
    fraction = label - low
    target = [0.0] * n_levels
    target[low] = 1.0 - fraction
    target[high] = fraction
    return target


def _score_metrics(
    records: Sequence[dict[str, Any]], temperature: float = 1.0
) -> dict[str, Any]:
    """MAE, RPS and QWK for score items (metrics that respect level order)."""
    if not records:
        return {"mae": 0.0, "rps": 0.0, "qwk": 0.0, "qwk_by_levels": {}}

    probabilities = [_probs(record, temperature) for record in records]
    mae = sum(
        abs(
            sum(index * p for index, p in enumerate(probability))
            - record["score_label"]
        )
        for probability, record in zip(probabilities, records)
    ) / len(records)

    # RPS compares distributions of the same length, and score sources use
    # different level counts (5 for sst5/yelp, 6 for stsb), so group before
    # stacking and weight the group means by item count.
    rps_groups: dict[int, tuple[list[list[float]], list[list[float]]]] = defaultdict(
        lambda: ([], [])
    )
    for probability, record in zip(probabilities, records):
        group_probs, group_targets = rps_groups[record["n_levels"]]
        group_probs.append(probability)
        group_targets.append(_score_target(record["score_label"], record["n_levels"]))
    rps = sum(
        rps_from_probs(
            torch.tensor(group_probs, dtype=torch.float32),
            torch.tensor(group_targets, dtype=torch.float32),
        ).item()
        * len(group_probs)
        for group_probs, group_targets in rps_groups.values()
    ) / len(records)

    # QWK is defined within one level scale, and the score sources use
    # different counts (5 for sst5/yelp, 6 for stsb).  Report each scale
    # separately and average over items so the summary is not dominated by a
    # source with more rows.
    by_levels: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for probability, record in zip(probabilities, records):
        predictions, golds = by_levels[record["n_levels"]]
        predictions.append(_argmax(probability))
        golds.append(record["label_idx"])
    qwk_by_levels = {
        str(n_levels): {
            "qwk": quadratic_weighted_kappa(predictions, golds, n_levels),
            "n": len(predictions),
        }
        for n_levels, (predictions, golds) in sorted(by_levels.items())
    }
    if len(records):
        qwk = sum(group["qwk"] * group["n"] for group in qwk_by_levels.values()) / len(
            records
        )
    else:
        qwk = 0.0

    return {
        "mae": mae,
        "rps": rps,
        "qwk": qwk,
        "qwk_by_levels": qwk_by_levels,
    }


def _bucket_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy and ECE split by option count."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[option_count_bucket(record["option_count"])].append(record)
    order = ["2", "3-5", "6-20", ">20"]
    return {
        bucket: _type_metrics(buckets[bucket])
        for bucket in order
        if buckets.get(bucket)
    }


def _temperature_report(
    records: Sequence[dict[str, Any]],
    fit_frac: float,
    seed: int,
    fit_steps: int,
) -> dict[str, Any]:
    """Fit a temperature on one split and score the other split."""
    if len(records) < 4:
        return {
            "fitted": 1.0,
            "applied": 1.0,
            "n_fit": 0,
            "n_holdout": len(records),
            "holdout": {},
        }

    q_type = records[0]["type"]
    rng = random.Random(f"{seed}:{q_type}:temperature")
    order = list(range(len(records)))
    rng.shuffle(order)
    n_fit = int(round(len(records) * fit_frac))
    n_fit = max(2, min(n_fit, len(records) - 2))
    fit_records = [records[index] for index in order[:n_fit]]
    holdout_records = [records[index] for index in order[n_fit:]]

    fitted = fit_temperature(
        [record["logits"] for record in fit_records],
        [record["label_idx"] for record in fit_records],
        steps=fit_steps,
    )
    temperature = fitted["temperature"]
    before = _type_metrics(holdout_records, 1.0)
    after = _type_metrics(holdout_records, temperature)
    if q_type == "score":
        score_before = _score_metrics(holdout_records, 1.0)
        score_after = _score_metrics(holdout_records, temperature)
    else:
        score_before = score_after = {}

    return {
        "fitted": temperature,
        "applied": temperature,
        "n_fit": n_fit,
        "n_holdout": len(holdout_records),
        "nll_before": fitted["nll_before"],
        "nll_after": fitted["nll_after"],
        "holdout": {
            "ece_before": before["ece"],
            "ece_after": after["ece"],
            "brier_before": before["brier"],
            "brier_after": after["brier"],
            "accuracy_before": before["accuracy"],
            "accuracy_after": after["accuracy"],
        },
        "holdout_score": (
            {
                "mae_before": score_before["mae"],
                "mae_after": score_after["mae"],
                "rps_before": score_before["rps"],
                "rps_after": score_after["rps"],
            }
            if q_type == "score"
            else {}
        ),
    }


def _order_report(
    model: Any,
    records: Sequence[dict[str, Any]],
    seed: int,
    probe: str,
) -> dict[str, Any]:
    """Run one reordering probe over a type and aggregate its three numbers."""
    if not records:
        return {
            "n": 0,
            "label_flip_rate": 0.0,
            "mean_tv": 0.0,
            "accuracy_before": 0.0,
            "accuracy_after": 0.0,
        }

    flips = 0
    tv_total = 0.0
    max_tv = 0.0
    correct_before = 0
    correct_after = 0
    for record in records:
        original_probs = _probs(record)
        original_map = dict(zip(record["labels"], original_probs))
        probe_logits, source_labels = _predict_probe(
            model, record["_item"], seed, probe
        )
        probe_probs = softmax(probe_logits)
        probe_map = remap_probe_probs(probe_probs, source_labels)

        original_pred = record["labels"][_argmax(original_probs)]
        probe_pred = source_labels[_argmax(probe_probs)]
        flips += probe_pred != original_pred
        distance = total_variation(original_map, probe_map)
        tv_total += distance
        max_tv = max(max_tv, distance)
        correct_before += original_pred == record["gold_label"]
        correct_after += probe_pred == record["gold_label"]

    return {
        "n": len(records),
        "label_flip_rate": flips / len(records),
        "mean_tv": tv_total / len(records),
        "max_tv": max_tv,
        "accuracy_before": correct_before / len(records),
        "accuracy_after": correct_after / len(records),
    }


# ---------------------------------------------------------------------------
# Loading and CLI
# ---------------------------------------------------------------------------


def _warn_architecture_mismatch(model: Any, checkpoint: Path) -> None:
    """Warn when the checkpoint and the constructed model disagree.

    ``load_heads`` uses ``strict=False`` so that a checkpoint may omit the
    frozen backbone.  That leniency also hides a real mistake: a checkpoint
    trained with ``readout=weighted`` has a ``readout.layer_logits`` tensor
    that a model built with the default ``last`` readout silently drops, and
    the evaluation then looks plausible while scoring a different network.
    This check names the tensors that would be ignored or left at their
    initial values.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = model.state_dict()
    ignored = sorted(key for key in state if key not in expected)
    missing = sorted(
        key
        for key in expected
        if key not in state
        and key.startswith(
            (
                "noul_head.",
                "choice_head.",
                "score_head.",
                "unified.",
                "readout.",
                "projector.",
            )
        )
    )
    if ignored:
        print(
            f"WARNING: {len(ignored)} checkpoint tensors do not exist in the "
            f"constructed model and will be ignored: {ignored[:6]}",
            flush=True,
        )
        if any(key.startswith("readout.") for key in ignored):
            print(
                "  hint: this checkpoint was trained with a weighted layer "
                "readout; pass --readout weighted",
                flush=True,
            )
    if missing:
        print(
            f"WARNING: {len(missing)} head tensors are absent from the "
            f"checkpoint and stay at initialization: {missing[:6]}",
            flush=True,
        )


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle)


def _resolve(cli_value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def _load_eval_items(cfg: dict[str, Any], args: argparse.Namespace) -> list[Any]:
    """Load eval splits exactly as train_multitask lays them out."""
    if args.items_cache and args.items_cache.exists():
        with open(args.items_cache, "rb") as handle:
            cached = pickle.load(handle)
        print(f"  loaded {len(cached):,} items from {args.items_cache}", flush=True)
        return cached

    if args.data_root:
        data_root = Path(args.data_root)
        # load_source_items resolves relative paths through this module global
        # and load_synthetic reads pipeline.DATA_DIR; point both at the
        # checkout that actually holds the (gitignored) benchmark corpus.
        sources_mod.ROOT = data_root
        pipeline_mod.DATA_DIR = data_root / "data" / "benchmarks"

    sources_cfg = cfg.get("sources", {})
    if not sources_cfg:
        raise SystemExit("Error: no sources defined in config")
    wanted = None
    if args.sources:
        wanted = {name.strip() for name in args.sources.split(",") if name.strip()}
        unknown = wanted - set(sources_cfg)
        if unknown:
            raise SystemExit(f"Error: unknown sources: {sorted(unknown)}")
    rng = random.Random(args.seed)
    eval_items: list[Any] = []
    for source_name, source_cfg in sources_cfg.items():
        if wanted is not None and source_name not in wanted:
            continue
        source_cfg = source_cfg or {}
        _, source_eval = load_source_items(source_name, source_cfg, rng)
        eval_items.extend(source_eval)
        print(f"  {source_name:<20} {len(source_eval):>6} eval items", flush=True)

    if args.items_cache:
        args.items_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.items_cache, "wb") as handle:
            pickle.dump(eval_items, handle)
        print(f"  cached items to {args.items_cache}", flush=True)
    return eval_items


def _cap_items_per_type(items: list[Any], cap: int | None, seed: int) -> list[Any]:
    """Keep at most ``cap`` eval items per question type.

    Capping before the forward pass is what makes a capped run cheap: scoring
    everything and dropping the surplus afterwards still pays the full
    inference cost, which is the opposite of what someone passing
    ``--max-items-per-type`` wants.
    """
    if not cap:
        return items
    grouped: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        grouped[item.question.get("type", "noul")].append(item)
    capped: list[Any] = []
    for q_type, group in grouped.items():
        if len(group) <= cap:
            capped.extend(group)
            continue
        rng = random.Random(f"{seed}:{q_type}:cap")
        rng.shuffle(group)
        capped.extend(group[:cap])
    return capped


def _print_table(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"Calibration: {summary['checkpoint']}", flush=True)
    print("=" * 78, flush=True)
    print(
        f"{'Type':<8} {'n':>6} {'Acc':>8} {'ECE':>8} {'Brier':>8} "
        f"{'MAE':>8} {'QWK':>8} {'RPS':>8} {'T':>7}",
        flush=True,
    )
    print("-" * 78, flush=True)
    for q_type in TYPES:
        metrics = summary["by_type"].get(q_type)
        if not metrics or not metrics["n"]:
            continue
        extra = metrics.get("score_metrics", {})
        temperature = metrics.get("temperature", {}).get("fitted", 1.0)
        print(
            f"{q_type:<8} {metrics['n']:>6} {metrics['accuracy']:>8.4f} "
            f"{metrics['ece']:>8.4f} {metrics['brier']:>8.4f} "
            f"{extra.get('mae', float('nan')):>8.4f} "
            f"{extra.get('qwk', float('nan')):>8.4f} "
            f"{extra.get('rps', float('nan')):>8.4f} "
            f"{temperature:>7.3f}",
            flush=True,
        )

    print("\nTemperature scaling (fit / holdout split):", flush=True)
    print(
        f"{'Type':<8} {'T':>7} {'ECE_before':>11} {'ECE_after':>10} "
        f"{'Brier_before':>13} {'Brier_after':>12}",
        flush=True,
    )
    for q_type in TYPES:
        metrics = summary["by_type"].get(q_type)
        if not metrics or not metrics.get("temperature", {}).get("holdout"):
            continue
        temp = metrics["temperature"]
        hold = temp["holdout"]
        print(
            f"{q_type:<8} {temp['fitted']:>7.3f} {hold['ece_before']:>11.4f} "
            f"{hold['ece_after']:>10.4f} {hold['brier_before']:>13.4f} "
            f"{hold['brier_after']:>12.4f}",
            flush=True,
        )

    order = summary.get("order_sensitivity", {})
    if order:
        print("\nOption-order sensitivity:", flush=True)
        print(
            f"{'Type':<8} {'Probe':<8} {'n':>6} {'Flip':>8} {'MeanTV':>8} "
            f"{'Acc_before':>11} {'Acc_after':>10}",
            flush=True,
        )
        for q_type in ("choice", "score"):
            for probe in ("order", "text"):
                values = order.get(q_type, {}).get(probe)
                if not values or not values["n"]:
                    continue
                print(
                    f"{q_type:<8} {probe:<8} {values['n']:>6} "
                    f"{values['label_flip_rate']:>8.4f} {values['mean_tv']:>8.4f} "
                    f"{values['accuracy_before']:>11.4f} "
                    f"{values['accuracy_after']:>10.4f}",
                    flush=True,
                )

    print("\nAccuracy / ECE by option count:", flush=True)
    print(f"{'Type':<8} {'Bucket':<8} {'n':>6} {'Acc':>8} {'ECE':>8}", flush=True)
    for q_type in TYPES:
        metrics = summary["by_type"].get(q_type)
        if not metrics:
            continue
        for bucket, values in metrics.get("by_option_count", {}).items():
            print(
                f"{q_type:<8} {bucket:<8} {values['n']:>6} "
                f"{values['accuracy']:>8.4f} {values['ece']:>8.4f}",
                flush=True,
            )

    skipped = summary.get("skipped", {})
    if any(skipped.values()):
        print(f"\nSkipped: {skipped}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibration and robustness evaluation for a decision model"
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument(
        "--load-heads",
        type=Path,
        required=True,
        help="Checkpoint with the trained heads",
    )
    parser.add_argument("--model", default=None, help="Override backbone model")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument(
        "--rank", type=int, default=None, help="Override attention rank"
    )
    parser.add_argument(
        "--mlp-layers", type=int, default=None, help="Override projector layers"
    )
    parser.add_argument(
        "--noul-rank", type=int, default=None, help="Override noul head rank"
    )
    parser.add_argument(
        "--max-length", type=int, default=None, help="Override max sequence length"
    )
    parser.add_argument(
        "--readout", choices=["last", "mean_last_n", "weighted"], default=None
    )
    parser.add_argument("--readout-n", type=int, default=None)
    parser.add_argument(
        "--lora-layers", type=int, default=None, help="-1 = all blocks, N = last N"
    )
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-targets", type=str, default=None)
    shared = parser.add_mutually_exclusive_group()
    shared.add_argument("--shared-attention", action="store_true", default=None)
    shared.add_argument(
        "--no-shared-attention", action="store_false", dest="shared_attention"
    )
    unified = parser.add_mutually_exclusive_group()
    unified.add_argument("--unified-head", action="store_true", default=None)
    unified.add_argument("--no-unified-head", action="store_false", dest="unified_head")
    flash = parser.add_mutually_exclusive_group()
    flash.add_argument("--flash-attention", action="store_true", default=None)
    flash.add_argument(
        "--no-flash-attention", action="store_false", dest="flash_attention"
    )
    bf16 = parser.add_mutually_exclusive_group()
    bf16.add_argument("--bf16", action="store_true", default=None)
    bf16.add_argument("--no-bf16", action="store_false", dest="bf16")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Checkout that holds data/benchmarks (default: this repo)",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source names to evaluate (default: all in config)",
    )
    parser.add_argument(
        "--items-cache",
        type=Path,
        default=None,
        help="Pickle the assembled eval items here and reuse them on later runs",
    )
    parser.add_argument(
        "--fit-frac", type=float, default=0.5, help="Fraction used to fit temperature"
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=200,
        help="Coarse grid size for temperature fit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-options",
        type=int,
        default=JEV_MAX_OPTIONS,
        help="Skip items with more options (0 disables; Jev allows 255)",
    )
    parser.add_argument(
        "--max-items-per-type",
        type=int,
        default=None,
        help="Cap evaluated items per type for a faster report",
    )
    parser.add_argument(
        "--no-order-probes", action="store_true", help="Skip reordering probes"
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the full report here"
    )
    args = parser.parse_args()

    if not 0.0 < args.fit_frac < 1.0:
        parser.error("--fit-frac must be strictly between 0 and 1")

    cfg = _load_config(args.config)
    model_name = args.model or cfg.get("model", "Qwen/Qwen3-0.6B")
    use_encoder = bool(cfg.get("encoder", False))
    qwen35_base = bool(cfg.get("qwen35_base", False))
    rank = _resolve(args.rank, cfg, "rank", 64)
    rival_aware = bool(cfg.get("rival_aware", False))
    mlp_layers = _resolve(args.mlp_layers, cfg, "mlp_layers", 0)
    mlp_dim = cfg.get("mlp_dim")
    noul_rank = _resolve(args.noul_rank, cfg, "noul_rank", None)
    max_length = _resolve(args.max_length, cfg, "max_length", None)
    shared_attention = _resolve(args.shared_attention, cfg, "shared_attention", False)
    unified_head = _resolve(args.unified_head, cfg, "unified_head", False)
    readout = _resolve(args.readout, cfg, "readout", "last")
    readout_n = _resolve(args.readout_n, cfg, "readout_n", 4)
    lora_layers = _resolve(args.lora_layers, cfg, "lora_layers", None)
    if lora_layers == 0:
        lora_layers = None
    lora_rank = _resolve(args.lora_rank, cfg, "lora_rank", 16)
    lora_alpha = _resolve(args.lora_alpha, cfg, "lora_alpha", 32)
    lora_dropout = _resolve(args.lora_dropout, cfg, "lora_dropout", 0.0)
    lora_targets_raw = _resolve(args.lora_targets, cfg, "lora_targets", "q_proj,v_proj")
    lora_targets = tuple(
        part.strip()
        for part in (
            lora_targets_raw.split(",")
            if isinstance(lora_targets_raw, str)
            else lora_targets_raw
        )
        if part.strip()
    )
    device = args.device or cfg.get("device")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=== Calibration evaluation ===", flush=True)
    print(f"Backbone: {model_name}", flush=True)
    print(f"Config:   {args.config}", flush=True)
    print(f"Heads:    {args.load_heads}", flush=True)

    model_billions = estimate_model_billions(model_name)
    gpu_config = GPUConfig.auto_detect(model_billions)
    gpu_config = gpu_config.apply_overrides(
        flash_attention=args.flash_attention,
        bf16=args.bf16,
        max_length=max_length,
    )
    print_gpu_info(gpu_config)

    model = build_decision_model(
        model_name=model_name,
        gpu_config=gpu_config,
        device=device,
        use_encoder=use_encoder,
        qwen35_base=qwen35_base,
        rank=rank,
        rival_aware=rival_aware,
        mlp_layers=mlp_layers,
        mlp_dim=mlp_dim,
        noul_rank=noul_rank,
        max_length=max_length,
        shared_attention=shared_attention,
        unified_head=unified_head,
        readout=readout,
        readout_n=readout_n,
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
    )
    print(f"Loading heads from {args.load_heads}", flush=True)
    _warn_architecture_mismatch(model, args.load_heads)
    model.load_heads(args.load_heads)
    model.eval()

    started = time.perf_counter()
    eval_items = _load_eval_items(cfg, args)
    print(f"\nLoaded {len(eval_items):,} eval items", flush=True)
    eval_items = _cap_items_per_type(eval_items, args.max_items_per_type, args.seed)

    records: dict[str, list[dict[str, Any]]] = {q_type: [] for q_type in TYPES}
    skipped = {"option_limit": 0, "label_not_in_options": 0, "unknown_type": 0}
    for item in eval_items:
        q_type = item.question.get("type", "noul")
        if q_type not in records:
            skipped["unknown_type"] += 1
            continue
        if args.max_options and _option_count(item) > args.max_options:
            skipped["option_limit"] += 1
            continue
        record = _predict_original(model, item)
        if record is None:
            skipped["label_not_in_options"] += 1
            continue
        records[q_type].append(record)

    for q_type in TYPES:
        print(f"  {q_type:<7} {len(records[q_type]):>6} items", flush=True)
    if any(skipped.values()):
        print(f"  skipped {skipped}", flush=True)

    summary: dict[str, Any] = {
        "checkpoint": str(args.load_heads),
        "config": str(args.config),
        "model": model_name,
        "seed": args.seed,
        "fit_frac": args.fit_frac,
        "max_options": args.max_options,
        "max_items_per_type": args.max_items_per_type,
        "skipped": skipped,
        "by_type": {},
        "order_sensitivity": {},
    }

    for q_type in TYPES:
        type_records = records[q_type]
        metrics = _type_metrics(type_records)
        metrics["temperature"] = _temperature_report(
            type_records, args.fit_frac, args.seed, args.fit_steps
        )
        metrics["by_option_count"] = _bucket_metrics(type_records)
        if q_type == "score":
            metrics["score_metrics"] = _score_metrics(type_records)
        summary["by_type"][q_type] = metrics

    if not args.no_order_probes:
        for q_type in ("choice", "score"):
            summary["order_sensitivity"][q_type] = {
                "order": _order_report(model, records[q_type], args.seed, "order"),
                "text": _order_report(model, records[q_type], args.seed, "text"),
            }

    summary["elapsed_s"] = time.perf_counter() - started
    _print_table(summary)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nReport saved to {args.json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
