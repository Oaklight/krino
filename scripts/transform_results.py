#!/usr/bin/env python3
"""Transform raw experiment JSONs into a standardized format for the web dashboard.

Reads model/experiments/*.json and writes web/public/results/eval_results.json.
"""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "model" / "experiments"
OUTPUT_PATH = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.evaluation.paired_power import detectable_delta  # noqa: E402
from model.scripts.paired_mcnemar import mcnemar  # noqa: E402


def source_file_label(filepath: Path) -> str:
    """Return the path a dashboard row can be traced back to.

    Repo-relative where possible so the deploy guard can check it against
    ``git ls-files``. A path outside the repository is returned verbatim and
    the guard rejects it.
    """
    try:
        return filepath.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(filepath.resolve())


# Benchmark type classification
BENCHMARK_TYPES: dict[str, str] = {
    "agnews": "choice",
    "arc": "choice",
    "authored144": "choice",
    "banking77": "choice",
    "codesearchnet": "choice",
    "contractnli": "noul",
    "fever": "choice",
    "hellaswag": "choice",
    "mednli": "noul",
    "mnli": "noul",
    "multirc": "noul",
    "race": "choice",
    "sst2": "noul",
    "sst5": "score",
    "stsb": "score",
    "swag": "choice",
    "tabfact": "noul",
    "typed_decisions": "mixed",
    "yelp": "score",
}

# Model metadata: filename stem -> (display_name, params, model_type)
MODEL_META: dict[str, tuple[str, str, str]] = {
    "logit_baseline_qwen_0.6b": ("Qwen2.5-0.5B (baseline)", "0.5B", "causal"),
    "logit_Qwen_Qwen2.5-0.5B": ("Qwen2.5-0.5B", "0.5B", "causal"),
    "logit_Qwen2.5-1.5B": ("Qwen2.5-1.5B", "1.5B", "causal"),
    "logit_Qwen_Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct", "7B", "causal"),
    "logit_Qwen_Qwen3-0.6B": ("Qwen3-0.6B", "0.6B", "causal"),
    "logit_Qwen_Qwen3-1.7B": ("Qwen3-1.7B", "1.7B", "causal"),
    "logit_Qwen_Qwen3-4B": ("Qwen3-4B", "4B", "causal"),
    "logit_Qwen_Qwen3-8B": ("Qwen3-8B", "8B", "causal"),
    "logit_Qwen_Qwen3-30B-A3B": ("Qwen3-30B-A3B", "30B (3B active)", "moe"),
    "logit_Qwen_Qwen3.5-4B": ("Qwen3.5-4B", "4B", "causal"),
    "logit_Qwen_Qwen3.5-9B": ("Qwen3.5-9B", "9B", "causal"),
    "logit_HuggingFaceTB_SmolLM2-1.7B": ("SmolLM2-1.7B", "1.7B", "causal"),
    "logit_microsoft_Phi-4-mini-instruct": ("Phi-4-mini-instruct", "3.8B", "causal"),
    "logit_openbmb_MiniCPM5-2B-Base": ("MiniCPM5-2B-Base", "2B", "causal"),
    "logit_mistralai_Ministral-3-3B-Base-2512": ("Ministral-3-3B-Base-2512", "3B", "causal"),
    "logit_openbmb_MiniCPM3-4B": ("MiniCPM3-4B (instruct)", "4B", "causal"),
    "reranker_cross-encoder_ettin-reranker-68m-v1": ("Ettin-68m", "68M", "reranker"),
    "reranker_cross-encoder_ettin-reranker-150m-v1": ("Ettin-150m", "150M", "reranker"),
    "reranker_cross-encoder_ettin-reranker-400m-v1": ("Ettin-400m", "400M", "reranker"),
    "encoder_modernbert_base": ("ModernBERT-base", "149M", "encoder"),
    "encoder_modernbert_large": ("ModernBERT-large", "395M", "encoder"),
    "cross_bert": ("BERT cross-encoder", "110M", "encoder"),
    "cross_qwen": ("Qwen cross-encoder", "0.6B", "causal"),
    "openjev_Qwen_Qwen3-0.6B": ("OpenJev Qwen3-0.6B", "0.6B", "causal"),
    "openjev_Qwen_Qwen3.5-4B": ("OpenJev Qwen3.5-4B", "4B", "causal"),
    "jev_api": ("Jev (TypeSafe API)", "unknown", "api"),
}

# phaseA files share models with non-phaseA — strip prefix for metadata lookup
PHASE_A_META: dict[str, tuple[str, str, str]] = {
    "phaseA_logit_Qwen_Qwen2.5-0.5B": ("Qwen2.5-0.5B", "0.5B", "causal"),
    "phaseA_logit_Qwen_Qwen2.5-1.5B": ("Qwen2.5-1.5B", "1.5B", "causal"),
    "phaseA_logit_Qwen_Qwen2.5-7B-Instruct": ("Qwen2.5-7B-Instruct", "7B", "causal"),
    "phaseA_logit_Qwen_Qwen3-0.6B": ("Qwen3-0.6B", "0.6B", "causal"),
    "phaseA_logit_Qwen_Qwen3-1.7B": ("Qwen3-1.7B", "1.7B", "causal"),
    "phaseA_logit_HuggingFaceTB_SmolLM2-1.7B": ("SmolLM2-1.7B", "1.7B", "causal"),
    "phaseA_logit_microsoft_Phi-4-mini-instruct": ("Phi-4-mini-instruct", "3.8B", "causal"),
    "phaseA_reranker_cross-encoder_ettin-reranker-68m-v1": ("Ettin-68m", "68M", "reranker"),
    "phaseA_reranker_cross-encoder_ettin-reranker-150m-v1": ("Ettin-150m", "150M", "reranker"),
    "phaseA_reranker_cross-encoder_ettin-reranker-400m-v1": ("Ettin-400m", "400M", "reranker"),
}

# Training run metadata: filename stem -> (display_name, backbone_params, benchmark, extra_info)
TRAINING_META: dict[str, tuple[str, str, str, str]] = {
    "train_banking77": ("Qwen3-0.6B r64", "0.6B", "banking77", "rank=64"),
    "train_ettin150m_r64": ("Ettin-150m r64", "150M", "banking77", "rank=64"),
    "train_ettin150m_r128": ("Ettin-150m r128", "150M", "banking77", "rank=128"),
    "train_ettin400m_r64": ("Ettin-400m r64", "400M", "banking77", "rank=64"),
    "train_modernbert_base": ("ModernBERT-base r64", "149M", "banking77", "rank=64"),
    "train_modernbert_large": ("ModernBERT-large r64", "395M", "banking77", "rank=64"),
    "train_modernbert_r128": ("ModernBERT-base r128", "149M", "banking77", "rank=128"),
    "train_sst2": ("Ettin-150m r128 (SST-2)", "150M", "sst2", "rank=128"),
    "train_v2_indomain": ("Ettin-150m r128 (multi-task)", "150M", "19-benchmarks", "rank=128, multi-task"),
}


def extract_benchmark_result(benchmark_data: dict, benchmark_name: str) -> dict:
    """Extract accuracy/ECE/MAE from a single benchmark's raw data."""
    items = benchmark_data.get("items") or benchmark_data.get("items_evaluated", 0)
    bm_type = BENCHMARK_TYPES.get(benchmark_name, "choice")

    latency = benchmark_data.get("latency", {})
    latency_ms = round(latency.get("total_mean_s", 0) * 1000, 1) if latency else None

    if bm_type == "mixed":
        # typed_decisions has noul + choice + score subtypes
        result: dict = {"n": items, "type": "mixed", "subtypes": {}}
        for subtype in ("noul", "choice", "score"):
            if subtype in benchmark_data:
                sub = benchmark_data[subtype]
                if subtype == "score":
                    result["subtypes"][subtype] = {"n": sub.get("n", 0), "mae": sub.get("mae")}
                else:
                    acc_data = sub.get("accuracy", {})
                    ece_data = sub.get("ece", {})
                    result["subtypes"][subtype] = {
                        "n": acc_data.get("n", 0),
                        "accuracy": acc_data.get("accuracy"),
                        "ece": ece_data.get("ece"),
                    }
        if latency_ms:
            result["latency_ms"] = latency_ms
        return result

    if bm_type == "score":
        score_data = benchmark_data.get("score", {})
        result = {"n": score_data.get("n", items), "type": "score", "mae": score_data.get("mae")}
        if latency_ms:
            result["latency_ms"] = latency_ms
        return result

    # choice or noul — both have accuracy and ece
    type_key = bm_type
    type_data = benchmark_data.get(type_key, {})
    acc_data = type_data.get("accuracy", {})
    ece_data = type_data.get("ece", {})

    result = {
        "n": acc_data.get("n", items),
        "type": bm_type,
        "accuracy": acc_data.get("accuracy"),
        "ece": ece_data.get("ece"),
    }
    if latency_ms:
        result["latency_ms"] = latency_ms
    return result


def get_accuracy_for_aggregate(bm_result: dict) -> float | None:
    """Get the accuracy value for aggregate computation (choice + noul only)."""
    if bm_result["type"] == "mixed":
        accs = []
        for st in ("noul", "choice"):
            sub = bm_result.get("subtypes", {}).get(st)
            if sub and sub.get("accuracy") is not None:
                accs.append(sub["accuracy"])
        return statistics.mean(accs) if accs else None
    if bm_result["type"] in ("choice", "noul"):
        return bm_result.get("accuracy")
    return None


def determine_method(stem: str) -> str:
    """Determine evaluation method from filename."""
    if stem.startswith("phaseA_reranker_") or stem.startswith("reranker_"):
        return "reranker"
    if stem.startswith("phaseA_logit_") or stem.startswith("logit_"):
        return "logit"
    if stem.startswith("encoder_"):
        return "encoder"
    if stem.startswith("cross_"):
        return "cross-encoder"
    if stem.startswith("openjev_"):
        return "openjev"
    if stem == "jev_api":
        return "jev_api"
    return "unknown"


def transform_eval_file(filepath: Path) -> dict | None:
    """Transform a single eval result file into standardized format."""
    stem = filepath.stem
    if stem.startswith("sf2_") or stem.startswith("train_"):
        return None

    raw = json.loads(filepath.read_text())

    all_meta = {**MODEL_META, **PHASE_A_META}
    if stem not in all_meta:
        print(f"  WARNING: no metadata for {stem}, skipping")
        return None

    display_name, params, model_type = all_meta[stem]
    method = determine_method(stem)
    is_phase_a = stem.startswith("phaseA_")

    benchmarks = {}
    for bm_name, bm_data in raw.items():
        if bm_name.startswith("_"):
            # ``_provenance`` and any future metadata block are not benchmarks.
            continue
        benchmarks[bm_name] = extract_benchmark_result(bm_data, bm_name)

    accuracies = []
    for bm_result in benchmarks.values():
        acc = get_accuracy_for_aggregate(bm_result)
        if acc is not None:
            accuracies.append(acc)

    aggregate_accuracy = round(statistics.mean(accuracies), 4) if accuracies else None

    run_id = f"{'phaseA-' if is_phase_a else ''}{method}-{stem.split('_', 2 if is_phase_a else 1)[-1]}"

    return {
        "run_id": run_id,
        "model": {"name": display_name, "params": params, "type": model_type},
        "method": method,
        "phase": "A" if is_phase_a else None,
        "source_file": source_file_label(filepath),
        "n_benchmarks": len(benchmarks),
        "results": {
            "aggregate": {"accuracy": aggregate_accuracy, "n_benchmarks_with_accuracy": len(accuracies)},
            "by_source": benchmarks,
        },
    }


def transform_training_file(filepath: Path) -> dict | None:
    """Transform a training history file."""
    stem = filepath.stem
    if stem not in TRAINING_META:
        print(f"  WARNING: no training metadata for {stem}, skipping")
        return None

    raw = json.loads(filepath.read_text())
    display_name, params, benchmark, extra = TRAINING_META[stem]
    history = raw.get("history", [])

    epochs = []
    for h in history:
        epoch_data: dict = {
            "epoch": h["epoch"],
            "train_loss": round(h["train"]["mean_loss"], 4),
            "eval_loss": round(h["eval"]["mean_loss"], 4) if "eval" in h else None,
            "eval_accuracy": h.get("eval", {}).get("accuracy"),
            "lr": h.get("lr"),
            "elapsed_s": round(h.get("elapsed_s", 0), 1),
        }
        if h.get("best"):
            epoch_data["best"] = True
        epochs.append(epoch_data)

    best_epoch = next((e for e in reversed(epochs) if e.get("best")), epochs[-1] if epochs else None)

    return {
        "run_id": stem,
        "model": display_name,
        "params": params,
        "benchmark": benchmark,
        "info": extra,
        "source_file": source_file_label(filepath),
        "best_accuracy": best_epoch["eval_accuracy"] if best_epoch else None,
        "best_loss": best_epoch["eval_loss"] if best_epoch else None,
        "n_epochs": len(epochs),
        "history": epochs,
    }


def build_jev_comparison(eval_runs: list[dict]) -> dict:
    """Build comparison table between each model and Jev API."""
    jev_run = next((r for r in eval_runs if r["method"] == "jev_api"), None)
    if not jev_run:
        return {}

    jev_benchmarks = jev_run["results"]["by_source"]
    comparison = {}

    for bm_name, jev_data in jev_benchmarks.items():
        jev_acc = get_accuracy_for_aggregate(jev_data)
        if jev_acc is None:
            continue

        best_model = None
        best_acc = -1.0
        for run in eval_runs:
            if run["method"] == "jev_api":
                continue
            bm = run["results"]["by_source"].get(bm_name)
            if not bm:
                continue
            acc = get_accuracy_for_aggregate(bm)
            if acc is not None and acc > best_acc:
                best_acc = acc
                best_model = run["model"]["name"]

        comparison[bm_name] = {
            "jev": round(jev_acc, 4),
            "best_model": best_model,
            "best_accuracy": round(best_acc, 4) if best_model else None,
        }

    return comparison


JEVBENCH_DIR = REPO_ROOT / "results" / "jevbench"
LOGIT_PROV_DIR = REPO_ROOT / "model" / "experiments" / "logit_prov"

# Display-only metadata for the trained-head and native-reranker files, which
# record no model id of their own. Every number still comes from the file.
JEVBENCH_META: dict[str, tuple[str, str, str]] = {
    "krino-ettin-150m-heads": ("Ettin-150m (trained)", "150M", "reranker"),
    "krino-modernbert-base-heads": ("ModernBERT-base (trained)", "149M", "encoder"),
    "krino-qwen3-0.6b-heads": ("Qwen3-0.6B (trained)", "0.6B", "causal"),
    "krino-qwen3.5-4b-heads": ("Qwen3.5-4B (trained)", "4B", "causal"),
    "krino-qwen3-reranker-4b-heads": ("Qwen3-reranker-4B (trained)", "4B", "reranker"),
    "krino-qwen3-reranker-0.6b-heads": ("Qwen3-reranker-0.6B (trained)", "0.6B", "reranker"),
    "raw-qwen3-reranker-4b": ("Qwen/Qwen3-Reranker-4B (zero-shot)", "4B", "reranker"),
    # Third-party trained model, served natively. Display-only; the row's
    # numbers come from the file.
    "h2o-lightning-4b": ("H2O-Lightning-4B (third-party)", "4B", "causal"),
    "h2o-lightning-4b-oneway": ("H2O-Lightning-4B (third-party)", "4B", "causal"),
}

# ``_provenance.strategy`` -> the readout label the dashboard renders.
READOUT_BY_STRATEGY: dict[str, str] = {
    "judgement": "judgement",
    "description": "continuation",
    "reranker_native": "native-reranker",
    "panel2_vocab_code": "panel2",
    "native_decision": "native-decision",
    "native_decision_oneway": "native-decision-oneway",
}


def derive_readout(filepath: Path, raw: dict) -> str | None:
    """Name the instrument that produced a JevBench cell.

    ``_provenance.strategy`` is the record for zero-shot runs. The trained-head
    files predate provenance and are identified by their ``-heads`` suffix; the
    raw reranker files record ``mode``. ``None`` means the file cannot be
    labelled, and the caller skips it instead of guessing.
    """
    provenance = raw.get("_provenance") or {}
    strategy = provenance.get("strategy")
    if strategy in READOUT_BY_STRATEGY:
        return READOUT_BY_STRATEGY[strategy]
    if raw.get("mode") == "raw_reranker":
        return "native-reranker"
    if filepath.stem.endswith("-heads"):
        return "trained-heads"
    return None


def add_head_version(display_name: str, head_version: str) -> str:
    """Render a generation beside a trained-head display name.

    ``Ettin-150m (trained)`` becomes ``Ettin-150m (trained, v1)``.
    """
    if display_name.endswith(")"):
        return f"{display_name[:-1]}, {head_version})"
    return f"{display_name} ({head_version})"


def build_jevbench_run(filepath: Path) -> dict | None:
    """Build one dashboard row from one committed JevBench result file."""
    stem = filepath.stem
    raw = json.loads(filepath.read_text())
    provenance = raw.get("_provenance") or {}

    readout = derive_readout(filepath, raw)
    if readout is None:
        print(f"  WARNING: cannot derive a readout for jevbench/{stem}, skipping")
        return None

    # Files written by eval_jevbench_logit.py nest the aggregate under
    # "summary"; the trained-head files put the same keys at the top level.
    # The panel-2 cells nest it one level deeper, under metrics.natural.
    summary = raw.get("summary", raw)
    panel2_natural = (raw.get("metrics") or {}).get("natural")
    if isinstance(panel2_natural, dict):
        summary = panel2_natural.get("summary", summary)
    overall = summary.get("overall") or raw.get("overall") or {}
    by_type = summary.get("by_type", {})
    by_tier = summary.get("by_tier", {})
    latency = raw.get("latency", {})

    model_id = provenance.get("model_id") or raw.get("model")
    head_version = raw.get("head_version")
    file_run_id = raw.get("run_id")
    backbone_model_id = (raw.get("backbone") or {}).get("model_id")
    code_commit = raw.get("code_commit")

    meta = JEVBENCH_META.get(stem)
    if meta:
        display_name, params, model_type = meta
    elif model_id:
        variant = provenance.get("variant")
        display_name = f"{model_id} ({variant})" if variant else model_id
        # Provenance records the checkpoint, not its parameter count or
        # architecture class. Neither is asserted here.
        params, model_type = "—", "unknown"
    else:
        print(f"  WARNING: no model id for jevbench/{stem}, skipping")
        return None

    if head_version:
        display_name = add_head_version(display_name, head_version)

    return {
        "run_id": file_run_id or f"jevbench-{stem}",
        "model": {"name": display_name, "params": params, "type": model_type},
        "model_id": model_id,
        "head_version": head_version,
        "backbone_model_id": backbone_model_id,
        "code_commit": code_commit,
        "method": (
            "trained-heads" if readout == "trained-heads"
            else "third-party" if readout.startswith("native-decision")
            else "zero-shot"
        ),
        "readout": readout,
        "readout_detail": provenance.get("readout"),
        "source_file": source_file_label(filepath),
        "benchmark": "jevbench",
        "n": raw.get("n") or overall.get("total", 0),
        "accuracy": round(overall.get("accuracy", 0), 6),
        "correct": overall.get("correct", 0),
        "total": overall.get("total", 0),
        "by_type": {k: {"accuracy": round(v["accuracy"], 4), "correct": v["correct"], "total": v["total"]}
                    for k, v in by_type.items()},
        "by_tier": {k: {"accuracy": round(v["accuracy"], 4), "correct": v["correct"], "total": v["total"]}
                    for k, v in by_tier.items()},
        "latency_ms": round(latency.get("mean_s", 0) * 1000, 1) if latency else None,
    }


def jevbench_source_files() -> list[Path]:
    """Every committed file that supplies a JevBench row.

    ``results/jevbench`` holds the trained-head, native-reranker and three
    continuation cells. ``logit_prov`` holds the judgement and continuation
    cells that carry ``_provenance``. This follows #311 item 2, which asks for
    judgement and continuation rows only: the four ``jevbench_reranker_*``
    files are native-reranker cells and stay out. Only the reranker
    checkpoint's native row would collide with
    ``raw-qwen3-reranker-4b.json``; the other three are native-reranker cells
    for checkpoints that have no row here.
    """
    paths: list[Path] = []
    if JEVBENCH_DIR.exists():
        for filepath in sorted(JEVBENCH_DIR.glob("*.json")):
            if (
                filepath.stem in JEVBENCH_META
                or filepath.stem.startswith("logit_")
                or filepath.stem.startswith(("panel2_Qwen", "continuation_", "judgement_"))
            ):
                paths.append(filepath)
    if LOGIT_PROV_DIR.exists():
        for pattern in ("jevbench_judgement_*.json", "jevbench_logit_*.json"):
            paths.extend(sorted(LOGIT_PROV_DIR.glob(pattern)))
    return paths


def transform_jevbench() -> list[dict]:
    """Transform JevBench result files into dashboard format."""
    runs = []
    for filepath in jevbench_source_files():
        run = build_jevbench_run(filepath)
        if run:
            runs.append(run)
            print(f"  {run['run_id']}: {run['readout']} "
                  f"{run['accuracy']:.1%} ({run['correct']}/{run['total']})")
    return runs


LANDING_BAND = (
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-Reranker-4B",
    "microsoft/Phi-4-mini-instruct",
)
DEV_RESOLUTION_MODELS = (
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-8B",
    "zeroentropy/zerank-2-reranker",
)
RESOLUTION_INPUTS_PATH = EXPERIMENTS_DIR / "resolution_inputs.json"
DEV_MANIFEST_PATH = REPO_ROOT / "data" / "benchmarks" / "dev" / "dev_800.json"


def _prediction_map(filepath: Path) -> dict[str, bool]:
    raw = json.loads(filepath.read_text())
    return {row["id"]: bool(row["correct"]) for row in raw["predictions"]}


def _logit_prov_path(prefix: str, model_id: str) -> Path:
    return LOGIT_PROV_DIR / f"{prefix}_{model_id.replace('/', '_')}.json"


def _discordance(first: dict[str, bool], second: dict[str, bool]) -> tuple[float, int]:
    shared = sorted(set(first) & set(second))
    if not shared:
        return 0.0, 0
    return sum(1 for item in shared if first[item] != second[item]) / len(shared), len(shared)


def _mde_pp(n_items: int, discordance: float) -> float:
    return round(detectable_delta(n_items, discordance) * 100, 1)


def build_landing(jevbench_runs: list[dict], eval_runs: list[dict], training_runs: list[dict]) -> dict:
    """Summarise the landing-page claims from rows that are already traceable."""
    cells: dict[str, dict[str, dict]] = {}
    for run in jevbench_runs:
        model_id = run.get("model_id")
        if model_id:
            cells.setdefault(model_id, {})[run["readout"]] = run

    pairs: list[dict] = []
    tier_gaps: dict[str, list[float]] = {"easy": [], "hard": [], "original": []}
    for model_id, by_readout in cells.items():
        judgement = by_readout.get("judgement")
        continuation = by_readout.get("continuation")
        if not (judgement and continuation):
            continue
        pairs.append({
            "model_id": model_id,
            "name": judgement["model"]["name"],
            "judgement_pp": round(judgement["accuracy"] * 100, 1),
            "continuation_pp": round(continuation["accuracy"] * 100, 1),
            "gap_pp": round((judgement["accuracy"] - continuation["accuracy"]) * 100, 1),
        })
        for tier in tier_gaps:
            j = judgement["by_tier"].get(tier)
            c = continuation["by_tier"].get(tier)
            if j and c:
                tier_gaps[tier].append((j["accuracy"] - c["accuracy"]) * 100)

    wins = [pair for pair in pairs if pair["gap_pp"] > 0]
    losses = [pair for pair in pairs if pair["gap_pp"] <= 0]
    above = [pair for pair in pairs if pair["gap_pp"] > 10.0]
    largest_tier = max(
        tier_gaps,
        key=lambda tier: statistics.mean(tier_gaps[tier]) if tier_gaps[tier] else float("-inf"),
    )

    band = []
    for model_id in LANDING_BAND:
        judgement = cells.get(model_id, {}).get("judgement")
        if judgement:
            band.append({
                "model_id": model_id,
                "name": judgement["model"]["name"],
                "pp": round(judgement["accuracy"] * 100, 1),
            })
    base = cells.get("Qwen/Qwen3-4B-Base", {}).get("judgement")

    inputs = json.loads(RESOLUTION_INPUTS_PATH.read_text())
    jev_first, jev_second = inputs["jevbench"]["pair"]
    jev_discordance, jev_n = _discordance(
        _prediction_map(_logit_prov_path("jevbench_judgement", jev_first)),
        _prediction_map(_logit_prov_path("jevbench_judgement", jev_second)),
    )
    jev_mde = _mde_pp(jev_n, jev_discordance)

    dev_first, dev_second = inputs["dev"]["pair"]
    dev_first_preds = _prediction_map(_logit_prov_path("dev_judgement", dev_first))
    dev_second_preds = _prediction_map(_logit_prov_path("dev_judgement", dev_second))
    dev_shared = sorted(set(dev_first_preds) & set(dev_second_preds))
    dev_first_only, dev_second_only, dev_p = mcnemar(dev_first_preds, dev_second_preds, dev_shared)

    dev_discordances = []
    for model_id in DEV_RESOLUTION_MODELS:
        rate, _ = _discordance(
            _prediction_map(_logit_prov_path("dev_judgement", model_id)),
            _prediction_map(_logit_prov_path("dev_logit", model_id)),
        )
        dev_discordances.append(rate)
    dev_n = len(dev_first_preds)
    manifest = json.loads(DEV_MANIFEST_PATH.read_text())
    mean_cluster = dev_n / len(manifest["cluster_sizes"])
    rho_low, rho_high = inputs["dev"]["rho"]
    n_eff_low = dev_n / (1 + (mean_cluster - 1) * rho_high)
    n_eff_high = dev_n / (1 + (mean_cluster - 1) * rho_low)
    dev_mdes = [
        _mde_pp(n_eff, rate)
        for n_eff in (n_eff_low, n_eff_high)
        for rate in (min(dev_discordances), max(dev_discordances))
    ]
    dev_mde_low, dev_mde_high = min(dev_mdes), max(dev_mdes)

    judgement_runs = sorted(
        (run for run in jevbench_runs if run["readout"] == "judgement"),
        key=lambda run: -run["accuracy"],
    )
    top_four = [
        {"model_id": run["model_id"], "name": run["model"]["name"], "pp": round(run["accuracy"] * 100, 1)}
        for run in judgement_runs[:4]
    ]

    jev_pair_rows = [cells[model_id]["judgement"] for model_id in (dev_first, dev_second) if model_id in cells]
    jev_item_diff = (
        abs(jev_pair_rows[0]["correct"] - jev_pair_rows[1]["correct"])
        if len(jev_pair_rows) == 2 else None
    )

    trained = max(
        (run for run in training_runs if run["benchmark"] == "banking77" and run.get("best_accuracy") is not None),
        key=lambda run: run["best_accuracy"],
        default=None,
    )
    jev_run = next((run for run in eval_runs if run["method"] == "jev_api"), None)
    jev_banking = None
    if jev_run:
        jev_banking = jev_run["results"]["by_source"].get("banking77", {}).get("accuracy")

    panel2_pairs: list[dict] = []
    for model_id, by_readout in cells.items():
        judgement = by_readout.get("judgement")
        panel2 = by_readout.get("panel2")
        if not (judgement and panel2):
            continue
        panel2_pairs.append({
            "model_id": model_id,
            "name": judgement["model"]["name"],
            "judgement_pp": round(judgement["accuracy"] * 100, 1),
            "panel2_pp": round(panel2["accuracy"] * 100, 1),
            "gap_pp": round((panel2["accuracy"] - judgement["accuracy"]) * 100, 1),
        })

    panel2_type_delta: dict[str, dict] = {}
    for panel2_type in ("choice", "noul", "score"):
        delta = 0
        total = 0
        for by_readout in cells.values():
            judgement = by_readout.get("judgement")
            panel2 = by_readout.get("panel2")
            if not (judgement and panel2):
                continue
            judgement_type = judgement["by_type"].get(panel2_type)
            panel2_cell = panel2["by_type"].get(panel2_type)
            if judgement_type and panel2_cell:
                delta += panel2_cell["correct"] - judgement_type["correct"]
                total += judgement_type["total"]
        if total:
            panel2_type_delta[panel2_type] = {
                "items": delta,
                "total": total,
                "pp": round(delta / total * 100, 1),
            }

    pair_sources = sorted({
        by_readout[readout]["source_file"]
        for by_readout in cells.values()
        if by_readout.get("judgement") and by_readout.get("continuation")
        for readout in ("judgement", "continuation")
    })
    band_sources = sorted({cells[entry["model_id"]]["judgement"]["source_file"] for entry in band})
    panel2_sources = sorted({
        by_readout[readout]["source_file"]
        for by_readout in cells.values()
        if by_readout.get("judgement") and by_readout.get("panel2")
        for readout in ("judgement", "panel2")
    })

    resolution_sources = sorted({
        source_file_label(RESOLUTION_INPUTS_PATH),
        "model/evaluation/paired_power.py",
        source_file_label(DEV_MANIFEST_PATH),
        source_file_label(_logit_prov_path("jevbench_judgement", jev_first)),
        source_file_label(_logit_prov_path("jevbench_judgement", jev_second)),
        *(
            source_file_label(_logit_prov_path(prefix, model_id))
            for model_id in DEV_RESOLUTION_MODELS
            for prefix in ("dev_judgement", "dev_logit")
        ),
        source_file_label(_logit_prov_path("dev_judgement", dev_first)),
        source_file_label(_logit_prov_path("dev_judgement", dev_second)),
    })

    # Every file the page names, once. The cards link here instead of each
    # repeating its own list.
    page_sources = sorted({
        source
        for source in (
            *pair_sources,
            *panel2_sources,
            *band_sources,
            *resolution_sources,
            source_file_label(OUTPUT_PATH),
            trained["source_file"] if trained else None,
            jev_run["source_file"] if jev_run else None,
        )
        if source
    })

    return {
        "judgement_vs_continuation": {
            "measured": len(pairs),
            "wins": len(wins),
            "losses": [
                {"model_id": pair["model_id"], "name": pair["name"], "gap_pp": pair["gap_pp"]}
                for pair in losses
            ],
            "above_10_count": len(above),
            "above_10_min_pp": min((pair["gap_pp"] for pair in above), default=None),
            "above_10_max_pp": max((pair["gap_pp"] for pair in above), default=None),
            "largest_tier": largest_tier,
            "source_files": pair_sources,
        },
        "panel2_vs_judgement": {
            "type_delta": panel2_type_delta,
            "measured": len(panel2_pairs),
            "wins": sum(1 for pair in panel2_pairs if pair["gap_pp"] > 0),
            "min_pp": min((pair["gap_pp"] for pair in panel2_pairs), default=None),
            "max_pp": max((pair["gap_pp"] for pair in panel2_pairs), default=None),
            "pairs": panel2_pairs,
            "source_files": sorted({
                by_readout[readout]["source_file"]
                for by_readout in cells.values()
                if by_readout.get("judgement") and by_readout.get("panel2")
                for readout in ("judgement", "panel2")
            }),
        },
        "post_trained_4b": {
            "base_pp": round(base["accuracy"] * 100, 1) if base else None,
            "min_pp": min((entry["pp"] for entry in band), default=None),
            "max_pp": max((entry["pp"] for entry in band), default=None),
            "checkpoints": band,
            "source_files": band_sources,
        },
        "resolution": {
            "jevbench_pp": jev_mde,
            "jevbench_pair": [jev_first, jev_second],
            "dev_min_pp": dev_mde_low,
            "dev_max_pp": dev_mde_high,
            "dev_card_min_pp": round(dev_mde_low),
            "dev_card_max_pp": round(dev_mde_high),
            "dev_pair": [dev_first, dev_second],
            "dev_pair_diff_pp": round((dev_first_only - dev_second_only) / len(dev_shared) * 100, 2),
            "dev_pair_p": round(dev_p, 4),
            "jevbench_pair_item_diff": jev_item_diff,
            "top_four": top_four,
            "top_four_spread_pp": round(
                (top_four[0]["pp"] - top_four[-1]["pp"]) if len(top_four) == 4 else 0.0, 1
            ),
            "source_files": resolution_sources,
        },
        "instrumented_rows": len(jevbench_runs),
        "banking77": {
            "trained_pp": round(trained["best_accuracy"] * 100, 1) if trained else None,
            "trained_model": trained["model"] if trained else None,
            "trained_source_file": trained["source_file"] if trained else None,
            "jev_pp": round(jev_banking * 100, 1) if jev_banking is not None else None,
            "jev_source_file": jev_run["source_file"] if jev_run else None,
        },
        "source_files": page_sources,
    }


def main() -> None:
    eval_runs = []
    training_runs = []

    print("Processing eval files...")
    for filepath in sorted(EXPERIMENTS_DIR.glob("*.json")):
        if filepath.stem.startswith("sf2_") or filepath.stem.startswith("train_"):
            continue
        result = transform_eval_file(filepath)
        if result:
            eval_runs.append(result)
            print(f"  {filepath.stem}: {result['n_benchmarks']} benchmarks, "
                  f"avg_acc={result['results']['aggregate']['accuracy']}")

    print(f"\nProcessing training files...")
    for filepath in sorted(EXPERIMENTS_DIR.glob("train_*.json")):
        result = transform_training_file(filepath)
        if result:
            training_runs.append(result)
            print(f"  {filepath.stem}: {result['n_epochs']} epochs, "
                  f"best_acc={result['best_accuracy']}")

    print(f"\nProcessing JevBench files...")
    jevbench_runs = transform_jevbench()

    jev_comparison = build_jev_comparison(eval_runs)
    landing = build_landing(jevbench_runs, eval_runs, training_runs)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_runs": eval_runs,
        "training_runs": training_runs,
        "jevbench_runs": jevbench_runs,
        "jev_comparison": jev_comparison,
        "landing": landing,
        "benchmarks": dict(sorted(BENCHMARK_TYPES.items())),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  {len(eval_runs)} eval runs, {len(training_runs)} training runs")


if __name__ == "__main__":
    main()
