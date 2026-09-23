#!/usr/bin/env python3
"""Transform raw experiment JSONs into a standardized format for the web dashboard.

Reads model/experiments/*.json and writes web/public/results/eval_results.json.
"""

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "model" / "experiments"
OUTPUT_PATH = REPO_ROOT / "web" / "public" / "results" / "eval_results.json"

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

    jev_comparison = build_jev_comparison(eval_runs)

    output = {
        "generated_at": None,  # filled by CI or manually
        "eval_runs": eval_runs,
        "training_runs": training_runs,
        "jev_comparison": jev_comparison,
        "benchmarks": dict(sorted(BENCHMARK_TYPES.items())),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  {len(eval_runs)} eval runs, {len(training_runs)} training runs")


if __name__ == "__main__":
    main()
