# Krino

*κρίνω — to judge, to separate, to decide*

Probing, analysis, and open replication of [TypeSafe.ai](https://typesafe.ai)'s Jev — the first System One model for typed, probabilistic decisions.

## Highlights

- **Black-box probing** of the Jev API (21 probes, 5,620 API calls) identified the architecture: causal Qwen-family backbone + typed decision heads + RLCD calibration
- **Open replication** achieving **95.2% on Banking77** (vs Jev's 75.0%) with a 150M reranker-pretrained encoder + trained decision heads
- **Novel finding:** cross-encoder reranker pretraining transfers to decision scoring — Ettin-150m beats vanilla ModernBERT by +4.8pp at identical params
- **19-benchmark evaluation suite** across 3 question types (noul, choice, score) and 6 domains

## Results

### Trained heads (Banking77, 77-class intent classification)

| Backbone | Params | Accuracy |
|----------|--------|----------|
| **Ettin-150m r128** | 150M + ~800K | **95.2%** |
| Qwen3-0.6B r64 | 596M + 402K | 93.2% |
| ModernBERT-base r64 | 149M + 301K | 89.0% |
| Jev (reference) | unknown | 75.0% |

### Zero-training baselines (average across the 16 choice+noul benchmarks)

| Backbone | Params | Avg Accuracy |
|----------|--------|-------------|
| Qwen2.5-7B-Instruct | 7B | 67.1% |
| Ettin-400m | 400M | 59.0% |
| Qwen2.5-1.5B | 1.5B | 58.4% |

The average is the unweighted mean over the 16 `choice` and `noul` benchmarks, computed by `scripts/transform_results.py::get_accuracy_for_aggregate` from the committed `model/experiments/phaseA_*.json` files. The three `score` benchmarks are reported as MAE and are not folded into this column. The values previously quoted here (68.5 / 59.4 / 59.2) were not reproducible from any committed file.

### JevBench status (v1.7.3 snapshot)

The official board (protocol `jevbench::v1.7`, 270 ranked rows; snapshot fetched 2026-10-30, source sha256 `95251a6582156ce62f01ffd8810eda48c034f52420ba17ea6d378952f03b88fd`) places Jev 1.13.0 at rank 21 with Intelligence 65.60 and composite 71.45. The best open-weights entry is H2O-Lightning-4B at rank 11 with Intelligence 61.05 and composite 73.78; the leader is singularity-super-decider at composite 78.48.

All local numbers are JevBench-public-231 scores under our own harness, not official JevBench scores. Official reference (v1.7.3 snapshot): Jev 1.13.0 Intelligence 65.60 / composite 71.45.

## What is Jev?

Jev is a "System One" model that returns typed, probabilistic decisions instead of generated text. It evaluates a `state` against typed `questions` and returns structured `answers` with calibrated probabilities.

| Type | Purpose | Returns |
|------|---------|---------| 
| **Noul** | Yes/no probability | `noul` (0–1 float) |
| **Choice** | Pick one from a set | `choice`, `probabilities`, `confidence` |
| **Score** | Rate on ordered levels | `score`, `probabilities`, `legend`, `confidence` |

## Site Contents

- [**Overview**](overview.md) — TypeSafe.ai background, Jev primitives, and API design
- [**Probing**](probing/index.md) — Black-box API probing methodology and architecture analysis
- [**Replication**](replication/index.md) — Open model replication: backbone sweep, trained heads, Jev comparison

## Links

- [GitHub Repository](https://github.com/Oaklight/krino)
- [TypeSafe docs](https://docs.typesafe.ai)
- [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe)
