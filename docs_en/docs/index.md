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

### Zero-training baselines (average across 19 benchmarks)

| Backbone | Params | Avg Accuracy |
|----------|--------|-------------|
| Qwen2.5-7B-Instruct | 7B | 68.5% |
| Ettin-400m | 400M | 59.4% |
| Qwen2.5-1.5B | 1.5B | 59.2% |

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
