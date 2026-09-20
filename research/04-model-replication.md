# Model Replication: Findings and Results

**Date:** 2026-09-20
**Status:** Step 2 complete, Step 3 calibration in progress
**Epic:** [#19](https://github.com/Oaklight/jev-explore/issues/19)

## Overview

This document consolidates findings from our effort to build an open typed decision model reproducing the pipeline identified through black-box probing of TypeSafe's Jev:

```
State → shared encoding → per-option token-level scoring → softmax → calibrated probabilities
```

We evaluated 10 backbone models across 19 benchmarks, trained lightweight decision heads, and tested cross-benchmark generalization and surface-form sensitivity. We then benchmarked the Jev API on the exact same eval suite to get authoritative comparison targets (PR #45, issue #44). The headline results: a 150M reranker-pretrained encoder with trained heads achieves 95.2% on Banking77 (vs Jev's 75.0%), but Jev dominates on reasoning-heavy tasks (ARC 99.0%, RACE 95.5%, authored144 97.9%).

## Step 1: Zero-Training Baselines

### Approach

Two scoring protocols, no training:

- **Causal LM logit readout (Route B):** Present state + question as a prompt prefix, score each option as a continuation by summing per-token log-probabilities, normalize across options.
- **Encoder cross-scoring:** Encode (state+question, option) pairs via a cross-encoder and score by embedding similarity (vanilla encoder) or relevance score (reranker).

### Results: 19 benchmarks × 10 models

Average accuracy ranking across choice + noul benchmarks (200 items per benchmark, H200 bf16):

| Rank | Model | Params | Type | Avg Accuracy |
|------|-------|--------|------|-------------|
| 1 | Qwen2.5-7B-Instruct | 7B | Causal | 68.5% |
| 2 | Phi-4-mini | 3.8B | Causal | 65.0% |
| 3 | **Ettin-400m** | **400M** | **Reranker** | **59.4%** |
| 4 | Qwen2.5-1.5B | 1.5B | Causal | 59.2% |
| 5 | Qwen3-1.7B | 1.7B | Causal | 54.2% |
| 6 | Ettin-150m | 150M | Reranker | 53.5% |
| 7 | Qwen2.5-0.5B | 0.5B | Causal | 53.5% |
| 8 | SmolLM2-1.7B | 1.7B | Causal | 51.1% |
| 9 | Ettin-68m | 68M | Reranker | 50.4% |
| 10 | Qwen3-0.6B | 0.6B | Causal | 47.8% |

Vanilla ModernBERT (base/large) scored near random on most benchmarks — embedding-similarity scoring without training is not viable for typed decisions.

### Key findings

1. **Reranker pretraining is remarkably effective.** Ettin-400m (400M) matches Qwen2.5-1.5B (1.5B) at 1/4 the params. Cross-encoder relevance scoring naturally maps to option discrimination.

2. **Qwen2.5 > Qwen3 at matched scale.** Qwen2.5-1.5B beats Qwen3-1.7B on 12/15 benchmarks. The Qwen3 "thinking" architecture doesn't help for logit readout — its internal reasoning tokens are not exposed in the continuation log-probabilities.

3. **Banking77 (77-class) is the hardest discriminator.** All causal models score 0–3% (random = 1.3%) — logit readout completely fails at this cardinality due to surface-form bias. Only Ettin rerankers show meaningful zero-shot discrimination: Ettin-400m (49.0%), Ettin-68m (27.0%).

4. **Rerankers underperform on noul.** Binary yes/no scoring via cross-encoder relevance doesn't work well — the "documents" are just "yes" and "no," which don't carry enough semantic signal for relevance matching.

5. **CodeSearchNet 100% for rerankers is a length-bias artifact.** The correct code snippet is 2× longer on average than distractors. Rerankers exploit length/lexical overlap rather than semantic understanding. This benchmark needs distractor length matching (#29).

### Latency (batched inference, H200 bf16)

| Model | Params | Banking77 (77-opt) | SST-2 (2-opt) |
|-------|--------|-------------------|---------------|
| Ettin-68m | 68M | **29ms** | 6ms |
| Ettin-150m | 150M | 31ms | 6ms |
| Qwen2.5-0.5B | 0.5B | 35ms | 15ms |
| SmolLM2-1.7B | 1.7B | 34ms | 14ms |
| Ettin-400m | 400M | 38ms | 9ms |
| Qwen2.5-7B-Inst | 7B | 50ms | 19ms |

All models meet the <100ms target. Batched inference (PR #42) reduced Banking77 latency from 1848ms to ~35ms per item — a 39× speedup via padding options and expanding the prefix KV cache with `batch_repeat_interleave`.

### Measured Jev API latency (for comparison)

From 5,564 timed API calls across 21 probing experiments (16 standard + 5 controlled):
- **Median:** 166ms (end-to-end, includes network)
- **Mean:** 174ms
- **p5–p95:** 83ms–288ms
- **Estimated server-side compute:** ~60–100ms (network overhead varies by payload size)

Controlled probes use larger payloads (multi-question, multi-option) than standard probes, shifting the distribution upward. Standard probes alone: median 106ms, p5–p95 74–195ms.

Data: `probing/results/*.jsonl` (`elapsed_s` for standard, `total_time_s` for controlled).

## Jev API Benchmark Results

We ran the Jev API against all 19 benchmarks using the exact same 200-item eval subsets (PR #45, issue #44). This gives authoritative comparison targets on identical data.

### Full results

**Choice benchmarks:**

| Benchmark | Options | Jev API | Best zero-shot (ours) | Best model (ours) |
|-----------|---------|---------|----------------------|-------------------|
| ARC | 4 | **99.0%** | 73.5% (Ettin-400m / Phi-4-mini) | — |
| authored144 | 3 | **97.9%** | 54.9% (Qwen2.5-7B) | — |
| RACE | 4 | **95.5%** | 65.5% (Phi-4-mini) | — |
| FEVER | 3 | **91.0%** | 66.5% (Qwen2.5-7B) | — |
| HellaSwag | 4 | **89.5%** | 68.0% (Qwen2.5-7B) | — |
| AG News | 4 | 79.0% | 80.0% (SmolLM2-1.7B) | — |
| SWAG | 4 | 77.5% | 71.0% (Qwen2.5-7B) | — |
| Banking77 | 77 | 75.0% | 49.0% (Ettin-400m) | **95.2%** (Ettin-150m r128 trained) |
| CodeSearchNet | 4 | **100.0%** | 100.0% (Ettin, length bias) | — |

**Noul benchmarks:**

| Benchmark | Jev API | Best zero-shot (ours) | Gap |
|-----------|---------|----------------------|-----|
| SST-2 | 90.5% | 87.5% (Qwen2.5-1.5B) | +3.0pp |
| MultiRC | **90.0%** | 88.0% (Qwen2.5-7B) | +2.0pp |
| MedNLI | **89.5%** | 86.0% (Qwen2.5-7B) | +3.5pp |
| TabFact | 82.0% | 78.0% (Qwen3-1.7B / Ettin-400m) | +4.0pp |
| ContractNLI | **80.5%** | 79.5% (Phi-4-mini) | +1.0pp |

**Score benchmarks (MAE, lower is better):**

| Benchmark | Jev API | Best zero-shot (ours) |
|-----------|---------|----------------------|
| Yelp | **0.432** | 0.85+ (all models) |
| SST-5 | **0.489** | 0.85+ (all models) |
| STS-B | **0.494** | 0.85+ (all models) |

**Jev calibration (ECE):**

| Benchmark | Jev ECE | Notes |
|-----------|---------|-------|
| ARC | **0.009** | Near-perfect calibration |
| authored144 | 0.031 | Excellent |
| RACE | 0.025 | Excellent |
| Banking77 | 0.163 | Weaker — 77-class is hard to calibrate |
| SST-2 | 0.133 | Moderate |
| AG News | 0.153 | Moderate |

**Jev latency:**

Consistent ~305ms TTFB across all benchmarks (measured from our client). Higher than the probing-era measurements (median 166ms) — likely due to benchmark payloads being larger than typical probing payloads, or API infrastructure changes.

### Key observations

1. **Jev dominates on reasoning-heavy tasks.** ARC (99.0%), authored144 (97.9%), RACE (95.5%), HellaSwag (89.5%) — these all require multi-hop reasoning, reading comprehension, or commonsense inference. Our best zero-shot models (Qwen2.5-7B, Phi-4-mini) are 20-43pp behind. This strongly suggests Jev uses a large causal backbone with substantial reasoning capability — an encoder-only model cannot achieve 99% on ARC.

2. **We beat Jev on Banking77 with trained heads** (95.2% vs 75.0%), but this is in-domain trained. Jev's 75.0% is zero-shot generalization across 77 classes — impressive given no task-specific training. The comparison is not apples-to-apples until we have multi-task trained heads that generalize.

3. **Jev's calibration varies by task.** ARC ECE=0.009 (excellent) vs Banking77 ECE=0.163 (weaker). High-cardinality choice tasks are harder to calibrate, consistent with information-theoretic expectations.

4. **Score-type tasks are where Jev's advantage is clearest.** Our zero-shot models get MAE 0.85+ on SST-5/STS-B/Yelp; Jev gets 0.43-0.49. The logit-readout and cross-encoder approaches don't handle ordinal regression well. Jev's ScoreHead (or equivalent) is substantially better calibrated for continuous scores.

5. **The causal backbone question is settled.** A pure encoder (Ettin-150m) cannot reach Jev-level performance on ARC, RACE, or authored144 — these require the kind of sequential reasoning that causal LMs excel at. A general-purpose decision model needs either a causal backbone or a hybrid architecture.

### Implications for training strategy

The Jev API results reshape our priorities:

- **Multi-task training on all 19 benchmarks** is the immediate next step — single-benchmark heads score 95.2% in-domain but would likely score <50% on ARC/RACE
- **Causal backbone (Qwen) may be needed** for reasoning-heavy tasks, despite encoder's efficiency advantage on comparison tasks. A hybrid approach — or simply accepting a larger model — may be necessary
- **Calibration is Jev's core advantage.** Even where our accuracy is competitive (SST-2, MultiRC), Jev's calibration (ECE) is likely better. Step 3 calibration training is the path to closing this gap
- **Score-type performance needs a different approach.** MAE 0.85 vs Jev's 0.43 — the current ScoreHead + CE loss is insufficient. Ordinal losses (ranked probability score) or RLCD with score-specific rewards are needed

## Step 2: Trained Decision Heads

### Architecture

Three lightweight heads on a frozen backbone:

- **NoulHead:** Linear projection → sigmoid → P(yes). ~1K params.
- **ChoiceHead:** AttentionHead (option queries cross-attend to context, dot-product scoring) → softmax. Optional rival-aware attention where options attend to each other before scoring (adapted from jevbetter, equivalent to SetRank inter-document attention). ~200K params at rank 64.
- **ScoreHead:** Same as ChoiceHead over ordered levels, plus expected-value computation. ~200K params at rank 64.

Total trainable: ~301K–402K params depending on backbone hidden size. Backbone remains frozen — no gradients flow through it.

### Training protocol

- Loss: cross-entropy (BCE for noul, CE for choice/score — equivalent to ListNet top-1 probability)
- Optimizer: AdamW with linear warmup + cosine decay
- Data: Banking77 (5000 train, 500 eval), 5 epochs
- Hardware: single H200 GPU, ~3 min/epoch

### Results: Banking77 in-domain accuracy

| Backbone | Type | Backbone params | Head rank | Head params | Accuracy |
|----------|------|----------------|-----------|-------------|----------|
| **Ettin-150m r128** | **Reranker** | **150M** | **128** | **~800K** | **95.2%** |
| Ettin-150m r64 | Reranker | 150M | 64 | ~400K | 93.8% |
| Qwen3-0.6B r64 | Causal | 596M | 64 | 402K | 93.2% |
| Ettin-400m r64 | Reranker | 400M | 64 | ~400K | 91.6% |
| ModernBERT-large r64 | Encoder | 395M | 64 | 402K | 89.2% |
| ModernBERT-base r64 | Encoder | 149M | 64 | 301K | 89.0% |
| ModernBERT-base r128 | Encoder | 149M | 128 | ~600K | 88.4% |
| Jev API (measured) | — | — | — | — | 75.0% |

### Key findings

1. **Reranker pretraining transfers to decision scoring.** Ettin-150m (93.8%) beats vanilla ModernBERT-base (89.0%) at identical param count (150M). Same architecture, different pretraining — the only variable is whether the backbone was pretrained on 143M relevance-scoring triples. This is a +4.8pp improvement, confirming that relevance-matching pretraining provides a useful initialization for decision heads.

2. **Head capacity matters for reranker backbones but not vanilla encoders.** Ettin-150m improves from 93.8% (rank 64) to 95.2% (rank 128). ModernBERT-base stays flat (89.0% → 88.4%). The reranker backbone has richer representations that only manifest with sufficient head capacity.

3. **Training > scaling.** A 0.6B causal LM with 402K trained heads (93.2%) beats a 7B causal LM with zero-training logit readout (68.5% average across 19 benchmarks). For typed decisions, small specialized heads are more effective than large general-purpose models.

4. **Ettin-400m underperforms Ettin-150m** (91.6% vs 93.8%). Training instability at epoch 4 (accuracy dropped 89.0% → 75.2% before recovering) suggests the 1e-3 learning rate is too aggressive for the larger backbone. The 150M model trained monotonically.

### Cross-benchmark generalization

Heads trained on Banking77 (77-class intent), evaluated on other tasks without retraining:

| Eval dataset | Type | Qwen3-0.6B | ModernBERT-base |
|-------------|------|-----------|----------------|
| authored144 | Choice (3-opt) | 34.7% | 36.8% |
| SST-2 | Noul (binary) | 48.2% | 51.8% |
| typed-decisions | Choice | 20.0% | 19.5% |

**Heads trained on one benchmark do not generalize.** They memorize the task structure (77 banking intents), not general decision-making. This directly motivates multi-task training and the calibration stage (Step 3).

### Surface-form sensitivity

R² of surface-form features vs assigned probability, measured on semantically equivalent options with varied wording (Probe 3 methodology, partially deconfounded design):

| Model | Length R² | Grammaticality R² | Commonness R² |
|-------|----------|-------------------|---------------|
| Qwen logit readout | 0.024 | 0.959 | 0.040 |
| Qwen + trained heads | 0.715 | 0.101 | 0.645 |
| ModernBERT + trained heads | 0.354 | 0.252 | 0.335 |

**Trained heads change which surface-form dimensions matter but do not eliminate sensitivity:**

- **Logit readout** is dominated by grammaticality (R²=0.96) — the causal LM assigns near-zero probability to ungrammatical continuations, regardless of semantic content.
- **Trained causal heads** shift to length and commonness (R²=0.72, 0.65) — the heads partially learn to ignore grammaticality but pick up different biases from the backbone's hidden representations.
- **Trained encoder heads** show moderate sensitivity across all dimensions (R²=0.25–0.35) — no single dominant bias, but none eliminated either.

**Implication:** Architecture choice (causal vs encoder) reduces surface-form sensitivity but does not solve it. Calibration training (Step 3) must explicitly include a surface-form invariance objective.

## Connection to Learning-to-Rank

The decision model's ChoiceHead is structurally equivalent to a listwise learning-to-rank model:

- The softmax-over-options IS ListNet's top-1 probability model (Cao et al. 2007).
- The rival-aware attention IS SetRank's inter-document attention (Pang et al. 2020).
- The cross-entropy loss IS the ListNet loss.

This connection has practical implications:

| Training stage | Loss | LTR equivalent |
|---------------|------|---------------|
| Step 2 (supervised warmup) | Cross-entropy | ListNet top-1 |
| Step 3 Phase A (calibration) | CE + Brier/MMCE/focal | ListNet + calibration |
| Step 3 Phase B (RL) | REINFORCE + proper scoring rules | — (no LTR equivalent) |

The key difference from standard LTR: decision models need **calibrated probabilities**, not just correct ranking. A reranker only needs to order documents correctly; a decision model's probabilities must be trustworthy. This is what Step 3 addresses.

## Infrastructure and Performance

### Batched inference (PR #42)

Replaced sequential per-option forward passes with batched inference. Options are padded to equal length, the prefix KV cache is expanded via `batch_repeat_interleave`, and all options are scored in a single forward pass per sub-batch.

| Benchmark | Options | Sequential | Batched | Speedup |
|-----------|---------|-----------|---------|---------|
| Banking77 | 77 | 1,848ms | 35ms | **53×** |
| SST-2 | 2 | 42ms | 15ms | 2.8× |

### KV cache optimization (PR #41)

Replaced `copy.deepcopy(cache)` per option with `DynamicCache.crop()` — a zero-copy operation that slices the cache back to prefix length after each option's forward pass. Eliminated 15,400 multi-GB tensor copies per Banking77 evaluation run.

### Data pipeline (PR #34, #40)

19 benchmarks across 6 domains converted to unified TypedQuestion format:

| Type | Benchmarks | Total items |
|------|-----------|-------------|
| Choice | Banking77, AG News, ARC, RACE, HellaSwag, CodeSearchNet, SWAG, FEVER, authored144 | ~690K |
| Noul | SST-2, TabFact, MultiRC, MedNLI, ContractNLI, MNLI | ~655K |
| Score | SST-5, STS-B, Yelp | ~90K |

## Current Status and Next Steps

### Completed

- **Step 0:** Data pipeline, eval suite, API server (#20, closed)
- **Step 1:** Zero-training baselines — 10 models × 19 benchmarks (#21, closed)
- **Step 2:** Trained heads — Ettin-150m r128 = 95.2% on Banking77, best overall (#22, closed)
- **Backbone sweep:** Causal + encoder + reranker comparison (#27)

### In progress

- **Step 3 Phase A:** Calibration training infrastructure built (losses.py, train_calibrated.py, sweep config). Actual sweep runs pending. Key losses: Brier, MMCE, focal, label smoothing (#23)
- **Data pipeline Phase B/C:** Synthetic data generation and hard negative mining not yet started (#29)

### Not started

- **Step 3 Phase B:** RLCD-style RL with proper scoring rules
- **Step 4:** Compact custom architecture exploration (partially addressed by Ettin findings — reranker-pretrained encoder is the current winner)

### Open questions

1. **Does multi-task training fix generalization?** Heads trained on Banking77 alone don't transfer. Training on all 19 benchmarks simultaneously should help, but the balance across types and domains needs tuning.

2. **Can calibration training reduce surface-form sensitivity?** Architecture alone doesn't solve it (encoder R²=0.35). An explicit invariance loss — `KL(P(options | original), P(options | paraphrased))` — is planned for Step 3 but not yet tested.

3. **Encoder vs causal backbone for production?** Ettin-150m wins on comparison tasks (Banking77 95.2%), but Jev API results show reasoning-heavy tasks (ARC 99.0%, RACE 95.5%) require causal LM representations that encoders cannot provide. A general-purpose model may need a causal backbone or a hybrid encoder-decoder architecture.

4. **Can we close the score-type gap?** Jev scores MAE 0.43–0.49 on SST-5/STS-B/Yelp; our models get 0.85+. The ScoreHead + CE loss is insufficient for ordinal regression. Ranked probability score (Laya's RLCD) or ordinal-specific losses may be needed.

5. **What is Jev's calibration advantage?** Jev achieves ECE 0.009 on ARC and 0.031 on authored144. Our models' ECE is unmeasured on trained heads — this is the key metric for Step 3 to target.

## References

| Project | Relevance |
|---------|-----------|
| [Laya](https://huggingface.co/convaiinnovations/laya) | Only true RLCD reproduction. ModernBERT-large + REINFORCE. typed-decisions 0.766 vs Jev 0.727 |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Qwen backbone, decision heads, RLCD experiment, game eval |
| [jevlike](https://github.com/vinnylarouge/jevlike) | AttentionHead architecture we adapted |
| [jevbetter](https://github.com/olanotolu/jevbetter) | Rival-aware attention we incorporated |
| [Ettin rerankers](https://huggingface.co/cross-encoder) | ModernBERT-arch rerankers trained on 143M triples. Our top backbone |
| [SetRank](https://dl.acm.org/doi/10.1145/3397271.3401104) | Inter-document attention for listwise LTR — same mechanism as rival-aware |
