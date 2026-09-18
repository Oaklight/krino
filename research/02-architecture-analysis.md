# Jev Architecture Analysis

**Date:** 2026-09-18
**Updated:** 2026-09-18 (3 rounds of probing + cross-model validation, 15 probes, ~2,000 API calls + H200 GPU runs)
**Status:** Empirically validated — high confidence on base model, attention, and RLCD debiasing

## What TypeSafe Has Confirmed

1. **RLCD (Reinforcement Learning for Calibrated Decisions)** — a new post-training method. Optimizes for probability calibration: if the model says 70% across many predictions, roughly 70% should be correct. Distinct from RLHF (human preference) and RLVR (verifiable correctness).

2. **Single forward pass, no autoregressive decoding** — because the output space is fixed and small (up to 255 choices, a scalar score, or a boolean), the model scores all possible answers in one pass. No token-by-token decode loop.

3. **New model architecture** — TypeSafe's launch post says "a new model architecture" but does not disclose what it is.

4. **Parallel sampler** — described as "hardware-aware," designed for "maximum efficiency."

5. **Starts from a pretrained language model** — the AI primer positions RLCD as a *post-training* method applied to pretrained language models, alongside RLHF and RLVR.

6. **No architecture paper published** as of September 2026.

## Our Probing Results

We ran 13 systematic probes across 3 rounds against the Jev API (jev-1.13.0), totaling ~1,800 API calls. Each round included a rigor self-audit, and flawed probes were re-designed and re-run. Raw data in `probing/results/`. Full methodology and scripts in `probing/scripts/`.

### Rigor Methodology

After each round, we asked: "Does this result necessarily imply what we claimed, or could a different architecture produce the same output?" Probes that failed this test were re-designed with stronger discriminators. Key corrections:
- Probe 3 (char-level masking) → superseded by 3b/3c (word-level masking, scaled code definitions)
- Probe 4 (easy 3-option ordering) → superseded by 4b (ambiguous 5/8-option, 190 permutations)
- Probe 6 (easy domain questions) → superseded by 6b (counterintuitive myths near capability boundary)
- Probe 9 (easy translated facts) → supplemented by 9b (language-specific idioms/grammar)

### Probe 1: Token Accounting (34 records)

Verified and refined Hume's token formula:

**Output tokens:** `4 (shared) + 15 × N_questions + Σ tokenized_len(question_id_i)`
- Hume's formula is structurally correct but the ID term should use **tokenized length**, not raw character length (~2 chars per token for repeated characters)
- Output token cost varies by question type: noul=20, score=17, choice=38+ (scales with option count at ~7 tokens/option)
- Score output is constant regardless of level count (3-level and 5-level both produce 17 tokens)

**Input tokens:** Baseline template overhead is ~271 tokens. Each additional noul question costs a stable **~12.6 input tokens**, perfectly linear from 1 to 20 questions. State length scales input tokens linearly with zero effect on output tokens.

**Architecture signal:** Token accounting is exactly additive, consistent with a shared prefix + branched suffix model. The fixed per-question overhead (~12.6 input tokens, 15 output tokens) matches tool-call framing patterns.

### Probe 2: Noul Precision Quantization (1,450 values)

**All probability values — noul, choice, and score — are quantized to an exact 1/100 grid (0.01 steps) with zero residual.**

| Type | Values | Unique | Min nonzero | Range |
|------|--------|--------|-------------|-------|
| Noul | 500 | 88 | 0.01 | 0.01–0.99 |
| Choice prob | 450 | 72 | 0.01 | 0.00–1.00 |
| Score prob | 500 | 75 | 0.01 | 0.00–1.00 |

Noul values never reach exactly 0.0 or 1.0 (capped at 0.01–0.99). Choice and score probabilities can hit exact 0.0 and 1.0.

**Architecture signal:** At most 101 distinct output values. Most likely API-layer rounding rather than model-intrinsic quantization, but could indicate a coarse output head if the model genuinely uses ≤100 bins.

### Probe 3: Diffusion vs Autoregressive Signature (188 records)

**Masking robustness (20 states × 6 masking levels):**

| Masking % | Mean noul | Delta from 0% |
|-----------|-----------|---------------|
| 0% | 0.973 | — |
| 10% | 0.929 | -0.044 |
| 20% | 0.868 | -0.106 |
| 30% | 0.682 | -0.292 |
| 40% | 0.571 | -0.402 |
| 50% | 0.450 | -0.523 |

Near-linear collapse with no plateau. A diffusion model trained with random masking would show much greater resilience.

**Word scrambling (20 states × 3 orderings):**

| Order | Mean noul | Delta |
|-------|-----------|-------|
| Original | 0.974 | — |
| Reversed | 0.762 | -0.211 |
| Shuffled | 0.831 | -0.143 |

Strong word-order sensitivity. Reversal (which completely breaks left-to-right flow) hurts more than shuffling (which preserves some local adjacency).

**Cloze (fill-in-the-blank):** 8/8 correct. Not strongly discriminative — any large model should ace these.

**Verdict (Round 1): autoregressive, not diffusion.** However, this probe was flawed — character-level masking breaks BPE tokens regardless of architecture. Superseded by Probes 3b and 3c.

### Probe 3b: Corrected Diffusion Test (125 records)

Word-level masking, bidirectional context placement (hint before vs after ambiguous content), and token-level noise.

**Bidirectional context (critical test):** hint_last ≥ hint_first in ALL 5 non-trivial cases (+0.04 mean). Token-level noise: only 4.7% degradation at 50% noise (vs 47% for word removal).

**However**, this probe is also inconclusive — in causal attention with KV caching, the question suffix attends to ALL state positions. A strong model can compose information from any position at question-level attention.

### Probe 3c: Rigorous Bidirectional Test (162 records)

Scaled code-definition test: N made-up codes (1→10) where definitions are placed AFTER usage. The causal prediction: def_last degrades as N grows. The bidirectional prediction: flat.

| N codes | def_first noul | def_last noul | delta |
|---|---|---|---|
| 1 | 0.940 | 0.920 | -0.020 |
| 3 | 0.970 | 0.967 | -0.003 |
| 5 | 0.973 | 0.977 | +0.003 |
| 10 | 0.973 | 0.973 | +0.000 |

def_last stays flat. But again, code-lookup is a retrieval/matching task solvable at query level for ANY architecture. **Suggestive but not conclusive on its own.**

**Resolution: Probe 4b's recency bias provides the strongest causal signal — see below.**

### Probe 4: Ordering Bias on Ambiguous Cases (72 records) — SUPERSEDED

**No systematic position bias:**

| Position | Mean probability |
|----------|-----------------|
| 0 (first) | 0.3300 |
| 1 (middle) | 0.3392 |
| 2 (last) | 0.3308 |

All within 1pp of the 0.333 baseline.

**Choice flips:** 2 out of 12 ambiguous cases (16.7%) had their top choice change depending on option order. Both flipped cases had top-two options within ~10pp of each other. Clear-cut cases showed zero sensitivity.

**Probability swings:** Mean 0.065, max 0.220, median 0.040.

**Architecture signal (Round 1):** No position bias on easy 3-option cases. But this finding was **overturned by Probe 4b**.

### Probe 4b: Rigorous Ordering Bias (190 records) — KEY FINDING

5-option (8 cases × 20 permutations) and 8-option (1 case × 30 permutations) on genuinely ambiguous inputs.

**5-option position bias (160 records):**

| Position | Mean probability | Delta from 0.200 |
|---|---|---|
| 0 (first) | 0.179 | -0.021 (suppressed) |
| 1 | 0.199 | -0.001 (neutral) |
| 2 | 0.179 | -0.022 (suppressed) |
| 3 | 0.212 | +0.012 (mild boost) |
| 4 (last) | **0.231** | **+0.031 (recency bias)** |

**Practical impact:** 44% flip rate (4/9 cases). Max probability swing: 0.39.

**This is the strongest causal-vs-bidirectional discriminator we have:** In a causal model, each option's KV is computed left-to-right. The last option attended to all previous options → richest KV → natural recency advantage. A bidirectional model would show no systematic position preference.

### Probe 6b: Hard Domain Variance (60 records)

Counterintuitive myths near capability boundary. 59/60 correct (98%). Only miss: spinach/iron myth. No meaningful domain variance (σ=0.037). MoE question remains **inconclusive**.

### Probe 7: Temporal Cutoff (30 records)

Sharp cliff at **mid-2025**. 100% through 2025-H1, 0% from 2025-H2. Zero self-awareness (noul 0.16–0.28 on TypeSafe/Jev). Pretraining data cutoff: ~June 2025.

### Probe 8: Readout Head Cross-Type (20 records)

100% binary agreement across types. Choice and score nearly identical (diff 0.003). Noul diverges (diff 0.06), higher floor on degenerate inputs, 12% faster. Suggests choice/score share a readout path; noul uses a separate one.

### Probe 9 + 9b: Multilingual Parity (199 records) — KEY FINDING

Easy facts (Probe 9): 100% in 10/12 languages, EN=ZH parity, token cost parity (284≈285).

Hard idioms/grammar (Probe 9b): 100% across all 8 languages. **ZH confidence (0.952) EXCEEDS EN (0.862)** on hard, language-specific questions. This +0.090 gap is a strong Qwen fingerprint — Qwen models have Chinese performance matching or exceeding English.

## Architecture Hypotheses — Updated

### ~~Hypothesis A: Encoder-Only Transformer (BERT-style)~~ — Ruled Out

- Probe 5c: Jev shares the same anti-primacy bias direction as causal Qwen — bidirectional models would not show this pattern
- Token accounting matches autoregressive tool-call framing

### ~~Hypothesis B: Text Diffusion Model (LLaDA-style)~~ — Ruled Out

- Probe 5c: Jev's bias pattern matches causal Qwen, not a diffusion model
- The LLaDA fork was an exploration that did not make it into production

### ~~Hypothesis C: Purpose-Built Architecture~~ — Partially Correct

The "new architecture" refers to the output mechanism (readout heads, parallel sampler, RLCD debiasing), not a novel backbone. The backbone is a standard causal Transformer.

### Hypothesis D: Causal Qwen + Readout Heads + RLCD — Confirmed ✅

**This is our current best hypothesis, supported by all four probes and external evidence.**

Jev's API maps directly onto LLM parallel tool calling with three critical constraints:

| Standard LLM Tool Calling | Jev System One |
|---------------------------|----------------|
| System message / context | `state` |
| Tool definitions (JSON Schema) | Question definitions (noul/choice/score) |
| Parallel tool calls | Parallel questions (forced, never sequential) |
| Tool call arguments (arbitrary JSON) | Probability distribution only (fixed shape) |
| Tool results feed back into next turn | No chaining (one-shot, read-only state) |

The constraint stack that enables single-pass inference:
1. **Only 3 tools** — noul (sigmoid), choice (softmax over K options), score (softmax over N levels)
2. **Every tool output is a probability vector** — never text, never variable-length
3. **All calls are forced-parallel** — no tool can see another tool's output
4. **State is read-only** — no tool modifies shared context

Because the output is always a fixed-size probability distribution, autoregressive decoding is unnecessary. The output head is:
- Noul: single sigmoid → `P(yes)`
- Choice: softmax over K options → `{option: probability}`
- Score: softmax over N levels → `{level: probability}`, then `score = Σ(level × P(level))`

**Evidence supporting this hypothesis:**

| Evidence | Source |
|----------|--------|
| Token accounting is exactly additive (shared prefix + per-question overhead) | Our Probe 1 |
| Output token formula matches tool-call framing (4 shared + 15/answer + tokenized ID) | Our Probe 1 + Hume |
| Autoregressive backbone (masking fragility, word-order sensitivity) | Our Probe 3 |
| Listwise option processing (no position bias, IIA violation) | Our Probe 4 + Hume |
| Probabilities quantized to 1/100 grid | Our Probe 2 |
| Tokenizer closest to Qwen (348/415 match) | Hume (445 probes) |
| Context window: ~32k/branch, ~65k total | Hume (35 probes) |
| Question isolation confirmed (can't see sibling question's content) | Hume (visibility experiment) |
| Likely sparse MoE, ~10B active params | Hume (latency inference) |
| Confidence field is `(p_max - 1/K) / (1 - 1/K)`, not learned | Hume (reverse-engineered) |

**Proposed full architecture:**

```
                    ┌─── Question 1 suffix ──→ Readout Head → P(options)
                    │
State prefix ───→ Shared KV Cache ─┼─── Question 2 suffix ──→ Readout Head → P(yes/no)
  (causal LM)      │               │
                    │               └─── Question 3 suffix ──→ Readout Head → P(levels)
                    │
                    (Prefix computed once, reused for all branches)
```

- **Backbone:** Causal autoregressive Transformer, likely Qwen-family (tokenizer evidence), possibly sparse MoE (~10B active params)
- **Inference:** Shared state prefix KV cache (Hydragen/DeFT pattern) + isolated question suffix branches
- **Output:** Listwise readout head per question (softmax/sigmoid over options/levels), not autoregressive text generation
- **Training:** RLCD — calibration-aware RL applied to the readout head probabilities
- **Quantization:** Output probabilities rounded to 0.01 steps (API layer)

## Consolidated Evidence Summary

| Finding | Source | Confidence | Notes |
|---------|--------|------------|-------|
| **Qwen-family base model** | Hume tokenizer (348/415) + Probe 9/9b (EN=ZH parity, ZH>EN confidence) + Probe 7 (mid-2025 cutoff) | **Very high** | Three independent signals converge. Likely Qwen3.5 |
| **Pretraining cutoff ~mid-2025** | Probe 7 (sharp cliff 2025-H1→H2) | **Very high** | Zero self-awareness |
| **Shared prefix + isolated question branches** | Probe 1 (additive tokens) + Hume (isolation, latency) | **Very high** | Tool-call pattern |
| **Output quantized to 1/100 grid** | Probe 2 (zero residual, 1,450 values) | **Very high** | Observation solid; interpretation unclear (API rounding vs model quantization) |
| **Causal attention confirmed** | Probe 5c: Jev and Qwen share anti-primacy bias direction. Jev 4× less biased (RLCD debiasing). | **Very high** | Resolves causal-vs-bidirectional question |
| **RLCD debiasing is measurable** | Probe 5c: 6.4pp→1.6pp position bias reduction vs raw Qwen | **High** | 75% reduction in ordering artifacts |
| **Recency bias in option processing** | Probe 4b (+3.1pp at last position, 44% flip rate) | **High** | Overturned Probe 4's "no bias" finding |
| **Choice/score share readout path; noul separate** | Probe 8 (diff 0.003 vs 0.06) | **High** | Caveat: different prompt templates could partially explain |
| **Tokenizer novel, closest to Qwen** | Hume (445 probes) | **Very high** | — |
| **Context window: ~32k/branch, ~65k total** | Hume (35 probes) | **Very high** | — |
| **IIA violation (options interact)** | Hume (50 probes) | **High** | — |
| **Likely sparse MoE (~10B active)** | Hume (latency inference) | **Moderate** | Our Probe 6/6b couldn't confirm (model too capable) |
| **Calibration: compression-toward-middle** | Sacco (800 items) | **High** | — |
| **Confidence = `(p_max - 1/K) / (1 - 1/K)`** | Hume | **Very high** | Arithmetic, not learned |
| **Open Qwen reproductions reach ~85% of Jev** | OpenJev, open-alternative-jev | **High** | — |

### Probe 5c: Qwen2.5-7B Direct Logit Comparison (36 records) — RESOLUTION

Ran the EXACT same 6 ordering-bias cases through Qwen2.5-7B-Instruct using direct logit scoring (Route B) on an H200 GPU. This is the definitive comparison: same model family (Qwen), same test cases, direct logit readout (no prompted probabilities).

**Position bias comparison (3-option, delta from 0.333):**

| Model | Position 0 | Position 1 | Position 2 | Max |delta| |
|---|---|---|---|---|
| **Jev** | -0.016 | +0.014 | +0.001 | **0.016** |
| GPT-4.1-mini | -0.036 | +0.040 | -0.004 | 0.040 |
| **Qwen2.5-7B** | **-0.064** | **+0.040** | **+0.024** | **0.064** |

**Key findings:**
1. **Qwen's raw bias is 4× larger than Jev's** (6.4pp vs 1.6pp first-position suppression). Same anti-primacy direction.
2. **Both share the same qualitative bias pattern** — consistent with a shared base model family.
3. **RLCD debiasing is real** — going from 6.4pp to 1.6pp is a 75% reduction in position bias. A well-calibrated model should not change its probability assignment based on irrelevant factors like option ordering.
4. **This resolves the causal-vs-bidirectional question** — Jev shows the same bias direction as known causal Qwen, just reduced. The Probe 3b/3c results (flat def_last) are explained by strong composition at query level, not by bidirectional attention.

## Resolved and Open Questions

### ✅ Causal vs Bidirectional — RESOLVED: Causal + RLCD Debiasing

Probes 3b and 3c initially suggested bidirectional attention (hint_last ≥ hint_first, flat def_last with scaling N). Probe 4b found recency bias favoring causal. These appeared contradictory.

**Probe 5c resolved the contradiction:** Jev and Qwen2.5-7B share the same anti-primacy bias direction, confirming the same causal attention mechanism. Jev's 4× smaller bias magnitude is the signature of **RLCD calibration training debiasing the base model's position artifacts**. The Probe 3b/3c results are explained by the model being strong enough to compose information at query level (which any capable causal model can do), not by bidirectional attention.

### ⬜ MoE vs Dense — Inconclusive

59/60 on hard counterintuitive questions, 120/120 on standard questions. The model is too capable at accessible difficulty levels to surface routing-pattern differences. Hume inferred sparse MoE (~10B active) from latency scaling — our probes neither confirm nor refute this.

### ⬜ Noul vs Choice/Score Readout — Partial

Choice and score track identically (diff 0.003) while noul diverges (diff 0.06). Likely separate readout mechanisms, but different prompt templates per type could partially explain the divergence.

## Known Limitations (from jaggedness doc)

These limitations provide architecture clues:

| Limitation | What It Suggests |
|-----------|-----------------|
| Literal reading (answers what you wrote, not what you meant) | Model is optimized for precise evaluation, not intent inference |
| Cannot count or do math | Not a reasoning model — pure classification/scoring |
| Date comparison unreliable | Reads dates as text, not ordered quantities |
| Struggles with indirection (double negatives, multi-hop) | Single-pass evaluation, no chain-of-thought |
| Context rot with large irrelevant state | Attention-based architecture where unrelated content acts as distractor |
| Adversarial content can steer answers | No adversarial robustness training (acknowledged, planned for improvement) |
| P(noul) and 1-P(not noul) don't sum to 1 | Each question is evaluated truly independently — no structural invariance across questions |
| Cannot generate text at all | Fundamental architectural constraint, not a training choice |

## The RLCD Training Gap

RLCD is the most critical and least disclosed component. What we know:

- **Objective:** calibration — P(stated confidence) ≈ P(actual correctness) across populations
- **Method:** reinforcement learning with a reward signal based on calibration metrics
- **Not disclosed:** loss function, reward model construction, training data, calibration measurement during training
- **Independent finding (Sacco):** Jev's probabilities show a fixed compression-toward-middle distortion — overstating low probabilities, understating high ones. Recommendation: "treat as a monotone score, not a probability, and fit your own calibration map."

Calibration-aware training is not entirely novel in the literature (e.g., "Rewarding Doubt" and related work on epistemic calibration), but the full product-level implementation at this scale appears to be new.

## Reproduction Roadmap

### What's Reproducible

1. **The interface pattern** — sending state + typed questions, getting probabilities back. Multiple open-source projects already do this.

2. **Zero-shot option scoring (Route B)** — reading log-probabilities from an existing LM's output distribution. No training needed. OpenJev achieves 0.845 agreement with Jev on a 102-row subset (vs Jev's 0.883).

3. **Trained scorer heads (Route A)** — freezing a pretrained backbone and training a lightweight scoring head on labeled data. jevlike and jevbetter demonstrate this.

4. **The API shape** — daseinlabs/open-jev implements the full `/v1/systemone` contract with all three primitives.

### What's Not Reproducible (Without More Information)

1. **RLCD training** — the loss function, reward model, and training procedure are undisclosed.

2. **The exact model architecture** — the readout head design and training are not public.

3. **The calibration quality** — Jev's calibration has been independently measured (Sacco, calibre) and found to have a compression-toward-middle distortion. Reproducing even this level of calibration requires the RLCD recipe.

4. **The training data** — undisclosed.

### Proposed Reproduction Strategy

Given access to HPC resources (ANL), we can explore:

#### Phase 1: Baseline Reproduction
- Set up open-jev (daseinlabs) with a 4B–8B model (e.g., Qwen3.5-4B, LLaMA 3.1 8B)
- Benchmark zero-shot logit scoring (Route B) against Jev on the same test cases
- Measure accuracy, calibration (ECE), and latency

#### Phase 2: Trained Head
- Use jevlike/jevbetter to train scorer heads on labeled decision data
- Freeze various backbones (0.5B to 8B) and compare head quality
- Experiment with rival-aware attention (jevbetter's approach)

#### Phase 3: ~~LLaDA Exploration~~ Cancelled
- ~~Load LLaDA-8B-Base and test its `get_log_likelihood()` for option scoring~~
- Probe 5c confirmed Jev uses a causal Qwen backbone, not diffusion. LLaDA is cancelled.

#### Phase 4: RLCD Approximation
- Design a calibration-aware training objective (ECE loss, Brier score loss, or RL-based)
- Train end-to-end: backbone + scoring heads optimized for calibrated probabilities
- This is the most speculative phase — RLCD's details are undisclosed

## References

- LLaDA paper: [arxiv:2502.09992](https://arxiv.org/abs/2502.09992)
- TypeSafe AI primer: https://docs.typesafe.ai/introduction/machine-learning-primer
- TypeSafe launch blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- explainx.ai analysis: https://explainx.ai/blog/how-does-jev-work-rlcd-system-one-model-explained-2026
- Archer Hume probing study: https://archerhume.com/posts/jevs-architecture-unmasked/
- SamuelSacco calibration audit: https://github.com/SamuelSacco/jev-exploration
- calibre routing study: https://github.com/FirasSX914/calibre
- jev-rerank-bench: https://github.com/anessbelbati/jev-rerank-bench
- OpenJev (TheoLeeCJ): https://github.com/TheoLeeCJ/openjev
- open-jev (daseinlabs): https://github.com/daseinlabs/open-jev
- jevlike (vinnylarouge): https://github.com/vinnylarouge/jevlike
- jevbetter (olanotolu): https://github.com/olanotolu/jevbetter
