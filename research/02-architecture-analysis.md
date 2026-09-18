# Jev Architecture Analysis

**Date:** 2026-09-18
**Updated:** 2026-09-18 (probe results incorporated)
**Status:** Partially validated through empirical probing

## What TypeSafe Has Confirmed

1. **RLCD (Reinforcement Learning for Calibrated Decisions)** — a new post-training method. Optimizes for probability calibration: if the model says 70% across many predictions, roughly 70% should be correct. Distinct from RLHF (human preference) and RLVR (verifiable correctness).

2. **Single forward pass, no autoregressive decoding** — because the output space is fixed and small (up to 255 choices, a scalar score, or a boolean), the model scores all possible answers in one pass. No token-by-token decode loop.

3. **New model architecture** — TypeSafe's launch post says "a new model architecture" but does not disclose what it is.

4. **Parallel sampler** — described as "hardware-aware," designed for "maximum efficiency."

5. **Starts from a pretrained language model** — the AI primer positions RLCD as a *post-training* method applied to pretrained language models, alongside RLHF and RLVR.

6. **No architecture paper published** as of September 2026.

## Our Probing Results

We ran four systematic probes against the Jev API (jev-1.13.0), totaling 744 API calls. Raw data in `probing/results/`. Full methodology and scripts in `probing/scripts/`.

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

**Verdict: autoregressive backbone, not diffusion.** The steep masking degradation and strong word-order sensitivity are classic autoregressive signatures. The LLaDA fork on TypeSafe's GitHub was likely an exploration path that was not used for the production model.

### Probe 4: Ordering Bias on Ambiguous Cases (72 records)

**No systematic position bias:**

| Position | Mean probability |
|----------|-----------------|
| 0 (first) | 0.3300 |
| 1 (middle) | 0.3392 |
| 2 (last) | 0.3308 |

All within 1pp of the 0.333 baseline.

**Choice flips:** 2 out of 12 ambiguous cases (16.7%) had their top choice change depending on option order. Both flipped cases had top-two options within ~10pp of each other. Clear-cut cases showed zero sensitivity.

**Probability swings:** Mean 0.065, max 0.220, median 0.040.

**Architecture signal:** The lack of systematic position bias disfavors simple left-to-right option scoring. Instead, it's consistent with **listwise processing** where all options are seen simultaneously before the decision is made. This aligns with Hume's IIA violation finding — options interact within the decision computation.

## Architecture Hypotheses — Updated

### ~~Hypothesis A: Encoder-Only Transformer (BERT-style)~~ — Unlikely

Our probing evidence argues against this:
- Strong word-order sensitivity (Probe 3) is inconsistent with bidirectional encoder models, which are relatively order-agnostic
- The token accounting pattern (Probe 1) matches autoregressive tool-call framing, not encoder classification

### ~~Hypothesis B: Text Diffusion Model (LLaDA-style)~~ — Ruled Out

Our Probe 3 results directly contradict this:
- Steep masking degradation (0.97→0.45 at 50% masking) is the opposite of what a diffusion model trained with random masking would show
- Strong word-order sensitivity is inconsistent with diffusion models that process all positions simultaneously
- The LLaDA fork was likely an exploration that did not make it into the production architecture

### Hypothesis C: Purpose-Built Architecture — Partially Supported

The "new architecture" claim may refer to the output mechanism rather than the backbone.

### Hypothesis D: Causal LM + Constrained Parallel Tool Calls — Best Fit ✅

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

## Community Evidence Summary

| Finding | Source | Confidence |
|---------|--------|------------|
| Autoregressive backbone (not diffusion, not encoder) | Our Probe 3 | **High** |
| Shared prefix + isolated question branches | Our Probe 1 + Hume | **Very high** |
| Listwise option processing (no position bias, IIA violation) | Our Probe 4 + Hume | **High** |
| Output quantized to 1/100 grid | Our Probe 2 | **Very high** |
| Tokenizer novel, closest to Qwen (348/415) | Hume | **Very high** |
| Context window: ~32k/branch, ~65k total | Hume | **Very high** |
| Likely sparse MoE (~10B active) | Hume (latency) | **Moderate** |
| Calibration has fixed compression-toward-middle distortion | Sacco (800 items) | **High** |
| Confidence = `(p_max - 1/K) / (1 - 1/K)` | Hume | **Very high** |
| Open Qwen reproductions reach ~85% of Jev's accuracy | OpenJev, open-alternative-jev | **High** |

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

#### Phase 3: ~~LLaDA Exploration~~ Deprioritized
- ~~Load LLaDA-8B-Base and test its `get_log_likelihood()` for option scoring~~
- Our Probe 3 results indicate Jev does NOT use a diffusion backbone, so LLaDA exploration is deprioritized
- May still be worth a quick comparison to confirm the diffusion hypothesis is wrong at scale

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
