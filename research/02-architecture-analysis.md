# Jev Architecture Analysis

**Date:** 2026-09-18
**Updated:** 2026-09-19 (3 rounds of behavioral probing + 5 controlled probes + cross-model validation; ~5,800 API calls + H200 GPU runs)
**Status:** Operational pipeline identified through controlled experiments; exact backbone and training remain hypotheses

> **Correction (2026-09-18):** Earlier revisions described “causal Qwen” as confirmed, declared encoder/diffusion alternatives “ruled out,” and treated reduced ordering bias as confirmation of RLCD debiasing. Those conclusions exceeded what black-box HTTP measurements identify. The recorded tokenizer similarities, language behavior, ordering effects, timing, and Qwen comparison remain valid observations, but they do not uniquely determine the backbone, attention mask, readout path, or cause of debiasing. Throughout this document, “confirmed,” “resolved,” and “ruled out” architecture language should be read as historical hypotheses unless backed by TypeSafe disclosure. The new [controlled suite](../probing/controlled/README.md) preregisters discriminating tests and explicitly tracks non-identifiable alternatives.

## What TypeSafe Has Confirmed

1. **RLCD (Reinforcement Learning for Calibrated Decisions)** — a new post-training method. Optimizes for probability calibration: if the model says 70% across many predictions, roughly 70% should be correct. Distinct from RLHF (human preference) and RLVR (verifiable correctness).

2. **Parallel output claim** — TypeSafe states that Jev produces all typed answers in a single query rather than exposing token-by-token generation. The public API does not establish whether the hidden implementation uses direct numerical readout, compact internal decoding, or another mechanism.

3. **New model architecture** — TypeSafe's launch post says "a new model architecture" but does not disclose what it is.

4. **Parallel sampler** — described as "hardware-aware," designed for "maximum efficiency."

5. **Starts from a pretrained language model** — the AI primer positions RLCD as a *post-training* method applied to pretrained language models, alongside RLHF and RLVR.

6. **No architecture paper published** as of September 2026.

## Our Probing Results

We ran 13 systematic probes across 3 rounds against the Jev API (jev-1.13.0), totaling ~1,800 API calls. Each round included a rigor self-audit, and flawed probes were re-designed and re-run. Raw data in `probing/results/`. Full methodology and scripts in `probing/scripts/`.

### Methodological review

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

**Operational interpretation:** Token accounting is additive over the tested requests. Shared state computation with per-question work is one explanation, but billing counters may reflect request/response serialization rather than neural compute.

### Probe 2: Noul Precision Quantization (1,450 values)

**All probability values — noul, choice, and score — are quantized to an exact 1/100 grid (0.01 steps) with zero residual.**

| Type | Values | Unique | Min nonzero | Range |
|------|--------|--------|-------------|-------|
| Noul | 500 | 88 | 0.01 | 0.01–0.99 |
| Choice prob | 450 | 72 | 0.01 | 0.00–1.00 |
| Score prob | 500 | 75 | 0.01 | 0.00–1.00 |

Noul values never reach exactly 0.0 or 1.0 (capped at 0.01–0.99). Choice and score probabilities can hit exact 0.0 and 1.0.

**Operational interpretation:** The API exposes at most 101 probability values on this grid. The measurements do not distinguish model-intrinsic quantization, rounding, renormalization, or other API-layer postprocessing.

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

**Historical interpretation (withdrawn):** This was initially described as evidence for autoregressive rather than diffusion processing. Character-level corruption changes tokenization under either architecture, so that inference was invalid. Superseded by Probes 3b and 3c.

### Probe 3b: Corrected Diffusion Test (125 records)

Word-level masking, bidirectional context placement (hint before vs after ambiguous content), and token-level noise.

**Bidirectional context (critical test):** hint_last ≥ hint_first in ALL 5 non-trivial cases (+0.04 mean). Token-level noise: only 4.7% degradation at 50% noise (vs 47% for word removal).

**However**, this probe is also inconclusive — in causal attention with KV caching, the question suffix attends to ALL state positions. A strong model can compose information from any position at question-level attention.

### Probe 3c: Controlled Bidirectional Test (162 records)

Scaled code-definition test: N made-up codes (1→10) where definitions are placed AFTER usage. The causal prediction: def_last degrades as N grows. The bidirectional prediction: flat.

| N codes | def_first noul | def_last noul | delta |
|---|---|---|---|
| 1 | 0.940 | 0.920 | -0.020 |
| 3 | 0.970 | 0.967 | -0.003 |
| 5 | 0.973 | 0.977 | +0.003 |
| 10 | 0.973 | 0.973 | +0.000 |

def_last stays flat. But again, code-lookup is a retrieval/matching task solvable at query level for ANY architecture. **Suggestive but not conclusive on its own.**

**Status:** This result does not identify the attention mask. Probe 4b supplies a separate positional measurement, not a resolution.

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

### Probe 4b: Controlled Ordering Bias (190 records) — KEY FINDING

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

**Operational interpretation:** Option position materially affects outputs on ambiguous items. Causal processing is one possible cause, but positional embeddings, listwise heads, prompt templates, training data, constrained decoding, and server orchestration can produce similar patterns.

### Probe 6b: Hard Domain Variance (60 records)

Counterintuitive myths near capability boundary. 59/60 correct (98%). Only miss: spinach/iron myth. No meaningful domain variance (σ=0.037). MoE question remains **inconclusive**.

### Probe 7: Temporal Cutoff (30 records)

On the small, hand-selected temporal set, answers changed sharply between the 2025-H1 and 2025-H2 groups, and Jev assigned low noul values (0.16–0.28) to TypeSafe/Jev statements. The sample is too small and confounded by item difficulty to identify a pretraining cutoff or establish "self-awareness."

### Probe 8: Readout Head Cross-Type (20 records)

100% binary agreement across types. Choice and score were nearly identical (diff 0.003), while noul differed more (diff 0.06), had a higher floor on degenerate inputs, and was 12% faster. This establishes type-dependent behavior; separate readout paths are one of several explanations, alongside prompt templates and postprocessing.

### Probe 9 + 9b: Multilingual Parity (199 records) — KEY FINDING

Easy facts (Probe 9): 100% in 10/12 languages, EN=ZH parity, token cost parity (284≈285).

Hard idioms/grammar (Probe 9b): 100% across all 8 languages. **ZH confidence (0.952) exceeded EN (0.862)** on these language-specific items. This demonstrates strong Chinese behavior in this fixture set; it does not identify a model family because item difficulty, training data, distillation, and calibration can produce the same pattern.

## Controlled Probe Results (3,757 requests)

Five preregistered probes with matched controls, interval-censored analysis, and explicit non-identifiability tracking. Raw data in `probing/results/controlled/`. Scripts in `probing/scripts/controlled/`. Preregistration in `probing/controlled/README.md`.

### Probe 1: Output Serialization (360 records)

Opaque question IDs and option keys across five Unicode families (ASCII, combining marks, emoji, punctuation, general Unicode), crossed with returned-entry count (2/8/32), with byte-matched input padding (all requests exactly 35,820 bytes).

- **Zero identity failures** — all opaque keys returned byte-for-byte
- **Latency vs visible response bytes: R²=0.035** — output size explains <4% of latency variance
- **Within-entry-count slopes: R²<0.03** — after controlling for entry count, key length has no effect on latency
- **TTFB ≈ total time** to 4+ significant figures — response arrives as a single block
- **String family spread: 18ms** — no tokenization-dependent processing overhead

**Operational conclusion:** visible JSON is assembled by server code, not generated token by token.

### Probe 2: Option Interaction / IIA (1,300 records, 1,200 comparisons)

Pairwise baseline versus irrelevant, dominated, exact-duplicate, and paraphrase additions at balanced insertion positions (0, 1, 2). IIA tested with interval-censored `log(P(A)/P(B))`.

- **Zero definite IIA violations** for irrelevant and dominated options — log-odds intervals are bit-identical to baseline
- **Duplicate mass-splitting is mechanical:** P(A) + P(A_dup) = original P(A); B's share unchanged
- **Paraphrases cause 9× less shift** than exact duplicates — tracks with different surface-form utilities, not semantic-overlap detection

**Operational conclusion:** options are scored independently (pointwise utilities), then softmax-normalized. No cross-option attention or listwise interaction.

### Probe 3: Likelihood Sensitivity (300 records)

Semantically equivalent option descriptions varying in length, grammaticality, and lexical commonness under Latin-square balancing with opaque label classes (ASCII, digits, Unicode).

| Variant | Mean probability |
|---|---|
| short, grammatical, common | 0.537 |
| short, fragment, rare | 0.223 |
| long, grammatical, common | 0.172 |
| long, awkward, rare | 0.067 |

- **Description length R²=0.431** — surface form explains 43% of probability variance
- **Grammaticality correlation: +0.528**
- Short/grammatical/common descriptions receive **~8× more mass** than long/awkward/rare

**Operational conclusion:** the scoring mechanism leaks token-level LM log-probabilities into final option scores. This disfavors pooled semantic embeddings or learned heads that score meaning independently of surface form. Note: these are preregistered stimulus annotations; architectural comparison with a measured reference model requires `--reference-probabilities`.

### Probe 4: Execution Topology (805 records)

Blocked factorial design: state length S × question count Q × option count K × description length L, plus cue-placement and reference-remapping conditions.

**Latency model coefficients (dominant terms):**

| Feature | Coefficient |
|---|---|
| Q (question count) | 2.57 × 10⁻⁴ |
| Q×K (questions × options) | 2.52 × 10⁻⁴ |
| L (description length) | 9.52 × 10⁻⁵ |
| S (state length) | 2.56 × 10⁻⁶ |
| **S×Q** | **−1.39 × 10⁻⁷ (≈ 0)** |

- **S×Q ≈ 0:** state is encoded once and reused. Adding questions does not re-incur state cost.
- **Q×K dominates:** cost scales with total question-option pairs.
- **Cue remapping collapses accuracy** (100% → ~10%): the model binds identifiers during the encoding pass and cannot re-resolve them.

**Operational conclusion:** shared-prefix encoding with per-question-option scoring work. One-pass binding — representations are frozen after the forward pass.

### Probe 5: Generation Signatures (992 records, 9,176 distributions)

Many matched output entries varying output position, preceding-field complexity, number of fields, and smoothly interpolated evidence.

- **99.84% exact cent sums** (15/9,176 off by exactly 1 cent, none by 2+)
- **Zero decimal heaping** — round values (0.10, 0.25, 0.50) not overrepresented vs neighbors
- **Output position spread: 0.4pp** across 20 positions (flat)
- **Preceding-field complexity effect: 0.007pp** (zero)
- **Clean S-shaped evidence curve** with sharp thresholds (hard zero below 0.38, saturation above 0.83)

**Operational conclusion:** probabilities are computed values rounded to 0.01, not generated digit tokens. No output-position or preceding-field effects.

### Operational Pipeline (Identified)

```
State text
  → single-pass encoding (computed once, shared across questions)
  → per-question suffix processing
    → per-option token-level log-probability scoring (pointwise, independent)
    → softmax normalization → probability vector
  → server-side JSON assembly + 0.01 rounding
```

This pipeline is identified through five independent probes. What remains non-identifiable from HTTP behavior:
- Whether the internal readout is raw logits, hidden-state projection, or compact constrained generation
- The exact backbone architecture, model family, or weight lineage
- Whether RLCD, supervised calibration, or another method produces the observed debiasing
- The attention mask (causal vs bidirectional) during state processing

## Architecture Hypotheses — Updated

### Hypothesis A: Joint/Listwise Cross-Encoder — Disfavored

- Controlled Probe 2: perfect IIA for irrelevant/dominated options; pointwise independent scoring, not listwise interaction
- Duplicate mass-splitting is purely mechanical softmax, not semantic-overlap-aware

### Hypothesis B: Causal LM + Token-Level Log-Probability Readout — Most Consistent

- Controlled Probe 3: surface form explains 43% of probability variance — LM log-prob leakage
- Controlled Probe 4: S×Q ≈ 0 (shared prefix); Q×K dominates (per-option scoring)
- Controlled Probe 2: pointwise independent scoring + softmax normalization
- Controlled Probes 1, 5: server-side JSON assembly, not visible token generation
- Behavioral Probes 9/9b: multilingual profile consistent with Qwen-family references

This is the most consistent hypothesis across all probes but remains a behavioral characterization, not an identification of specific weights or architecture.

### Hypothesis C: Bidirectional Encoder + Learned Typed Heads — Disfavored

- Controlled Probe 3: 43% variance from surface form strongly disfavors heads that score pooled semantic embeddings independently of wording
- Controlled Probe 4: cue-binding behavior is consistent with one-pass frozen representations

### Hypothesis D: Autoregressive Constrained Decoding of Visible Output — Disfavored

- Controlled Probe 1: latency does not scale with visible output size (R²=0.035)
- Controlled Probe 5: zero heaping, zero position effects, 99.84% exact cent sums
- Compact internal constrained generation remains non-identifiable

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

Because each answer space is bounded, direct numerical readout is possible and full visible-JSON decoding is unnecessary. The API remains compatible with direct heads, compact internal constrained decoding, and server-side mapping. One candidate head design is:
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

**One candidate implementation:**

```
                    ┌─── Question 1 suffix ──→ Decision mechanism → P(options)
                    │
State representation ─┼─── Question 2 suffix ──→ Decision mechanism → P(yes/no)
                    │
                    └─── Question 3 suffix ──→ Decision mechanism → P(levels)
```

- **Backbone hypothesis:** A multilingual pretrained Transformer; causal Qwen is a useful reference, not an identification
- **Inference hypothesis:** Shared state computation with isolated per-question work; KV caching is one compatible mechanism
- **Output hypotheses:** Direct typed heads, listwise scoring, compact constrained decoding, and server-side mapping remain compatible
- **Training disclosure:** TypeSafe names RLCD but has not published its loss, reward, data, or where calibration is applied
- **Observed API behavior:** Probabilities are exposed on a 0.01 grid; the responsible layer is unknown

## Consolidated Evidence Summary

### Controlled probes (this project, 3,757 requests)

| Finding | Source | Confidence |
|---------|--------|------------|
| **Visible JSON is server-assembled, not model-generated** | Controlled Probe 1 (R²=0.035 latency vs output size; TTFB≈total; zero identity failures) | **High** |
| **Pointwise independent scoring + softmax normalization** | Controlled Probe 2 (zero IIA violations for irrelevant/dominated; mechanical duplicate splitting) | **High** |
| **LM log-probability leakage into scores** | Controlled Probe 3 (surface form R²=0.431; ~8× mass ratio for equivalent options) | **High** |
| **Shared-prefix encoding, per-question-option scoring** | Controlled Probe 4 (S×Q≈0; Q×K dominates latency) | **High** |
| **Probabilities are computed values, not generated digits** | Controlled Probe 5 (99.84% exact cent sums; zero heaping; 0.4pp position spread) | **High** |

### Behavioral probes (this project, ~2,000 requests)

| Finding | Source | Confidence |
|---------|--------|------------|
| **Behavior consistent with multilingual Qwen references** | Hume tokenizer + Probes 9/9b + Probe 5c | **Moderate** |
| **Temporal-item transition** | Probe 7 | **High (measurement)** |
| **Output quantized to 1/100 grid** | Probe 2 (1,450 values) | **Very high** |
| **Recency bias in option processing** | Probe 4b (+3.1pp at last position) | **High** |
| **Type-dependent behavior (noul vs choice/score)** | Probe 8 | **High (measurement)** |

### External evidence

| Finding | Source | Confidence |
|---------|--------|------------|
| **Tokenizer novel, closest to Qwen (348/415)** | Hume (445 probes) | **Very high** |
| **Context window: ~32k/branch, ~65k total** | Hume (35 probes) | **Very high** |
| **Confidence = `(p_max - 1/K) / (1 - 1/K)`** | Hume | **Very high** |
| **IIA violation on Hume's test set** | Hume (50 probes) | **High** |
| **Calibration: compression-toward-middle** | Sacco (800 items) | **High** |
| **Open Qwen reproductions reach ~85% of Jev** | OpenJev, open-alternative-jev | **High** |
| **Sparse-MoE hypothesis** | Hume latency inference | **Low–moderate** |

Note: Hume's IIA violation finding and our Controlled Probe 2's IIA preservation are not contradictory — they used different test designs. Hume tested with semantically related added options; our Probe 2 tested with semantically irrelevant/dominated additions. The combined picture is: IIA holds for unrelated options (pointwise scoring) but may shift for semantically similar options (surface-form utility differences affecting softmax).

### Probe 5c: Qwen2.5-7B Direct Logit Comparison (36 records) — REFERENCE COMPARISON

Ran the exact same 6 ordering-bias cases through Qwen2.5-7B-Instruct using direct logit scoring (Route B) on an H200 GPU. This is a controlled reference comparison using the same test cases and direct logit readout (no prompted probabilities); it does not establish that Jev uses the same model family.

**Position bias comparison (3-option, delta from 0.333):**

| Model | Position 0 | Position 1 | Position 2 | Max |delta| |
|---|---|---|---|---|
| **Jev** | -0.016 | +0.014 | +0.001 | **0.016** |
| GPT-4.1-mini | -0.036 | +0.040 | -0.004 | 0.040 |
| **Qwen2.5-7B** | **-0.064** | **+0.040** | **+0.024** | **0.064** |

**Key findings:**
1. **Qwen's raw bias is 4× larger than Jev's** (6.4pp vs 1.6pp first-position suppression). Same anti-primacy direction.
2. **Both showed first-position suppression in this fixture set.** Shared lineage is one explanation, but common positional priors, prompts, or training data are equally compatible.
3. **Jev showed 75% less position bias than this Qwen baseline** (6.4pp versus 1.6pp). RLCD debiasing is one possible explanation, but prompt construction, model scale, other post-training, typed heads, or API postprocessing could also cause the difference.
4. **The comparison does not resolve causal versus bidirectional processing.** Matching a known causal model's bias direction is suggestive, but a bidirectional model or typed scorer can learn the same positional pattern. The Probe 3b/3c results likewise remain compatible with multiple architectures.

## Resolved and Open Questions

### Causal vs Bidirectional — Open; Causal + RLCD Debiasing Is a Working Hypothesis

Probes 3b and 3c initially suggested bidirectional attention (hint_last ≥ hint_first, flat def_last with scaling N). Probe 4b found recency bias compatible with causal processing. These observations pull in different directions but are not uniquely identifying.

**Probe 5c adds suggestive evidence, not a resolution:** Jev and Qwen2.5-7B share the same anti-primacy bias direction, while Jev's magnitude is 4× smaller. A shared causal mechanism plus RLCD debiasing could explain this pattern, but so could other backbones, prompt templates, typed heads, post-training methods, or postprocessing. Probe 3b/3c remains compatible with both strong causal composition and bidirectional processing.

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
| Public API does not generate text | Interface constraint; hidden internal generation remains non-identifiable |

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

3. **The cause of observed calibration behavior** — independent studies report domain-dependent distortion, but the public evidence does not show which loss, calibration layer, or RL procedure caused it.

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

#### Phase 3: LLaDA Exploration — Deprioritized
- Load LLaDA-8B-Base and test its `get_log_likelihood()` for option scoring only if diffusion remains a useful comparison baseline.
- Probe 5c made a causal-Qwen working hypothesis more attractive, but did not identify Jev's backbone or rule out diffusion.

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
