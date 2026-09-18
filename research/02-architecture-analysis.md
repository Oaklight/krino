# Jev Architecture Analysis

**Date:** 2026-09-18
**Status:** Informed speculation based on public evidence

## What TypeSafe Has Confirmed

1. **RLCD (Reinforcement Learning for Calibrated Decisions)** — a new post-training method. Optimizes for probability calibration: if the model says 70% across many predictions, roughly 70% should be correct. Distinct from RLHF (human preference) and RLVR (verifiable correctness).

2. **Single forward pass, no autoregressive decoding** — because the output space is fixed and small (up to 255 choices, a scalar score, or a boolean), the model scores all possible answers in one pass. No token-by-token decode loop.

3. **New model architecture** — TypeSafe's launch post says "a new model architecture" but does not disclose what it is.

4. **Parallel sampler** — described as "hardware-aware," designed for "maximum efficiency."

5. **Starts from a pretrained language model** — the AI primer positions RLCD as a *post-training* method applied to pretrained language models, alongside RLHF and RLVR.

6. **No architecture paper published** as of September 2026.

## Architecture Hypotheses

Three candidate architectures have been discussed in the community:

### Hypothesis A: Encoder-Only Transformer + Classification Heads (BERT-style)

**Fit:** Encoder-only models compute a representation of the full input in one pass and attach task-specific heads for fixed-size output distributions — exactly the shape Choice/Score/Noul need.

**Evidence for:**
- Mechanically the most natural fit for the described behavior
- Single forward pass is the default for encoder models
- Classification heads over a fixed output set is standard practice

**Evidence against:**
- TypeSafe says "new architecture" — BERT-style is not new
- BERT-scale models (~340M params) may not have the semantic depth needed for "frontier intelligence" claims
- Would need significant scaling beyond traditional BERT sizes

### Hypothesis B: Text Diffusion Model (LLaDA-style)

**Fit:** Diffusion models compute outputs via iterative denoising rather than left-to-right autoregression. Could potentially produce calibrated probabilities over fixed output sets.

**Evidence for:**
- TypeSafe's GitHub includes a [fork of LLaDA](https://github.com/typesafe-ai/LLaDA) (Large Language Diffusion Models, arxiv:2502.09992), predating launch by over a year
- LLaDA natively has a `get_log_likelihood()` function that evaluates conditional probabilities without generating text
- TypeSafe's launch blog includes the phrase "continuously diffuse" (possible Easter egg)
- LLaDA uses a standard Transformer architecture with a different probabilistic modeling approach (masked diffusion vs autoregression)
- LLaDA's FAQ explicitly says it is NOT improved BERT — it's a generative model with a proper log-likelihood bound

**Evidence against:**
- Text diffusion models typically need multiple denoising steps, making 70ms latency challenging
- LLaDA's own FAQ says it's *slower* than autoregressive baselines at generation
- However, for classification/scoring (not generation), a single diffusion step might suffice

### Hypothesis C: Purpose-Built Architecture

**Fit:** Given how narrow Jev's three primitives are, TypeSafe may have built something that doesn't map cleanly onto either existing category.

**Evidence for:**
- "New model architecture" is explicitly stated
- The output constraints are so specific that a general-purpose architecture may be wasteful
- TypeSafe also forked [vllm](https://github.com/typesafe-ai/vllm), suggesting custom inference engineering

**Evidence against:**
- Building a new architecture from scratch is extremely expensive
- Most "new architectures" in practice are modifications of existing ones

## Our Best Guess

**Most likely: a large pretrained LLM backbone (possibly diffusion-based, given the LLaDA fork), with custom classification/scoring heads, trained end-to-end with RLCD.**

The reasoning:

1. **The backbone is a pretrained language model** — the AI primer explicitly says RLCD is a post-training method starting from pretrained LMs. The model needs frontier-level language understanding, which requires billions of parameters and web-scale pretraining.

2. **LLaDA as a potential base** — LLaDA is architecturally attractive because:
   - It uses a standard Transformer but with masked diffusion instead of autoregression
   - Its `get_log_likelihood()` function naturally evaluates conditional probabilities
   - It can compute representations in parallel (no sequential dependency)
   - It's an 8B parameter model — big enough for strong language understanding
   - TypeSafe forked it and has had it since June 2025

3. **Custom heads for the three primitives** — on top of the backbone, lightweight heads that:
   - For Choice: compute softmax over user-defined option embeddings
   - For Score: compute probabilities over user-defined levels
   - For Noul: compute a single yes/no probability

4. **RLCD training** — the key innovation. Uses RL to optimize the entire pipeline (backbone + heads) for calibrated probabilities. The reward signal measures the match between stated confidence and actual accuracy across populations of predictions.

5. **Engineering optimizations** — the "parallel sampler" and "hardware-aware" claims suggest deep inference optimization, likely including:
   - KV-cache sharing across questions (questions share the same state)
   - Batched evaluation of multiple questions in one pass
   - Quantization / pruning for latency
   - Custom CUDA kernels (hence the vllm fork)

## The RLCD Training Gap

RLCD is the most critical and least disclosed component. What we know:

- **Objective:** calibration — P(stated confidence) ≈ P(actual correctness) across populations
- **Method:** reinforcement learning with a reward signal based on calibration metrics
- **Not disclosed:** loss function, reward model construction, training data, calibration measurement during training

Calibration-aware training is not entirely novel in the literature (e.g., "Rewarding Doubt" and related work on epistemic calibration), but the full product-level implementation at this scale appears to be new.

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

## Reproduction Roadmap

### What's Reproducible

1. **The interface pattern** — sending state + typed questions, getting probabilities back. Multiple open-source projects already do this.

2. **Zero-shot option scoring (Route B)** — reading log-probabilities from an existing LM's output distribution. No training needed. OpenJev achieves 0.845 agreement with Jev on a 102-row subset (vs Jev's 0.883).

3. **Trained scorer heads (Route A)** — freezing a pretrained backbone and training a lightweight scoring head on labeled data. jevlike and jevbetter demonstrate this.

4. **The API shape** — daseinlabs/open-jev implements the full `/v1/systemone` contract with all three primitives.

### What's Not Reproducible (Without More Information)

1. **RLCD training** — the loss function, reward model, and training procedure are undisclosed.

2. **The exact model architecture** — "new model architecture" is a black box.

3. **The calibration quality** — Jev's claimed calibration has not been independently verified against public benchmarks.

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

#### Phase 3: LLaDA Exploration
- Load LLaDA-8B-Base and test its `get_log_likelihood()` for option scoring
- Compare LLaDA vs autoregressive backbones for the scoring task
- If LLaDA shows promise, explore RLCD-style calibration training on top

#### Phase 4: RLCD Approximation
- Design a calibration-aware training objective (ECE loss, Brier score loss, or RL-based)
- Train end-to-end: backbone + scoring heads optimized for calibrated probabilities
- This is the most speculative phase — RLCD's details are undisclosed

## References

- LLaDA paper: [arxiv:2502.09992](https://arxiv.org/abs/2502.09992)
- TypeSafe AI primer: https://docs.typesafe.ai/introduction/machine-learning-primer
- TypeSafe launch blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- explainx.ai analysis: https://explainx.ai/blog/how-does-jev-work-rlcd-system-one-model-explained-2026
- OpenJev (TheoLeeCJ): https://github.com/TheoLeeCJ/openjev
- open-jev (daseinlabs): https://github.com/daseinlabs/open-jev
- jevlike (vinnylarouge): https://github.com/vinnylarouge/jevlike
- jevbetter (olanotolu): https://github.com/olanotolu/jevbetter
