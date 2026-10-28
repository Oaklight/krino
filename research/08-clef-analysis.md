# Clef Analysis: Cloudflare's Open-Source Decision Model

**Date:** 2026-10-12
**Status:** Reference analysis — external prior art comparison
**Sources:**
- [Introducing Clef (blog)](https://blog.cloudflare.com/clef-decision-models/)
- [Cloudflare/clef on Hugging Face](https://huggingface.co/Cloudflare/clef)

## Why This Matters

Cloudflare released Clef and Clef-flash on 2026-10-01: two open-source decision
models that are API-compatible with Jev and SystemOne. This is the first
directly comparable open artifact from a major lab, and it validates several
choices we have already made (frozen backbone, attention scoring head,
non-autoregressive inference) while diverging in specific areas we can learn
from.

## Model Facts

| Property | Clef | Clef-flash |
|----------|------|------------|
| Backbone | Qwen/Qwen3.8-27B (+ vision encoder) | Qwen/Qwen3.5-9B |
| Modality | text, JSON, images, video | text, JSON, images, video |
| Context | 64k | 64k |
| Head | joint schema head (small transformer) | same |
| Trainable | rank-256 LoRA + routing head, jointly | rank-256 LoRA + head |
| License | Apache-2.0 | Apache-2.0 |
| Median latency | 209 ms | 38.8 ms |

Clef is post-trained from Qwen3.8-27B. The backbone is frozen; LoRA adapters
and the joint schema head are optimized together.

## Training Recipe

Three components, in order of novelty:

1. **Label-smoothed cross-entropy** over valid schema outputs. This is close to
   our teacher-distillation KL loss, but with explicit label smoothing.
2. **Brier loss** as an auxiliary calibration term. Optimizes probability
   calibration directly — not just ranking.
3. **RLCD (Reinforcement Learning for Calibrated Decisions)** — a secondary
   RL objective that:
   - grants partial credit to adjacent ordinal choices (score questions);
   - rewards fully-precise record outputs (all questions in a record correct);
   - applies a reference penalty to prevent distribution shift.

They train on internal synthetic data that permutes field order, prompts, and
schema structure.

## Joint Schema Head vs Our AttentionHead

### Shared foundations

Both follow the same recipe:

- Frozen large backbone, small trainable scoring head.
- Non-autoregressive inference: one forward pass emits one logit per option.
- Option/field representations act as queries over context hidden states
  (cross-attention).
- Per-question softmax to obtain probabilities.
- Low-rank projections to keep the head small.
- Jev / SystemOne API compatible.

### Differences

| Dimension | Clef joint schema head | Our AttentionHead |
|-----------|------------------------|-------------------|
| Scoring scope | All questions in a record scored jointly in one forward pass | One question per item, scored independently |
| Cross-field attention | Questions attend to each other | None — questions are isolated |
| Routing | Two-stage: field→state, field↔field, then back to payload | Single stage: option→context cross-attention |
| Lexical prior | Explicit, preserves semantic intent across options | None — raw option embeddings |
| Attention rank | 256 | 64 (config) |
| Trainable params | rank-256 LoRA + head, jointly optimized | Head only; backbone fully frozen |
| Modality | Text + image + video | Text only |

### The two structural gaps

**1. No multi-question records.** Our `TypedQuestion` holds exactly one
question. Even though a synthetic family carries several questions,
`family_to_typed_questions()` splits them into independent items. Clef's head
is schema-level: it ingests the whole record and scores every question
together.

**2. No cross-field attention.** Clef lets field representations cross-attend
before scoring, so "is it urgent" is informed by "which team". We have no
analog. Our `rival_aware` mode does inter-option attention, but only *within*
one question's options and it is off by default.

The practical consequence: for workflows where questions in a record are
semantically dependent (urgency + team + severity), our head cannot express
that dependency. This is the most concrete architectural gap to consider
closing.

## Benchmark Position

Clef leads the Jev Decision Index overall but the split is informative:

| Benchmark family | Winner | Note |
|------------------|--------|------|
| Classification (BANKING77, CLINC150, API-Bank) | Clef | Large margins (+14 pp on BANKING77) |
| Reasoning (BBH, MMLU-Pro, GPQA Diamond) | Jev | Jev retains a clear lead |
| Latency | Clef-flash | 38.8 ms median vs Jev 524 ms |

Clef's gains concentrate in classification and routing; Jev's in reasoning.
This suggests Clef's recipe (classification-oriented post-training with
calibration losses) trades away some reasoning ability.

## Implications For Krino

1. **Backbone choice validated.** Clef uses Qwen3.8-27B; our Qwen3-Reranker
   line is the same family at smaller scale. Scaling up is a defensible next
   step.
2. **LoRA + head joint training.** Clef trains rank-256 LoRA alongside the
   head. We currently freeze the backbone entirely (#93 covers LoRA as
   pending). Clef's results argue for prioritizing LoRA.
3. **Calibration losses.** Brier loss and RLCD map directly onto our
   calibration work (#23). Clef gives a concrete, published recipe to
   replicate.
4. **Joint schema scoring.** If we ever need multi-question workflows, the
   head needs cross-field attention. Not required for single-question
   benchmarks, but a real capability gap.

## Unsloth (tooling note)

Unsloth is an efficiency library for LoRA/QLoRA fine-tuning: hand-written
Triton kernels, roughly 2x faster LoRA training with 50-70% less VRAM than
stock PEFT. It supports the Qwen family. For the LoRA experiments in #93 it
is a practical way to fit larger backbones on a single A100. Their documented
starting point is `alpha = r = 16`; Clef's r=256 is far more aggressive,
which makes sense given they adapt a 27B backbone for a specialized task.

## Open Questions

- How exactly does the lexical prior enter the attention computation? The
  HF repo ships `joint_schema_model.py`; reading it would settle this.
- Does Clef's cross-field attention scale with schema size, or is it capped?
- Is RLCD's reference penalty computed against the pre-RL checkpoint or a
  frozen teacher?
