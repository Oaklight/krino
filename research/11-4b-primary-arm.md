# 4B primary arm: preserve the base decision, add only what transfers

Status: design. No trained arm yet.

## Why this note exists

The v1.6.1 goal makes the 4B decoder the primary arm and demotes 150-600M
encoders to an ablation with a kill line (Intelligence < 40). The encoder
screen is measuring that kill line now. This note covers only the primary arm.

## What the evidence says

Local JevBench public, 231 items, `model/scripts/eval_gate.py`:

| Arm | Raw acc | Chance-corrected | easy | hard | original |
|---|---:|---:|---:|---:|---:|
| raw Qwen3-Reranker-4B | 0.6407 | **0.4459** | 1.000 | 0.459 | 0.681 |

By type on the same arm: choice 0.6187, noul 0.6351, score 0.8333.
Chance is the per-item uniform baseline (1/K; 0.5 for noul), mean 0.3176.

The same raw arm on the internal eval split (400 items drawn from the run
config) scores 0.6325 raw and 0.4137 chance-corrected, close to its JevBench
numbers. The base model is therefore equally strong on both distributions.
The trained 4B head reaches 0.595 on the internal split (research/07), which
is 3.8 points *below* doing nothing there, and 20+ points below on JevBench.
Head training is not a transfer problem alone; it is a loss even
in-distribution.

Trained heads on the 26-task corpus score 0.325-0.429 raw and 0.045-0.192
chance-corrected on the same suite. The board's strict encoders sit at
Intelligence median 3.76 and max 7.43. The target is 50.

Two facts drive the design:

1. The gap to acceptance is +5.4 chance-corrected points over the raw base.
2. Easy is already 1.000 and original is 0.681. All headroom is in hard
   (0.459), which is the long-context policy tier (~3.7K state tokens).

Head training so far loses 25-40 points. Any arm that cannot reproduce the
raw base at step 0 is measuring a different model, not an improvement.

## Design rules

1. Do not train on JevBench public. The within-corpus schema holdout is the
   seen-vs-held-out instrument; JevBench is scored once per finished arm.
2. Initialize to the base. The head enters as a residual on the base yes/no
   logits with a zero-initialized gate: `logits = base + alpha * head`,
   `alpha = 0`. Step 0 reproduces 0.4459 exactly; the gate is the only thing
   that can move away from it.
3. Preserve the base distribution. Add `KL(p_base || p_theta)` on the
   training corpus with a fixed weight, so the label loss is the only force
   that changes accuracy.
4. Early stop on a within-corpus held-out split. Report JevBench only for a
   finished arm.
5. Do not regress easy or original. A gain on hard that costs easy is not a
   gain; report all three tiers every time.

## Arms

| Arm | Backbone | Readout | Training | Kill rule |
|---|---|---|---|---|
| B0 | Qwen3-Reranker-4B | raw LM head | none | reference |
| B1 | Qwen3-Reranker-4B | pooled + zero-gated residual on base logits, KL-preserved | 1 epoch, small lr | cc < 0.4459 on JevBench |
| B2 | Qwen3-Reranker-4B | marker, chunk 1, zero-gated residual | 1 epoch | must beat B1 on hard |
| B3 | Qwen3.5-4B | pooled + residual | 1 epoch | alternate base only if B1 fails structurally |

B2 is only worth its cost if B1 shows the gate learns anything at all.

## Open problems

- The base logits per option cost one forward per (context, option) pair.
  Precompute for a fixed 20-40K training subset and cache by item id.
- The internal corpus caps at 2K tokens, while hard items run to ~3.7K. A
  length arm may be required before hard can move.
- The corpus label space and JevBench's differ. KL-preservation protects the
  base ranking but not the label mapping; the label loss must be small
  enough not to rewrite the option order.

## Concrete residual formulation

The base arm already produces a distribution `p_base` over options: P(yes)
per option, normalised across options. Its logit vector is therefore
`b_i = log(p_base_i + eps)`, which softmaxes back to `p_base` exactly.

The trained arm is `logits_i = b_i + alpha * h_i`, with `h_i` the head's
option logits and `alpha` a scalar initialised to 0. At step 0 the arm is
bit-identical to the reference. The loss is

    CE(logits, label) + beta * KL(p_base || softmax(logits))

with `alpha` and the head trained jointly from the first step, and `beta`
fixed, not tuned per run.

`p_base` is precomputed per training item and cached by item id, because one
4B forward per (context, option) pair dominates the cost.
