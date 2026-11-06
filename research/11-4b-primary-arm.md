# 4B primary arm: preserve the base decision, add only what transfers

Status: B1 measured and killed (2026-11-03). No arm yet meets the primary gate.

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

## Corpus audit (commit e02ab0c)

`model/experiments/corpus_audit.json`, built by
`model/scripts/audit_corpus.py` on the run config, 195,706 train+eval items:

| Property | Corpus | JevBench public |
|---|---|---|
| noul / choice / score | 47.8 / 40.3 / 12.0 % | 32.0 / 60.2 / 7.8 % |
| teacher_probs coverage | 10,916 (5.6%), synthetic only | n/a |
| context p90 | 2,084 chars | hard tier ~3.7K tokens |
| >8K chars | 4,901 (2.5%), 4,790 of them contractnli | |
| choice K | 2, 3, 4, 5, 80 (banking77) | includes K=6 |
| score levels | 3, 4, 5, 6 (stsb 6,249) | 3-5 |
| hard label vs teacher argmax | 21.0% disagree | n/a |

Consequences:

1. The corpus is 94.4% hard-label CE. An arm trained on it is generic SFT,
   not Jev distillation. Do not report it as distillation.
2. Teacher-labeling 40-50K benchmark items would take coverage to ~25-30%
   and is the cheapest way to make a distillation claim credible.
3. Type mix must be reweighted toward choice for the 4B arm; 0.32 / 0.60 /
   0.08 is feasible from the current pools at 15K x 5.
4. Long context is a contractnli monoculture. Raising `synthetic.max_train`
   helps only if the long synthetic domains are upweighted.

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

## Arm B1 (running)

`model/scripts/train_residual.py`, `model/experiments/residual/B1`:

- 5,000 cached items, stratified 80/20 by (type, option count).
- rank 32, no MLP, lr 1e-3 for the head, 1e-2 for the per-type alphas.
- KL weight 1.0, CE weight 0.2, teacher probabilities as the CE target when
  present, hard labels otherwise.
- Early stop on held-out chance-corrected accuracy, patience 1.
- Each epoch logs `kl_mean`, `ce_mean`, `kl_ce_ratio`, and every alpha. An
  alpha that stays near zero is a result: the base is already at its ceiling
  for that type.
- The gate reports pre (raw 4B JSON) and post (residual JSON) overall, per
  type, per tier, and per K bucket.

## Arm B1 result (2026-11-03)

`model/experiments/residual/B1/jevbench.json`, 231 public items, same scorer as B0:

| Arm | Raw acc | cc | ECE | easy | hard | original |
|---|---:|---:|---:|---:|---:|---:|
| B0 raw 4B | 0.6407 | 0.4459 | 0.065 | 1.000 | 0.459 | 0.681 |
| B1 residual | 0.6320 | 0.4395 | 0.088 | 1.000 | 0.423 | 0.708 |

Per type, B1 vs B0: choice 0.597 vs 0.619, noul 0.662 vs 0.635, score 0.778
vs 0.833. The alphas at the saved epoch are noul -0.036, choice 0.006,
score 0.001: the gate barely moved, and the movement does not transfer.

Held-out (n=999) chance-corrected went 0.447 -> 0.510 -> 0.459 -> 0.448 over
four epochs. The 0.510 peak is one draw on a 999-item split; JevBench, the
pre-registered instrument, lands 0.64 points below the base.

Verdict: killed by its own rule (cc < 0.4459). The residual keeps the base
distribution intact (ECE 0.088, QWK 0.893) but adds nothing on hard, which is
where the 5.4-point gap lives. B2 is conditional on B1 showing transfer and
therefore does not start. B3's precondition ("B1 fails structurally") is not
met: B1 reproduced the base at step 0 and stayed within 0.9 points of it.

Two facts for the next attempt: 5,000 cached items is one thirtieth of the
designed 20-40K subset, and 94.4% of the corpus is hard-label CE. A rerun at
the same scale with the same recipe would repeat this result; the lever is
data (teacher-labeled items, long-context hard items), not another epoch.

## Arm B2 result (2026-11-03): the marker readout clears the gate

`model/experiments/residual/B2/jevbench.json`. Same cache, split, rank, KL and
CE weights as B1; only the readout differs (marker, chunk 1).

| Arm | Raw acc | cc | ECE | easy | hard | original | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 raw 4B | 0.6407 | 0.4459 | 0.065 | 1.000 | 0.459 | 0.681 | 0.619 | 0.635 | 0.833 |
| B1 pooled residual | 0.6320 | 0.4395 | 0.088 | 1.000 | 0.423 | 0.708 | 0.597 | 0.662 | 0.778 |
| B2 marker residual | **0.6494** | **0.4603** | 0.075 | 1.000 | 0.459 | 0.708 | 0.626 | 0.649 | 0.833 |

B2 beats the base by 1.44 cc points and B1 by 2.08, ties hard exactly, and
keeps B1's original gain. The alphas engage this time: noul -0.041, choice
-0.048, score -0.127, against B1's +0.006 for choice. Held-out cc reached
0.531 at epoch 4 without early stopping.

The pre-registered gate was `cc >= 0.4459 and hard >= 0.459`. Both hold, so
B2 is promoted to the confirm stage.

Read the size honestly: 1.44 points on 231 items is inside the per-item noise
band (about 3.3 points at one standard error). Promotion means it cleared a
bar fixed before the run, not that the improvement is established. The confirm
stage is the instrument for that, and it is registered as `confirm_frozen` in
`model/experiments/residual/manifest.jsonl` before it starts.

## Confirm stage (2026-11-03, running)

Two seeds (43, 44), 20 epochs, patience 1, same cache and recipe as B2.
Gate: confirmed if both seeds reach cc >= 0.4459; rejected if either falls
below 0.4395; otherwise inconclusive.

**Split note (2026-11-06).** At the time of these runs, `train_residual.py`
passed `--seed` to `split_items`, so seeds 43 and 44 moved the validation split
as well as init and order. Split 42 and split 43 select the same counts (3,999;
choice 1,617 / noul 1,908 / score 474) but share only 804 items; 3,195 differ
each way. At the canonical scale (243,366 rows, val 0.2) splits 42/43 share 9,863 of 48,672 validation rows and 42/44 share 9,660; about 80 percent moves when --seed changes. The s43/s44 runs are therefore replications plus a different
validation set, and their spread includes split variance on top of init+order
variance. The JevBench gates are unaffected (same 231 items). The trainer now
has a separate `--split-seed` (default 42); these historical numbers stand as
recorded with this caveat.

§7's confirm stage also lists a LoRA arm. `train_residual.py` has no LoRA path
yet, and the marker readout disables gradient checkpointing by design, so a
LoRA confirmation needs a memory check first. It is not started.

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
