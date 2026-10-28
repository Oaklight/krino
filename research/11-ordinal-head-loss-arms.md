# Ordinal head × loss: four arms on a frozen 0.6B

The score primitive is the project's weakest and most foundational one, so two
design choices were worth testing early: whether score should be modelled as a
distribution over independent levels with a softmax head, or as a cumulative
(ordinal) head, and whether the training loss should be cross-entropy or a
ranked-probability score (RPS), which is the metric the literature uses for
ordinal targets.

This note reports a 2×2 at 0.6B. It is a negative result on accuracy: the plain
baseline — softmax head, cross-entropy — wins on every type, and neither the
ordinal head nor RPS recovers any of the gap. The caveat that matters is at the
end: RPS targets *calibration*, and this experiment did not measure it.

## Setup

| | |
|---|---|
| Backbone | `Qwen/Qwen3-Reranker-0.6B`, frozen (no LoRA) |
| Data | `model/configs/r2_reranker0.6b.yaml` — 182,262 train / 13,444 eval items, `epoch_size` 15,000, `type_ratios` 1:1:1 |
| Training | 5 epochs, seed 42, `accumulation_steps 8`, bf16, flash attention, `last` readout, rank 64 |
| Arms | `--score-head {softmax,cumulative}` × `--score-loss {ce,rps}` |
| Wall clock | 76.5-78.0 minutes per arm on one A100-40GB |

The four arms are matched controls of each other: identical data, seed, schedule
and architecture, differing only in the score head and the score loss. The
choice and noul heads are untouched, which is why their accuracies move at all —
they share the backbone with the score head, so a change to the score objective
changes what the shared trunk learns.

Command per arm (substitute the two flags):

```bash
python model/scripts/train_multitask.py \
    --config model/configs/r2_reranker0.6b.yaml \
    --epochs 5 --eval-every 5 --seed 42 \
    --score-head cumulative --score-loss rps \
    --checkpoint-dir model/experiments/ordinal_cumulative_rps \
    --output model/experiments/ordinal_cumulative_rps.json
```

## Results

Final-epoch eval accuracy on the shared 13,444-item set:

| arm | overall | noul | choice | score | score loss |
|---|---:|---:|---:|---:|---:|
| **softmax_ce** (control) | **0.4932** | **0.6566** | **0.3985** | **0.4276** | 1.8606 |
| softmax_rps | 0.4836 | 0.6583 | 0.3922 | 0.3775 | 1.6522 |
| cumulative_ce | 0.4735 | 0.6483 | 0.3810 | 0.3710 | 1.7463 |
| cumulative_rps | 0.4794 | 0.6521 | 0.3904 | 0.3689 | 1.6266 |

**Read the accuracy columns only.** The loss column mixes two different scales —
cross-entropy and RPS are not comparable, and an arm cannot be ranked by its own
training objective. RPS reports a lower number than cross-entropy on every arm
by construction; that is not evidence of anything.

## Findings

**1. The plain baseline wins on all three types.** Score accuracy is 5.0 pp
higher under softmax_ce than under softmax_rps, 5.7 pp higher than
cumulative_ce, and 5.9 pp higher than cumulative_rps. Against 0.29-0.40 pp seed
noise at this scale, all three gaps are real. Overall accuracy is 0.96 to 1.97 pp
higher, in the same order.

**2. RPS only hurts under the softmax head.** Under cumulative, RPS and
cross-entropy differ by 0.2 pp on score (0.3689 vs 0.3710) — inside noise. Under
softmax they differ by 5.0 pp. So the two changes are not independent: RPS is
neutral under the head designed for ordinality and harmful under the one that
treats levels as unordered, which is consistent with RPS's gradient pushing a
softmax head toward a geometry it cannot represent. A 2×2 was needed to see
this; testing either change alone would have produced a misleading summary.

**3. The ordinal head costs more than the loss does.** Between the two heads
at cross-entropy, cumulative loses 5.7 pp of score accuracy. Between the two
losses at the cumulative head, 0.2 pp. Whatever the cumulative head buys, it is
not accuracy on this data.

## The calibration half

Accuracy was never what RPS claims. The two softmax arms -- same architecture,
same data, same seed, only the loss differs -- were therefore scored with the
calibration harness (PR #167) on the same 13,444 items:

| type | metric | softmax_ce | softmax_rps |
|---|---|---:|---:|
| noul | accuracy | 0.6578 | 0.6576 |
| | ECE (T=1) | 0.0557 | **0.0503** |
| | Brier | 0.2133 | **0.2116** |
| | ECE after fitted T | **0.0172** | 0.0221 |
| choice | accuracy | **0.4015** | 0.3936 |
| | ECE (T=1) | **0.0722** | 0.0864 |
| | Brier | **0.6823** | 0.6915 |
| | ECE after fitted T | 0.0519 | **0.0425** |
| score | accuracy | **0.4265** | 0.3802 |
| | ECE (T=1) | 0.0628 | **0.0595** |
| | Brier | **0.7148** | 0.7515 |
| | ECE after fitted T | 0.1267 | **0.0844** |
| | MAE | **0.9404** | 1.0481 |
| | QWK (item-weighted, 4-6 levels) | **0.5272** | 0.4592 |
| | RPS | **0.1557** | 0.1715 |

**RPS does not buy a calibration advantage that pays for its accuracy cost.**
On the score items -- where the loss is supposed to matter -- raw ECE is a tie
(0.0628 vs 0.0595) and every other proper metric is worse: Brier 0.7515 vs
0.7148, MAE 1.048 vs 0.940, QWK 0.459 vs 0.527, RPS 0.1715 vs 0.1557. Accuracy
is 4.6 pp lower. On noul the two are indistinguishable in accuracy and RPS is
marginally better calibrated; on choice RPS is worse on both.

**Temperature scaling is not rescuing the score primitive in either arm.** ECE
after the fitted temperature rises above its value at T=1 in both:
0.0628 -> 0.1267 for cross-entropy at T=3.796, and 0.0595 -> 0.0844 for RPS at
T=2.445. The temperature is fitted by NLL on one half and evaluated by ECE on
the other, and ECE is not a monotone function of T with a minimum at the NLL
optimum, so a large fitted T can make bin-level calibration worse. Goal item 2
asks for ECE <= 0.05 after per-type temperature scaling; no arm reaches it on
score, and the fitting criterion is a candidate explanation worth testing
before the head or the loss is blamed again.

### What the two halves together say

| | accuracy | calibration |
|---|---|---|
| ordinal head | -5.7 pp score | not measured (the harness cannot build a cumulative head yet) |
| RPS loss | -5.0 pp score | a tie on raw ECE, worse on Brier/MAE/QWK/RPS |

Neither change is worth adopting on this evidence. The default -- softmax head,
cross-entropy -- remains both the most accurate and, on the metrics measured, no
worse calibrated. The open question is no longer "which head or loss" but
"why does temperature scaling fail on score", which is a property of the
distribution rather than of the architecture.

## Caveats

* **One seed per arm.** These are matched controls, which is the evidence bar's
  alternative to a second seed, but a single seed each means the 0.2 pp
  cumulative_ce/cumulative_rps difference is not resolvable and the 5 pp gaps
  rest on one draw per arm.
* **Frozen backbone, 5 epochs.** Every arm is a lower bound: full LoRA is worth
  +5.35 pp over frozen at 0.6B, and five epochs truncates the schedule. The
  ordering of the arms is what this measures, not the absolute level.
* **Pre-#138 split.** These runs predate the cluster-aware synthetic split, so
  the eval set may contain near-duplicates of training text — the same caveat as
  `research/10-architecture-sweep-r2.md`, and it applies to all four arms
  equally so it does not affect the comparison.
* **Accuracy is not what RPS is for.** RPS is a proper scoring rule for ordinal
  distributions: its claim is about calibration and about penalising
  distance-wrong answers, not about argmax accuracy. A head that is 5 pp less
  accurate but substantially better calibrated is a different trade, and this
  experiment did not measure ECE, Brier, MAE or QWK on the score items. The
  calibration harness (PR #167) measures exactly those, and running the four
  checkpoints through it is the experiment that would settle RPS.

## What this means for the goal

The goal's score clause is calibration and QWK (ECE, Brier, QWK >= 0.85), not
accuracy alone. This note removes one hypothesis from the accuracy path — an
ordinal head or an RPS loss does not buy score accuracy at 0.6B — and leaves the
calibration question open, with the tool to answer it now in review. Until the
four checkpoints are scored for ECE and QWK, the default should stay softmax +
cross-entropy, which is also what every existing checkpoint uses.
