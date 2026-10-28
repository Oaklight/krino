# Ordinal head × loss: four arms on a frozen 0.6B

The score primitive is the project's weakest and most foundational one, so two
design choices were worth testing early: whether score should be modelled as a
distribution over independent levels with a softmax head, or as a cumulative
(ordinal) head, and whether the training loss should be cross-entropy or a
ranked-probability score (RPS), which is the metric the literature uses for
ordinal targets.

This note reports a 2×2 at 0.6B. It is a negative result on accuracy: the plain
baseline — softmax head, cross-entropy — wins on every type, and neither the
ordinal head nor RPS recovers any of the gap. RPS targets *calibration* rather
than accuracy, so the two softmax arms were then scored with the calibration
harness; that second half is below, and it does not rescue RPS either.

## Setup

| | |
|---|---|
| Backbone | `Qwen/Qwen3-Reranker-0.6B`, frozen (no LoRA) |
| Data | `model/configs/r2_reranker0.6b.yaml` — 182,262 train / 13,444 eval items, `epoch_size` 15,000, `type_ratios` 1:1:1 |
| Training | 5 epochs, seed 42, `accumulation_steps 8`, bf16, flash attention, `last` readout, rank 64 |
| Arms | `--score-head {softmax,cumulative}` × `--score-loss {ce,rps}` |
| Wall clock | 76.5-78.0 minutes for the five-epoch loop, per arm — the sum of the per-epoch `elapsed_s` in each committed JSON — and ~92 minutes end to end, from the runner's per-arm start/stop timestamps |

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

The four `--output` files are committed as `model/experiments/ordinal_<arm>.json`,
one per arm; each holds the per-epoch history and the final per-type and
per-source breakdown the accuracy table below is read from.

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
calibration harness (merged as #167) on the same 13,444 items. The two reports
are committed as `model/experiments/calibration_ordinal_softmax_ce.json` and
`model/experiments/calibration_ordinal_softmax_rps.json`; the ECE-fitted variant
of the first is the separately committed `calibration_ecefit_softmax_ce.json`
that #167 added. The two `cumulative` arms are not scored: `eval_calibration.py`
has no `--score-head` flag, so it would build a softmax head and report on a
network that does not exist (#145).

| type | metric | softmax_ce | softmax_rps |
|---|---|---:|---:|
| noul | accuracy | 0.6578 | 0.6576 |
| | ECE (T=1) | 0.0557 | **0.0503** |
| | Brier | 0.2133 | **0.2116** |
| | ECE after fitted T (holdout) | **0.0172** | 0.0221 |
| choice | accuracy | **0.4015** | 0.3936 |
| | ECE (T=1) | **0.0722** | 0.0864 |
| | Brier | **0.6823** | 0.6915 |
| | ECE after fitted T (holdout) | 0.0519 | **0.0425** |
| score | accuracy | **0.4265** | 0.3802 |
| | ECE (T=1) | 0.0628 | **0.0595** |
| | Brier | **0.7148** | 0.7515 |
| | ECE after fitted T (holdout) | 0.1267 | **0.0844** |
| | MAE | **0.9404** | 1.0481 |
| | QWK (item-weighted, 4-6 levels) | **0.5272** | 0.4592 |
| | RPS | **0.1557** | 0.1715 |

### Why the two tables' accuracies differ

They are two inference passes over the same items, not two definitions of
"correct", and the difference is small enough to state as a floor rather than a
caveat. Measured, not assumed:

* **The item lists are identical.** The loaders in `train_multitask.py` and
  `eval_calibration.py` were run side by side on the same config and seed and
  produced byte-identical ordered lists (13,444 items, compared by id, source,
  type and state text). The per-type `n` agree in both reports as well.
* **The rules are identical.** Applying `_eval_item` (training) and
  `_predict_original` (calibration) to one loaded checkpoint in one process gives
  **0 disagreements over all 13,444 items**, and both reproduce this note's
  Results table exactly: 3,101 / 2,735 / 794 correct.
* **The calibration reports disagree with that by 25 items net**: 3,107 noul
  (+6), 2,756 choice (+21), 792 score (−2), i.e. 0.19% of the set.

So what moves is the inference pass — near-tied logits resolved differently in
bf16 across two processes — not the scoring code and not the data. Tie-breaking
is ruled out as the main cause directly: noul has `tie_count` 0 in the report and
still moves six items, and choice's six ties cannot account for its 21.

The calibration column is the authoritative one for anything probability-shaped,
because those reports carry the tie, duplicate-option and per-bucket metadata;
the accuracy rows are quoted from them for that reason. The residual is ±0.2 pp,
inside every seed band this note uses (the 2.00 pp LoRA spread at 0.6B, the
0.29-0.40 pp frozen spread), so no conclusion depends on which column is read.

**RPS does not buy a calibration advantage that pays for its accuracy cost.**
On the score items -- where the loss is supposed to matter -- raw ECE is a tie
(0.0628 vs 0.0595) and every other proper metric is worse: Brier 0.7515 vs
0.7148, MAE 1.048 vs 0.940, QWK 0.459 vs 0.527, RPS 0.1715 vs 0.1557. Accuracy
is 4.6 pp lower. On noul the two are indistinguishable in accuracy and RPS is
marginally better calibrated; on choice RPS is worse on both.

**Temperature scaling is not rescuing the score primitive in either arm.** ECE on
the half the temperature does not see rises in both: 0.0765 -> 0.1267 for
cross-entropy at T=3.796, and 0.0523 -> 0.0844 for RPS at T=2.445. The
temperature is fitted by NLL on one half and evaluated by ECE on the other, and
ECE is not a monotone function of T with a minimum at the NLL optimum, so a large
fitted T makes bin-level calibration worse. Goal item 2 asks for ECE <= 0.05
after per-type temperature scaling; no arm reaches it on score, and the fitting
criterion is the explanation: fitted for ECE instead of NLL, the same
cross-entropy checkpoint reaches 0.0325 on score (`calibration_ecefit_softmax_ce.json`,
reported in #167). So the head and the loss are not what fails that clause; the
fitting objective is.

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
  distance-wrong answers, not about argmax accuracy. That is why the two softmax
  arms were scored for ECE, Brier, MAE, QWK and RPS as well; the result is in
  "The calibration half" above. The two `cumulative` arms remain unscored, so
  the ordinal head's calibration is still an open question rather than a
  measured loss.

## What this means for the goal

The goal's score clause is calibration and QWK (ECE, Brier, QWK >= 0.85), not
accuracy alone. This note removes one hypothesis from the accuracy path — an
ordinal head or an RPS loss does not buy score accuracy at 0.6B — and, with the
calibration half, one more from the calibration path: RPS does not buy better
calibration either. The default stays softmax + cross-entropy, which is also what
every existing checkpoint uses. What remains open is the temperature-fitting
criterion (#172), not the head, the loss, or the backbone.
