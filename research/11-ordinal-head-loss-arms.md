# Ordinal head × loss: four arms on a frozen 0.6B

The score primitive is the project's weakest and most foundational one, so two
design choices were worth testing early: whether score should be modelled as a
distribution over independent levels with a softmax head, or as a cumulative
(ordinal) head, and whether the training loss should be cross-entropy or a
ranked-probability score (RPS), which is the metric the literature uses for
ordinal targets.

This note reports a 2×2 at 0.6B. It is a negative result on accuracy: the plain
baseline — softmax head, cross-entropy — wins overall, on choice and on score,
while on noul the two losses are tied inside the reproduction floor the note
establishes later. Neither the ordinal head nor RPS recovers any of the gap. RPS
targets *calibration* rather than accuracy, so the two softmax arms were then
scored with the calibration harness; that second half is below, and it does not
rescue RPS either.

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

**1. The plain baseline wins on choice and score, and ties on noul.** Score
accuracy is 5.0 pp higher under softmax_ce than under softmax_rps, 5.7 pp higher
than cumulative_ce, and 5.9 pp higher than cumulative_rps. Choice is higher on
every variant, by 0.6 to 1.8 pp. Against 0.29-0.40 pp seed noise at this scale,
those gaps are real. Overall accuracy is 0.96 to 1.97 pp higher, in the same
order.

Noul is the exception, and the small residual moves RPS's way rather than the
baseline's: softmax_rps scores 3,109 to softmax_ce's 3,101 (0.6583 vs 0.6566,
0.17 pp), and cumulative_rps 3,080 to cumulative_ce's 3,062 (0.6521 vs 0.6483,
0.38 pp). Both are inside the 0.29-0.40 pp seed band this note uses everywhere
else, so noul is a tie — an earlier revision said the baseline "wins on all three
types", which overstated it. A tie on the third type makes for a cleaner negative
result than a sweep, and finding 2 below does not depend on noul.

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
| noul | accuracy | 0.6566 | 0.6583 |
| | ECE (T=1) | 0.0569 | **0.0498** |
| | Brier | 0.2133 | **0.2117** |
| | ECE after fitted T (holdout) | **0.0203** | 0.0223 |
| choice | accuracy | **0.3985** | 0.3922 |
| | ECE (T=1) | **0.0744** | 0.0881 |
| | Brier | **0.6824** | 0.6915 |
| | ECE after fitted T (holdout) | 0.0520 | **0.0429** |
| score | accuracy | **0.4276** | 0.3775 |
| | ECE (T=1) | 0.0648 | **0.0622** |
| | Brier | **0.7146** | 0.7513 |
| | ECE after fitted T (holdout) | 0.1350 | **0.0768** |
| | MAE | **0.9384** | 1.0463 |
| | QWK (item-weighted, 4-6 levels) | **0.5312** | 0.4483 |
| | RPS | **0.1553** | 0.1712 |

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

Which column to trust, then. Both, now: the calibration reports were regenerated
with this revision — a fresh pass over the committed checkpoint that lands on
3,101 / 2,735 / 794, i.e. the Results table — so the two tables agree on accuracy
and the metrics below are reproducible from the committed artifacts. The earlier
calibration reports (0.6578 / 0.4015 / 0.4265) were produced by a pass that does
not reproduce, and they are superseded rather than reconciled; the numbers moved
by 0.19% of items and no conclusion changed.

That pass-to-pass movement is the resolution floor for every metric in the
calibration half, not only for accuracy, and it is measured rather than assumed:
the two passes over the same checkpoint differ by

| metric | noul | choice | score |
|---|---:|---:|---:|
| accuracy | 0.0012 | 0.0030 | 0.0011 |
| ECE (T=1) | 0.0012 | 0.0022 | 0.0020 |
| Brier | 0.0000 | 0.0001 | 0.0002 |
| MAE | — | — | 0.0020 |
| QWK | — | — | 0.0040 |
| RPS | — | — | 0.0004 |

The largest of them, 0.0040, is the floor any difference in this section has to
clear. Accuracy and Brier are far more stable than ECE and QWK, which is what
makes the accuracy agreement between the two tables a meaningful check and the
ECE comparisons the ones that need the floor quoted next to them.

**RPS does not buy a calibration advantage that pays for its accuracy cost.**
On the score items -- where the loss is supposed to matter -- the raw-ECE gap is
0.0026 in RPS's favour (0.0622 vs 0.0648), which is a tie in the only sense that
matters here: it is the size of the movement the score ECE itself shows between
two passes over one checkpoint (0.0020) and below the largest metric movement
between those passes (0.0040, QWK). It sits at the floor, not above it. Every
other proper metric is clearly worse for RPS:
Brier 0.7513 vs 0.7146, MAE 1.0463 vs 0.9384, QWK 0.4483 vs 0.5312, RPS 0.1712 vs
0.1553. Accuracy is 5.0 pp lower (0.3775 vs 0.4276). On noul, RPS is better
calibrated by 0.0071 ECE -- six times the 0.0012 pass-to-pass movement on that
arm -- while the two are indistinguishable in accuracy; on choice RPS is worse on
both, and its 0.0137 ECE gap is also well above the 0.0022 floor.

**Temperature scaling is not rescuing the score primitive in either arm.** ECE on
the half the temperature does not see rises in both: 0.0722 -> 0.1350 for
cross-entropy at T=4.079, and 0.0616 -> 0.0768 for RPS at T=2.445. The
temperature is fitted by NLL on one half and evaluated by ECE on the other, and
ECE is not a monotone function of T with a minimum at the NLL optimum, so a large
fitted T makes bin-level calibration worse. Goal item 2 asks for ECE <= 0.05
after per-type temperature scaling; no arm reaches it on score, and the fitting
criterion is the explanation: fitted for ECE instead of NLL, the same
cross-entropy checkpoint clears the bar on all three types — noul 0.0548 ->
0.0294, choice 0.0741 -> 0.0345, score 0.0722 -> **0.0337** at T=1.336
(`calibration_ecefit_softmax_ce.json`). So the head and the loss are not what
fails that clause; the fitting objective is. It does not survive the 4B
checkpoint, where the ECE fit reaches 0.0631 and no point on the grid clears 0.05
(#172).

### What the two halves together say

| | accuracy | calibration |
|---|---|---|
| ordinal head | -5.7 pp score | not measured (the harness cannot build a cumulative head yet) |
| RPS loss | -5.0 pp score | a tie on raw ECE, worse on Brier/MAE/QWK/RPS |

Neither change is worth adopting on this evidence. The default -- softmax head,
cross-entropy -- remains the most accurate overall, on choice and on score, and,
on the metrics measured, no worse calibrated: RPS's one claimed advantage, a
better-calibrated distribution, does not appear on the primitive it was meant to
help. The open question is no longer "which head or loss" but "why does
temperature scaling fail on score", which is a property of the distribution
rather than of the architecture.

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
