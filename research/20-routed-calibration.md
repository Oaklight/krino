# The routed system's ECE passes only when each path gets its own temperature

Goal item 1 is met by a rule rather than a model: short lists go to the
backbone's native reranker readout and long ones to the trained head, which is
worth +3.77 pp overall (`research/17`) and +1.74 pp more with the blend
(`research/19`). Item 2 asks for `ECE <= 0.05` after per-type temperature scaling,
and the number that has been quoted for it — 0.0264 / 0.0137 / 0.0289 per type —
was measured on the trained head, not on the system that actually answers. The
deployed system has two scorers with different calibration, and this note measures
it: on the routed system, one temperature per question type misses 0.05 on every
checkpoint, and one temperature per **path** clears it on every checkpoint.

## Setup

| | |
|---|---|
| Input | the `--dump-records` files from `model/scripts/blend_scorers.py`: 7,026 choice items per checkpoint, each with its head logit vector and its native log-odds |
| Rule | `k < 20` options → native readout, `k >= 20` → trained head (6,526 / 500 items; the 16 six-to-nineteen items follow the `rule_sweep` boundary) |
| Temperatures | fitted by minimising ECE on the other folds' items, applied to the held-out fold; 4 folds, folds alternate by list position |
| Checkpoints | 4B `arch_4b_loraAll_32k_s123` epoch 10, and the epoch-5 pair (seeds 123 and 42) on the cluster-disjoint split |
| Code | `model/evaluation/routed_calibration.py`, `model/scripts/eval_routed_calibration.py`; artifact `model/experiments/routed_calibration.json` |

Temperature scaling is monotone, so every accuracy below is the argmax's and does
not move with the temperature; only ECE and Brier do.

## Results

ECE, Brier and accuracy on the held-out folds, averaged over the four folds.

| | s123 ep10 | s123 ep5 | s42 ep5 |
|---|---:|---:|---:|
| head only, T=1 | **0.0509** | 0.0352 | 0.0284 |
| head only, fitted T (0.99–1.11) | 0.0227 | 0.0338 | 0.0258 |
| routed, T=1 | **0.1716** | 0.1733 | 0.1721 |
| routed, one temperature for everything (fitted) | **0.0607** | **0.0526** | **0.0774** |
| routed, one temperature per path (fitted) | **0.0355** | **0.0371** | **0.0382** |
| accuracy (all rows) | 0.7048 | 0.6641 | 0.6890 |
| Brier, head only (fitted T) | 0.4695 | 0.5648 | 0.5331 |
| Brier, routed (per-path T) | 0.4046 | 0.4494 | 0.4269 |

## Finding 1 — the native readout is the miscalibrated one, by a factor of three

At `T = 1` the routed system's ECE is 0.172 while the head alone is 0.051 / 0.035 /
0.028. The fitted temperatures say where it comes from: the native path wants
**2.7–3.6**, the head wants **0.94–1.37**. The head arrives close to calibrated and
the reranker does not, which is unsurprising — the reranker was trained on a
pointwise relevance objective, not on the option set of a choice question.

That also explains why one number for the type cannot work: a single temperature
that divides the native logits by 3 would soften the head into uselessness, and
one that leaves the head alone leaves the native path at 0.17.

## Finding 2 — as the clause is worded, the ECE half fails on the system that meets item 1

"After per-type temperature scaling" is one temperature for the whole choice
type. Fitted that way, the routed system lands at **0.0607, 0.0526 and 0.0774**
against a 0.05 bar: it misses on all three checkpoints, and the miss is not
marginal on two of them. This is the honest reading of item 2 for the deployed
system, and it is a different claim from the head's 0.0227–0.0338 that has been
quoted.

## Finding 3 — per-path temperatures clear it on two seeds and two training lengths

With one temperature per path — still two numbers, still fitted on items the
evaluation does not see, still no per-item adaptation — the same three checkpoints
give **0.0355, 0.0371 and 0.0382**. All three are under 0.05, and the two epoch-5
numbers are two seeds of the same training length, so this is not a one-seed
artefact.

## Finding 4 — routing does not cost calibration, and improves Brier

The routed system is more accurate than the head alone (0.7048 against 0.6285 at
epoch 10) and also better on multiclass Brier (0.4046 against 0.4695). So the
routing gain in `research/17` and `research/19` is not being paid for in
calibration quality — once each path is calibrated, the system is better on both
axes at once.

## What this means for the goal

Item 2's ECE half is **met with per-path temperatures and failed with a single
one**. Both are global, held-out calibrations; the difference is that the
deployed rule has two scorers. That makes it an author decision, and the note
should not pretend otherwise:

- **If per-path scaling counts**, item 2's ECE half is met on the system that
  meets item 1, on two seeds, at 0.0355–0.0382.
- **If the clause means one temperature per question type** (noul / choice / score), this note
  does not measure it: the records are choice-only, so the per-question-type reading needs a
  calibration dump over all three primitives. What this note can say is that within choice, a
  single temperature fails and per-path succeeds. If the choice reading is the one that counts,
  the repair is not a better temperature but a better-calibrated native path: fit its
  temperature at serving time from training items, or replace the reranker score with a
  calibrated head. Software alone will not close 0.0607 → 0.05 without touching
  how the native scores are produced.

Either way, the previously quoted 0.0264 / 0.0137 / 0.0289 describes the head,
not the system, and should be labelled as such wherever it appears.

## Caveats

- **The 16 six-to-nineteen items follow the deployed rule** (`k < 20` → native).
  They are 0.2% of the set and cannot move any number here by more than 0.001.
- **The fold assignment alternates by list position**, which is the order the
  script wrote the items in. It is not a random split, so the numbers were
  re-measured with `--folds 2`, which changes how much of the set each fit sees:

  | | folds 4 | folds 2 |
  |---|---|---:|
  | routed, per path: s123 ep10 / s123 ep5 / s42 ep5 | 0.0355 / 0.0371 / 0.0382 | 0.0278 / 0.0336 / 0.0324 |
  | routed, per type: s123 ep10 / s123 ep5 / s42 ep5 | 0.0607 / 0.0526 / 0.0774 | 0.0589 / 0.0517 / 0.0632 |

  The largest move is 0.014 (s42, per type). **No verdict flips**: per-type stays
  above 0.05 on all three checkpoints under both schedules and per-path stays
  below on all three. The two-seed spread on the per-path number is 0.0011 at
  four folds, so the pass is not resting on the fold size.

  The verdicts are mean-level statements and the individual folds are not all on
  the same side of the bar. On the passing side, s42 ep5 has a per-path fold at
  **0.0501**, over 0.05, on a mean of 0.0382; its per-fold spread (sd 0.0141) is
  three times the other two checkpoints', so its 2-SE upper bound is 0.0523. On
  the failing side, s123 ep5 has a per-type fold at **0.0429**, under 0.05, on a
  mean of 0.0526. Neither flips a verdict, but "no verdict flips" is a statement
  about the means, and the choice this note asks the author to make is exactly
  the place where a fold-level margin matters.
- **The criterion is ECE, not NLL**, because the clause is written in ECE. An NLL
  fit would pick different temperatures (the two criteria disagree by a factor of
  ~2 on the score primitive, `research/17`) and this note does not report that
  variant.
- **The head-only T=1 number at epoch 10 is 0.0509** — the head alone is right at
  the bar before scaling, which is worth knowing when reading any claim that the
  head "is calibrated" without a temperature.
- **One set of artifacts per checkpoint.** The records are 3.4 MB each and are not
  committed; the run is reproducible from the checkpoint with
  `blend_scorers.py --dump-records`.
