# The score primitive's calibration ceiling

Goal item 2 asks for ECE ≤ 0.05 after per-type temperature scaling. On the 4B
checkpoint a temperature fitted by ECE on half the score items and scored on the
other half moves ECE from 0.1027 to 0.0811 and stops (#172), and an earlier note
showed the fit criterion is not the reason (#171): fitted for ECE rather than
NLL, the 0.6B checkpoint clears 0.05 on all three types, but the 4B one does not.

This note asks two questions. Why does a scale factor fail on score, and how far
does calibration get if the method is allowed to be *anything* that uses the
model's confidence? The second is a ceiling: a monotone map of top-1 confidence
is at least as expressive as temperature scaling for ECE, because ECE is computed
on top-1 confidence alone, so nothing that ECE can measure is out of its reach.
If the ceiling is under the bar, the problem is upstream of calibration.

Answering it does not need a new inference pass. `eval_calibration.py` writes
`--predictions`, and each row carries `top_prob` and `correct` — everything ECE
uses. The dump is the one behind #176's clean figures
(`cal_clean3_4b_predictions.jsonl`, 13,444 items, 4B full-LoRA weighted
checkpoint on the rebuilt stsb corpus).

## Setup

| | |
|---|---|
| Checkpoint | `arch_4b_loraAll_weighted`, the arm behind #171's headline |
| Data | the rebuilt corpus after #177, 13,444 eval items; 928 score holdout items |
| Split | the harness's own: `random.Random("<seed>:<type>:temperature")` over `range(n)`, `fit_frac` 0.5 |
| Script | `model/scripts/score_calibration_ceiling.py` |
| Derived | `model/experiments/score_calibration_map_4b.json` |

The split reproduces to the digit — recomputing ECE at T=1 over the dump's
holdout half gives 0.1027 for score, which is the `ece_before` in
`cal_clean3_4b.json` — so every column below is measured on the same items the
reported temperature was fitted and scored on. The script warns and refuses to
quote the temperature column when it cannot reproduce that number.

Three recalibrations, all fitted on the mirror half and scored on the holdout:

| type | holdout | ECE at T=1 | after ECE-fitted temperature | after monotone map (CV) | 95% interval on the map |
|---|---:|---:|---:|---:|---|
| noul | 2,344 | 0.0400 | **0.0274** | 0.0309 | [0.0205, 0.0465] |
| choice | 3,450 | 0.0352 | **0.0126** | 0.0272 | [0.0199, 0.0449] |
| score | 928 | 0.1027 | 0.0811 | **0.0537** | [0.0282, 0.0865] |

The map is isotonic regression on (confidence, outcome) pairs with two
regularisations — blocks under 20 items absorbed, every block pulled toward the
base rate — and its two parameters are chosen by a two-fold cross-validation
*inside* the fitting half. Choosing them on the holdout instead is not worth
much on this grid: the best of the thirty configurations when scored against the
holdout gives 0.0535 at `alpha=5, min_n=40`, against 0.0537 at the CV choice. The
figure is therefore not a product of the selection protocol. The grid still
matters in the other direction — the fixed default of `alpha=10, min_n=20` gives
0.0666 — which is why the chosen configuration is reported rather than a default.

## Finding 1 — the miscalibration is one band, not a scale factor

Score items on the holdout half, at T=1:

| bin | mean confidence | accuracy | gap | n |
|---|---:|---:|---:|---:|
| 2 | 0.280 | 0.400 | +0.120 | 10 |
| 3 | 0.359 | 0.361 | +0.002 | 36 |
| 4 | 0.462 | 0.526 | +0.064 | 95 |
| 5 | 0.547 | 0.572 | +0.025 | 215 |
| 6 | 0.652 | **0.453** | **−0.199** | 203 |
| 7 | 0.748 | **0.604** | **−0.144** | 182 |
| 8 | 0.846 | **0.733** | **−0.113** | 135 |
| 9 | 0.938 | 0.923 | −0.015 | 52 |

Below 0.55 the model is roughly calibrated or slightly underconfident. Between
0.60 and 0.85 — 520 of the 928 items, 56% of the set — it is 11 to 20 points too
confident, and the excess is largest in the middle of that band rather than at
its top. At 0.94 it is right again.

A temperature is a single multiplicative change to the logits, which moves the
whole curve; this curve is flat, then drops 20 points, then recovers. Fixing
0.65 → 0.45 with a scale factor would put the 0.94 items near 0.70. That is why
the fitted T of 1.77 gains only 0.022: it is the best compromise across a curve
that has no single scale factor in it.

The shape is the same in all three score sources, which is why it is one map and
not three: over all 500 items of each source, sst5, stsb and yelp each have their
worst bin at mean confidence 0.65, with gaps of −0.226, −0.184 and −0.142 against
overall ECEs of 0.1300, 0.1260 and 0.0841. Nothing about the failure is specific
to one corpus.

## Finding 2 — a monotone map beats a temperature on score, and misses the bar

On the same 928 holdout items the map lands at 0.0537 against the temperature's
0.0811. That is a real improvement and it is still not the goal's 0.05 — the
point estimate is 0.0037 above the bar and the interval [0.0282, 0.0865] spans
it. So the honest reading is not "calibration can reach 0.05" but "the cheapest
method available closes a third of the gap, and the bar sits inside the noise of
a 928-item measurement."

The fitted map makes the overconfidence legible. Its breakpoints, in confidence:

| stated | 0.38 | 0.48 | 0.59 | 0.69 | 0.74 | 0.79 | 0.85 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| re-calibrated | 0.37 | 0.46 | 0.48 | 0.56 | 0.61 | 0.65 | 0.69 | 0.74 |

Items the model calls 95% are worth 74%. That is not a sharpening problem — the
model already spreads its mass — it is that the head's confidence ranking is
miscalibrated against its own accuracy.

## Finding 3 — where a temperature works, a map is worse

noul and choice already clear the bar after a temperature (0.0274 and 0.0126).
There the map is worse (0.0309, 0.0272): 13 and 28 blocks pay a variance cost for
a bias that is not there. The comparison is therefore not "isotonic beats
temperature". It is that the right instrument depends on the shape of the
reliability curve, and only score has a shape a scale factor cannot express.

## What the ceiling says

The residual after the best monotone map is not calibration error that a better
calibrator would remove. A monotone map applied to 0.65 sets 0.47 no matter what;
whatever ECE remains comes from *which* items sit in that band — items the model
cannot distinguish and should not be confident about. That is an accuracy and
ranking problem in the score head, and it is the one goal item 2's phrasing
implicitly assumes away.

Two things follow, in cost order:

1. **Train the head to be less confidently wrong.** Label smoothing or an
   entropy penalty on the score head is a one-parameter change whose control is
   already on disk — the frozen 0.6B softmax arm of #171, 76.5 minutes per arm —
   so it is the cheapest next measurement. It attacks the 20-point band
   directly rather than post hoc.
2. **Measure the full distribution, not top-1.** The ceiling above is over maps
   of top-1 confidence, which is all ECE reads. Vector scaling, or a per-level
   recalibration of the whole option vector, can change a maximum in ways that no
   function of the maximum can, so it is not covered by the bound. It needs
   logits, which the dump does not carry, so it is a separate run.

## Caveats

* **One checkpoint, one split, 928 items.** The interval is quoted next to every
  score figure for that reason; the 0.0037 gap to the bar is smaller than the
  interval's width.
* **The map is fitted on half the eval set.** That is the harness's own protocol
  for temperature, so the comparison is matched, but a deployed calibrator would
  be fitted on a separate corpus and would do no better.
* **The regularisation grid was chosen by inner CV, not by holdout.** The grid
  itself (six alphas, five minimum block sizes) was not pre-registered. On this
  data the CV choice is 0.0002 from the best point on the grid scored against the
  holdout, so the reported figure is not tuned; a fixed `alpha=10, min_n=20`
  default would have given 0.0666, and the note quotes the chosen configuration
  for that reason.
* **The map uses top-1 confidence only.** A better score model would not change
  that bound; a different calibration *family* would, which is why item 2 above
  is named rather than assumed away.
* **Pre-#138 checkpoint.** This is #171's arm, trained before the cluster-aware
  split, so it is not a goal result — it is a diagnosis of the calibration
  behaviour the goal has to fix.

## What this means for the goal

Goal item 2 is not met on score, and this note closes off the calibration side of
the explanation: no post-hoc map of top-1 confidence clearly reaches 0.05, and a
per-type temperature does not come close (0.0811). The remaining hypotheses are
in training — overconfident logits in a band where the model is only 45%
accurate — and in the calibration family, where vector scaling on the full level
vector is the untested one. The next cheap test is label smoothing on the 0.6B
score head against the frozen arm this note's instruments were validated on.
