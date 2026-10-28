# Score calibration is a property of the arm, not of the score primitive

Goal item 2 asks for ECE ≤ 0.05 after per-type temperature scaling. This note has
been rewritten once: its first version measured one 4B arm and concluded that a
temperature could not calibrate the score primitive at 4B. That conclusion does
not survive the second arm. On `arch_4b_loraAll_weighted` — the checkpoint that
carries the rest of the goal — the same per-type temperature procedure reaches
**0.0250** on score, under the bar. On `arch_4b_loraAll_32k` it stalls at 0.0811.

Both arms have the same backbone, data, schedule and seed, and they land within
0.13 pp of each other on overall accuracy (0.6581 against 0.6594, inside the
0.35 pp seed floor). So this is not "one model is better". It is that the
reliability curve of the score head is shaped differently, and a scale factor
fits one shape and not the other.

## Setup

| | `arch_4b_loraAll_32k` | `arch_4b_loraAll_weighted` |
|---|---|---|
| Backbone | Qwen3-Reranker-4B, full LoRA | same |
| Readout | `last` | `weighted`, 37 hidden states |
| Training `max_length` | 32,768 (its log) | 16,384 (its log) |
| Eval `max_length` | 32,768 | 8,192 |
| Overall accuracy, rebuilt corpus | 0.6594 | 0.6581 |
| Score items | 1,856 (928 holdout) | 1,856 (928 holdout) |

The derived artifacts are `model/experiments/score_calibration_map_4b_32k.json`
and `score_calibration_map_4b_weighted.json`, each tagged with its arm.

Data is the corpus rebuilt after #177 (stsb's withheld labels removed). The split
is the harness's own — a shuffle of `range(n)` keyed by
`"<seed>:score:temperature"`, `fit_frac` 0.5 — and both runs reproduce its
`ece_before` to four decimals, so every number below is on the same 928 holdout
items. Score items are short documents; the eval-time `max_length` difference
affects long contractnli rows and cannot touch these 1,856 items.

The weighted arm's numbers were produced twice, by two independent passes over
the same checkpoint, and every accuracy, fitted temperature and holdout ECE below
agrees to four decimals between the two reports
(`cal_clean_4b_weighted.json` and `cal_clean_4b_weighted_pred.json`). The 32k
arm was measured once.

The temperatures are fitted by ECE, not NLL. That is the harness's
`--fit-criterion ece`, and it is the criterion the goal is stated in: on the 0.6B
arm the two criteria pick 1.5 and 3.6 and disagree about whether the bar is met
(#171).

## Results

Per type, at T=1 and after a temperature fitted by ECE on the mirror half:

| type | arm | accuracy | ECE at T=1 | fitted T | holdout ECE before → after |
|---|---|---:|---:|---:|---:|
| noul | 32k | 0.8242 | 0.0329 | 1.140 | 0.0400 → 0.0274 |
| noul | weighted | 0.8238 | 0.0333 | 1.162 | 0.0396 → 0.0290 |
| choice | 32k | 0.5732 | 0.0395 | 1.264 | 0.0352 → 0.0126 |
| choice | weighted | 0.5722 | 0.0373 | 1.209 | 0.0373 → 0.0161 |
| score | 32k | 0.5636 | 0.0897 | 1.772 | 0.1027 → **0.0811** |
| score | weighted | 0.5593 | 0.0784 | 1.518 | 0.0765 → **0.0250** |

## Finding 1 — the arms agree on noul and choice, and only score moves

Noul and choice are indistinguishable across the two arms: fitted temperatures of
1.140 against 1.162 and 1.264 against 1.209, holdout ECE after scaling of 0.0274
against 0.0290 and 0.0126 against 0.0161. Both are under the bar on both arms.

Score is not: the temperature moves from 1.772 to 1.518 and the holdout ECE after
scaling moves by a factor of 3.2. Whatever differs between these arms, it shows
up in the score head's calibration and nowhere else.

## Finding 2 — on the 32k arm the reliability curve is bent

Score holdout, 32k arm, at T=1:

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

Below 0.55 the model is roughly calibrated and at 0.94 it is right again, but
between 0.60 and 0.85 — 520 of the 928 holdout items — it is 11 to 20 points too
confident, worst in the middle of the band. A temperature is one multiplicative
change applied to the whole curve, so reaching the middle costs the ends: that is
why 1.772 recovers only 0.022 of the 0.053 needed.

The ceiling for anything that reads top-1 confidence, on this arm, is a
regularised monotone map fitted on the mirror half: **0.0537** (95% interval
[0.0282, 0.0865]) against the temperature's 0.0811. The map closes a third of the
gap and still misses the bar, so on this arm the residual is not calibration error
a better calibrator would remove.

## Finding 3 — on the weighted arm a temperature is enough, and the map does not improve on it

Score holdout, weighted arm, at T=1:

| bin | mean confidence | accuracy | gap | n |
|---|---:|---:|---:|---:|
| 2 | 0.257 | 0.375 | +0.118 | 8 |
| 3 | 0.361 | 0.405 | +0.045 | 37 |
| 4 | 0.460 | 0.465 | +0.005 | 99 |
| 5 | 0.547 | 0.450 | −0.096 | 242 |
| 6 | 0.652 | 0.574 | −0.078 | 251 |
| 7 | 0.742 | 0.669 | −0.073 | 175 |
| 8 | 0.848 | 0.704 | −0.144 | 71 |
| 9 | 0.935 | 0.889 | −0.046 | 45 |

Compare the two tables. The 32k arm is calibrated at 0.55 (+0.025) and then falls
off a cliff at 0.65 (−0.199). The weighted arm is undercalibrated at 0.55 (−0.096)
and overconfident by a similar amount at every bin above it (−0.07 to −0.14). One
is a dent; the other is a slope. A single scale factor cannot fix a dent — moving
0.65 down to 0.47 drags the well-calibrated 0.55 bin with it — but it absorbs a
slope, and T=1.518 does: 0.0765 to 0.0250 on the holdout.

The map does not improve on that. It reaches 0.0316 at the fixed regularisation
and 0.0350 with the regularisation chosen by inner CV, against the temperature's
0.0250. Nor is the map useless: on noul it wins (0.0143 against 0.0290), and on
choice the temperature wins (0.0161 against 0.0259), so which instrument is
better depends on the cell. That is the reason both are reported rather than one
being proposed as the answer.

Across the two arms the map is the better of the two instruments in two of six
cells: score on the 32k arm and noul on the weighted arm. In the cell that
matters here — score — the map helps only where the curve is bent, and there it
still lands above the bar, at 0.0537 against 0.05.

## What this means for the goal

**Goal item 2's ECE clause is met on the weighted arm**: 0.0250 on score, 0.0290
on noul and 0.0161 on choice, all after a per-type temperature fitted on held-out
items. This is the first time any 4B checkpoint has cleared that clause on score,
and it clears it by half the bar.

It is not yet a goal result. The checkpoint predates the cluster-disjoint split
(#138), so it trained on data that overlaps the eval set; item 2 has to be
re-measured on a #138-trained checkpoint before it counts. What this note
establishes is that the target is reachable and that the reading depends on which
arm is measured — a difference of 0.056 on a metric whose pass-to-pass floor is
0.0020.

The practical consequence is about reporting: a calibration number should name the
arm it came from, because "the 4B model" covers arms that differ by 3x on this
metric while agreeing to 0.13 pp on accuracy.

## Caveats

* **Two things differ between the arms**, the readout and the training
  `max_length` (32,768 against 16,384, from the runs' own logs), so this note
  cannot attribute the calibration difference to either. The obvious story — the
  readout changed what the score head sees — is also incomplete: the model shares
  one attention head between choice and score (`Shared attention: choice+score
  share AttentionHead` in both runs' logs), so a readout change reaches choice
  too, and choice's calibration does not move. A readout-only pair at fixed
  length is the clean ablation, and it is not run.
* **The eval `max_length` also differs** (32,768 against 8,192). It cannot affect
  the 1,856 score items, all short; it could in principle touch long contractnli
  rows, where choice accuracy differs by 0.001.
* **One seed per arm.** These are matched controls, but a single draw each.
* **The map ceiling is over top-1 confidence**, which is all ECE reads. Vector
  scaling on the full level vector can change a maximum in ways no function of
  the maximum can, so it is outside the bound and untested — it needs logits,
  which the prediction dump does not carry.
* **The numbers here are one split.** The 0.0020 pass-to-pass floor and the
  0.0282-0.0865 interval are quoted where they bear on a claim.
