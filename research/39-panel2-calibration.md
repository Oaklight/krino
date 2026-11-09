# Vocabulary-code calibration for panel 2 (#338) — 2026-11-09

Panel 2 reads the next-token distribution at `Answer:` and keeps the K code
columns, so every number it has produced is an argmax accuracy over
uncalibrated logits. This note fits a per-type temperature and bias on the
development set, freezes them, and reports ECE, Brier and accuracy on the
committed JevBench cells before and after. Two checkpoints, `Qwen/Qwen3-
Reranker-4B` and `Qwen/Qwen3-32B`; nothing is fitted on JevBench.

## The parameterisation, and two corrections

The fitted model is `p = softmax(z / T + b)`, with `T` and `b` per question
type, fitted on dev by minimising the mean negative log-likelihood.

**A temperature cannot move accuracy.** `softmax(z / T)` is a strictly
monotone transform of `softmax(z)`, so the argmax is identical for every
`T > 0`. The issue's hypothesis — that a badly scaled distribution can
suppress the correct option in close calls, so a temperature correction may
move accuracy — is false under this parameterisation, and no rescaling of the
logits can produce that effect. The script compares the before and after
predictions item by item and aborts if one moves; it did not.

**A per-type scalar bias is not identifiable.** `softmax(z + c) =
softmax(z)`, so a bias equal on every code cancels and has no effect at all.
The bias fitted here is therefore a per-slot vector over code positions, with
a ridge (`l2 = 1e-3`) and a box at ±6. That is the only member of the family
that can reorder codes, and it is a position prior as much as a calibration
term; it is reported separately for that reason.

**Production reference.** H2O-Lightning-4B's `serve_config.json` carries
per-type temperatures on its own readout — noul 0.8, choice 0.75, score 0.65
— with `noul_floor: 0` and no bias term. The family matches; the direction
does not: the served model is sharpened (`T < 1`), and the panel-2 code
distribution is over-confident and needs smoothing (`T > 1`). The two are
different heads, so this is a parameterisation reference, not a like-for-like
comparison.

## Results on JevBench

Fitted on dev's 788 scorable natural-order rows (noul 391, choice 298,
score 99; 16 items have a gold label outside the option set and 24 have 80
options against 26 one-token codes). Evaluated once on the 231 committed
JevBench predictions.

| checkpoint | mode | accuracy | ECE | Brier |
|---|---|---:|---:|---:|
| Reranker-4B | before | 162/231 = 70.13% | 0.1348 | 0.4516 |
| Reranker-4B | temperature | 162/231 = 70.13% | 0.0547 | 0.4158 |
| Reranker-4B | temperature + slot bias | 162/231 = 70.13% | 0.0561 | 0.4266 |
| 32B | before | 178/231 = 77.06% | 0.0808 | 0.3386 |
| 32B | temperature | 178/231 = 77.06% | 0.0650 | 0.3204 |
| 32B | temperature + slot bias | 173/231 = 74.89% | 0.0748 | 0.3295 |

The aggregate ECE falls by 0.080 (−59%) on the 4B and 0.016 (−20%) on the
32B. Accuracy is bit-identical under the temperature on both. The slot bias
moves accuracy by +6 −6 items on the 4B (net 0, exact McNemar p = 1.0) and
+4 −9 on the 32B (net −5 = 2.16 pp, p = 0.267); both are far inside the
9.3 pp detectable delta of 231 items, so neither direction is resolved.

## Per type

| checkpoint | type | n | accuracy | ECE before → after | Brier before → after |
|---|---|---:|---:|---|---|
| Reranker-4B | noul | 74 | 64.86% | 0.2740 → 0.1351 | 0.6030 → 0.4777 |
| Reranker-4B | choice | 139 | 73.38% | 0.0972 → 0.1349 | 0.3705 → 0.3805 |
| Reranker-4B | score | 18 | 66.67% | 0.1961 → 0.1751 | 0.4558 → 0.4338 |
| 32B | noul | 74 | 81.08% | 0.1515 → 0.1306 | 0.3264 → 0.3004 |
| 32B | choice | 139 | 75.54% | 0.0736 → 0.0798 | 0.3364 → 0.3248 |
| 32B | score | 18 | 72.22% | 0.1844 → 0.2183 | 0.4058 → 0.3686 |

Accuracy is unchanged for every type under the temperature. The aggregate
gain is entirely noul: on both checkpoints it is the type with the largest
calibration gap and the one the correction helps most. Choice, the largest
type, is worse in top-1 ECE after the correction on both checkpoints while
its multiclass Brier improves on the 32B and worsens slightly on the 4B
(0.3705 to 0.3805) — the distribution gets better probabilities over the
options on one checkpoint while the top-1 confidence moves away from the hit
rate. Score has 18 items, so its ECE moves are worth a couple of items and
are not read as a trend.

## The fitted parameters

| checkpoint | type | T (temperature) | T (joint) | bias |
|---|---|---:|---:|---|
| Reranker-4B | noul | 3.009 | 2.942 | 0.00, 0.00 |
| Reranker-4B | choice | 2.501 | 2.325 | −0.62, −0.20, 0.37, −0.15, −0.20, 0.16, 0.72, −0.17 |
| Reranker-4B | score | 1.349 | 1.297 | −0.60, 0.31, 0.09, 0.06, −0.09 |
| 32B | noul | 2.187 | 2.266 | −0.08, 0.01 |
| 32B | choice | 1.650 | 1.577 | −0.63, −0.09, 0.29, 0.05, −0.16, 0.51, 0.28, −0.67 |
| 32B | score | 1.862 | 1.550 | −1.61, 0.17, 0.01, 0.53, −0.15 |

No slot sits on the ±6 box. The bias is invariant to an additive shift, so
only its spread is interpretable; the largest spread is the 32B's score
vector, fitted on 99 rows across five slots and not transferred safely.

## Why the aggregate improves while two types regress

The dev and JevBench calibration profiles are not the same, and the fitted
temperature carries the dev profile across. On dev before any fitting, choice
ECE is 0.221 (4B) and 0.142 (32B); on JevBench it is 0.097 and 0.074. Dev
choice items are also much harder (59.4% and 69.1% accuracy against 73.4%
and 75.5%), so the raw distribution is more over-confident there. Fitting on
dev therefore over-smooths choice on JevBench: the bulk of choice items moves
from a confidence near 0.97, where its accuracy is 0.92, into the 0.65-0.86
range, where accuracy is 0.87-1.00. Noul runs the other way — JevBench is
more miscalibrated than dev (0.274 and 0.152 against 0.149 and 0.130) — so
the same temperature under-corrects it but still helps.

This is the direct argument for the issue's own rule: a temperature fitted on
the acceptance instrument would have looked better on every column and
would have said nothing about transfer.

## Reading against the pre-registered branches

**Branch 1 — accuracy unchanged, ECE drops materially — fires at the
aggregate and for noul, not for choice.** The 4B's −59% is material by any
reading; the 32B's −20% is smaller. Panel 2 becomes usable as a probability
in the sense the issue asks for, but the useful form is per type, and the
correction is fitted on data whose per-type profile differs from the
deployment candidate. A production temperature should be fitted on data that
resembles the deployment distribution, which JevBench is not allowed to be.

**Branch 2 — accuracy moves by more than the floor — does not fire.** The
temperature cannot move it, and the slot bias moves it by 0 and −2.16 pp
against the 9.3 pp floor. The raw panel-2 accuracy numbers in #326 and #329
are not scale-dependent and do not need restating.

**Branch 3 — neither moves — does not fire**: ECE moves.

## What this does not say

It does not say the code distribution is well calibrated after the correction.
The reliability tables are in the result files; the residual gaps are a
per-type shape that neither a temperature nor a slot bias can remove, and a
monotone confidence map is the ceiling for anything that uses confidence
alone. It also does not say the slot bias is useless in general: what it says
is that a position prior fitted on dev and applied to JevBench does not
improve accuracy on either checkpoint.

## Provenance

Config hash `4752c103e6cf2d48d97fceab313a357435a3d5bc86dd8bad6bd46c7eb68a92e9`,
recorded in #338 before the runs; dev-run code commit
`420d3ee96488fd1c05f565903921d36b92ee054c`, fitting script commit
`327842e926583501189b3d2c8da867fb02e0cf38`. Dev runs at revision
`22e6836` (4B) and `9216db5` (32B), bf16, identical prompt template and code
form as the committed JevBench cells. The JevBench side is the committed
panel-2 dump; calibration transforms its stored logits, so the instrument was
not re-read. Fit: `criterion=nll`, `l2=1e-3`, bias box ±6, ECE over 10
equal-width bins on top-1 confidence, multiclass Brier summed over the codes.

Files: `panel2_dev_Qwen_Qwen3-Reranker-4B.json`,
`panel2_dev_Qwen_Qwen3-32B.json`, `panel2_calibration_Qwen_Qwen3-Reranker-4B.json`,
`panel2_calibration_Qwen_Qwen3-32B.json`, all under `results/jevbench/`.
