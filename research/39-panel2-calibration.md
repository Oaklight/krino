# Vocabulary-code calibration for panel 2 (#338) — 2026-11-09

Panel 2 reads the next-token distribution at `Answer:` and keeps the K code
columns, so every number it has produced is an argmax accuracy over
uncalibrated logits. This note fits a per-type temperature and bias on the
development set, freezes them, and reports ECE, Brier and accuracy on the
committed JevBench cells before and after. Two checkpoints, `Qwen/Qwen3-
Reranker-4B` and `Qwen/Qwen3-32B`; nothing is fitted on JevBench.

**The result in one line.** A per-type temperature fitted on dev lowers
aggregate ECE on both checkpoints, but the gain is entirely noul, a 74-item
binary type: choice — the 139-item type panel 2 exists to improve — is worse
after the correction on both checkpoints (ECE 0.097 → 0.135 and 0.074 →
0.080). The aggregate overstates the cell. The per-type split is the result,
and the reason for it is a dev/JevBench profile difference that constrains
the epic, not only this cell. The slot bias is reported as the negative it
is; the recommendation is temperature-only.

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
32B, and accuracy is bit-identical under the temperature on both. That
aggregate is a weighted sum dominated by noul, whose 74 items carry an ECE of
0.274 (4B) and 0.152 (32B) before the fit; the per-type table below is the
result. The slot bias moves accuracy by +6 −6 items on the 4B (net 0, exact
McNemar p = 1.0) and +4 −9 on the 32B (net −5 = 2.16 pp, p = 0.267), both far
inside the 9.3 pp detectable delta of 231 items.

## Per type

| checkpoint | type | n | accuracy | ECE before → after | Brier before → after |
|---|---|---:|---:|---|---|
| Reranker-4B | noul | 74 | 64.86% | 0.2740 → 0.1351 | 0.6030 → 0.4777 |
| Reranker-4B | choice | 139 | 73.38% | 0.0972 → 0.1349 | 0.3705 → 0.3805 |
| Reranker-4B | score | 18 | 66.67% | 0.1961 → 0.1751 | 0.4558 → 0.4338 |
| 32B | noul | 74 | 81.08% | 0.1515 → 0.1306 | 0.3264 → 0.3004 |
| 32B | choice | 139 | 75.54% | 0.0736 → 0.0798 | 0.3364 → 0.3248 |
| 32B | score | 18 | 72.22% | 0.1844 → 0.2183 | 0.4058 → 0.3686 |

Accuracy is unchanged for every type under the temperature. Choice is the
type this readout exists to improve, and on both checkpoints the dev-fitted
temperature makes its top-1 ECE worse: 0.097 → 0.135 on the 4B and 0.074 →
0.080 on the 32B. Its multiclass Brier improves on the 32B (0.3364 → 0.3248)
and worsens slightly on the 4B (0.3705 → 0.3805): the distribution over the
options gets better on one checkpoint while the top-1 confidence moves away
from the hit rate. Noul is the only type that improves on both, 0.274 → 0.135
and 0.152 → 0.131, and it is what lifts the aggregate. Score has 18 items, so
its moves are worth a couple of items and are not read as a trend.

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
only its spread is interpretable. The vectors are reported because the issue
asked for a bias; they improve no reported quantity on either checkpoint, and
the recommendation is temperature-only.

## Why the aggregate improves while choice regresses

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

### A constraint for the epic

Dev and JevBench disagree about choice in a specific way: dev choice is easier
(59.4% and 69.1% accuracy against 73.4% and 75.5%) and less well calibrated
(ECE 0.221 and 0.142 against 0.097 and 0.074) at the same time. Dev therefore
ranks choice accuracy plausibly but is not a calibration-transfer set for that
type. Any later cell that fits a calibration parameter on dev and reports it
on JevBench for choice has to carry that caveat, not rediscover it; #342's
threshold sweep is the next one that will meet it.

## Reading against the pre-registered branches

**Branch 1 — accuracy unchanged, ECE drops materially — fires for noul
only.** Accuracy is unchanged on both checkpoints, as the monotonicity
argument requires. ECE drops materially for noul (0.274 → 0.135 and 0.152 →
0.131) and rises for choice on both checkpoints. The aggregate falls by 59%
and 20%, but only because noul's pre-fit ECE is the largest in the table;
quoting the aggregate as the cell's result would invert what the cell says
about the type the readout exists to improve. Panel 2's noul probabilities
become usable; its choice probabilities do not.

**Branch 2 — accuracy moves by more than the floor — does not fire.** The
temperature cannot move it, and the slot bias moves it by 0 and −2.16 pp
against the 9.3 pp floor. The raw panel-2 accuracy numbers in #326 and #329
are not scale-dependent and do not need restating.

**Branch 3 — neither moves — does not fire**: ECE moves.

## What this does not say

It does not say the code distribution is well calibrated after the correction,
and it does not recommend the slot bias. On both checkpoints the bias improves
no reported quantity: on the 4B it raises ECE (0.0547 → 0.0561) and Brier
(0.4158 → 0.4266), and on the 32B it costs five items with no detected loss
inside the floor but no gain either. A position prior fitted on dev is the
least transferable object in the table. The residual calibration gaps are a
per-type shape that neither a temperature nor a slot bias can remove, and a
monotone confidence map is the ceiling for anything that uses confidence
alone.

## Provenance

Config hash `4752c103e6cf2d48d97fceab313a357435a3d5bc86dd8bad6bd46c7eb68a92e9`,
recorded in #338 before the runs; dev-run code commit
`420d3ee96488fd1c05f565903921d36b92ee054c`, fitting script commit
`1a672a7aecbe1387ac8a25b77e1e08cd458124b1`. Dev runs at revision
`22e6836` (4B) and `9216db5` (32B), bf16, identical prompt template and code
form as the committed JevBench cells. The JevBench side is the committed
panel-2 dump; calibration transforms its stored logits, so the instrument was
not re-read. Fit: `criterion=nll`, `l2=1e-3`, bias box ±6, ECE over 10
equal-width bins on top-1 confidence, multiclass Brier summed over the codes.

Files: `panel2_dev_Qwen_Qwen3-Reranker-4B.json`,
`panel2_dev_Qwen_Qwen3-32B.json`, `panel2_calibration_Qwen_Qwen3-Reranker-4B.json`,
`panel2_calibration_Qwen_Qwen3-32B.json`, all under `results/jevbench/`.

## Appendix: reliability by type on JevBench

Ten equal-width bins on top-1 confidence; each entry is
`(items, mean confidence, accuracy)`. Empty bins are omitted. The same tables,
with bin edges, are in the calibration result files.

```
Qwen3-Reranker-4B
noul    before  (4, 0.55, 1.00) (5, 0.65, 0.40) (14, 0.75, 0.71) (7, 0.85, 0.57) (44, 0.97, 0.64)
        after   (18, 0.56, 0.67) (17, 0.65, 0.59) (18, 0.75, 0.56) (14, 0.86, 0.64) (7, 0.91, 1.00)
choice  before  (3, 0.29, 0.00) (3, 0.39, 0.33) (13, 0.44, 0.46) (12, 0.54, 0.25) (7, 0.64, 0.71) (7, 0.74, 0.71) (11, 0.84, 0.55) (83, 0.97, 0.92)
        after   (10, 0.27, 0.30) (20, 0.35, 0.50) (17, 0.45, 0.47) (16, 0.55, 0.63) (30, 0.65, 0.87) (18, 0.76, 0.94) (26, 0.86, 1.00) (2, 0.91, 1.00)
score   before  (1, 0.46, 0.00) (3, 0.54, 0.67) (2, 0.61, 0.00) (2, 0.78, 0.50) (1, 0.86, 1.00) (9, 0.97, 0.89)
        after   (1, 0.39, 0.00) (3, 0.47, 0.67) (2, 0.53, 0.00) (2, 0.68, 0.50) (1, 0.76, 1.00) (1, 0.87, 1.00) (8, 0.93, 0.88)
```

```
Qwen3-32B
noul    before  (5, 0.56, 1.00) (5, 0.66, 0.40) (7, 0.74, 0.86) (13, 0.86, 0.54) (44, 0.97, 0.91)
        after   (11, 0.56, 0.73) (10, 0.64, 0.90) (24, 0.75, 0.62) (15, 0.85, 0.93) (14, 0.93, 1.00)
choice  before  (5, 0.37, 0.20) (6, 0.47, 0.33) (15, 0.55, 0.47) (13, 0.65, 0.38) (11, 0.75, 0.64) (12, 0.85, 0.67) (77, 0.97, 0.97)
        after   (2, 0.29, 0.00) (9, 0.36, 0.22) (17, 0.45, 0.47) (19, 0.54, 0.53) (13, 0.66, 0.62) (19, 0.77, 0.95) (24, 0.86, 0.96) (36, 0.94, 1.00)
score   before  (1, 0.48, 0.00) (2, 0.57, 0.50) (2, 0.66, 0.00) (1, 0.75, 1.00) (1, 0.87, 0.00) (11, 0.98, 1.00)
        after   (1, 0.37, 0.00) (4, 0.46, 0.25) (1, 0.55, 1.00) (1, 0.65, 0.00) (2, 0.75, 1.00) (6, 0.86, 1.00) (3, 0.90, 1.00)
```

The 32B's choice top bin is the clearest case: 77 of 139 items at 0.97
confidence with 0.97 accuracy before the correction, spread to 0.86-0.94 with
accuracy 0.96-1.00 after it. The fitted temperature is correcting a
mis-calibration that type does not have.
