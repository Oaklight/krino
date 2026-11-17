# The score-loss arms: one real but different, one null — 2026-11-16

Status: measured, one run per arm. Three full-parameter arms differ only in the loss applied to score rows. B is null on every endpoint; C improves rank probability score with an interval that excludes zero, holds exact accuracy flat and loses on quadratic-weighted kappa, and does not meet its pre-registered exact trigger. Nothing here changes the admission verdict of any earlier arm.

Every arm uses Qwen3-Reranker-4B with every parameter trainable (LoRA off), the panel-2 readout with listwise cross-entropy, the v3 a20k plan at 303 steps, seed 42, activation checkpointing on, and at most 4,096 tokens per row. All three share one harness and one plan. The only difference is the target applied to score rows. The primary cells are the 1,111 non-synthetic validation score rows with K from each row's own codes. Paired tests are exact two-sided McNemar on exact accuracy and bootstrap intervals on the rank probability score, with the paired n printed.

## 1. Design

The three arms are A, B and C, in the order of the loss modes: A is `--score-loss ce`, B is `mix` with the lambda below, and C is `rps`. All three ran the same score target, the default one-hot; the soft-target arm (scoreB2) was held behind these three and did not run. Choice and noul keep the shipped loss bit for bit; only score rows change, and all arms decode by argmax.

The rank probability score term is weighted by a single global lambda, a pinned constant at 1.90243, quoted here to five decimals. The pin comes from step 0 on the 1,111 non-synthetic validation score rows: mean cross-entropy 1.403504, mean rank probability score 0.184435, ratio 7.609737. A's final weights give 1.866514, a ratio of 7.466057, within 2% of the pin, so it is not a step-0 artefact.

The realised share is not uniform. By gold position the RPS term is 26.3% (gold 0, n = 227), 17.7%, 14.1%, 18.7%, 20.8% and 21.2% (gold 5, n = 35) of the score-row loss, so 14.1 to 26.3% rather than a flat 20%. By K it is 21.7% at K = 4, 19.7% at K = 5 and 17.4% at K = 6. A per-row weight that equalises the share was rejected: it would make the loss depend on the gold's position, which is the confound the plan-level permutation exists to remove.

## 2. Payload cells and the bands

Validation differences sit inside the 0.52 pp band measured in the noise-floor work: B is +38 rows and C is -7 of 9,923. JevBench differences sit inside 1 to 2 items. Neither is a result, and no cell in this section is read as one.

ECE is reported only where it clears the two-run spread of up to 2.0 pp. C's score-row ECE starts at 0.1817 under both definitions: the payload definition improves to 0.1110, which clears the spread, while the p(gold) definition worsens to 0.2024 on the same baseline, so the sign disagreement is established on both sides. B's baseline is 0.1723, and it does not clear. Every ECE sentence in this note names its definition.

Order self-consistency falls for C by 14 items, with 31 discordant items against A's 17, so the natural-minus-reversed gap shrinks because the flips grew rather than because the orders agree more.

## 3. Paired endpoints on the 1,111 rows

| endpoint | A | B | C |
|---|---:|---:|---:|
| exact correct | 530 | 530 | 521 |
| rank probability score | 0.174367 | 0.174091 | 0.168172 |
| quadratic-weighted kappa | 0.3922 | 0.3885 | 0.3577 |
| mean cross-entropy | 1.3018 | 1.2967 | 1.2446 |

A and B have identical exact counts on different rows, exact McNemar p = 1.0. C is -9 exact against A, p = 0.58. The A-to-C rank probability score gain is +0.006195 with a bootstrap interval of [0.000555, 0.011881] and p = 0.0328. C loses 0.035 on quadratic-weighted kappa.

## 4. Verdicts against the pre-registration

B is null on every endpoint with a stated resolution: its exact pair is b = 25, c = 25, 50 discordant, exact McNemar p = 1.0, and the minimum detectable effect at that discordance is about 14 items, so the finding is no measurable effect at this resolution. Rank probability score, kappa and cross-entropy sit inside the bands as well.

C is real but different, not a pass. Its rank probability score improves with an interval that excludes zero, its exact accuracy is flat within noise, and its kappa is worse by 0.035. The repeat trigger was registered on the 1,111 score rows as a 12-item exact gain; at that size the paired minimum detectable effect is 41 to 47 items, so the trigger was under-powered. C's non-confirmation therefore rests on its exact p = 0.58 and on being a single run, not on the trigger failing to fire.

## 5. Calibration, by definition

C's score rows move in opposite directions under the two ECE definitions: the payload definition improves to 0.1110 and the p(gold) definition worsens to 0.2024. A sentence about ECE that does not say which definition it uses is therefore not interpretable here.

The per-K move is located rather than averaged: +5.5 pp at K = 4 and -4.6 pp at K = 6. An average over K would hide a gain at the short lists and a loss at the long ones.

## 6. The K restriction and renormalisation

The decision-relevant K restriction is nearly lossless. The label is the global top-1 vocabulary token in the median row of both models: non-code top-1 appears in 1.43% of frozen rows and 0.88% of adapter rows, with median mass outside the K codes of 10.4% and 19.9%, and median gold rank 0 in both. The masked tokens are newline and punctuation continuations, dominated by " \n\n" (116), then " ?\n\n" (7), " $." (4) and " $\n" (2), about one row in seventy. Outside mass by type runs 7.7% and 15.1% for choice, 15.3% and 29.1% for noul, 11.2% and 17.8% for score, frozen and adapter.

Every reported probability is renormalised over the offered codes, p_reported = p_raw / (1 - outside). Median outside mass of 10.4% inflates reported probabilities by about 12%, the adapter's 19.9% by about 25%, and its noul rows at 29.1% by about 41%. Part of any frozen-to-adapter ECE movement is therefore the renormalisation, not the raw distribution. This is a caveat on ECE comparisons, not a defect: renormalising over the offered options is what the API contract requires.

## 7. The frozen-sample caveat

The reweighted stop reference rests on a frozen sample that carries 5 score rows of 64, against a target weight of 18.45%, which is 3,569 of 19,348 rows on the a20k plan rather than the canonical plan's 13,024 of 69,464 distinct rows at 18.7%, so its score component is n = 5 and `stopped` is not a score-quality check in practice. Raising `--frozen-loss-sample` to 512 buys n = 31, since the first 512 rows are 6.1% score, at about 90 s per rank with no change to step counts; that is pre-registered for a future arm family, and the three arms here keep 64 so their references stay identical.

## Provenance

| number group | status | source |
|---|---|---|
| design, lambda pin, realised shares | measured; probe output on the run host, not committed | step-0 measurement on the 1,111 score rows, plan df32899 |
| payload cells, paired endpoints, verdicts, ECE, per-K moves | measured; run payloads on the run host, not committed | arms A, B and C of 2026-11-16, seed 42 |
| K restriction and renormalisation | measured; probe output on the run host, not committed | 9,923 val rows plus 231 JevBench natural items |
| frozen-sample caveat | measured; run payload on the run host, not committed | frozen-loss sample |

## Scope

One backbone, one readout, one plan, one seed, one run per arm. The pre-registered exact trigger did not fire, so C is not a confirmed arm. Nothing here measures a second seed, a full epoch, a per-row weighted loss, or the official JevBench Intelligence score.
