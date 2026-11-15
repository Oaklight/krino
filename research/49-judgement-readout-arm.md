# The judgement-readout arm regresses JevBench — 2026-11-15

Status: measured, one seed, one epoch. The adapted arm is a significant regression on the local 231-item JevBench proxy, and the calibration explanation is excluded by construction. The 20k-versus-full-corpus comparison is not measured yet; its placeholder states what it must report.

## Arm

| field | value |
|---|---|
| backbone | `Qwen/Qwen3-Reranker-4B`, frozen |
| adapter | LoRA on the last 8 blocks, rank 16, alpha 32, targets `q_proj k_proj v_proj o_proj` |
| readout | per-option yes/no margin, `log P(" yes") - log P(" no")` through a learned scale and bias |
| loss | row-normalised BCE: each row's option losses are averaged within the row, then across rows |
| data | canonical corpus, `answerable=false` masked, 166,208 train rows, one epoch, 2,608 steps |
| batch | 8 rows per step per rank, 8 ranks |
| length | at most 4,096 tokens per row and 20 options per row |
| seed | 42 (training and torch seed) |

Every number below belongs to this arm unless the table names the frozen control. Unless stated otherwise the length is the 231-item JevBench-public set (easy 48, hard 111, original 72) or the 42,667-row validation set, the readout is the per-option yes/no margin, and the seed is 42. The local 231-item number is a proxy over the public rows; it is not an official JevBench Intelligence score, and no official score is computed here.

## Training and full validation

| split | rows | loss | accuracy | ECE | gold P mean | all P mean |
|---|---:|---:|---:|---:|---:|---:|
| train | 166,208 | 0.48022 | — | — | — | — |
| val, all | 42,667 | 0.47426 | 0.6675 | 0.1662 | 0.4661 | 0.2447 |
| val, noul | 21,125 | 0.5516 | 0.7078 | 0.2115 | 0.4937 | 0.4913 |
| val, choice | 15,660 | 0.3860 | 0.6691 | 0.1366 | 0.4683 | 0.1896 |
| val, score | 5,882 | 0.4315 | 0.5184 | 0.0826 | 0.3610 | 0.2278 |

The train loss is the mean over the 2,608 steps; 2,947 rows were masked by the answerable flag and 0 were skipped. Accuracy is argmax over each item's options. The noul cell is above its 0.5 chance level. The score cell is not interpretable against chance without the level count. The run's by-source breakdown needs a pooling step before it can be read. The row id carries the exact source only for ids ending in `-<split>-<digits>`; synthetic, `typed_decisions` and apigen rows are identified only by prefix, so their cells are prefix pools. The pooling is additive over the run's shards, so a pooled table would reproduce.

The per-source table is pending trainer's run-level eval report. Two procedural caveats will apply to it. Only families with at least 90 rows are shown, because the rest are too thin to read. And a benchmark source with a large option set is confounded with that option set, so its cell is not a pure source effect.

## JevBench public 231: frozen against adapted

Both columns use the same backbone and readout. `frozen` is the unchanged Qwen3-Reranker-4B judgement readout; `adapted` adds the LoRA adapter above. Labels are corrected (see the next section). `cc` is chance-corrected accuracy with the set's chance of 0.3176.

| arm | correct | accuracy | cc | easy | hard | original |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 149/231 | 0.6450 | 0.4798 | 1.0000 | 0.4324 | 0.7361 |
| adapted | 135/231 | 0.5844 | 0.3910 | 1.0000 | 0.3514 | 0.6667 |

Paired flips, frozen to adapted, exact two-sided McNemar on the paired items:

| subset | n | both | lost | gained | neither | net | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 231 | 127 | 22 | 8 | 74 | -14 | 0.0161 |
| easy | 48 | 48 | 0 | 0 | 0 | 0 | 1.0000 |
| hard | 111 | 36 | 12 | 3 | 60 | -9 | 0.0352 |
| original | 72 | 43 | 10 | 5 | 14 | -5 | 0.3018 |
| choice | 139 | 77 | 12 | 3 | 47 | -9 | 0.0352 |
| noul | 74 | 41 | 6 | 4 | 23 | -2 | 0.7539 |
| score | 18 | 9 | 4 | 1 | 4 | -3 | 0.3750 |

On items whose decision margin gap is at least 0.25 (the "decisive" subset), the direction survives: 117 both, 14 lost, 4 gained, 57 neither, 39 skipped, net -10, exact McNemar p = 0.0309; the hard tier alone is 9 lost against 1 gained, p = 0.0215. The drop is therefore not carried by ties or near-ties.

By question type, corrected: frozen choice 0.6403 (cc +0.5331), noul 0.6351 (+0.2703), score 0.7222 (+0.6310); adapted choice 0.5755 (+0.4490), noul 0.6081 (+0.2162), score 0.5556 (+0.4096). Two earlier statements are retracted: noul is not below chance in either arm, and the apparent +4 noul gain was a net loss of 2 once the labels were corrected.

The cells come from a fresh deterministic re-score of the run's per-item records; two fresh runs were identical. The in-run aggregate and the re-score can differ by one or two items, so the re-score is the number of record and the paired analysis uses it.

## The eval bug behind the first numbers

`candidates()` in `model/evaluation/judgement_prompt.py` decided which noul option was gold by comparing the label to the string `"true"`. The public JevBench files carry `expected` as `"yes"` and `"no"`, so neither matched, every one of the 74 noul items was built with gold index 1, and unknown strings fell through silently. The correction in #378 routes both call sites through one `label_polarity()` that accepts booleans, 0/1, and true/yes/y/false/no/n in any case with surrounding space, and raises on an unknown form instead of defaulting.

The correction changes the frozen arm's cc from 0.3212 to 0.4798. The noul cells that were reported through this path are superseded: frozen noul 24/74 becomes 47/74, adapted noul 28/74 becomes 45/74, and the aggregate flips above replace the earlier ones. The audit in #378 checked every committed JevBench JSON for the fingerprint (gold index 1 on every noul row) and found none, because the committed files pass through `load_jevbench()`, which maps "yes" to `True` before `candidates()` sees it. No earlier research note requires a superseded marker. The `148/231` figure in the marker-ablation note is a different arm, scored through `load_jevbench`, and is not the same measurement as the `149/231` frozen cell here.

## Mechanism: sharper margins, worse ranking

The adapter raises the decision margin without improving the ordering. Median raw top-two gap 1.188 to 3.938; median effective gap (after the scale and bias) 0.241 to 0.798, and mean effective gap 0.472 to 1.525, about 3.2x. On the hard tier, items that are confident and wrong (effective gap above 0.5) go from 10/111 to 42/111, while hard accuracy falls from 0.4324 to 0.3514.

The attribution is the objective, not the data: per-option BCE rewards the magnitude of the correct log-odds margin, and a larger margin on a wrong option still lowers the per-option loss even when the argmax over options stays wrong. The adapter therefore learns to be decisive without learning to rank, which is the opposite of a calibrated transfer.

## Calibration is not the explanation

The calibration gate is not met: ECE 0.1662 overall, noul 0.2115, choice 0.1366, score 0.0826. That is a real calibration defect, but it cannot explain the JevBench regression. Each JevBench decision is an argmax over at least two options; the learned scale is positive (final value 0.2028); sigmoid followed by a positive affine map is monotone per item, so no affine recalibration can change a decision. The flip tables with and without the learned scale and bias are identical. The cause is the LoRA weights.

## Marker token, cross-reference

The marker-token ablation (#362, merged as #372) trains six heads identical except for the marker id. All five non-default markers are unresolved against the default `<|im_end|>`; the largest raw delta is +1.30 pp, the realized 80% minimum detectable effect is 2.8-4.3 pp, and the pre-registered adoption bar remains 9.3 pp. The swap pass moves no cell by more than +1.73 pp. The head reads the option-boundary position, not the token identity. The cells are in `research/44-marker-token-ablation.md`; they are one seed per arm and are not repeated here.

## Throughput

The per-row loop idled the card: mean utilization 20.7%, with 57% of samples at zero and 25% at or above 50%, at 3.36 rows/s cold and 9.54 rows/s warm. Batching the chunked forwards raised the real full-corpus epoch to 52.6 rows/s at 8 ranks over the 166,208 rows and 2,608 steps. Data-parallel scaling is 7.8x from one rank to eight once the shapes are warm (9.54 to 74.37 rows/s, with per-step cost near 0.86 s at every world size); the earlier 1.0x reading was cold-shape compilation per process, not a scaling limit. Units are rows per second, one row being one judgement item with its option set.

## Pending: 20k against the full corpus

Not measured. The 20k arm draws a deterministic 20,000-row subset and runs one epoch, about 313 steps at 64 rows per step; the full arm is 2,608 steps. The comparison is therefore confounded by optimizer steps (about 8.3x) at the same one-epoch schedule, and no difference may be attributed to corpus size alone without both step counts. Evidence required before any conclusion: the same plan, seed, learning rate and batch; both arms scored through the corrected label path; JevBench paired flips against the frozen arm with the exact McNemar test and n; validation loss, accuracy and ECE; wall time. Until then this section is unverified.

## Provenance

| number group | status | source |
|---|---|---|
| arm definition, train loss, validation table, mechanism gaps, ECE, throughput | measured; run log not committed | training run of 2026-11-15, seed 42 |
| JevBench corrected cells, flips, decisive subset, by-type cells | measured; correction committed, per-item records not committed | #378, recomputed from the run's per-item records |
| by-source table | pending | awaiting trainer's run-level eval report for run C (166,208 rows, 2,608 steps, mean loss 0.48022) |
| marker-token ablation | measured; committed | #372 and `research/44-marker-token-ablation.md` |
| 20k against full | pending | not measured |

Committing the run's validation JSON and per-item records in a follow-up would give the first two rows artifact provenance. The repository currently holds the method for the correction, not the run's own records.

## Scope

One backbone, one adapter position, one readout, one corpus, one seed, one epoch. The JevBench number is the local 231-item proxy, not the official Intelligence metric. Nothing here measures a second seed, a different adapter position, a different corpus size, or a trained head on top of the adapted backbone.
