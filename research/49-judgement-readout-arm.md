# The judgement-readout arm regresses JevBench — 2026-11-15

Status: measured, one seed, one epoch. On the full corpus the adapted arm is a significant regression on the local 231-item JevBench proxy, and the calibration explanation is excluded by construction. A 20,000-row subset arm is 15 items better and does not regress, but it differs in both data volume and step count, so that contrast is confounded.

## Arm

| field | value |
|---|---|
| backbone | `Qwen/Qwen3-Reranker-4B`, frozen |
| adapter | LoRA on the last 8 blocks, rank 16, alpha 32, targets `q_proj k_proj v_proj o_proj` |
| readout | per-option yes/no margin, `log P(" yes") - log P(" no")` through a learned scale and bias |
| loss | row-normalised BCE: each row's option losses are averaged within the row, then across rows |
| data | canonical corpus, `answerable=false` masked, 166,208 train rows, one epoch, 2,597 steps |
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

The train loss is the mean over the 2,597 steps; 2,947 rows were masked by the answerable flag and 0 were skipped. Accuracy is argmax over each item's options. The noul cell is above its 0.5 chance level. The score cell is not interpretable against chance without the level count. The run's by-source breakdown is pooled from C's eight eval shards by source prefix; the shards group benchmark ids exactly (ids ending `-<split>-NNN`) and carry one key per row for synthetic, apigen and `typed_decisions` rows, whose ids end in a descriptor. Pooling is additive over the eight shards. Re-aggregating the shards reproduces C's validation totals (n = 42,667, accuracy 0.6675, loss 0.47426), which is the check that the table is C's split.

Families with at least 90 rows, accuracy / loss / ECE / rows:

| source | family | acc | loss | ECE | rows |
|---|---|---:|---:|---:|---:|
| synthetic | spatial_reasoning | 0.5855 | 0.54954 | 0.0872 | 2,878 |
| synthetic | multi_hop_reasoning | 0.7156 | 0.47839 | 0.1462 | 2,103 |
| synthetic | navigation_planning | 0.6366 | 0.50806 | 0.1661 | 2,083 |
| synthetic | education_assessment | 0.6137 | 0.52177 | 0.1384 | 1,952 |
| synthetic | safety_moderation | 0.6040 | 0.52855 | 0.1715 | 1,884 |
| synthetic | game_strategy | 0.6249 | 0.51420 | 0.1420 | 1,738 |
| synthetic | content_analysis | 0.7109 | 0.44626 | 0.1977 | 1,705 |
| synthetic | adversarial_inference | 0.7440 | 0.42306 | 0.2132 | 1,676 |
| synthetic | financial_analysis | 0.6398 | 0.53704 | 0.1140 | 1,663 |
| synthetic | sequential_action | 0.6718 | 0.48292 | 0.1868 | 1,639 |
| synthetic | resource_management | 0.6610 | 0.48747 | 0.1374 | 1,602 |
| synthetic | product_categorization | 0.8465 | 0.27827 | 0.2508 | 1,570 |
| synthetic | numerical_reasoning | 0.6015 | 0.53976 | 0.1523 | 1,566 |
| synthetic | passage_decision | 0.7562 | 0.42272 | 0.2156 | 1,530 |
| typed decisions | typed_decisions | 0.4878 | 0.60918 | 0.0685 | 1,476 |
| synthetic | code_review | 0.7128 | 0.44614 | 0.1972 | 1,466 |
| synthetic | academic_reasoning | 0.7217 | 0.41790 | 0.2142 | 1,398 |
| synthetic | multi_agent_coordination | 0.6548 | 0.50417 | 0.2010 | 1,295 |
| synthetic | scientific_reasoning | 0.7002 | 0.45280 | 0.2000 | 1,254 |
| synthetic | logical_inference | 0.7102 | 0.42205 | 0.2467 | 1,142 |
| synthetic | commonsense_decision | 0.6655 | 0.47433 | 0.1788 | 1,139 |
| synthetic | medical_triage | 0.6922 | 0.43217 | 0.1993 | 952 |
| benchmark | swag | 0.4277 | 0.54565 | 0.1587 | 816 |
| synthetic | legal_judgment | 0.6295 | 0.50183 | 0.1549 | 556 |
| benchmark | hellaswag | 0.5803 | 0.46716 | 0.1330 | 498 |
| benchmark | agnews | 0.6839 | 0.36088 | 0.0895 | 386 |
| benchmark | multirc | 0.7826 | 0.47386 | 0.3488 | 299 |
| benchmark | banking77 | 0.2891 | 0.08626 | 0.2668 | 294 |
| benchmark | tabfact | 0.8481 | 0.42415 | 0.1359 | 283 |
| synthetic | long_document | 0.6333 | 0.51422 | 0.1313 | 240 |
| benchmark | fever | 0.6575 | 0.49008 | 0.1191 | 219 |
| benchmark | piqa | 0.6348 | 0.71951 | 0.2048 | 178 |
| benchmark | logiqa | 0.3594 | 0.56171 | 0.0701 | 128 |
| benchmark | stsb | 0.2975 | 0.40560 | 0.0541 | 121 |
| benchmark | winogrande | 0.5500 | 0.69918 | 0.0985 | 100 |
| benchmark | yelp | 0.4624 | 0.40597 | 0.1781 | 93 |
| benchmark | apigen | 0.882 | — | — | 2,483 |

Two caveats apply. banking77's 0.289 is confounded with its 80-option set, so it is not a pure source effect. And the synthetic families are 82.1% of the validation rows, so per-source rows are not equal-weight evidence.

## JevBench public 231: frozen against adapted

Both columns use the same backbone and readout. `frozen` is the unchanged Qwen3-Reranker-4B judgement readout; `adapted` adds the LoRA adapter above. Labels are corrected (see the next section). `cc` is chance-corrected accuracy with the set's chance of 0.3176.

| arm | correct | accuracy | cc | easy | hard | original |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 149/231 | 0.6450 | 0.4798 | 1.0000 | 0.4324 | 0.7361 |
| adapted | 135/231 | 0.5844 | 0.3910 | 1.0000 | 0.3514 | 0.6667 |

Hard is 48 of 111 against 39 of 111; original is 53 of 72 against 48 of 72.

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

The numbers separate by pass. On the deterministic re-score, the buggy labels put the frozen arm at 126/231 (cc 0.3339); the corrected labels put the same pass at 149/231 (cc 0.4798), so the label fix alone accounts for 0.3339 to 0.4798. The earlier in-run aggregate with the buggy labels was 124/231 (cc 0.3212), which differs from the re-score by the usual one or two items and is not a second measure of the fix. The noul cells that were reported through this path are superseded: frozen noul 24/74 becomes 47/74, adapted noul 28/74 becomes 45/74, and the aggregate flips above replace the earlier ones. The audit in #378 checked every committed JevBench JSON for the fingerprint (gold index 1 on every noul row) and found none, because the committed files pass through `load_jevbench()`, which maps "yes" to `True` before `candidates()` sees it. No earlier research note requires a superseded marker. The `148/231` figure in the marker-ablation note is a different arm, scored through `load_jevbench`, and is not the same measurement as the `149/231` frozen cell here.

## Mechanism: sharper margins, worse ranking

The adapter raises the decision margin without improving the ordering. The claim rests on three measurements: median raw top-two gap 1.188 to 3.938; hard accuracy 0.4324 to 0.3514; and the learned scale 0.7418 to 0.2028 at comparable calibration targets. Together they show confidence rising while accuracy falls.

The fixed-threshold count points the same way but is not evidence. On the hard tier, items with an effective gap above 0.5 go from 10/111 to 42/111, and that count is scale-dependent: the adapted arm's median raw gap is 3.3x larger, so more items cross a fixed threshold mechanically. The scale-free version splits each arm at its own gap quartiles: the most confident quarter is wrong on 2/58 items in the frozen arm and 3/58 in the adapted arm, 3.4% against 5.2%. That is itself evidence that the damage is in the ordering, not in overconfidence: the arm is not more wrong where it is most sure, it is more sure everywhere.

The learned scale on the same readout is 0.7418 in A against 0.2028 in C, from each run directory's `history.json`. Both arms calibrate toward the same targets, so a scale 3.7x smaller implies C's raw margins are about 3.7x larger than A's. That is independent evidence for the sharpening, computed without the JevBench items.

The attribution is the objective, not the data: per-option BCE rewards the magnitude of the correct log-odds margin, and a larger margin on a wrong option still lowers the per-option loss even when the argmax over options stays wrong. The adapter therefore learns to be decisive without learning to rank, which is the opposite of a calibrated transfer.

## Calibration is not the explanation

The calibration gate is not met: ECE 0.1662 overall, noul 0.2115, choice 0.1366, score 0.0826. That is a real calibration defect, but it cannot explain the JevBench regression. Each JevBench decision is an argmax over at least two options; the learned scale is positive (final value 0.2028); sigmoid followed by a positive affine map is monotone per item, so no affine recalibration can change a decision. The flip tables with and without the learned scale and bias are identical. The cause is the LoRA weights.

## Marker token, cross-reference

The marker-token ablation (#362, merged as #372) trains six heads identical except for the marker id. All five non-default markers are unresolved against the default `<|im_end|>`; the largest raw delta is +1.30 pp, the realized 80% minimum detectable effect is 2.8-4.3 pp, and the pre-registered adoption bar remains 9.3 pp. The swap pass moves no cell by more than +1.73 pp. The head reads the option-boundary position, not the token identity. The cells are in `research/44-marker-token-ablation.md`; they are one seed per arm and are not repeated here.

## Throughput

The run logs give 52.62 rows/s for C over 166,208 rows and 2,597 steps, and 28.4 rows/s for A over 19,968 rows and 312 steps. A warm fixed-row scaling probe reported 7.8x from one rank to eight (9.54 to 74.37 rows/s, with per-step cost near 0.86 s at every world size), but its output was not retained, so that figure is unverified; an earlier 1.0x reading was cold-shape compilation per process, not a scaling limit. No utilisation cell is reported: the per-card traces were not retained, and the two reported traces come from different legs, 399 rows on one card against 176 one-Hz samples of all eight cards. Units are rows per second, one row being one judgement item with its option set.

## 20k subset against the full corpus

Arm A draws a deterministic 20,000-row subset and runs one epoch in 312 logged steps, 19,968 rows in whole steps; arm C is the full corpus at 2,597 steps. Labels are corrected throughout, and both arms are scored against the same frozen baseline.

| arm | JevBench correct | accuracy | cc | easy | hard | original | val loss | val accuracy | val ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A, 20k | 150/231 | 0.6494 | 0.4862 | 47/48 | 48/111 | 55/72 | 0.51037 | 0.62730 | 0.1681 |
| C, full | 135/231 | 0.5844 | 0.3910 | 48/48 | 39/111 | 48/72 | 0.47426 | 0.66750 | 0.1662 |

A's own frozen pass reproduces 149/231 exactly, so the contrast is not a scoring artefact. Paired against the frozen arm, A is net +1 (10 lost against 11 gained) with exact McNemar p = 1.0; C is net -14 with p = 0.0161.

By question type the advantage is in choice: A is choice 92/139, noul 48/74, score 10/18; C is choice 80/139, noul 45/74, score 10/18; the frozen readout is choice 89/139, noul 47/74, score 13/18 in both runs. The adapter therefore damages the score head at either data size, 10/18 against 13/18, and that is a separate finding.

The confound is explicit: 312 versus 2,597 optimizer steps, so the difference is data volume confounded with step count and does not isolate a data effect. What the comparison shows is the direction: the smaller, shorter arm is 15 items better, and it is the only one of the two not below its own frozen baseline. More data and more steps together made the result worse, which is evidence against a pure training-length artefact but is not proof.

On distribution, A fits the training distribution worse (validation loss 0.51037 against 0.47426) and still transfers better, so the effect is not explained by fit. Status: measured; neither run log is committed.

## Provenance

| number group | status | source |
|---|---|---|
| arm definition, train loss, validation table, by-source table, mechanism gaps, ECE, throughput | measured; run directory and eight eval shards on the run host, not committed | C, 2026-11-15, seed 42: 166,208 rows, 2,597 steps, mean train loss 0.48022 |
| JevBench corrected cells, flips, decisive subset, by-type cells, scale-free quartiles | measured; the label correction is PR #378, unmerged at the time of writing; per-item margins retained on the run host, not committed | #378, recomputed from the run's per-item records |
| marker-token ablation | measured; committed | #372 and `research/44-marker-token-ablation.md` |
| arm A, 20k subset | measured; run directory on the run host, not committed | A, 2026-11-15, seed 42: 20,000 rows (19,968 in 312 whole steps), mean train loss 0.5128, final logged step loss 0.5134 at step 310 |

The merged validation report is about 8 MB. JevBench per-item margins are retained; validation per-item margins are not. Committing the validation JSON and the merged report in a follow-up would give the first two rows artifact provenance. The repository currently holds the method for the correction, not the run's own records.

## Scope

One backbone, one adapter position, one readout, two corpus sizes, one seed, one epoch. The JevBench number is the local 231-item proxy, not the official Intelligence metric. Nothing here measures a second seed, a different adapter position, or a trained head on top of the adapted backbone. The corpus-size contrast is the confounded A-against-C comparison of the previous section, not a data-effect estimate.
