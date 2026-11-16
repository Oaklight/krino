# Full-parameter fine-tuning, the admission suite, and the score ceiling — 2026-11-16

Status: measured, one seed, one short schedule. Full-parameter training at 1e-5 is the first arm to clear every gate of the admission suite, and the suite now has measured floors and a measured noise band. Every panel-2 arm sits between 0.4662 and 0.4977 on the 1,111 score rows while capacity moves benchmark val by 11.4 pp and score by 12 items; the ceiling is high and flat, not untouched. Full-corpus scaling has no result yet and is unverified.

**PRs:** #387, open at the time of writing and not yet merged.

Every arm here uses Qwen3-Reranker-4B, at most 4,096 tokens per row, the panel-2 readout with listwise cross-entropy over the K code logits, the v3 a20k plan, 303 steps and seed 42 unless a cell says otherwise. JevBench is the local 231-item public proxy, reported as natural order, reversed order and their average; the frozen average is 159.0. "Benchmark val" in this note means every non-synthetic validation row, 9,923 after the 639-row `answerable:false` drop; that set includes typed and apigen rows, not only the benchmark-named sources. The frozen reference there is 5,652/9,923 = 0.5696, the same figure the previous note uses. Paired tests are exact two-sided McNemar, and the paired n is given with each p.

## 1. Full-parameter arms

LoRA is off and every parameter is trainable, with activation checkpointing on.

| lr | val | natural | reversed | averaged | agreement | ties | K = 80 | ECE noul / choice / score |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1e-5 | 0.6833 | 165 | 161 | 163.0 | 0.8831 | 7 | 0.3281 | 0.0440 / 0.0560 / 0.1652 |
| 2e-5 | 0.6845 | 163 | 159 | 161.0 | 0.9048 | 1 | 0.3596 | 0.1286 / 0.1253 / 0.3691 |
| 5e-5 | 0.5536 | 131 | 128 | 129.5 | 0.7835 | 0 | 0.1348 | +0.127 / +0.148 over frozen |

In the table, agreement is order self-consistency: the count of items whose natural-order and reversed-order choices name the same option, over the 231 items, with ties included. It is not agreement with any teacher.

The ECE column is absolute for the first two arms and a change over frozen for 5e-5, where the deltas are noul +0.127 (0.2679 against 0.1406) and score +0.148 (0.3752 against 0.2269). The choice cell is 0.0770 against frozen 0.2787, a 0.202 improvement, so it is the one ECE cell that improves and it is not a positive delta. The frozen ECE on the same rows is noul 0.1406, choice 0.2787 and score 0.2269. Every ECE cell rests on the 9,923-row benchmark comparator: the frozen and LoRA cells come from a separate pass over exactly those rows, and each of the four full-parameter runs has in-run bins that sum to exactly 9,923 rows, noul 3,482, choice 5,330 and score 1,111, identical to the comparator per type. The four runs are 1e-5, 1e-5_w, 2e-5 and 5e-5; 1e-5 and 1e-5_w are the same arm under two loss settings, which is why the table lists three arms and the bin count names four runs. The filter concern raised in #387's review does not change these cells.

The sweet spot is narrow. 1e-5 and 2e-5 both beat frozen on benchmark val by more than 11 pp and on JevBench by 2 to 4 items averaged; 5e-5 collapses on val, averaged JevBench and K = 80, while it still passes the agreement floor at 0.7835 against 0.7206 and improves the choice ECE at 0.0770 against 0.2787. Between the two good arms the choice is not made by val, which is 0.0012 apart, but by the per-cell ECE: 1e-5 is better on all three cells than both 2e-5 and frozen, and 2e-5 is worse than frozen on score.

The 1e-5 cells above are from the `p2_fullft_lr1e-5` launch, which saved no weights. Its re-scorable twin, `p2_fullft_lr1e-5_w`, runs the same configuration with weights saved: val 6,821/9,923 = 0.6874, averaged JevBench 163.5 (natural 166/231, reversed 161/231), agreement 204/231 = 0.8831 with 2 ties, K = 80 at 142/445 = 0.3191, and ECE 0.0462 / 0.0760 / 0.1817. All five gates pass for `_w` as they do for the launch quoted above, so the admission verdict does not depend on which launch is read. From here `_w` is the canonical artifact for later comparisons, because it carries weights and can be re-scored; the score-loss arms use it as their control.

## 2. Admission suite outcomes

The suite now has measured floors, all taken from the frozen readout on the same rows. The frozen row is natural 162/231, reversed 156/231, averaged 159.0; val 5,652/9,923 = 0.5696; agreement 178/231 = 0.7706 with 3 exact ties; K = 80 at 116/445 = 0.2607; ECE noul / choice / score at 0.1406 / 0.2787 / 0.2269.

| measure | frozen | floor |
|---|---:|---:|
| averaged JevBench | 159.0 | at least 156.0 |
| natural-versus-reversed agreement | 0.7706 (178/231) | at least 0.7206 |
| benchmark val, 9,923 rows | 0.5696 (5,652) | at least 0.5596 |
| K = 80 exact | 0.2607 (116/445) | at least 0.2407 |
| ECE, per cell | 0.1406 / 0.2787 / 0.2269 | at most 0.1606 / 0.2987 / 0.2469 |

The val gate is applied against 5,652/9,923 = 0.5696, the frozen cell count. The floor is therefore 0.5596.

Three arms are admissible: full fine-tuning at 1e-5, LoRA panel-2 on v3 at 3e-5, and LoRA panel-2 on the old plan at 1e-4. The v3 3e-5 arm is provisional at this point of use. Its reversed pass was run, because every panel-2 arm's JevBench evaluation scores both orders in the same process, and it records 153/231 reversed, 157.0 averaged and 174/231 agreement with 21 exact ties, which clears the 156.0 floor on the average. The only record is the run-directory payload: it is uncommitted and not reproducible from the tree, so the values are unverified and 157.0 is not cited as a committed result. The verdict carries the previous note's proviso and is not a settled admission. The outcomes are unverified at point of use; the eval artifacts are on the run host and are not committed.

Seven are rejected, each for the gate that ended it.

| arm | failing gate |
|---|---|
| LoRA v3, 1e-4 | averaged JevBench 149.0 against 156.0; agreement 0.6104 against 0.7206 |
| benchmark-only, 3e-5 | averaged JevBench 155.5 against 156.0, half an item below and inside the 0.5-item noise band |
| benchmark-only, 1e-4 | averaged JevBench 154.5 against 156.0 |
| LoRA old plan, 3e-5 | val 0.5510 against 0.5596; agreement 0.7186 against 0.7206 |
| LoRA v3, 3e-4 | averaged JevBench 150.0, val 0.4956, agreement 0.5498, K = 80 0.0382; ECE unmeasured |
| full FT, 2e-5 | score ECE 0.3691 against the 0.2469 floor, a gap of 0.122, and nothing else |
| full FT, 5e-5 | averaged 129.5, val 0.5536, noul ECE 0.2679, score ECE 0.3752, K = 80 0.1348 |

Only the JevBench allowance has a basis in the measured band: 3 items is half the 6-item order swing, and the 0.5-item band is why the 155.5 rejection is reported as inside noise rather than as a clean miss. The other floors are policy choices, not band multiples: benchmark val within 1 pp, per-cell ECE within 0.02, and the agreement and K = 80 floors anchored at the frozen values and set 0.05 and 0.02 below them, both wider than the measured band.

The rejections are the useful half of the suite: each names the measure that ended the arm, so a later recipe can be aimed at a specific gate rather than at the aggregate.

## 3. Noise floor, measured

Two runs of one configuration differed on 1,017 of 41,232 validation rows, on 6 of 231 JevBench items, and on the epoch mean loss, 0.9003 against 0.8942. The trajectories separate by step 25. The cause is training nondeterminism, suspected to be attention-backward atomics; that is unconfirmed. With a third run the band is 0.52 pp on benchmark val and 0.5 items on averaged JevBench.

The band is measured on 41,232 scored validation rows. The plan's val split is 41,871 rows; the loader's `answerable:false` rule removes 639 of them, which is the only difference between the two counts. The `--max-row-tokens` cap applies to train rows and is not the cause of the 639: it skips 180 of 69,644 train rows. The band and the projection therefore do not share a denominator, and every band figure here is on the 41,232 scored rows. There is no seed axis in this loop: it has no shuffle, dropout or random draw, so changing the seed reproduces the same trajectory. Every comparison in this note is read against the measured band, not against zero.

## 4. The score ceiling

On the 1,111 benchmark score rows, the panel-2 arms land between 0.4662 and 0.4977 exact. Frozen panel-2 is 518/1,111 = 0.4662, the best LoRA arm is 553/1,111 = 0.4977 (benchmark-only at 1e-4) and full fine-tuning at 1e-5 is 0.4770. The judgement readout sits below that band on the same rows: frozen 0.4455, best arm 0.4518, and two arms are significantly worse with exact McNemar on the 1,111 rows: p = 0.00750 for v3 at 1e-4, which lost 121 and gained 82, and p = 0.00082 for the old plan at 1e-4, which lost 133 and gained 83. Both come from the judgement per-K validation pass.

The benchmark-only arm's lead over frozen on these rows is significant: 52 lost against 87 gained, exact McNemar p = 0.00377. Per K it is positive but not individually significant: K = 4 is 18 against 31 with p = 0.085, K = 5 is 19 against 31 with p = 0.119, K = 6 is 15 against 25 with p = 0.154.

The direct data-version pair at the same learning rate sharpens the reading. v3 at 1e-4 against benchmark-only at 1e-4 on the same rows is 72 against 103, exact McNemar p = 0.0231; at 3e-5 it is 41 against 51, p = 0.348. Full fine-tuning at 1e-5 against frozen is 225 against 237, p = 0.609, so the full-parameter score movement is noise. On score rows the significant effect is the data version, not capacity: capacity does not move score, and the synthetic share costs about 31 net items at 1e-4 (103 against 72).

The caveat is that benchmark-only at 1e-4 is not an admissible arm: its averaged JevBench is 154.5 against the 156.0 floor. The finding is a mechanism result about synthetic rows, not a shippable arm.

Full fine-tuning moves exact by +1.1 pp, which its own pair puts at p = 0.609, and within-one by +1.9 pp. It moves the mass by under two points rather than reshaping it: 1.9 pp of predictions leave the two-or-more-levels group, and adjacent misses go from 22.8 to 23.6 pp. The score cells are unverified at point of use, as the artifacts sit on the run host. K = 6 is the worst cell in every arm.

## 5. Decode

Rounding the expected level loses decisively on both models: 0.3501 against 0.4770 for full fine-tuning at 1e-5, and 0.3780 against 0.4662 for frozen, with argmax-right/rounding-wrong counts of 201 against 60 and 149 against 51. Argmax stands, and expected-level decoding is excluded as an alternative.

## 6. Score-target arms, implemented and unrun

`--score-target {onehot,soft,expected}` applies an ordinal target to score rows only; choice and noul keep the shipped loss bit for bit, and argmax decodes in all modes. Soft uses exp(-|j-gold|), expected uses |sum_j j p_j - gold|.

The reading is pre-registered: exact accuracy on the 1,111 rows improves at p < 0.05, with mean level error no worse and every gate in section 2 kept. If the change moves only adjacent misses, that is a mechanism result rather than a ceiling break. No arm has been run in any mode, so every cell in this section is unrun and unverified.

## 7. Full-corpus scaling (unverified)

The step time is not constant and is not a property of the row mix. Leg 1 on rank 0 records cumulative elapsed 315 s at step 25, 578 s at 50, 807 s at 75 and 1,013 s at 100, which gives window rates of 12.60, 10.52, 9.16 and 8.24 s per step. The rate drifts down as start-up amortises, the last window matches the a20k weights run at 8.17 s, and an earlier killed run shows the same shape, 311 s and 573 s at steps 25 and 50. The earlier 11.3 s figure was start-up amortisation, not a longer row mix.

The projection this note previously carried, 3.96 to 4.52 h, was built from a per-step time times 1,086 steps, which ignores length ordering. The plan orders rows by length, so the early steps are cheap and the later ones are not. Leg 1 moved 10,319,786 tokens in 4,426 s of in-step clock, 2,331.6 tokens per second. Integrating the epoch's 33,098,619 tokens at that rate gives 14,195 s = 3.94 h of steps, and about 4.45 h with this cadence and the checkpoint and load terms. That lands at the top of the old range, not below it: the old basis was wrong, but the old number was not uniformly low.

The integration is arithmetic at one rate, not a new epoch figure and not a bound. The window rate rose from 2,089 to 5,495 tokens per second net across the leg, so the leg average is pulled down by start-up, and leg 2's rate is unmeasured with rows that carry 2.2x the tokens per step. The epoch wall time is therefore unmeasured.

Leg 1 itself is measured: 79.2 min of launcher wall for 543 steps and 10,319,786 tokens, 31.18% of the epoch, from 16:57:47Z to 18:16:57Z. About 2 min of that is start-up before the first step and about 3 min is the final validation pass, JevBench, weights write and payload; the trainer's in-step clock at step 543 reads 4,426 s = 73.8 min. That measures the first half, not the epoch.

The checkpoint cost is 3.2 to 3.8 min: 192 s at step 250, 223 s at 375 and 230 s at 500. The first checkpoint, at step 125, took about 700 s and is an outlier whose cause is not established; a cold page cache for the first 8.8 GB weights write and the first evaluation's allocator warm-up are the candidates, and that is an inference, not a measurement. Every checkpoint did the full work: eight rank files covering the 41,232 scored validation rows, JevBench on rank 0, and an 8.8 GB weights file. The 13.6-min figure that circulated earlier was not a measured cost: 931 s is 15 min 31 s and 803 s is 13 min 23 s, neither of them 13.6 min, so 13.6 was an interval estimate near the 250-to-375 span.

The row cap skips 180 of 69,644 train rows, 0.26%, and 2,169 train rows are masked as `answerable:false`. Every evaluation forwards the 41,232 scored validation rows; 41,871 is the plan's val split before the 639-row `answerable:false` drop, so the two counts are the same population at different stages. The run is executed in two legs of 543 steps with checkpoint evaluation every 125 steps and a resume path. The legs are separate processes: `optimizer_state: fresh`, because `--load-weights` starts AdamW cold and step 544 is not a continuation of step 543. No epoch result exists yet; the epoch wall time remains unverified, while leg 1's 79.2 min, the row-cap counts and the checkpoint cost are measured.

## 8. Mechanism

Full-parameter training needs activation checkpointing because a single oversized row cannot be bounded by a token budget when rows are processed one at a time. The longest row still peaks at 83.84 GiB with checkpointing on, which is why a 4,096-token cap exists. The cap is a memory device, not a data choice.

## 9. Engineering record

Four pieces landed in the working tree in this stretch: checkpoint evaluation, weight saving with a streamed sha256, resume with a global warmup step, and plan immutability with versioned paths. The code backing them, including the score-target flag of section 6, is PR #387, open at the time of writing and not yet on main.

## Provenance

| number group | status | source |
|---|---|---|
| full-parameter arms, admission outcomes, noise floor, score ceiling, decode | measured; run logs, eval shards and `SUITE_CELLS.json` on the run host, not committed | runs of 2026-11-16, seed 42 |
| pre-registered score-target reading | unrun; code in the working tree, no result | section 6 |
| full-corpus scaling | partially measured; leg 1 wall time, token shares and checkpoint cost measured, epoch wall time unverified | section 7 |
| engineering record | in the working tree; not merged at the time of writing | section 9 |

## Scope

One backbone, one readout, one plan, one seed, a 303-step schedule, and the local 231-item proxy. The noise band is measured on one configuration with three runs. Nothing here measures a full-corpus epoch, a score-target arm, a second seed, or the official JevBench Intelligence score.
