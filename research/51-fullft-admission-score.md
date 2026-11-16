# Full-parameter fine-tuning, the admission suite, and the score ceiling — 2026-11-16

Status: measured, one seed, one short schedule. Full-parameter training at 1e-5 is the first arm to clear every gate of the admission suite, the suite itself now has measured floors and a measured noise band, and the score head is shown to sit on a ceiling that no arm in this stretch moves. Full-corpus scaling has no result yet and is unverified.

**PRs:** none confirmed for this note; the engineering items in section 9 were not merged when the note was written.

Every arm here uses Qwen3-Reranker-4B, at most 4,096 tokens per row, the panel-2 readout with listwise cross-entropy over the K code logits, the v3 a20k plan, 303 steps and seed 42 unless a cell says otherwise. JevBench is the local 231-item public proxy, reported as natural order, reversed order and their average; the frozen average is 159.0. Benchmark validation is 9,923 rows, and the frozen reference there is 0.5697. The previous note quotes 0.5696 for the same config; the two are separate frozen runs and the difference is inside the noise band of section 3, so each note keeps its own figure rather than a reconciled one. Paired tests are exact two-sided McNemar, and the paired n is given with each p.

## 1. Full-parameter arms

LoRA is off and every parameter is trainable, with activation checkpointing on.

| lr | val | natural | reversed | averaged | agreement | ties | K = 80 | ECE noul / choice / score |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1e-5 | 0.6833 | 165 | 161 | 163.0 | 0.8831 | 7 | 0.3281 | 0.0440 / 0.0560 / 0.1652 |
| 2e-5 | 0.6845 | 163 | 159 | 161.0 | 0.9048 | 1 | 0.3596 | 0.1286 / 0.1253 / 0.3691 |
| 5e-5 | 0.5536 | 131 | 128 | 129.5 | 0.7835 | 0 | 0.1348 | +0.127 / +0.148 over frozen |

The ECE column is absolute for the first two arms and a change over frozen for 5e-5, where the deltas are noul +0.127 (0.2679 against 0.1406) and score +0.148 (0.3752 against 0.2269). The choice cell is 0.0770 against frozen 0.2787, a 0.202 improvement, so it is the one ECE cell that improves and it is not a positive delta. The frozen ECE on the same rows is noul 0.1406, choice 0.2787 and score 0.2269.

The sweet spot is narrow. 1e-5 and 2e-5 both beat frozen on benchmark val by more than 11 pp and on JevBench by 2 to 4 items averaged; 5e-5 is a collapse on every measure. Between the two good arms the choice is not made by val, which is 0.0012 apart, but by the per-cell ECE, where 1e-5 is better on two cells and 2e-5 is worse on score than frozen.

## 2. Admission suite outcomes

The suite now has measured floors, all taken from the frozen readout on the same rows: averaged JevBench at least 156.0, natural-versus-reversed agreement at least 0.7206, benchmark val within 1 pp of frozen, the long-list K = 80 cell at least 0.2407, and every ECE cell within 0.02 of its frozen value.

Three arms are admissible: full fine-tuning at 1e-5, LoRA panel-2 on v3 at 3e-5, and LoRA panel-2 on the old plan at 1e-4.

Seven are rejected, each for a named reason. LoRA v3 at 1e-4 fails on averaged JevBench, 149.0 against the 156.0 floor, and on agreement, 0.6104 against 0.7206. The benchmark-only arms at 3e-5 and 1e-4 fail on averaged JevBench, 155.5 and 154.5. The old plan at 3e-5 fails on val, 0.5510 against 0.5596. v3 at 3e-4 fails on JevBench, val, agreement and K = 80, which is 0.0382. Full fine-tuning at 2e-5 fails on the score ECE, 0.3691, which is 0.142 over frozen. Full fine-tuning at 5e-5 fails everything. The rejections are the useful half of the suite: each names the one measure that ended the arm, so a later recipe can be aimed at a specific gate rather than at the aggregate.

## 3. Noise floor, measured

Two runs of one configuration differed on 1,017 of 41,232 validation rows, on 6 of 231 JevBench items, and on the epoch mean loss, 0.9003 against 0.8942. The trajectories separate by step 25. The cause is training nondeterminism, suspected to be attention-backward atomics; that is unconfirmed. With a third run the band is 0.52 pp on benchmark val and 0.5 items on averaged JevBench.

There is no seed axis in this loop: it has no shuffle, dropout or random draw, so changing the seed reproduces the same trajectory. Every comparison in this note is read against the measured band, not against zero.

## 4. The score ceiling

On the 1,111 benchmark score rows, every arm lands between 0.45 and 0.50 exact. Frozen panel-2 is 0.4662, the best LoRA arm is 0.4977 (benchmark-only at 1e-4) and full fine-tuning at 1e-5 is 0.4770. The judgement readout is behind everywhere on the same rows: frozen 0.4455, best arm 0.4518, and two arms are significantly worse with exact McNemar on the 1,111 rows: p = 0.00750 for v3 at 1e-4, which lost 121 and gained 82, and p = 0.00082 for the old plan at 1e-4, which lost 133 and gained 83. Both come from the judgement per-K validation pass.

Full fine-tuning moves exact by +1.1 pp and within-one by +1.9 pp, and it does not reshape the errors. Adjacent misses are 23.6 pp and 28.7% of predictions are two or more levels away, against 22.8 pp and 30.6% for frozen. K = 6 is the worst cell in every arm. The ceiling is therefore a property of the score formulation or the labels, not of the readout: full fine-tuning, which moves every other measure, moves this one by about a point.

## 5. Decode

Rounding the expected level loses decisively on both models: 0.3501 against 0.4770 for full fine-tuning at 1e-5, and 0.3780 against 0.4662 for frozen, with argmax-right/rounding-wrong counts of 201 against 60 and 149 against 51. Argmax stands, and expected-level decoding is excluded as an alternative.

## 6. Score-target arms, implemented and unrun

`--score-target {onehot,soft,expected}` applies an ordinal target to score rows only; choice and noul keep the shipped loss bit for bit, and argmax decodes in all modes. Soft uses exp(-|j-gold|), expected uses |sum_j j p_j - gold|.

The reading is pre-registered: exact accuracy on the 1,111 rows improves at p < 0.05, with mean level error no worse and every gate in section 2 kept. If the change moves only adjacent misses, that is a mechanism result rather than a ceiling break. No arm has been run in any mode, so every cell in this section is unrun and unverified.

## 7. Full-corpus scaling (unverified)

The measured cost is 11.3 s per step on this row mix against 8.17 on the a20k, over 1,086 steps, so the epoch projects to 2.9 to 3.2 hours of steps and 4 to 4.5 hours with eight checkpoint evaluations. The row cap skips 180 of 69,644 train rows, 0.26%, and 2,169 rows are masked as `answerable:false`.

The run is executed in two legs of 543 steps with checkpoint evaluation every 125 steps and a resume path. No result exists yet, and every figure in this section is unverified.

## 8. Mechanism

Full-parameter training needs activation checkpointing because a single oversized row cannot be bounded by a token budget when rows are processed one at a time. The longest row still peaks at 83.84 GiB with checkpointing on, which is why a 4,096-token cap exists. The cap is a memory device, not a data choice.

## 9. Engineering record

Four pieces landed in the working tree in this stretch: checkpoint evaluation, weight saving with a streamed sha256, resume with a global warmup step, and plan immutability with versioned paths. The code backing them, including the score-target flag of section 6, is on the trainer branch and not yet on main. No PR numbers are recorded here because none could be confirmed against the repository at the time of writing; the note names the behaviour and not a merge.

## Provenance

| number group | status | source |
|---|---|---|
| full-parameter arms, admission outcomes, noise floor, score ceiling, decode | measured; run logs, eval shards and `SUITE_CELLS.json` on the run host, not committed | runs of 2026-11-16, seed 42 |
| pre-registered score-target reading | unrun; code in the working tree, no result | section 6 |
| full-corpus scaling | unverified; projection from a step-time measurement, no epoch run | section 7 |
| engineering record | in the working tree; not merged at the time of writing | section 9 |

## Scope

One backbone, one readout, one plan, one seed, a 303-step schedule, and the local 231-item proxy. The noise band is measured on one configuration with three runs. Nothing here measures a full-corpus epoch, a score-target arm, a second seed, or the official JevBench Intelligence score.
