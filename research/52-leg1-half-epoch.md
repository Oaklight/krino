# The first half of a full-parameter epoch on the panel-2 readout — 2026-11-16

Status: measured, one leg of a planned two-leg run. Leg 2 was skipped, so this is the first 543 of 1,086 steps and 31.18% of the epoch's tokens. At the exit checkpoint the arm passes all five admission floors while still missing the internal calibration target, and the order-consistency dip seen at the first checkpoint recovers by the middle of the leg.

Every row here uses Qwen3-Reranker-4B with every parameter trainable (LoRA off), the panel-2 readout with listwise cross-entropy, the v3 a20k plan, seed 42, activation checkpointing on, and at most 4,096 tokens per row. JevBench is the local 231-item public proxy, reported as natural, reversed and their average; the frozen average is 159.0. "Benchmark val" means every non-synthetic validation row, 9,923 after the 639-row `answerable:false` drop, including typed and apigen rows. Agreement is order self-consistency: the count of items whose natural-order and reversed-order choices name the same option, over 231, ties included, not agreement with any teacher. The frozen reference is val 0.5696, averaged JevBench 159.0, agreement 178/231 = 0.7706, K = 80 at 116/445 = 0.2607, and ECE 0.1406 / 0.2787 / 0.2269.

## 1. Run and schedule

Leg 1 ran steps 1 to 543 of 1,086, from 16:57:47Z to 18:16:57Z: 79.2 min of launcher wall time, of which the trainer's in-step clock at step 543 reads 4,426 s = 73.8 min. The leg covers 10,319,786 tokens, 31.18% of the epoch's 33,098,619, over 50.0% of its steps. The plan orders rows by length, so the second half carries 2.2x the tokens per step, 41,949 against 19,005, and leg 1 is by construction the cheap half. Leg 2 was skipped, so no epoch result exists.

## 2. Checkpoints

| step | val | natural | reversed | averaged JevBench | agreement | ties | K = 80 | ECE noul / choice / score |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 125 | 0.6113 | 162 | 153 | 157.5 | 156/231 = 0.6753 | 14 | 0.3303 | 0.0563 / 0.0572 / 0.0216 |
| 250 | 0.6261 | 164 | 152 | 158.0 | 142/231 = 0.6147 | 17 | 0.3888 | 0.1520 / 0.0358 / 0.0544 |
| 375 | 0.6781 | 166 | 165 | 165.5 | 196/231 = 0.8485 | 12 | 0.3798 | 0.0379 / 0.1077 / 0.0756 |
| 500 | 0.6667 | 167 | 164 | 165.5 | 194/231 = 0.8398 | 7 | 0.3798 | 0.1024 / 0.1643 / 0.1333 |
| 543 | 0.6725 | 168 | 163 | 165.5 | 195/231 = 0.8442 | 12 | 0.3820 | 0.0874 / 0.1197 / 0.0633 |

Within the plateau from step 375 to 543 the averaged JevBench is flat at 165.5 while natural rises from 166 to 167 to 168 and reversed falls from 165 to 164 to 163, so the natural-minus-reversed gap narrows to 1 item at 375 and then widens to 3 and 5. The average is the admission metric, so no admission verdict moves; what the average hides is increasing order asymmetry at a fixed average. Across the whole leg the gap is not monotone, 9 items at 125 and 12 at 250 before 1 at 375, so the gap narrows and then widens again within the plateau. A mechanism such as the model leaning more on natural-order position as training continues is a hypothesis, not a finding.

## 3. Best checkpoint and exit

Step 375 is the best checkpoint on val, 0.6781, and ties the exit on averaged JevBench at 165.5. The exit at 543 is 0.6725 on val, 165.5 averaged, agreement 0.8442, K = 80 0.3820 and ECE 0.0874 / 0.1197 / 0.0633.

The 375-against-543 paired test on the 9,923 validation rows is b = 729 (375 right, 543 wrong), c = 673 (543 right, 375 wrong), 8,521 unchanged, exact McNemar p = 0.1418. The 56-row deficit is not distinguishable from noise, so the exit is not a meaningful loss against the best checkpoint, and 375 is a checkpoint choice rather than a better arm.

## 4. Admission at the exit

At step 543 the arm clears all five floors: averaged JevBench 165.5 against 156.0; val 0.6725 against 0.5596; agreement 0.8442 against 0.7206; K = 80 0.3820 against 0.2407; ECE 0.0874 / 0.1197 / 0.0633 against 0.1606 / 0.2987 / 0.2469. The half epoch is therefore a new admissible arm. Step 375 clears the same five floors, with val 0.6781 and agreement 0.8485.

Admission is not the internal calibration gate. The working target inside the calibration work is ECE at or below 0.05 per cell, and that target is not met here.

## 5. Calibration

At step 543 all three ECE cells are above the 0.05 working target, 0.0874 / 0.1197 / 0.0633. At step 375 only noul is below it, 0.0379, while choice is 0.1077 and score is 0.0756. The correct summary is that choice and score are above 0.05 at both checkpoints, and noul is above it only at the exit.

Both statements can hold at once: the arm passes the admission floors, which are frozen-relative, and misses the absolute 0.05 target. A later note must not read one as the other.

## 6. The agreement dip is transient

Order self-consistency falls to 156/231 = 0.6753 at step 125, below the 0.7206 floor, then recovers to 196/231 = 0.8485 at step 375 and holds at 195/231 = 0.8442 at the exit. Exact ties move 14, 17, 12, 7 and 12 across the five checkpoints.

The mid-run dip is therefore transient, not a persistent failure: the arm recovers 40 items of self-consistency within the leg and finishes 17 above the frozen baseline of 178/231. The first checkpoint alone would have supported the opposite reading.

## 7. Checkpoint cost

A checkpoint evaluation costs 3.2 to 3.8 min: 192 s at step 250, 223 s at 375 and 230 s at 500. The first checkpoint, at step 125, took about 700 s and is an outlier whose cause is not established; a cold page cache for the first 8.8 GB weights write and the first evaluation's allocator warm-up are the candidates, and that is an inference. Every checkpoint did the full work: eight rank files covering the 41,232 scored validation rows, JevBench on rank 0, and an 8.8 GB weights file. The 13.6-min figure that circulated earlier was not a cost; it was the wall interval between checkpoint files, 931 s between steps 125 and 250 and 803 s between 250 and 375, which contains the training steps in between.

## Provenance

| number group | status | source |
|---|---|---|
| schedule, wall time, token shares | measured | leg 1 launcher timestamps and the trainer's in-step clock |
| checkpoint cells, McNemar, checkpoint cost | measured; exit report on the run host, not committed | exit report and run payloads, seed 42 |
| natural and reversed splits at all five checkpoints | measured; exit report on the run host, not committed | exit report, seed 42 |

## Scope

One backbone, one readout, one plan, one seed, half an epoch. Leg 2 was skipped, so nothing here measures a full epoch, a completed convergence curve, or a second seed. The local 231-item proxy is not the official JevBench Intelligence score.
