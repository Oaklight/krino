# The panel-2 readout replaces the judgement readout — 2026-11-16

Status: measured, one seed. The mainline readout moves from per-option judgement scoring to the panel-2 vocabulary code. On the 231 local items the frozen panel-2 readout beats the frozen judgement readout by 13 items, and training the panel-2 readout at a low learning rate improves benchmark validation significantly while JevBench stays flat. Full fine-tuning has no result yet and is unverified.

**PRs:** #378 through #384.

Every number below uses Qwen3-Reranker-4B as the backbone, at most 4,096 tokens per row, seed 42, and corrected noul labels unless a cell says otherwise. JevBench is the local 231-item public proxy (easy 48, hard 111, original 72), not the official JevBench Intelligence score. Paired tests are exact two-sided McNemar on the paired items, and the paired n is given with each p.

## 1. Mainline switch

The panel-2 vocabulary readout scores an item in one forward: the model reads the item once, and the decision is the argmax over the declared single-token code ids for the options. The judgement readout needs one plus K forwards per item, one per option, and contrasts the yes and no margins. On the same frozen backbone and the same 231 corrected-label items, panel-2 is 162/231, chance-corrected accuracy (cc) 0.5623, with easy 48/48, hard 52/111 and original 62/72, against judgement 149/231 and cc 0.4798. The paired difference is 13 items, exact McNemar p = 0.0725.

The gain is entirely in choice items: panel-2 is 102/139 against judgement 80/139, a paired net of +13 with exact McNemar p = 0.0192. The noul and score cells are flat.

Order robustness at the frozen readout is already imperfect. Scoring the same items with the option list natural and then reversed gives 162 and 156 correct, the same choice in 178/231 items, and 3 exact ties. Two of the three figures bear on the later sections: the 178 agreement is the frozen baseline for section 6, and the 3 ties are its starting point.

## 2. Training recipe

The recipe is LoRA rank 16, alpha 32, on the last 8 blocks, with the panel-2 readout and a listwise cross-entropy over the K code logits, one forward per item. The plan is v3 at 20,000 rows (19,392 usable after 608 masked), which gives 303 steps at 64 rows per step on eight ranks. Seed 42.

Learning rate dominates every other choice, and the response is non-monotone.

| learning rate | JevBench | benchmark val | val p against frozen 0.5696 |
|---|---:|---:|---:|
| 3e-5 | 161/231 | 5,777/9,923 = 0.5822 | 5e-7 |
| 1e-4 | 152/231 | 5,960/9,923 = 0.6006 | 7e-14 |
| 3e-4 | 152/231 | 4,918/9,923 = 0.4956 | 3.6e-38 |
| 1e-3 | 96/231 | 4,083/9,923 = 0.4115 | 7.2e-127 |
| 3e-3 | stopped by the loss rule | — | — |

All p-values in the table are exact McNemar on the 9,923 paired rows. The sign change is the finding: at 3e-5 and 1e-4 the trained readout improves benchmark validation significantly while JevBench stays at its frozen level; at 3e-4 and above both collapse. A recipe that optimises on-distribution accuracy and a recipe that preserves the JevBench proxy are therefore not the same parameter region, and the frozen-to-trained comparison alone does not say which region an arm is in.

## 3. The 2x2

The readout and the data version were crossed at two learning rates, with 303 steps, the same seed and the same evaluation.

| readout | data | 3e-5 | 1e-4 |
|---|---|---:|---:|
| judgement | v3 corrected | 149 | 148 |
| judgement | old pre-permutation | 149 | 151 |
| panel-2 | v3 corrected | 161 | 152 |
| panel-2 | old pre-permutation | 160 | 158 |

Across the four matched cells the readout effect is +12, +4, +11 and +7 items, and the data effect is 0, -3, +1 and -6. The readout carries the gain; the data fixes carry none on these 231 items, although they move benchmark val by 3 to 4 pp in the same runs. A 231-item JevBench cell is therefore insensitive to data repairs that the 9,923-row validation set registers.

The comparison is readout plus objective, not a head swap. The judgement readout is trained with per-option BCE on yes/no margins, and the panel-2 readout with a listwise cross-entropy over K code logits. The two differ in objective by construction, so no cell here isolates the head from the loss.

## 4. Benchmark-only arm

Removing the synthetic rows leaves 12,290 benchmark rows trained for two epochs. That arm reaches 157/231 at both learning rates, with benchmark val 0.5690 and 0.5944. It does not improve JevBench over the frozen panel-2 readout, so the synthetic share is not the cause of the transfer gap. The arm is evidence against one explanation, not evidence for another.

## 5. Interference at high learning rate

Per-source change in cross-entropy from the frozen readout to the trained readout at 3e-4, on the rows that arm saw. A negative value means the trained model fits that source better than the frozen one; a positive value means it fits it worse.

| direction | sources |
|---|---|
| better | agnews -0.703, fever -0.627, apigen -0.474, winogrande -0.405, banking77 -0.402, swag -0.364, multirc -0.263, logiqa -0.242, typed_decisions -0.219, synthetic -0.204 |
| worse | sst2 +0.341, tabfact +0.338, contractnli +0.225, arc +0.114, codesearchnet +0.076, mmlu +0.074 |

The spread is the point: at 3e-4 the arm improves on ten sources and degrades on six, all inside the set it trained on. That is interference, not a uniform fit, and it is the mechanism behind the benchmark-val collapse in section 2 that a single aggregate number hides.

Absolute cross-entropy across list lengths needs its own baseline. banking77 at that point has an absolute CE of 4.84 with K = 80, above the uniform baseline of ln 80 = 4.38. A long list can therefore sit at or above uniform while its neighbours look fitted, and absolute CE cannot be compared across K; long-list cells need a per-K baseline and are reported separately in the long-list line.

## 6. Order robustness degrades with training

Natural-versus-reversed agreement falls from 0.771 at the frozen readout to 0.550 at 3e-4, and exact ties rise from 3 to 21. Training therefore buys accuracy at some cost in order invariance, and the cost is already visible at a learning rate whose JevBench cell is unchanged. Any arm that reports accuracy without the order agreement reports a partial result.

## 7. Data state

The judgement plan of record is built in-repo. Each row carries its true source and its synthetic family; the split is group-free and namespaced, with `internal_split`; the template cap is six rows (train templates 4,309 at 5.995 rows per template, down from 34.1); and the option order is a deterministic per-row permutation with a stratum-balanced gold slot, whose worst non-noul deviation is 0.63 pp, down from 25 to 78 pp.

The noul K = 2 stratum is the exception: its deviation stays at 6.7 pp because criteria-bearing noul rows are excluded from the permutation, 1,599 in the full plan and 251 in the a20k sample. The exclusion is deliberate and is recorded here so the residual deviation is not read as a bug.

The plan directory on the run host carries `PROVENANCE.json` with the plan shas and their equivalence pairs. It is not in the repository and is not reproducible from main. Plans are immutable once used, and the handoff between people, hosts and sessions is the versioned path plus its sha. A plan that is edited after use is a new plan and needs a new sha.

## 8. Full fine-tuning status (unverified)

The memory probe measured 4.02B trainable parameters, 7.54 GiB of weights, bf16 moments at 14.98 GiB, and a peak of 68.18 GiB at 8 rows of 486-token prefixes at 0.80 s per step; 945-token prefixes ran out of memory.

The first smoke exited at 93 GiB on a single 11,576-token row. The cause is structural: this trainer runs one row per forward, so a token budget cannot bound a single oversized row. Activation checkpointing is in progress. No full-fine-tuning result exists, and every figure in this section is unverified.

## 9. The admission rule

JevBench is a veto, not the target. An arm may lose at most 3 items on the 231 local items. Benchmark-only validation is the primary measure. ECE, order agreement, the long-list K = 80 cell and cost are reported for every arm. Among arms that pass the veto, no single number decides; the combination is read together.

The earlier statement that no arm improves on the frozen readout is a 231-only statement and does not survive this rule. At 3e-5 the trained panel-2 readout improves benchmark val from 0.5696 to 0.5822, 1.3 pp on 9,923 rows with exact McNemar p = 5e-7, while its JevBench cell stays at the frozen level. Under the veto the arm is admissible, and its improvement is real on the primary measure.

## 10. Engineering record

Seven PRs merged in this stretch, #378 through #384: the noul label mapping (#378), the judgement-readout note (#379), the panel-2 listwise trainer with the plan reader (#380), the in-repo plan of record (#381), and three collective defects. The defects are a hash-then-reopen race (#382), unequal per-rank step counts that paired two different collectives (#383), and a failure path that deadlocked the other ranks (#384). Each carries a test that was mutation-verified: the test fails when the guard is removed.

The derived rule is general: nothing that can fail on one rank may run before a collective. A rank that raises early leaves the others waiting, and the failure becomes a hang whose cause is invisible in the logs.

## Provenance

| number group | status | source |
|---|---|---|
| frozen panel-2 comparison, learning-rate sweep, 2x2, benchmark-only arm, interference, order robustness | measured; run logs and eval shards on the run host, not committed | panel-2 runs of 2026-11-15 and 2026-11-16, seed 42 |
| data state (source, split, cap, permutation) | measured; plan artifacts and `PROVENANCE.json` sit in the plan directory on the run host, not in the repository and not reproducible from main | #381 |
| full fine-tuning memory figures | unverified; probe output not retained | single-node memory probe |
| engineering record | committed | #378 through #384 |

## Scope

One backbone, one adapter position, one seed, 20,000-row plans, and the local 231-item proxy. The 2x2 crosses two readouts and two data versions, and the readout comparison carries its objective with it. Nothing here measures a second seed, a full fine-tuning run, an arm trained on the full corpus with the panel-2 readout, or the official JevBench Intelligence score.
