# The panel-2 readout replaces the judgement readout — 2026-11-16

Status: measured, one seed. The mainline readout moves from per-option judgement scoring to the panel-2 vocabulary code. On the 231 local items the frozen panel-2 readout beats the frozen judgement readout by 13 items, and training the panel-2 readout at a low learning rate improves benchmark validation significantly while JevBench stays flat. Full fine-tuning has no result yet and is unverified.

**PRs:** #378 through #384.

Every number below uses Qwen3-Reranker-4B as the backbone, at most 4,096 tokens per row, seed 42, and corrected noul labels unless a cell says otherwise. JevBench is the local 231-item public proxy (easy 48, hard 111, original 72), not the official JevBench Intelligence score. Paired tests are exact two-sided McNemar on the paired items, and the paired n is given with each p.

## 1. Mainline switch

The panel-2 vocabulary readout scores an item in one forward: the model reads the item once, and the decision is the argmax over the declared single-token code ids for the options. The judgement readout needs one plus K forwards per item, one per option, and contrasts the yes and no margins. On the same frozen backbone and the same 231 corrected-label items, panel-2 is 162/231, chance-corrected accuracy (cc) 0.5623, with easy 48/48, hard 52/111 and original 62/72, against judgement 149/231 and cc 0.4798. The paired difference is 13 items, exact McNemar p = 0.0725, with 29 discordant pairs in one direction and 16 in the other. The 149/231 judgement cell is the earlier note's re-score pass; the in-process pass inside the evaluator reads 150/231 on the same items, and section 3 pairs its judgement arms against that 150. The p = 0.0725 belongs to the 149 re-score pairing; against the in-process 150 the difference is 12 items and no test statistic is computed for that pairing in this note. Every cell names the pass it comes from.

The gain is entirely in choice items, and its size depends on which frozen judgement pass is the reference. Against the 149 re-score pass, panel-2 is 102/139 against 89/139, a paired net of +13 with exact McNemar p = 0.0192 on 20/7 discordant pairs. Against the in-process 150 pass, whose choice cell is 90/139, the difference is 12 items and no test statistic is computed here. Non-choice is net-flat under either pass, 60 against 60: noul 48/74 against 47/74, score 12/18 against 13/18. The two passes differ by one item overall and one in choice, 149 with choice 89 and 150 with choice 90. That is run-to-run variation on the same frozen model, not a correction, and the noul and score cells agree.

Order robustness at the frozen readout is already imperfect. Scoring the same items with the option list natural and then reversed gives 162 and 156 correct, the same choice in 178/231 items, and 3 exact ties, from `SUITE_CELLS.json` in the run directory of the arm concerned, derived from its shard items. The file is not committed, and the committed reverse-order artifact scores 157 and cannot yield the agreement cell, so the order cells carry the run-directory artifact rather than a repository one. The order average of the frozen panel-2 readout is 159.0/231.

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

All p-values in the table are exact McNemar on the 9,923 paired rows, with b the rows the frozen readout gets right and the trained readout gets wrong and c the reverse. The discordant pairs are b/c = 244/369 at 3e-5, 691/999 at 1e-4, 1,991/1,257 at 3e-4 and 2,974/1,405 at 1e-3. Only 3e-5 has c > b; the other three are losses with b well above c. The sign change is the finding. The two low learning rates part company on JevBench: 3e-5 scores 161 against the frozen 162, one item below and admissible; 1e-4 scores 152, ten below, and fails the veto despite having the best benchmark val of any arm at 0.6006. At 3e-4 and 1e-3 both measures collapse. That contrast is the case for the veto: the arm with the highest on-distribution accuracy is the one that loses the proxy by ten items, so a recipe that optimises on-distribution accuracy and a recipe that preserves the proxy are not the same parameter region, and the frozen-to-trained comparison alone does not say which region an arm is in.

## 3. The 2x2

The readout and the data version were crossed at two learning rates, with 303 steps, the same seed and the same evaluation. The frozen comparisons use the in-process pass, 150/231 for judgement and 162/231 for panel-2.

| readout | data | 3e-5 | 1e-4 | 3e-4 |
|---|---|---:|---:|---:|
| judgement | v3 corrected | 148 | 148 | 154 |
| judgement | old pre-permutation | 149 | 151 | — |
| panel-2 | v3 corrected | 161 | 152 | 152 |
| panel-2 | old pre-permutation | 160 | 158 | — |

Readout effect at matched data and learning rate: frozen +12, v3 +13 at 3e-5, +4 at 1e-4 and -2 at 3e-4, old +11 at 3e-5 and +7 at 1e-4. Data effect: judgement -1 and -3, panel-2 +1 and -6. The readout carries the gain; the data fixes carry none on these 231 items, although they move benchmark val by 3 to 4 pp in the same runs. A 231-item JevBench cell is therefore insensitive to data repairs that the 9,923-row validation set registers. At 3e-4 the readout effect reverses to -2, so the readout advantage belongs to the low-learning-rate region, not to the readout on its own. On choice, the frozen readout effect is +12 against the in-process pass, 102 against 90; the +13 with p = 0.0192 in section 1 belongs to the 149 re-score pairing, so the two numbers are not in conflict.

The three judgement cells at 3e-5 are three separate runs and they do not all read the same. The frozen in-process pass is 150; the judgement arm on v3 a20k (run judgement_v3_a20k_lr3e-5, plan fa83a43f) is 148; the judgement arm on old a20k (run judgement_old_a20k_lr3e-5, plan 6dd503cc) is 149. The committed 149/231 frozen cell from the earlier note is a separate re-score pass on the same items, not a trained arm, and is not one of these three.

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

Order robustness is measured at both ends. At the admissible arm, panel-2 on v3 a20k at 3e-5, natural order is 161, reversed 153, order-averaged 157.0, the same choice in 174/231 items, and 21 exact ties counted over both orders. The frozen readout is 162 natural, 156 reversed, 159.0 averaged, 178/231 agreement and 3 ties, all from the same uncommitted run-directory file, so the 0.771 frozen agreement is unverified. The adapter therefore cuts agreement by 4 items and multiplies exact ties by 7, while losing one item natural and three reversed, which is the 2-item gap on the average. At 3e-4 the agreement falls further to 0.550 with 21 ties, but that arm scores 152 against 162, a collapse, so it is not an admissible point and the note does not use it as one.

The averaged cell is what the veto uses. The arm averages 157.0 against the frozen 159.0, a 2-item loss inside the 3-item allowance, where the natural order alone would show a 1-item loss and the reversed order a 3-item one. The reversed pass behind the arm's averaged cell is not committed, so its admissibility is provisional: it needs an averaged cell of at least 156 and clears at 157.0 by one item, but that cell must be reproduced from a committed pass before the arm is treated as admitted. Its margin against the frozen average is 2 items, so a one-item change in either pass puts it exactly on the allowance and a two-item change fails it. Any arm that reports accuracy without the order agreement still reports a partial result, and the instrument is the reversed pass.

## 7. Data state

The judgement plan of record is built in-repo. Each row carries its true source and its synthetic family; the split is group-free and namespaced, with `internal_split`; the template cap is six rows. Before the cap the train split has 4,309 templates and 121,210 rows, 28.1 rows per template. After the cap it has the same 4,309 templates and 25,833 rows, 6.0 per template. Both figures use one population and one denominator, and the cap alone accounts for the 95,377 rows removed. The option order is a deterministic per-row permutation with a stratum-balanced gold slot, whose worst non-noul deviation is 0.63 pp, down from 25 to 78 pp.

The plans used by the runs in this note are `f369be84ca3af9061a7f2188584d1a482f5a416e998fc64466f31fb1b4a2d858` (v3 cap-6 full, 130,701 rows and 677,369 prompts), `fa83a43f11d33518eddd60820a2c14a576ab9ad551145c7506c90a933f6d4abf` (v3 cap-6 a20k subset) and `6dd503cc97276921f7248ee64e624ab86d98df1857c2deaa212821fc7366e566` (v2 restricted to the same 20,000 row ids). `fa83a43f` is equivalent to `e191cd4cda93d25f8cb347902995377389decf285106813090be1b29edc1c844`, and `f369be84` to `6388641e49a6d034e06b52d9fb4feac4e2c0fdaacbcdfa0e055bdfae02c0f334`: identical on every field except the added polarity key.

The noul K = 2 stratum is the exception: its deviation stays at 6.7 pp because criteria-bearing noul rows are excluded from the permutation, 1,599 in the full plan and 251 in the a20k sample. The exclusion is deliberate and is recorded here so the residual deviation is not read as a bug.

All seven shas and their one-line descriptions are in `research/50-plans.json`, which is path-free; the full plan records, including `PROVENANCE.json`, sit in the plan directory on the run host, outside this repository, and are not reproducible from main. Plans are immutable once used, and the handoff between people, hosts and sessions is the versioned path plus its sha. A plan that is edited after use is a new plan and needs a new sha.

## 8. Full fine-tuning status (unverified)

The memory probe measured 4.02B trainable parameters, 7.54 GiB of weights, bf16 moments at 14.98 GiB, and a peak of 68.18 GiB at 8 rows of 486-token prefixes at 0.80 s per step; 945-token prefixes ran out of memory.

The first smoke exited at 93 GiB on a single 11,576-token row. The cause is structural: this trainer runs one row per forward, so a token budget cannot bound a single oversized row. Activation checkpointing is in progress. No full-fine-tuning result exists, and every figure in this section is unverified.

## 9. The admission rule

JevBench is a veto, not the target. The veto is stated on order-averaged accuracy over the 231 local items, the mean of the natural-order and reversed-order scores. The instrument moves 6 items between the two orders at the frozen readout, so a 3-item threshold on a single order would sit inside the order noise. The frozen panel-2 readout scores 162 natural and 156 reversed, 159.0 averaged, from `SUITE_CELLS.json` in the run directory; the committed reverse-order artifact scores 157, which would give 159.5. The threshold is 156 on the 159.0 average, and it is provisional until the run-directory file is committed, because the repository artifact alone cannot reproduce it. The 3-item allowance is a choice, not a derivation. One binomial standard error at 0.70 on 231 items is sqrt(231 x 0.7 x 0.3) = 7.0 items, and the paired standard error at 45 discordant pairs is sqrt(45) = 6.7, so both land near 7. Order averaging removes order noise and does not remove sampling noise. The allowance is deliberately tighter than one standard error because the cost is asymmetric: admitting an arm that has quietly lost three items is worse for the programme than rejecting a neutral one, since every later arm is compared against the admitted baseline. Benchmark-only validation is the primary measure. Averaged cells are printed only where a reversed pass exists; an arm without one has its averaged cell marked — rather than approximated. ECE, order agreement, the long-list K = 80 cell and cost are reported for every arm. Among arms that pass the veto, no single number decides; the combination is read together.

A 231-only reading, that no arm improves on the frozen readout, does not survive this rule, with the same proviso: the arm's averaged cell and the frozen average both rest on uncommitted order passes. At 3e-5 the trained panel-2 readout improves benchmark val from 0.5696 to 0.5822, 1.3 pp on 9,923 rows with exact McNemar p = 5e-7, while its JevBench cell is 161 natural and 157.0 averaged against the frozen 162 and 159.0, a 2-item loss inside the veto allowance. Under the veto the arm is provisionally admissible, subject to a committed reversed pass, and its improvement is real on the primary measure.

## 10. Engineering record

Seven PRs merged in this stretch, #378 through #384: the noul label mapping (#378), the judgement-readout note (#379), the panel-2 listwise trainer with the plan reader (#380), the in-repo plan of record (#381), and three collective defects. The defects are a hash-then-reopen race (#382), unequal per-rank step counts that paired two different collectives (#383), and a failure path that deadlocked the other ranks (#384). Each carries a test that was mutation-verified: the test fails when the guard is removed.

The derived rule is general: nothing that can fail on one rank may run before a collective. A rank that raises early leaves the others waiting, and the failure becomes a hang whose cause is invisible in the logs.

## Provenance

| number group | status | source |
|---|---|---|
| frozen panel-2 comparison, learning-rate sweep, 2x2, benchmark-only arm, interference, order robustness | measured; run logs, eval shards, `SUITE_CELLS.json` and `judgement_frozen_pass.json` on the run host, not committed | panel-2 runs of 2026-11-15 and 2026-11-16, seed 42 |
| data state (source, split, cap, permutation) | measured; plan artifacts and `PROVENANCE.json` sit in the plan directory on the run host, not in the repository and not reproducible from main | #381 |
| full fine-tuning memory figures | unverified; probe output not retained | single-node memory probe |
| engineering record | committed | #378 through #384 |

## Scope

One backbone, one adapter position, one seed, 20,000-row plans, and the local 231-item proxy. The 2x2 crosses two readouts and two data versions, and the readout comparison carries its objective with it. Nothing here measures a second seed, a full fine-tuning run, an arm trained on the full corpus with the panel-2 readout, or the official JevBench Intelligence score.
