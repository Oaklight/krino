# More epochs buy in-distribution accuracy; the readout buys JevBench — and neither crosses over

Two levers were tested against the goal today and they turn out to be disjoint. Training past the
project's 5-epoch convention raises in-distribution accuracy by 6.8 pp and leaves JevBench exactly
where it was. Scoring choice with the backbone's own readout instead of the trained head buys
12.6 pp on JevBench and 3.8 pp on in-distribution choice (+1.9 pp overall). Nothing trades one for
the other.

Together they reach **73.01% on the 13,444-item held-out cluster-split eval** — 69.24% from the
trained heads alone, and +3.77 pp from routing short option lists to the backbone's readout. That
is 8.0 pp over goal item 1's 65% bar, on one seed.

All numbers below are on the cluster-disjoint split after #138, one fixed draw (seed 123, full
13,444-item eval, no per-type cap) unless stated, checkpoint `arch_4b_loraAll_32k_s123`
(full LoRA, `last` readout, 32k context).

## The schedule

| epoch | noul | choice | score | overall | JevBench (head) |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.8036 | 0.5307 | 0.5142 | 0.6244 | — |
| 8 | 0.8237 | 0.6084 | 0.5322 | 0.6737 | 0.515 |
| 10 | **0.8362** | **0.6322** | **0.5480** | **0.6924** | 0.506 |

The in-training curve is monotone from epoch 3 (0.5734 → 0.6055 → 0.6212 → 0.6385 → 0.6580 →
0.6700 → 0.6862 → 0.6909) and the harness reproduces it within 0.4 pp at every checkpoint it was
run on. **69.24% clears goal item 1's 65% bar by 4.2 pp and beats the previous best by 4.35 pp** —
64.89, the 4B full-LoRA `weighted` arm of the round-2 sweep (`#150`, merged as `ef557d2`), not the
62.85 8B number an earlier revision of this note quoted. Both come from the pre-#138 corpus, so the
gap is if anything understated, but the comparison is no longer a scale result: the previous best
was also 4B, so what this establishes is a schedule and readout effect at one scale. Against the
lower and more leaky of the two candidates the gain would look like 6.4 pp; picking that one makes
the gap look larger, not smaller, which is the opposite of conservative.

Every type improves across the last two epochs, so this is not one head carrying the total.

### How much of that is seed?

The project has been reading 4B deltas against a 0.35 pp seed-noise floor. That floor came from
*frozen-backbone* arms, and it does not describe a full-LoRA run. Measured on this arm:

| measurement | method | value |
|---|---|---:|
| seed 42 vs seed 123 at epoch 5 | harness, fixed draw | **2.47 pp** |
| same seed twice, epoch 1 | training loop | 0.67 pp |
| same seed twice, epoch 2 | training loop | 2.52 pp |
| same seed twice, epoch 3 | training loop | 0.08 pp |
| epoch 5 → 10, same seed | harness, fixed draw | **+6.80 pp** |

The three training-loop rows are single epochs and are shown to make a point rather than to be
averaged: they disagree with each other by a factor of thirty. Only the two harness rows measure
anything, and they are the ones the rest of this note relies on.

Two of those rows appeared later than the first, and together they change what the top row means.
The "same seed twice" pair is `arch_4b_loraAll_32k_s123` against `arch_4b_32k_split138_s123`:
identical seed, corpus and schedule, differing only in `--no-gradient-checkpointing` and an explicit
`--model`. They agree to 0.67 pp at epoch 1 and diverge to 2.52 pp by epoch 2.

Once the seed-42 arm reached epoch 2 the comparison became a 2x2, and a third epoch then dissolved
it again — which is the most instructive thing about it:

| run | seed | gradient checkpointing | epoch 1 | epoch 2 | epoch 3 |
|---|---|---:|---:|---:|---:|
| `arch_4b_loraAll_32k_s42` | 42 | default | 0.5440 | **0.5124** | — |
| `arch_4b_loraAll_32k_s123` | 123 | default | 0.5466 | **0.5144** | 0.5734 |
| `arch_4b_32k_split138_s123` | 123 | off | 0.5399 | 0.5396 | **0.5726** |

At epoch 2 the two *different* seeds agree to 0.20 pp while the two runs of the *same* seed differ
by 2.52 pp, which invites the conclusion that the flag matters twelve times as much as the seed. At
epoch 3 the same pair agree to **0.08 pp**. The epoch-2 divergence was transient, and neither
reading survives: single-epoch differences between runs of the same configuration fluctuate by
~2.5 pp in either direction, whatever differs between them.

That is the rule this project already applies to conclusions — mid-schedule epochs decide nothing —
applied to noise instead of to arms. **The honest statement is not "seed noise is 2.47 pp" and not
"the flag dominates"; it is that a single training-loop epoch is not a measurement of run-to-run
spread at 4B full LoRA.** Estimate it from the full curve or from harness numbers at a fixed epoch,
not from one row. I made the flag claim myself after epoch 2 and had to withdraw it one epoch later,
which is the same mistake in miniature.

The +6.80 pp schedule gain is measured the way the rest of this note measures things — harness
numbers at fixed checkpoints on one draw, five epochs apart — so it is not exposed to the
single-epoch noise above. What the two-seed run in flight can and cannot settle is narrower than
the definition of done implies: it answers "does ~73% reproduce", not "is 73% the right number".

## The readout

Per-option-count, head against native (per-option P(yes) through the same backbone), 1,500 choice
items, epoch 8:

| options | n | head | native | delta |
|---|---:|---:|---:|---:|
| 2 | 225 | 0.636 | 0.662 | +2.67 |
| 3-5 | 1,166 | 0.595 | **0.680** | **+8.49** |
| >20 | 109 | **0.807** | 0.312 | **−49.54** |

The mechanism is unchanged from #180: native scores each option independently, with no term that
lets options compete, so it wins where a few candidates must be compared and collapses where
twenty must be ranked. Routing by option count — native for ≤5 options, head above — is
therefore the natural rule, and it is worth **+7.00 pp on choice** on the sample.

That rule has to be defensible against the obvious objection that it was chosen after seeing the
result. Splitting the choice items into two random halves and running the rule on each:

| half | 2 opts | 3-5 | >20 | routed vs head |
|---|---:|---:|---:|---:|
| A (n=750) | +2.80 | +8.73 | −49.15 | **+7.20** |
| B (n=750) | +2.54 | +8.25 | −50.00 | **+6.80** |

The sign of every bucket delta is the same in both halves and the magnitudes agree to within half
a point, so the rule is stable rather than a fork of the data. The threshold itself is not a free
parameter on this corpus: there are **no choice items between 6 and 20 options**, so "route at 5"
and "route at 20" are the same rule, and nothing depends on where inside the gap it is placed.

On the full eval set the native readout alone scores **0.7118 overall against the head's 0.6924**
(+1.94 pp overall, +3.79 pp on choice — `overall_epoch_10`, the only full-eval readout comparison
that is in the artifact), with `noul` and `score` identical by construction. The epoch-8 figure an
earlier revision quoted for the choice gain (+5.88 pp) came from a pass whose full-eval dump was
not committed; the 1,500-item sample at that epoch is `readout_epoch_8.choice` (head 0.6167 against
native 0.6507), and this note now quotes the epoch-10 pair, which reproduces from the tree.
At epoch 10 the same comparison, on both dumps' own items:

| epoch 10, 13,444 items | choice | overall |
|---|---:|---:|
| trained head only | 0.6322 | 0.6924 |
| native readout only | 0.6701 | 0.7118 |
| **routed by option count** | **0.7058** | **0.7301** |

So routing is worth **+7.36 pp on choice and +3.77 pp overall**, and **73.01% is the best number
this project has produced** — 8.0 pp over goal item 1's bar, on **one seed, one draw, against a
2.47 pp seed spread**. The routing gain is measured within a single checkpoint, so it is less
exposed to that spread than the absolute number is; every later quotation of 73.01 in this project
should carry the same sentence. It costs nothing in ECE: the choice
ECE on the native path is 0.2050 before a fitted temperature, and the fitted temperature brings it
under 0.05 as it does for the head path.

**The rule replicates on a checkpoint that shares neither seed nor schedule.** The seed-42
five-epoch checkpoint, scored the same way against its own head dump on the same draw, gives head
0.6490, native 0.7035 and **routed 0.7138** — +6.48 pp overall and +12.63 pp on choice, with the
same sign in every bucket. The gain is *larger* on the weaker checkpoint, which is what the
mechanism predicts: routing recovers what the head has not yet learned to do, and the head catches
up as it trains.

## JevBench does not move

| checkpoint | head | native |
|---|---:|---:|
| epoch 8 | 0.515 (hard 0.351) | **0.641** (hard 0.414) |
| epoch 10 | 0.506 (hard 0.351) | 0.632 (hard 0.414) |

The hard tier sits at 0.351 for the head and 0.414 for native at both checkpoints, to three
digits. Against teacher 84.9 and best open student 74.1, the best figure here is 64.1, so **goal
item 4 remains 10 pp short** and nothing tested today closes it. The readout buys 12.6 pp and is
the only lever that moved; the schedule buys nothing.

That asymmetry is the useful result: JevBench's remaining gap is a capability gap in the model,
not a configuration one, and further epochs spent on the in-distribution mixture will not pay.

## Calibration on the best checkpoint

Full 13,444-item eval, epoch 10, fitted on one half and reported on the other. "Map (CV)" picks
the map's regularisation by an inner cross-validation on the fitting half, so the number is not
selected on the data it is reported on:

| type | ECE at T=1 | after per-type T | after map (fixed 10/20) | after map (CV) |
|---|---:|---:|---:|---:|
| noul | 0.0368 | 0.0273 | 0.0286 | **0.0264** |
| choice | 0.0488 | **0.0180** | 0.0170 | **0.0137** |
| score | 0.1733 | 0.1244 | 0.0313 | **0.0289** |

**Goal item 2's ECE clause is met on every type of this checkpoint**: a temperature alone already
brings noul and choice under 0.05, and the map brings score from 0.1733 to 0.0289. That is a
change from the epoch-8 sample, where the map *raised* noul and choice (0.0297 → 0.0384, 0.0533 →
0.0545) — so the map's effect on those two types is small and its sign is not stable across
checkpoints, while its effect on score is large and robust (0.1733 → 0.0289 here, 0.2036 → 0.0215
at epoch 8). The defensible claim is narrower than "#179 makes calibration a solved problem": the
score primitive needs the map, and the other two types do not care much either way.

**QWK is what blocks item 2**, and the mixture is dragged by more than one group. Per level count,
the score eval splits cleanly by source:

| levels | n | sources | QWK |
|---|---:|---|---:|
| 4 | 218 | typed_decisions 161, synthetic 57 | 0.631 |
| 5 | 1,113 | sst5 500, yelp 500, synthetic 92, typed_decisions 21 | 0.827 |
| 6 | 500 | stsb 500 | 0.287 |

Per source, yelp 0.880 and sst5 0.832 are at or near the 0.85 bar; stsb 0.287,
typed_decisions and synthetic are well below. The mixture is 0.6562.

STS-B is the worst and the reason is visible in the gold distribution — it is the only source whose
eval items are dominated by one level:

| source | n | most common gold | accuracy |
|---|---:|---:|---:|
| yelp | 500 | 21% | 0.670 |
| sst5 | 500 | 31% | 0.568 |
| synthetic | 150 | 31% | 0.440 |
| typed_decisions | 182 | 37% | 0.577 |
| **stsb** | 500 | **53%** | 0.428 |

STS-B's train split has mean 2.70 and its eval split 1.23, so the model trains on high-similarity
pairs and is evaluated on low ones. That is a within-source distribution shift created by the
split, not something more training fixes, and it is separate from the 7 withheld-label rows #177
removes.

The label distributions say how bad that shift is. STS-B is the only source whose train and test
marginals barely overlap:

| source | split | level 0 share | full distribution |
|---|---|---:|---|
| stsb | train | 0.11 | 0.11 / 0.14 / 0.16 / 0.23 / 0.25 / 0.10 |
| stsb | test | **0.56** | 0.56 / 0.08 / 0.09 / 0.12 / 0.10 / 0.04 |
| sst5 | train | 0.13 | 0.13 / 0.26 / 0.19 / 0.27 / 0.15 |
| sst5 | test | 0.13 | 0.13 / 0.28 / 0.19 / 0.24 / 0.17 |
| yelp | train | 0.19 | uniform |
| yelp | test | 0.20 | uniform |
| typed_decisions | train | 0.13 | 0.13 / 0.26 / 0.30 / 0.30 / 0.01 |
| typed_decisions | test | 0.15 | 0.15 / 0.20 / 0.32 / 0.32 / 0.01 |

Every other source's train and test marginals match to within two points. STS-B's test set is 56%
level 0 where its train set is 11% — a model trained on that train split is *correctly* prior-ed
for level 0 to be rare, and then scored on a set where it dominates. **Its QWK of 0.287 therefore
measures the benchmark's own train/test mismatch, not the model**, and its 500 items are 27% of
the score eval.

That is the strongest statement this note can make about clause 2: on the two sources whose
distributions match and which are real human judgements (sst5 0.832, yelp 0.880), the model is at
or near the 0.85 bar; the aggregate is pulled down by a benchmark that cannot be scored this way,
and by the four-level group where the bottom-level aversion below is a genuine model weakness.

**But fixing STS-B alone would not clear the bar.** Dropping it leaves 0.827 on the 1,113
five-level items and 0.631 on the 218 four-level ones — a mixture of roughly 0.80 against a 0.85
target. The 4-level group is our own data (typed_decisions and synthetic), and it is the second
thing to look at. I said earlier that STS-B was "the sole calibration blocker"; that was too strong
and this table is the correction. I also first called the 4-level group a label-quality question,
before looking at the confusion; the next paragraph is what the data actually shows.

The 4-level failure has a specific shape, and it is the same shape as STS-B's. On the 218
four-level items the gold is spread (level 0: 29, 1: 56, 2: 81, 3: 52) but the predictions are not:
the model predicts level 0 only **13 times against 29 golds**, pushing the rest up into level 1,
which it over-predicts 79 times against 56. The top level is fine (57 predicted, 52 gold). So the
error is not noise in the middle of the scale and it is not a label problem — the labels are
plausibly spread and the model systematically avoids one end of them. It is an **aversion to the
bottom level**, and it appears in both typed_decisions (11 predicted against 27 gold at level 0)
and, more weakly, in synthetic.

That single behaviour explains both QWK failures: level 0 is 53% of the STS-B items in the eval
(56% of that benchmark's whole test split) and the four-level group's lowest level is under-called,
while the five-level groups, whose gold is mid-scale, are at 0.827. A model that will not emit the
lowest level cannot score well on any scale whose lowest level is common. It is a plausible
downstream consequence of an imbalanced vote: if the lowest level is rare in training relative to
the middle, an argmax over a shared scale vocabulary will drift upward, and no temperature or map
changes an argmax.

## Clause 2's Brier criterion cannot be met by a good model

Goal item 2 asks for "Brier and QWK >= 0.85 on the score primitive". QWK is a chance-corrected
agreement statistic where 1 is perfect and 0.85 is a demanding bar, so the inequality makes sense
for it. **Brier runs the other way**: it is a mean squared error over the option vector, 0 is
perfect, and the uniform distribution over k options scores (k-1)/k — 0.75 at four levels, 0.83 at
six. A criterion of "Brier >= 0.85" is therefore satisfied by a model that is *worse than uniform*,
and is failed by a perfect one.

The measured value on this checkpoint is **0.6446**, against 0.75-0.83 for uniform on the same
items: better than uniform, far from perfect. Under the two readings that would make the clause
coherent, it fails both:

| reading | value | bar |
|---|---:|---:|
| 1 - Brier | 0.355 | >= 0.85 |
| Brier skill score against the uniform baseline (~0.80) | ~0.19 | >= 0.85 |

I am not claiming a pass on a literal reading, and I am not claiming a fail on a criterion that no
correct model can satisfy. **This is a defect in the definition of done that needs a decision** —
whether "Brier" meant a skill score, an inverted Brier (1 - Brier), or was a slip alongside a QWK
bar. Until it is resolved, the honest report is the number: score Brier 0.6446, lower is better,
better than uniform and short of any demanding skill threshold.

## Goal status

Rewritten for the cross-encoder arm (`research/21`, `#192`). That arm changes the
answer to clauses 1, 3 and 5, so the table below is the state after it; the
rule-based figures this table used to carry are kept only where they are still the
best measurement of their clause, and they are labelled as superseded where not.

| clause | state |
|---|---|
| 1. accuracy ≥ 65% | **met, by the head, on two seeds.** The 4B cross-encoder reaches **0.7626** on the 13,444-item cluster-disjoint eval at epoch 5 (s123), and the two seeds agree to **0.04 pp at matched epoch 3** — 0.7291 (s123) against 0.7287 (s42). The arm was still improving when it stopped (eval loss 0.727 → 0.590 across the run), so 0.7626 is a point on a rising curve, not a ceiling. The routing and blend rules from `research/19` are **superseded**: they were what met this clause when the head alone was 62.16 % / 64.18 %. |
| 2. ECE ≤ 0.05, Brier/QWK ≥ 0.85 | **ECE met on noul and choice on both seeds; the score primitive is unmeasurable at this size, and QWK fails on every scale.** Per-type temperature, fitted on one half and reported on the other: noul 0.0206 / 0.0265, choice 0.0250 / 0.0236 (s123 / s42). Score is **0.1024 on s123 and 0.0259 on s42** — a 4× seed spread that straddles the 0.05 bar, on the same primitive and the same data slice. QWK on the cross-encoder is below the tower on **every** scale (4-level 0.2862/0.0866 vs 0.4196, 5-level 0.7391/0.8028 vs 0.8314, 6-level 0.8281/0.8270 vs 0.9222, mixed 0.7062/0.7193 vs 0.7916; `research/18`'s repaired column), and the 4-level scale itself moves 0.20 between seeds on 233 items — ten times any lever's measured effect. The four post-hoc levers are measured null (`research/18`), and the ordinal-loss route is closed: RPS lowers QWK by 8 points (`research/11`). The **Brier criterion is unsatisfiable as written**. |
| 3. flip ≤ 5%, >20 options within 5 pp | flip **met, structurally** — 0 flips and 0.0000 total-variation on 6,900 cross-encoder choice items and 1,856 score items under both probes. Count **fails but improved**: on a matched 300-item banking77 sweep at epoch 3, K=4 → K=80 costs the cross-encoder **−31.7 pp** against the tower's −57.3, and it is 26 pp ahead at eighty options (0.420 vs 0.157). The tower at epoch 10 reaches −12.67 pp, so training length still dominates the level. |
| 4. JevBench vs 84.9 / 74.1 | **not met, and not resolvable on the public tier.** The 4B cross-encoder spans 0.405 (ep1) to 0.450 (ep5) on the hard tier and 0.615-0.632 overall; **no paired comparison on 111 items clears chance** (tower vs ep5 16/21, p = 0.51; ep3 vs ep5 11/12, p = 1.00), so the tier cannot resolve a difference below ~10 pp. Two mechanisms are measured on other instruments: the corpus contains 25 long-state choice items in 172,262 (`research/13`), and the tower's read-out loses 24 pp to 1,400 tokens of irrelevant padding where the cross-encoder loses 1 (`research/22`). Closing this needs a larger held-out set before any model change can be told from noise. |
| 5. ≤100 ms at 256 options; 0.6B within 3 pp | latency **met by the 0.6B tower, compiled**: 55.5 ms on the A100 the clause names and 67.4 ms on H100 (eager misses at 126.6 / 102.9). The 4B misses at every setting (183.6 compiled). Accuracy half **fails at about twice the allowance**: on matched 10-epoch schedules the 4B reaches 0.6871 and the 0.6B is heading for ~0.62, a gap near 6.5 pp against 3. On the cross-encoder the gap is wider still: 0.6048 / 0.6028 against 0.7231 / 0.7190 at matched epoch 1, 11.6-11.8 pp. |

## Clause 3's count half fails on the tower at epoch 10, and it got worse as that model improved

This needs stating plainly because an earlier note (#182) recorded the count half as passing, and
that was true only for the weaker arm it measured. The matched sweep in `research/21` is the
measurement to read for the head-shape question — at epoch 3 the cross-encoder costs −31.7 pp from
K=4 to K=80 against the tower's −57.3 — and the numbers below are the *bucket* comparison on the
tower at epoch 10, which mixes benchmarks and is kept here because the widening of the gap as that
model trained is its own finding. On the epoch-10 checkpoint:

| choice, epoch 10 | n | accuracy |
|---|---:|---:|
| ≤5 options | 6,387 | 0.6177 |
| >20 options | 500 | **0.8180** |
| gap | | **+20.03 pp** |

The clause asks for the two to be within 5 pp. They are 20 pp apart. The direction matters for how
to read it — the long-list bucket is *better*, not worse, so the model is not brittle in the sense
the clause was written to catch — but a 20 pp divergence is not "within 5 pp" on any reading, and
the noul primitive disagrees in the opposite direction (2 options 0.8619, 3-5 options 0.6399,
−22.2 pp).

**And the gap is a consequence of the model getting better.** The seed-42 epoch-5 checkpoint, put
through the same harness on the same draw, has choice accuracy 0.560 at ≤5 options against 0.596
at >20 — a **3.6 pp** gap that passes the clause. Between epoch 5 and epoch 10 the long-list bucket
improves 0.596 → 0.818 (+22 pp, banking77 at a mean of 80 options) while the short-list bucket
moves 0.560 → 0.618 (+6 pp). Training fixes long lists faster than it fixes short ones, so a
comparison that was inside the band at epoch 5 is 20 pp outside it at epoch 10.

That makes the clause's failure a measurement consequence rather than a regression: the model did
not become more option-brittle, it became much better at the one benchmark that has many options.
The matched sweep in the next section is the measurement that holds the option count fixed and
still finds a real 12.67 pp effect, which is the number worth acting on.

Two things follow. First, **the gap widened as the model got better**: +4.6 pp on the 5-epoch
weighted arm against +20.03 pp here, so a pass on a weak checkpoint says nothing about a strong
one. Second, comparing raw accuracies across these buckets is partly a comparison of tasks: the
>20-option items have a 1.25% chance floor against 50% at two options, so they are long lists with
mostly implausible distractors, not the same question with more competition. A clause that wants
option-count robustness is better tested by asking whether accuracy *degrades* with count, which
this model passes in the strong sense, than by a two-sided 5 pp band across buckets built from
different distributions. I have not changed the clause; I am reporting that the model does not
meet it as written.

### The bucket comparison is one benchmark against thirteen

Following that second point to its source: **every >20-option item is banking77** (500 items, mean
80 options, chance 1.25%), and the ≤5 bucket is thirteen different benchmarks. So the 20.03 pp gap
is banking77's accuracy against the average of agnews, arc, race, hellaswag, swag, fever and the
rest — a statement about how hard intent classification is compared with those tasks, not about
what happens to this model when option lists grow.

Where the corpus *can* hold the question fixed, it does, and the answer is that option count barely
matters:

| source | option counts present | accuracy |
|---|---|---|
| synthetic | 4 options (n=185) | 0.535 |
| synthetic | 5 options (n=113) | 0.558 |
| typed_decisions | 4 options (n=96) | 0.500 |
| typed_decisions | 5 options (n=46) | 0.609 |

Both move in the *favourable* direction, and the larger sample moves 2.3 pp. Nothing here degrades
with option count.

**A matched test does exist in the repository, and it needs re-running on this model.**
`model/experiments/optcount_pooled.json` holds exactly the comparison the clause wants — 300 fixed
banking77 items, padded to K ∈ {4, 8, 16, 32, 64, 80, 128, 256} — but it was measured on
`sweep_reranker0.6b_r64_mlp0` epoch 5, a much weaker checkpoint:

| K | 4 | 8 | 16 | 32 | 64 | 80 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| accuracy | 0.677 | 0.557 | 0.453 | 0.320 | 0.237 | 0.230 | 0.217 |

That curve collapses by 46 pp, and it does not describe the checkpoint this note is about: the
4B epoch-10 model scores **0.818 on banking77 at a mean of 80 options**, where the 0.6B sweep
scored 0.230 at K=80. So the current model is far more K-robust than the committed sweep suggests,
and the matched curve has to be re-measured on it before clause 3 can be judged either way.

Two further facts that matter for what to try next. Training-side option-count augmentation already
exists (`data/k_augment.py`, with padding that preserves the gold key and a `--k-augment-control`
flag for an A/B), its cost is measured (1.559× encodes at a fraction of 0.2), and **every headline
run so far trained with `--k-augment-fraction 0`** — the flag defaults to off and neither config
sets it. So the one lever designed for this clause has never been applied to a run that counts.

The honest report is therefore narrower than "the corpus cannot test clause 3": the clause fails
numerically at +20.03 pp, that number compares banking77 against thirteen benchmarks, the only
matched curve on record is from a weak checkpoint, and the augmentation that targets it is unused.

### The matched curve on the 4B, and it is not the collapse the repo records

`model/scripts/option_count_sweep.py` (new) holds 300 banking77 items fixed and varies only K,
shrinking the option list below the natural 80 and padding above it. On the epoch-10 checkpoint:

| K | 4 | 8 | 16 | 32 | 64 | 80 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | **0.9767** | 0.9700 | 0.9533 | 0.9133 | 0.8733 | 0.8500 |

Two things follow, and the first overturns a claim the repository has been carrying. The 0.6B
sweep falls 0.677 → 0.217 over the same direction; this model falls **0.977 → 0.850, 12.67 pp**.
The "accuracy collapses as options are added" narrative in `data/k_augment.py`'s docstring is a
description of that weaker checkpoint, not of the model this note is about, and the augmentation
built on it is aimed at a smaller problem than it advertises.

Second, the clause is **still not met on a matched test**: K=4 against K=80 is 12.67 pp against a
5 pp band (K=4 against K=32, the closest reading of "≤5 vs >20", is 6.33 pp). But that is a
different failure from the one the buckets suggest — two-thirds of the cross-bucket 20.03 pp was
the task mix, and what remains is the honest cost of 76 extra distractors.

The first version of this script reported `changed=0/300` at every K and would have "proved"
perfect robustness, because `pad_choice_item` only pads and banking77 items already carry all 80
intents. It now shrinks as well as pads, and prints the changed count per row so that failure
cannot pass quietly again.

## Caveats

* **One seed at 10 epochs.** The seed-42 run is in flight; until it lands, 69.24 / 73.01 are single
  draws with a known 2.47 pp seed spread. The routing *gain* (+3.77 pp) is measured within one
  checkpoint, so it is less exposed to that spread than the absolute number is.
* **Routing trades latency for accuracy** on the short-list path: native scores one forward per
  option rather than one per question, which matters for goal item 5 and is not accounted for here.
* **The routed system's own ECE is not measured.** Each path calibrates under 0.05 with its own
  fitted temperature (head 0.0180 at epoch 10, native 0.0406 on the 1,500-item sample), but a single
  temperature over the mixture would be wrong — the two paths have different confidences by
  construction. The routed ECE needs per-path temperatures fitted and reported together, which is a
  follow-up, not something this note has done.
* **The route is a product decision, not a training result.** A single head that learned to compete
  over options would be the better fix; routing is what works today.
* **One duplicated id** in the eval set (`synthetic-education_assessment-0160-noul-causal`, both
  copies correct), which is why the per-item dump has 13,443 unique ids over 13,444 rows. It is
  immaterial to every figure above but worth knowing before trusting a unique-id count.
