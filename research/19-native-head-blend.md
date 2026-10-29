# Blending the native score with the trained head beats both on short lists

`research/13` ended on a proposed experiment: the backbone's native per-option
score and the trained choice head trade places at about twenty options, so feed
the native score into the head as a feature and get both regimes. Before paying
for a training run, this measures the cheapest version of that idea — a weighted
blend of the two score vectors, one free parameter, no training — and asks
whether it already beats both scorers and the hard routing rule the model ships
with.

It does, on short lists. On long lists it does not, and there the routing rule
is already right.

Every number below is the 4B epoch-10 checkpoint (`arch_4b_loraAll_32k_s123`),
one seed, scored once per item through both readouts in a single process:
`model/scripts/blend_scorers.py`. Two artifacts: the 60-per-source sample the
blend was developed on (`model/experiments/blend_scorers_s123_ep10.json`) and the
full 7,026 choice items
(`model/experiments/blend_rule_full_s123_ep10.json`). The per-item score vectors
are not committed (1.1 MB and 3.3 MB); the run writes them with
`--dump-records` and both sweeps re-run over them without a GPU with
`--from-records`.

## The blend

Both scorers are rescaled by one number each — the median per-item margin on the
in-distribution items — so a single weight means the same thing on a two-option
item as on an eighty-option one:

```
blended_i = head_i / 1.2332 + lam * native_i / 3.8125
```

`lam = 0` is the head's own argmax and `lam = 1e9` is the native scorer's, so the
two endpoints of every sweep below are the two scorers rather than
approximations of them.

## Results

In-distribution (1,050 choice items, 60 per source, `>=20` is 150 banking77
items — it is the only source above 19 options) and all three JevBench tiers
(139 choice items):

| lam | in ≤5 | in ≥20 | in all | JevBench choice |
|---|---|---|---|---|
| 0 (head only) | 0.6411 | **0.8067** | 0.6648 | 0.4460 |
| 0.25 | 0.6833 | 0.8000 | 0.7000 | 0.5827 |
| 0.5 | 0.7089 | **0.8200** | 0.7248 | 0.6187 |
| 0.75 | 0.7167 | 0.7733 | 0.7248 | **0.6547** |
| 1.0 | 0.7278 | 0.7533 | **0.7314** | **0.6547** |
| 1.5 | **0.7300** | 0.6000 | 0.7114 | 0.6475 |
| 2.0 | 0.7233 | 0.5067 | 0.6924 | 0.6475 |
| 4.0 | 0.7211 | 0.3600 | 0.6695 | 0.6475 |
| 1e9 (native only) | 0.6911 | 0.2600 | 0.6295 | **0.6547** |

Three readings, and they point in different directions depending on the bucket.

**On short lists the blend beats both scorers.** At `lam = 1.5` the ≤5 bucket is
0.7300 against the head's 0.6411 and native-only's 0.6911 — +8.9 pp over the head
and +3.9 pp over the better of the two alone. That is the claim the blend exists
to test, and it holds on the bucket that carries 900 of the 1,050 items.

**On long lists the head wins and the blend cannot help.** The head alone is
0.8067 on banking77; a light blend (`lam = 0.5`) reaches 0.8200, +1.3 pp, and any
heavier weight falls off a cliff — 0.60 at `lam = 1.5`, 0.26 at native-only. The
crossover in `research/13` is visible here as a monotone loss in `lam`, and the
honest reading is that the +1.3 pp is inside the noise of 150 items.

**On JevBench the blend is exactly the native scorer.** 0.6547 at `lam` 0.75, 1.0
and native-only alike — the three agree on every one of the 139 items. So the
JevBench gain over the head (0.4460 → 0.6547, +20.9 pp) is `research/13`'s native
readout finding, not a new blend effect. The head adds nothing there; it is
pulling in the wrong direction and the weights that let it contribute are the
ones that cost in-distribution accuracy.

## The full eval

The sample above established the shape; the numbers that count come from scoring
every choice item in the 13,444-item eval — 6,887 in-distribution plus the 139
JevBench choice items — with the rule applied item by item. The head is kept at
`>= 20` options, so those items are never scored natively and the run costs one
forward pass per short-list option and nothing extra on banking77.

In-distribution choice, and the clause-3 bucket gap:

| lam | ≤5 | ≥20 | choice | gap | overall accuracy |
|---|---|---|---|---|---|
| 0 (head only) | 0.6177 | 0.8180 | 0.6322 | −20.03 pp | 69.24% |
| 0.5 | 0.6906 | 0.8180 | 0.6999 | −12.74 pp | 72.71% |
| 1.0 | 0.7188 | 0.8180 | 0.7260 | −9.92 pp | 74.05% |
| 1.5 | 0.7276 | 0.8180 | 0.7341 | −9.04 pp | 74.46% |
| 2.0 | 0.7321 | 0.8180 | 0.7383 | −8.59 pp | 74.68% |
| **2.5** | **0.7341** | 0.8180 | **0.7398** | −8.39 pp | **74.75%** |
| 3.0 | 0.7318 | 0.8180 | 0.7382 | −8.62 pp | 74.66% |
| native only (the shipped rule) | 0.6972 | 0.8180 | 0.7060 | −12.08 pp | 73.01% |

Overall accuracy is the whole 13,443-item eval: the 4,954 correct noul and score
items are untouched by a choice rule, so the movement is choice only.

**The rule is +1.73 pp overall over what ships, and +5.51 pp over the head
alone.** Choice accuracy goes 0.7060 → 0.7398, and the gain is entirely in the
short-list bucket: 0.6972 → 0.7341, +3.7 pp, while the long-list bucket is
untouched by construction.

**Clause 3's count gap roughly halves and still fails.** It is −8.39 pp at
`lam = 2.5` against the routing rule's −12.08 and the head's −20.03. The 5 pp
band needs the short-list bucket at 0.768; the blend gets to 0.734.

JevBench choice, which the whole note's out-of-sample claim rests on:

| lam | easy (36) | hard (67) | original (36) | choice (139) |
|---|---|---|---|---|
| 0 (head only) | 0.7222 | 0.2687 | 0.5000 | 0.4460 |
| 1.0 | 1.0000 | 0.3731 | 0.8333 | 0.6547 |
| 2.0 | 1.0000 | 0.3881 | 0.8056 | 0.6403 |
| 2.5 | 1.0000 | 0.4030 | 0.7778 | 0.6475 |
| native only | 1.0000 | 0.3731 | 0.8333 | 0.6547 |

The blend beats native-only on the hard tier (0.4030 against 0.3731, +2 items)
and loses one item on the original tier. Across all 139 it is 0.6475 against the
native scorer's 0.6547 — inside the noise of 139 items, and the honest statement
is that on JevBench the blend and the native readout are the same measurement.

**`lam` is chosen on the split it is scored on**, which is a real objection, so
it was checked: fitting `lam` on one half of the in-distribution items and
applying it to the other picks 2.5 and 3.0 on the two halves and scores
**0.7379** cross-fitted, against 0.7058 for the routing rule and 0.6322 for the
head. The gain is not a tuning artefact.

## The deployable rule

Keep the head for long lists and blend on short lists, at `lam = 2.5`:

| rule | ≤5 | ≥20 | choice | all | vs shipped |
|---|---|---|---|---|---|
| head only | 0.6177 | 0.8180 | 0.6322 | 69.24% | −3.77 pp |
| shipped routing (native ≤5, head ≥20) | 0.6972 | 0.8180 | 0.7060 | 73.01% | — |
| blend `lam = 2.5` on ≤5, head ≥20 | 0.7341 | 0.8180 | 0.7398 | **74.75%** | **+1.73 pp** |

It costs the native scorer's forward passes on short items — at most five per
item, against one for the head — and nothing extra on long lists.

## Three better fusion rules, all worse

`lam = 2.5` leaves the short-list bucket at 0.7341 where clause 3's 5 pp band
needs 0.768. Fusion is cheap to vary, so three alternatives were measured on the
same 6,387 short-list items before concluding anything about training:

| rule | ≤5 accuracy |
|---|---|
| linear blend, `lam = 2.5` | **0.7341** |
| a separate weight for 2-option and 3-5-option items | ≤ 0.735 at every pair tried (`lam` 1-4 each) |
| rank fusion (average the two rank vectors) | 0.6970 at best |
| margin gate (native alone when its margin is in the top 20%) | 0.7341 |

None beats the single linear weight. Rank fusion is worse by 3.7 pp because it
throws away the sizes of the margins, which is exactly the information the two
scorers disagree about. The gate does nothing because the native margin is
already monotone in what the blend uses.

**The linear blend is at the ceiling of what post-hoc combination of these two
scores can do.** Closing clause 3's gap needs the short-list bucket 3.4 pp
higher, and nothing that only reweights the two existing scores gets there. Test-
time augmentation over option order is also excluded in advance rather than by
experiment: clause 3's flip rate is 0/1,500, so the head has no order sensitivity
for augmentation to average away.

## It replicates on the second seed

Everything above is seed 123 at epoch 10. The claim that matters for the goal is
whether the rule beats what ships on a *second* seed, so the same full-eval run
was repeated on `arch_4b_32k_split138` epoch 5 — seed 42, the checkpoint the
routing rule was independently verified on in `research/17`.

| checkpoint | head | shipped routing | blend | blend − routing | blend − head |
|---|---|---|---|---|---|
| seed 123, epoch 10 | 0.6322 | 0.7060 | 0.7398 | **+3.38 pp** | +10.76 pp |
| seed 42, epoch 5 | 0.5627 | 0.6893 | 0.7131 | **+2.38 pp** | +15.04 pp |

Overall accuracy, converting each choice delta onto that checkpoint's own harness
counts:

| checkpoint | head alone | shipped routing | blend |
|---|---|---|---|
| seed 123, epoch 10 | 69.24% | 73.01% | **74.75%** |
| seed 42, epoch 5 | 64.19% | 70.67% | **71.89%** |

**The blend beats the routing rule on both seeds**, by 1.73 and 1.22 pp overall.
The two seeds disagree about the best weight — 2.5 at epoch 10, 4.0 at epoch 5 —
but not about the direction: every weight from 1 to the native endpoint is
positive against head-only on both, and a single fixed `lam = 4` is +3.10 pp on
seed 123 and +2.38 pp on seed 42. The choice between 2.5 and 4 is worth 0.28 pp
on seed 123 and 0.32 pp on seed 42, both far inside the per-seed spread.

The second seed also matters for what does *not* replicate: the long-list bucket
is much weaker at epoch 5 (0.5960 against 0.8180), and JevBench's hard tier is
higher (0.4627-0.4925 against 0.3731-0.4030). Neither changes the rule's
direction — the head is kept there regardless — but both say the epoch-10 figures
are a later-training snapshot, not a seed-independent constant.

## The matched pair: both seeds at epoch 5

The table above compares seed 123 at epoch 10 with seed 42 at epoch 5, which is
not like-for-like — the two differ in training length as well as in seed. The
matched comparison needs both seeds at the same epoch on the same split, and both
checkpoints exist at epoch 5:

| checkpoint (epoch 5, cluster-disjoint split) | head | shipped routing | blend | blend − routing | blend − head |
|---|---|---|---|---|---|
| seed 123 | 62.16% | 68.69% | **70.12%** | **+1.43 pp** | +7.96 pp |
| seed 42 | 64.18% | 70.66% | **71.88%** | **+1.22 pp** | +7.70 pp |

This is the pair that settles the claim. Same split, same training length, same
items, same rule: **the blend beats what ships by 1.43 and 1.22 pp overall, and
the head alone by 7.96 and 7.70 pp.** The two seeds agree on the effect to within
0.2 pp even though they disagree about the best weight, and both put the model
above clause 1's 65% bar through the shipped rule where the head alone is still
below it (62.16% and 64.18%).

The choice rows, for the same pair: seed 123 goes 0.5386 head → 0.6660 routed →
0.6941 blended; seed 42 goes 0.5627 → 0.6893 → 0.7131.

## Where the gain comes from, and why routing per source does not beat it

The two scorers are not uniformly ordered. Per source on the 6,387 short-list
items (seed 123, epoch 10), head against the backbone's native readout:

| source | head | native | n |
|---|---|---|---|
| ARC | 0.580 | **0.878** | 500 |
| race | 0.512 | **0.762** | 500 |
| authored144 | 0.514 | **0.792** | 144 |
| commonsenseqa | 0.538 | **0.720** | 500 |
| MMLU | 0.338 | **0.504** | 500 |
| LogiQA | 0.350 | **0.444** | 500 |
| synthetic | 0.538 | **0.691** | 301 |
| winogrande | 0.552 | **0.612** | 500 |
| codesearchnet | 0.933 | **0.977** | 300 |
| PIQA | 0.748 | 0.754 | 500 |
| SWAG | **0.660** | 0.594 | 500 |
| HellaSwag | **0.738** | 0.668 | 500 |
| FEVER | **0.804** | 0.752 | 500 |
| agnews | **0.886** | 0.844 | 500 |
| banking77 (>20 options) | **0.818** | 0.326 | 500 |

**The trained head is much worse than the untouched reranker readout on the
reasoning-flavoured short-list sources** — ARC by 30 points, race by 25,
authored144 by 28, CommonsenseQA by 18, MMLU by 17 — and much better on
banking77 (+49) and slightly better on SWAG, HellaSwag, FEVER and agnews. That is
the same pattern `research/13` found on JevBench from the outside: fine-tuning
buys the task it is trained on and costs general reranking ability elsewhere.

It also rules out the obvious alternative. If the sources were consistently one
way or the other, routing per source would beat blending. It does not:

| rule on the ≤5 bucket | accuracy |
|---|---|
| head only | 0.6177 |
| native only (what ships) | 0.6970 |
| **oracle per-source choice of the better scorer** | 0.7158 |
| blend, `lam = 2.5` | **0.7341** |

The oracle pick is *worse* than the blend, and it uses knowledge a deployment
does not have. Blending beats both scorers on individual sources because the two
disagree item by item, not just source by source — which is why a single global
weight is the right instrument and per-source routing is not.

## A third score vector from the untrained reranker: inconclusive, not adopted

If fine-tuning cost the backbone reranking ability on ARC and race, the *original*
reranker — never trained on our task — might be a better second opinion there.
That is testable: score the same 900 short-list sample items with the base
Qwen3-Reranker-4B and blend three ways.

| scorer on the 900 short-list items | accuracy |
|---|---|
| trained head | 0.6411 |
| fine-tuned backbone's native readout | **0.6911** |
| base reranker's native readout | 0.6489 |

**The base reranker is worse than the fine-tuned one, not better.** Whatever the
per-source table above shows, fine-tuning improved the native readout overall:
0.6911 against 0.6489. The per-source losses are real but they do not add up to a
net regression, and the "fine-tuning damaged the reranker" reading of that table
is wrong as a summary.

The three-way blend is not established either. Best over a 48-point grid is
0.7378 at `(fine-tuned 1.5, base 0.5)` against the two-way 0.7300, and a paired
bootstrap over 2,000 resamples puts the difference at **+0.0100 [−0.0022,
+0.0233], P(>0) = 0.944** — the interval includes zero, on one seed, on a
900-item sample. By this project's own bar that is not a finding, and it is not
adopted. Settling it would need both seeds on the full 6,387 items, which is a
further ~40 minutes of GPU per checkpoint for a +1 pp question.

## Caveats

- **`lam` is chosen on the eval split.** That is why the cross-fit paragraph
  above exists: the two halves agree (2.5 and 3.0), the cross-fitted accuracy is
  0.7379 against 0.7058 for the routing rule, and the overall number computed
  from it is 74.66% against 74.75% at a fixed 2.5. The gain survives, but a
  deployment should fit `lam` on held-out data rather than on the eval split.
- **The sample rows are a sample.** The 60-per-source table earlier in the note
  was the development instrument; small sources (`authored144` 60/144,
  `typed_decisions` 60/142) are over-weighted in it and its bucket accuracies
  carry ±1.5 to ±4 pp. The full-eval section is the one to quote, and it is
  larger in every direction.
- **JevBench is the clean test, and there the blend ties the native scorer.** It
  reaches 0.6475 against native-only's 0.6547 on the 139 choice items — a
  one-item loss. What the out-of-sample data supports is that the blend does not
  *hurt* relative to the native scorer on short lists; the +3.4 pp over
  native-only is in-distribution only.
- **`>=20` is banking77.** It is the only choice source with more than 19
  options, so the long-list column is one benchmark, as it is for the bucket
  comparisons in `research/17`.
- **The head's full-eval choice accuracy (0.6322) is not `research/13`'s 0.5667.**
  That table used 1,500 items per type from the clean corpus; this is the whole
  6,887-item choice eval. Neither should be quoted against the other.
- **Two scorers, one backbone.** The native score comes from the same fine-tuned
  weights through the reranker's own readout, not from an independent model, so
  this measures two readouts of one backbone rather than an ensemble.

## What this means for the goal

It is the first change this session that moves a clause in the right direction
without training. **Overall eval accuracy goes from 73.01% to 74.75%** — +1.73 pp
over the routing rule the model ships, +5.51 pp over the head alone — and clause
1's 65% bar is cleared by a wider margin than before. The short-list bucket, the
one where the head is weakest, moves +3.7 pp.

It moves clause 3 without meeting it: the ≤5 vs ≥20 gap falls from −12.08 pp to
−8.39 pp, still outside the 5 pp band, and the blend would need the short-list
bucket at 0.768 to close it.

It does not touch clause 4. On JevBench the blend reproduces the native readout's
0.6547 and stops there, so the gap to the 74.1 reference is unchanged; the hard
tier moves from 0.373 to 0.403, which is two items on 67.

The feature version that `research/13` proposed is still untested, and this
result makes it more interesting rather than less: a *learned* combination does
not have to pick one `lam` for every short item, and the two scorers disagree on
enough items for that to matter.
