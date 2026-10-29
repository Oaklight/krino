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
`model/scripts/blend_scorers.py`, artifact at
`model/experiments/blend_scorers_s123_ep10.json`. The per-item score vectors are
not committed (1.1 MB); the run that produced them writes them with
`--dump-records` and the sweep re-runs over them without a GPU with
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

## The deployable rule

The current model routes: native for ≤5 options, head above 20. On this sample
that scores **0.7076**. Two blends do better:

| rule | ≤5 | ≥20 | all | vs routing |
|---|---|---|---|---|
| routing (native ≤5, head ≥20) | 0.6911 | 0.8067 | 0.7076 | — |
| single `lam = 1` everywhere | 0.7278 | 0.7533 | 0.7314 | +2.4 pp |
| `lam = 1` on ≤5, head on ≥20 | 0.7278 | 0.8067 | 0.7390 | **+3.1 pp** |
| `lam = 1.5` on ≤5, head on ≥20 | 0.7300 | 0.8067 | 0.7410 | **+3.3 pp** |

The third row is the recommendation: keep the head for long lists, replace the
hard native switch with a blend on short lists. It costs the native scorer's
forward passes on short items — at most five per item, against one for the head —
and nothing extra on long lists. The fourth row is inside the noise of the first
(2 items on 900), so `lam = 1` is the value to ship; the ≤5 optimum is flat
between 1.0 and 1.5.

## Caveats

- **The in-distribution rows are in-sample for `lam`.** The scale is fitted on
  the same items and the weight is chosen on them. The JevBench rows are the
  clean test — different corpus, and `research/13`'s companion audit shows it is
  disjoint from training — and there the blend ties the native scorer rather than
  beating it. What the out-of-sample data supports is that a blend does not
  *hurt* relative to the native scorer on short lists; the +3.9 pp over
  native-only is in-distribution only.
- **1,050 in-distribution items, 60 per source.** Sampling is per source, so
  small sources (`authored144` 60/144, `typed_decisions` 60/142) are
  over-weighted relative to their share of the full 6,874-item choice eval. The
  ≤5 and ≥20 rows have 900 and 150 items, so the bucket accuracies carry roughly
  ±1.5 and ±4 pp of sampling noise. Every gap above that clears it.
- **`>=20` is banking77.** It is the only choice source with more than 19
  options, so the long-list column is one benchmark, as it is for the bucket
  comparisons in `research/17`.
- **The head's in-distribution accuracy here (0.6648) is not the 0.5667 quoted in
  `research/13`.** Different sample: that table used 1,500 per type from the
  clean corpus, this uses 60 per source. The two are not comparable and neither
  should be quoted against the other; what is comparable is within this run.
- **Two scorers, one backbone.** The native score comes from the same fine-tuned
  weights through the reranker's own readout, not from an independent model, so
  this measures two readouts of one backbone rather than an ensemble.

## What this means for the goal

It is a deployable gain with no training: +3.1 pp on in-distribution choice, on
top of the +3.77 pp that routing already gave (`research/17`), and it moves the
short-list bucket — the one where the head is weakest — by +8.9 pp. It does not
touch clause 4: on JevBench the blend reproduces the native readout's 0.6547 and
stops there, so the gap to the 74.1 reference is unchanged.

The feature version that `research/13` proposed is still untested, and this
result makes it more interesting rather than less: a *learned* combination does
not have to pick one `lam` for every short item, and the two scorers disagree on
enough items for that to matter.
