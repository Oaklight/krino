# The cross-encoder head: +18 points at 4B, +21 on the slot it replaces, and nothing on the hard tier

The choice head has been a tower: the state is encoded once, each option on its
own, and a rank-64 bilinear against the pooled state scores the option. The
alternative is the shape the backbone's own reranker uses — put the state, the
instructions and the option in **one prompt**, run it through every layer, and
read a single number out of the pooled hidden state. The softmax across options
stays where it was either way.

This note reports what that swap buys, on the two sizes the project trains, and
where it buys nothing. Head, code and flag: `#192`, `--choice-mode
cross_encoder`.

## In distribution, at the same budget

Both comparisons are matched on data, split, seed, LoRA scope and learning rate;
the only difference is `--choice-mode`. Epoch 1 for both arms, 13,444-item
cluster-disjoint eval:

| | overall | choice (6,887) | noul (4,725) | score (1,832) |
|---|---:|---:|---:|---:|
| 0.6B tower (`weighted`) | 0.4794 | 0.363 | 0.704 | 0.337 |
| **0.6B cross-encoder** | **0.6048** | **0.5718** | 0.7302 | 0.4056 |
| — | | | | |
| 4B tower (`last`) | 0.5399 | 0.391 | 0.797 | 0.438 |
| **4B cross-encoder, s123** | **0.7231** | **0.721** | 0.822 | 0.477 |
| **4B cross-encoder, s42** | **0.7190** | **0.715** | 0.823 | 0.470 |

**+12.5 pp at 0.6B and +18.3 pp at 4B**, and the gain sits in the slot the head
replaces: choice goes +20.9 pp at 0.6B and +33 pp at 4B. The 4B cross-encoder's
*choice* accuracy at epoch 1 (0.721) is 9.5 pp above the 4B tower's choice at
epoch **10** (0.626). The effect grows with the backbone rather than washing out,
which is the opposite of what a rank-limited scorer would do.

The 0.6B row replicates on a second seed: 0.6048 (s123) against **0.6028** (s42)
overall, choice 0.5718 against 0.5648, noul 0.7302 against 0.7184, score 0.4056
against 0.4520. Two seeds, 0.2 pp apart on the aggregate. The per-type counts
differ slightly between the two draws (choice 6,887 against 6,900) because the
eval mixture is sampled, so these are not identical item sets — which is the
right caveat on a 0.2 pp agreement, not a reason to distrust it.

`score` moves by much less (0.337 → 0.406 at 0.6B, 0.438 → 0.477 at 4B) and at
4B epoch 1 it is still below the tower's epoch-10 score (0.477 against 0.547).
The score head is untouched by this change and needs its epochs.

## What it costs

**Training: +5.7 % wall clock.** The design note that motivated the head assumed
it would cost "one full-depth forward per option instead of one per item". It
does not, because the tower already runs one forward per option — the extra is
the state prepended to each of them. 0.6B epoch 1 measured 7,075 s against the
tower's 6,695 s.

**Inference: it is faster at small K and much slower at large K.** One H100 per
row, eager, batch 1, ~200-token state:

| options | 4B tower | 0.6B cross-encoder |
|---:|---:|---:|
| 2 | 68.3 ms | **27.7 ms** |
| 5 | 70.8 | **29.1** |
| 20 | 75.0 | **42.4** |
| 64 | 97.9 | 133.3 |
| 256 | **244.6** | 490.2 |

The tower's cost grows with the option count alone; the cross-encoder's grows
with *state × options*, so it wins while the state is the expensive part and
loses badly once K dominates. Clause 5's 100 ms bar at 256 options is 2.4× away
for the 4B tower and 4.9× for the 0.6B cross-encoder — measured on H100, which is
the faster card, so the negative result only strengthens on the A100 the clause
names. `bench_latency.py` gained `--choice-mode` so this is measurable at all
(`#194`).

## It widens the gap clause 5 cares about

Clause 5 also asks for the 0.6B to sit within 3 pp of the 4B, and the cross-encoder
moves that the wrong way. At matched epoch 1 on the same split:

| | 0.6B | 4B | gap | allowance |
|---|---:|---:|---:|---:|
| tower (0.6B weighted, 4B `last`) | 0.4794 | 0.5399 | **6.1 pp** | 3 pp |
| **cross-encoder** | **0.6048 / 0.6028** | **0.7231 / 0.7190** | **11.6–11.8 pp** | 3 pp |

So the head that lifts the board by 12.5 points at 0.6B lifts it by 18.3 at 4B,
and the capacity gap doubles from 6.1 to about 11.7 points. That is the expected
shape of a change that lets the option interact with the state through the full
depth: more capacity to use, and more of it at the larger size. It does not make
clause 5's accuracy half reachable, and it is worth saying plainly that the two
clauses are now further apart than they were before this change — the arm that
meets clause 1 on the aggregate is not the arm that meets clause 5, and the
cross-encoder is not the arm that meets either clause at 0.6B.

## It halves the option-count degradation

Clause 3's count half asks accuracy at more than twenty options to sit within 5
points of accuracy at five or fewer. `research/17` measured a matched sweep on the
tower — 300 fixed banking77 items, the same questions padded and shrunk to each K —
and found 12.67 points at epoch 10. The same sweep on the two head shapes at the
**same training length** (4B, split138, epoch 3, seed 42, 300 items, the same
distractor pool):

| options | tower | cross-encoder |
|---:|---:|---:|
| 4 | 0.7300 | 0.7367 |
| 8 | 0.6467 | 0.6567 |
| 16 | 0.4433 | 0.5733 |
| 32 | 0.3300 | 0.4933 |
| 64 | 0.1700 | 0.4433 |
| **80** | **0.1567** | **0.4200** |
| K=4 → 80 | **−57.3 pp** | **−31.7 pp** |

At matched training the two shapes are level at four options and 26 points apart
at eighty, and the degradation is roughly halved. That is what a head that reads
each option out of the state's full context should do: the tower compares option
vectors against one encoded state through a rank-limited attention, so adding
distractors dilutes the comparison; the cross-encoder never has to compare, it
scores each option in place.

**Two comparisons that must not be confused.** The tower at epoch **10** scores
0.9767 / 0.9700 / 0.9533 / 0.9133 / 0.8733 / 0.8500 on this sweep — 12.67 points,
better than *either* epoch-3 arm. Training length dominates the level, so the
epoch-3 pair is the comparison that isolates the head shape and the epoch-10 pair
is the one that says how far training takes it. Whether the cross-encoder's
five-epoch run approaches that curve is not yet measured; at epoch 3 it is already
26 points ahead at eighty options, on 42 % against 16 %.

Clause 3's count half is still not met by either — 31.7 points against a 5-point
allowance — but the arm that meets clause 1 is now also the arm that comes
closest to this one.

## On held-out data it moves the board, and the hard tier moves too

JevBench-public, 231 items, states rendered through the loader (`#180`), LoRA
applied, same loader for every row:

| | easy (48) | hard (111) | original (72) | overall (231) |
|---|---:|---:|---:|---:|
| 4B tower, epoch 10 (s123) | 0.792 | 0.405 | 0.486 | 0.511 |
| 0.6B cross-encoder, epoch 1 | **1.000** | 0.396 | **0.611** | **0.589** |
| **4B cross-encoder, epoch 1** | **1.000** | **0.405** | **0.722** | **0.628** |
| **4B cross-encoder, epoch 3** | **1.000** | **0.441** | 0.681 | **0.632** |
| **4B cross-encoder, epoch 4** | **1.000** | **0.450** | 0.597 | 0.610 |

The in-distribution eval and the board do not move together: between epochs 3 and
4 the 13,444-item eval gained 1.7 points (0.7291 → 0.7462) while the board lost
2.2 (0.632 → 0.610), almost all of it on the `original` tier (0.681 → 0.597, six
items). Neither of those board moves is resolvable at 72 to 231 items, so the
honest reading is that the board is *not tracking* in-distribution gains rather
than that it is actively degrading — which is itself the useful fact, because it
means more training on this mixture is not a route to the authored tiers.

**Correction, twice.** The first version of this note said the change moves the
hard tier "by exactly nothing" — 0.405 at 4B before the change and 0.405 after it,
three models on one number. That was an unpaired reading of four point estimates.
The second version said the tier *does* move, because the same arm scored 0.441 at
epoch 3 and 0.450 at epoch 4. That was also wrong, for a reason the first version
shared: 111 items cannot resolve differences of this size. Paired on the items,
with `model/scripts/paired_mcnemar.py`:

| comparison | hard tier | discordant | McNemar p |
|---|---:|---:|---:|
| tower ep10 vs cross-encoder ep3 | 0.405 vs 0.441 | 16 / 20 | 0.62 |
| tower ep10 vs cross-encoder ep4 | 0.405 vs 0.450 | 16 / 21 | 0.51 |
| cross-encoder ep3 vs ep4 | 0.441 vs 0.450 | 9 / 10 | 1.00 |

**Every comparison on this tier is indistinguishable from a coin flip.** The four
models differ on 35 to 37 of 111 items and split those differences roughly evenly,
which is what two models of similar quality look like on a set this size. So the
honest statement is not that the tier is invariant and not that it moves: it is
that **this tier cannot resolve a difference below about ten points**, and every
hard-tier claim in this note and in `research/13` should be read as a point
estimate with that width. Closing clause 4 needs a larger held-out set before any
model change can be told apart from noise.

The leakage audit (`scripts/check_jevbench_leakage.py`) finds **0 exact
duplicates, 0 state repeats and 0 near-duplicates above 0.5 shingle containment
over 1,562,838 training items** (max containment 0.001), so none of the movement
that *is* visible elsewhere on the board is memorisation.

It also does not close the tier. The hard items are a **long-state wall**, and the
quartile cut below is recomputed so both architectures are on the same boundaries
(`scripts/jevbench_quartiles.py`, which prints them):

| hard-tier state length | n | 4B tower, ep10 | 4B cross-encoder, ep3 |
|---|---:|---:|---:|
| 350 – 987 chars | 27 | 0.630 | **0.704** |
| 995 – 1,553 | 28 | 0.393 | 0.393 |
| 1,583 – 8,619 | 28 | 0.357 | 0.250 |
| 8,676 – 14,986 | 28 | **0.250** | **0.429** |
| overall | 111 | 0.405 | 0.441 |

The tower loses 38 points from the shortest to the longest quartile; the
cross-encoder loses 27 and not monotonically, dipping in the third quartile and
recovering in the fourth. With 27–28 items per cell a quartile's standard error is
about 9 points, so only the tower's Q1-to-Q4 fall is clearly resolvable and the
cross-encoder's non-monotonicity is not. The architecture difference lands on the
longest quartile — the cross-encoder is 18 points ahead there (0.429 against
0.250) — which is the quartile a read-out attending over the whole context should
lose first, and it is the same phenomenon `research/22` measures directly. On 28
items, 18 points is a little over one standard error of the difference, so this
is a hypothesis the padding experiment supports rather than establishes on its
own; the padding experiment is the one that isolates it.

The earlier version of this table carried a single "trained head" column and a
native-readout column without naming the checkpoint behind the head, which is why
the 0.405-invariance claim survived as long as it did. It was the tower.

Neither loss is truncation: the longest state in the tier is ~3.7k tokens, inside
the 16,384 window. Structured states are not the problem either (0.486 on the 35
dict-state items against 0.368 on the 76 string-state ones; the structured items
are simply shorter), and option count is not (2–4 options 0.444, 5–10 0.238).
Whatever keeps clause 4 out of reach survives both heads and both read-outs, and
`research/22` isolates part of it: the tower loses 24 points when the same
question is padded with 1,400 tokens of irrelevant text, and the cross-encoder
loses 1. So the tower's long-state penalty is at least partly dilution, which is
the reading that fits it being 18 points behind on the longest quartile — and the
cross-encoder's remaining 27-point spread over the quartiles is something else,
on the same items, at the same lengths, with no padding involved.

## Epoch 2 dipped on one seed and not on the other

The 4B arm's first three epochs, both seeds:

| | overall | choice | noul | score | eval loss |
|---|---:|---:|---:|---:|---:|
| s123, epoch 1 | 0.7231 | 0.721 | 0.822 | 0.477 | 0.727 |
| s123, epoch 2 | **0.3761** | 0.292 | 0.555 | 0.229 | **1.362** |
| s123, epoch 3 | **0.7291** | **0.737** | 0.825 | 0.452 | 0.745 |
| s42, epoch 1 | 0.7190 | — | — | — | 0.752 |
| **s42, epoch 2** | **0.7174** | — | — | — | 0.741 |

**The second seed does not dip.** s42 moves 0.7190 → 0.7174 across the same epoch
— 0.2 points, in the direction of the training loss (0.9411 → 0.8437), with no
excursion. So the epoch-2 collapse is a property of one run, not of the arm, and
two earlier readings of it should be retired together: "the 4B arm collapses at
epoch 2" (wrong — it recovered), and "the cross-encoder is unstable at epoch 2"
(also wrong — it is unstable once, on s123, in a way that leaves no trace on s42).

**On s123, where the excursion lives.** Rebuilding that checkpoint with epoch 2's
adapter and epoch 1's heads and scoring it: `choice` **0.289**, the same as epoch
2's own pair (0.292). Putting the good head back does not recover the accuracy, so
*that* excursion was in the LoRA adapter, not in the linear read-out: at that
moment the adapter produced a representation neither head could read.

**What that means for how to read the epoch-1 rows.** They are two seeds agreeing
to 0.4 points, and s42's epoch 2 agrees with them too, so "epoch 1" is not a lucky
draw before a fall. What it still is: one point on each run's learning curve, taken
before either arm finished. The 0.6B JevBench row was scored at epoch 1 of a
one-epoch run and is unaffected either way.

## Caveats

- **Two seeds at 0.6B, two at 4B, compared at epoch 1; s42 also has epoch 2.** The
  0.6B rows are 0.6048 and 0.6028, the 4B rows 0.7231 and 0.7190, and s42's epoch
  2 is 0.7174. The per-type counts differ by up to 13
  items between seeds because the eval mixture is sampled, so the comparison
  across seeds is not on identical items.
- **The schedules differ.** The tower baselines ran 10 epochs; the 0.6B
  cross-encoder arms are **one-epoch runs**, and the 4B arms run 5. So "epoch 1"
  is the whole run at 0.6B and the first of five at 4B, and the comparison is at
  matched epoch and matched data, not at matched schedule.
- **The 0.6B JevBench row is one epoch**, so it is a floor, not a ceiling.
- **The option-count sweep's two arms differ in readout as well as in head.** The
  epoch-3 tower checkpoint is a `last` arm and the cross-encoder a `weighted` one,
  because that is how each was trained. Readout is not the manipulation, so part of
  the 26-point gap at eighty options could be the readout rather than the head.
- **Nothing here says the head helps `score` or `noul`**; the change is confined
  to choice, and the numbers above show it.
