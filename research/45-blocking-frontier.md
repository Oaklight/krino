# Blocking readout frontier at K = 150 (#356) — 2026-11-09

The blocking design of #356, run on CLINC150 at K = 150. **Scheme B — block,
then one panel-2 prompt over the block winners — is the best of the three
schemes at every block size.** It beats A at every size under both block
assignments and beats C at `b = 26` under both; the B-over-C margin is
significant at `b = 52` and `b = 32` only under the class-order assignment. The
best measured point is `b = 26`: 153/200 = 76.5%
at 7 forwards per item; `b = 32` is 146/200 = 73.0% at 6. On the same 200 items
full judgement over all 150 options scores 127/200 = 63.5% at 150 forwards, so
B at `b = 52` matches it at 4 forwards and B at `b = 26` exceeds it by 26 items.

Two pre-registered predictions failed. Scheme C was predicted to beat B; it
does not, at any block size (exact McNemar p = 0.019, 0.0090, 0.0017 on all 200
items). And the end-to-end accuracy at `b = 52` was predicted near 43% from a
final-pass anchor of 45.6%; the final pass measured 92.9%, so the cell lands at
65.0%.

## Design

CLINC150 `test`, in-scope rows only, shuffled with `random.Random(342)`, first
200 — the same items as the CLINC single-pass and tier cells. K = 150 = every
in-scope class, so the gold's position is its class index. Config hash
`5b7889815db0c7231f47cf1e0813a4a643ff865e9be3e43bf960bd643048aeab`, printed
before the run.

Blocks are consecutive slices of the class order: `ceil(150/b)` blocks of `b`
options for `b` = 52, 32, 26 (3, 5, 6 blocks). Within a block, options are
labelled `A`–`Z`, `a`–`z`. The gold is in exactly one block.

| scheme | final decision | forwards |
|---|---|---:|
| A | compare the block winners by their raw code logit | `blocks` |
| B | one panel-2 prompt over the block winners | `blocks + 1` |
| C | per-option judgement over the block winners | `2 × blocks` |

The forward counts follow the run's convention: a panel-2 prompt is one
forward, a judgement option is one forward. Each judgement call also pays a
context-prefix forward, which the convention does not count, so scheme C is
slightly dearer than the table shows.

## Results

All 200 items, exact McNemar against scheme B.

| scheme | b | blocks | forwards | correct | accuracy | vs B |
|---|---:|---:|---:|---:|---:|---:|
| A | 52 | 3 | 3 | 94/200 | 47.0% | 5.6e-9 |
| **B** | 52 | 3 | 4 | **130/200** | **65.0%** | — |
| C | 52 | 3 | 6 | 119/200 | 59.5% | 0.019 |
| A | 32 | 5 | 5 | 112/200 | 56.0% | 1.2e-6 |
| **B** | 32 | 5 | 6 | **146/200** | **73.0%** | — |
| C | 32 | 5 | 10 | 130/200 | 65.0% | 0.0090 |
| A | 26 | 6 | 6 | 108/200 | 54.0% | 3.0e-9 |
| **B** | 26 | 6 | 7 | **153/200** | **76.5%** | — |
| C | 26 | 6 | 12 | 133/200 | 66.5% | 0.0017 |

Scheme C is dominated by scheme B at every block size: fewer correct answers
at more forwards. Scheme A is below B by 18.0, 17.0 and 22.5 pp at b = 52, 32
and 26, so the extra pass is paid for by a wide margin.

B at `b = 26` and `b = 32` are not separable: 153 against 146, 16 items correct
under one only against 9 under the other, exact p = 0.23.

### Decomposition

End-to-end = `P(gold wins its block) × P(final picks the gold | it won)`.

| b | gold wins block | B final given won | B product | C final given won | C product | A final given won | A product |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 52 | 140/200 = 70.0% | 130/140 = 92.9% | 65.0% | 119/140 = 85.0% | 59.5% | 94/140 = 67.1% | 47.0% |
| 32 | 166/200 = 83.0% | 146/166 = 88.0% | 73.0% | 130/166 = 78.3% | 65.0% | 112/166 = 67.5% | 56.0% |
| 26 | 174/200 = 87.0% | 153/174 = 87.9% | 76.5% | 133/174 = 76.4% | 66.5% | 108/174 = 62.1% | 54.0% |

Smaller blocks raise the first factor as predicted (70.0 → 83.0 → 87.0%) and
barely move the second (92.9 → 88.0 → 87.9%). The second factor is the strong
one, at 88–93%, not the weak one the pre-registration assumed.

### Within-block position

Each cell's block is only `b` effective candidates if its tail is readable. The
block-win rate by the gold's within-block position, over all 200 items:

| b | first 20% of block | last 20% of block | difference | Fisher exact p |
|---:|---:|---:|---:|---:|
| 52 | 34/43 = 79.1% | 26/44 = 59.1% | +20.0 pp | 0.063 |
| 32 | 43/45 = 95.6% | 23/38 = 60.5% | +35.0 pp | 8.6e-5 |
| 26 | 46/54 = 85.2% | 35/39 = 89.7% | −4.6 pp | 0.76 |

At `b = 32` the tail is dead and the block is not 32 effective candidates. At
`b = 26` there is no measurable gradient. This is consistent with the `b = 32`
versus `b = 26` difference being the 32-option tail rather than a property of
block size alone; the bins are not a designed position manipulation, so that
reading is inference from the two measurements, not a controlled comparison.

A block size fixed by a class order rather than by position means the bins are
also confounded with which classes sit in them. The direction reproduces the
CLINC single-pass gradient (82.6% to 59.1% across gold-position buckets), so
the dead-tail reading is consistent, but it is not a designed position
manipulation. The shuffled assignment below is the control for that confound.

### The shuffled assignment

The declared sensitivity variant: the 150 class indices are permuted per item
with `random.Random(f"{342}:{item_id}")` and cut into consecutive slices of the
permuted order. Config hash
`9eab2515e8705abf4defe5a07a975989b02e85cf593f70880a96462add081149`, printed
before the run. Per item rather than one global permutation, because a global
permutation only replaces one fixed class-to-position assignment with another
and does not remove the confound; per-item shuffling makes the gold's local
position uniform across items by construction.

| b | won block | A | B | C | B, consecutive vs shuffled |
|---:|---:|---:|---:|---:|---:|
| 52 | 142/200 = 71.0% | 108/200 = 54.0% | 129/200 = 64.5% | 126/200 = 63.0% | 129 vs 130, p = 1.00 |
| 32 | 163/200 = 81.5% | 117/200 = 58.5% | 138/200 = 69.0% | 133/200 = 66.5% | 138 vs 146, p = 0.32 |
| 26 | 171/200 = 85.5% | 122/200 = 61.0% | 148/200 = 74.0% | 131/200 = 65.5% | 148 vs 153, p = 0.49 |

No cell differs from its consecutive counterpart: paired exact McNemar for
scheme B gives p = 1.00, 0.32 and 0.49; for the block-win rate 0.90, 0.74 and
0.69; no scheme–block-size pair falls below p = 0.12. **The recorded frontier is
not an artifact of the dataset's class order.**

The scheme ranking persists in point estimates at every block size, B above C
above A. B against A stays significant (p = 5.1e-4, 1.9e-3, 3.1e-4); B against
C is significant only at `b = 26` (p = 0.0060) and not at 52 or 32 (0.61, 0.46).
So the B-over-C margin is smaller under a random assignment than under the
class order, and only the `b = 26` margin survives as a significant one.

B at `b = 26` against `b = 32` under the shuffle: 148 against 138, +5.0 pp,
exact p = 0.0525; on the 50-item judgement subset both are 37/50 (41 against 40
under the class order). The pre-registered rule fires the "within 5 pp" branch:
the block-size choice is not resolvable and both points stand. The `b = 26` lead
over `b = 32` comes from the 150 items outside the judgement subset in both
assignments, so it cannot be read as a block-size effect either way.

Position under the shuffle: `b = 52` first 44/52 = 84.6% against last 19/33 =
57.6% (p = 0.010); `b = 32` 37/42 = 88.1% against 31/41 = 75.6% (p = 0.16);
`b = 26` 44/49 = 89.8% against 30/34 = 88.2% (p = 1.00). The 52-option tail
deficit survives the shuffle unchanged; the 32-option gradient is reduced by
the shuffle (60.5% to 75.6%) and is no longer significant; `b = 26` is flat in
both assignments. Part of the 32-option tail deficit is therefore the class
order, part is block width; the 52-option deficit is not the class order.

## The judgement point, on the same items

Full judgement over all 150 options, on all 200 items. Config hash
`55925975450fd1cb4465e98c1d4f39d40b7b8dac2890c793206e96eaf0843fdc`, printed
before the run. It scores **127/200 = 63.5%**, against 64.94% for the same
instrument on JevBench. The first 50 items are the same first 50 as the
blocking run and reproduce its 34/50 exactly, so the instrument is identical.

| point | correct/200 | accuracy | forwards | exact p vs judgement |
|---|---:|---:|---:|---:|
| full judgement | 127 | 63.5% | 150 | — |
| B, b = 52 | 130 | 65.0% | 4 | 0.80 |
| B, b = 32 | 146 | 73.0% | 6 | 0.0079 |
| B, b = 26 | 153 | 76.5% | 7 | 4.1e-4 |
| C, b = 52 | 119 | 59.5% | 6 | 0.28 |
| C, b = 32 | 130 | 65.0% | 10 | 0.66 |
| C, b = 26 | 133 | 66.5% | 12 | 0.26 |
| A, b = 52 | 94 | 47.0% | 3 | 5.1e-5 |
| A, b = 32 | 112 | 56.0% | 5 | 0.077 |
| A, b = 26 | 108 | 54.0% | 6 | 0.027 |

Nine comparisons against one reference: the Bonferroni threshold is 0.0056. B
at `b = 26` clears it; B at `b = 32` does not (0.0079) under the nine-way
family, though it does under a three-way family for the three B cells (0.0167).
No C or A cell is above judgement.

The same cells on the 50-item subset where the blocking run first measured
judgement: B at `b = 52` 33/50, `b = 32` 41/50, `b = 26` 40/50, judgement
34/50; paired exact p = 1.00, 0.039 and 0.11. The subset agrees in direction
with the 200-item cell and cannot resolve it.

**Each point's margin over judgement, under both block assignments.** Under the
per-item shuffle the same reference gives B at `b = 52` 129 (p = 0.90), `b = 32`
138 (p = 0.18) and `b = 26` 148 (p = 0.0046). The margin at `b = 52` is a
non-difference in both assignments; the margin at `b = 26` is significant in
both (4.1e-4 and 4.6e-3); the margin at `b = 32` is significant under the class
order (0.0079) and not under the shuffle (0.18). The `b = 26` and `b = 32`
difference itself stays unresolved: +5.0 pp under the shuffle (p = 0.0525) and
37 against 37 on the 50-item subset.

Measured wall time per item on the 50-item subset: scheme B totals 0.44 s
(`b = 52`), 0.57 s (`b = 32`) and 0.53 s (`b = 26`); scheme C totals 0.72 s,
0.90 s and 0.90 s; full judgement is 11.83 s (11.68 s averaged over all 200).
These are from one run on one machine; the ratio to full judgement (about 20×)
is the portable part.

## Pre-registered predictions and what happened

| prediction, written before the run | outcome |
|---|---|
| Scheme C beats B, so the shape is "block, then a judgement finish" | **failed**: B > C at all three sizes, p = 0.019 / 0.0090 / 0.0017 |
| B at `b = 52` ≈ 43% = 0.73 × P(final over 3) | **failed**: 65.0%. Block-win 70.0%, final over 3 = 92.9% |
| End-to-end well below the 62.5% single pass at K = 80 | **failed**: B at `b = 52` is 65.0%, B at `b = 26` is 76.5% |
| Full judgement ≈ 65% | **confirmed**: 63.5% on 200 items; 68% on the 50-item cell |
| If A is within 5 pp of B, the extra pass is not paid for | **decided**: A is 18.0–22.5 pp below B; the pass is paid for |
| The position split shows whether the block width is effective | **decided**: `b = 32` has a dead tail, `b = 26` does not |
| Verification: full judgement 60–72% on 200, first 50 reproduces 34/50 | **confirmed**: 127/200 = 63.5%; first 50 = 34/50 |
| Verification: shuffled B at `b = 32` within 5 pp of `b = 26`, so neither is selected | **confirmed**: +5.0 pp (148 against 138), p = 0.0525, 37 against 37 on the 50-item subset |
| Verification: shuffled cells keep B > C > A and the position pattern | **partly**: the ranking holds in point estimates, but B over C is significant only at `b = 26`; the `b = 52` tail survives the shuffle, the `b = 32` gradient is attenuated |

The reviewing session's two written predictions also failed: that blocking
would be dominated by a single pass below the 84-code alphabet, and that the
final-pass factor would be about 59%. Both are recorded here because this note
is the record of the forecasts as well as of the results.

### The product model, and the rule it teaches

The pre-registration anchored the second factor at 45.6%, from stored
development items with three to five options and random distractors. Measured
in this construction it is 88–93%, and the product reproduces the end-to-end
count exactly at every block size. The error was entirely in borrowing the
anchor from a similar-sounding but different problem: the final pass chooses
among the model's own block winners, which are already strong candidates, not
among random distractors.

**A product model is only as good as the construction each factor is measured
on. Both factors must be measured in the same construction as the end-to-end
cell; neither may be estimated from another task that merely resembles one of
them.** That rule applies to every two-factor frontier statement in this line
of work, not only this one.

The 45.6% anchor was the loosest number in the model, as the issue said. On
these items a 3-option final pass over intent descriptions is 92.9%: a shortlist
of the model's own block winners is a different task from the development
items' two-to-five-option cells, and the anchor does not transfer.

## The frontier

| forwards | point | accuracy | options covered |
|---:|---|---:|---:|
| 1 | single pass, 52 letters | 73.0% | 52 |
| 1 | single pass, shipped 84 | 62.5% | 80 |
| 3 | A, b = 52 | 47.0% | 150 |
| 4 | B, b = 52 | 65.0% | 150 |
| 6 | B, b = 32 | 73.0% | 150 |
| 7 | B, b = 26 | 76.5% | 150 |
| 150 | full judgement | 63.5% | 150 |

**The two single-pass rows are not frontier points for K = 150.** They cover 52
and 80 options — a different, easier problem — and no single pass exists at
150, where the code count exceeds the working alphabet. They are listed only to
fix the option count, and they must not be compared with the blocking rows: in
particular, blocking at K = 150 scoring 76.5% says nothing about the K = 80
single pass scoring 62.5%.

Within K = 150 the frontier is scheme B. Covering all 150 classes costs 4
forwards at 65.0% and 6–7 forwards at 73.0–76.5%; full judgement spends 150
forwards for 63.5%. **Judgement is not the accuracy endpoint the issue assumed.**
B at `b = 52` matches it at 4 forwards (130 against 127, p = 0.80) and B at
`b = 26` exceeds it by 26 items (p = 4.1e-4). The pre-registered expectation
that a K-forward budget should choose judgement does not hold on this
distribution.

At a 5-forward budget the choice is B at `b = 52`. At 6–7 forwards both
`b = 32` and `b = 26` stand: the pre-registered "within 5 pp" branch fires under
the shuffle (+5.0 pp, p = 0.0525), and the two are 37 against 37 on the 50-item
subset. `b = 32` costs one fewer forward and is the point the judgement subset
was measured against; `b = 26` has the larger point estimate and its margin over
judgement replicates under both block assignments, while `b = 32`'s does not.
No selection between the two is supported by the measurements.

Blocking's value interval is much wider than the issue expected. The issue
expected it to be narrow — 16–43%, below the single pass — on the strength of
the Banking77 cells. On CLINC the second factor is 88–93%, and blocking matches
or exceeds per-option judgement from 4 forwards upward: `b = 52` is a
non-difference and `b = 26` is a significant margin that replicates under both
assignments. At 150 forwards judgement is not a better point than a 4-forward
blocked readout.

## What this does not show

- **Banking77.** The issue's K = 80 sanity cell was not run. The product model
  was checked on CLINC instead, where the second factor turned out to be the
  mis-estimated one; the Banking77 check would test a different distribution,
  not this model.
- **Multiplicity.** Nine scheme cells are compared against one judgement
  reference. Under a nine-way Bonferroni (0.0056) only B at `b = 26` clears the
  threshold; B at `b = 32` clears a three-way family for the B cells (0.0167)
  but not the nine-way one. The `b = 26` margin over judgement also replicates
  under the shuffle (p = 0.0046); the `b = 32` margin does not (0.18), so the
  `b = 32` result is the fragile one.
- **Assignment scope.** The shuffle is per item on the repository's item seed.
  A single global permutation was not run: it was rejected as a control because
  it only replaces one fixed class-to-position assignment with another.
- **Code form.** One code form (`space`), one alphabet, one model.

## Scope

The measurements are the CLINC150 in-scope distribution, this 200-item draw,
consecutive class-order blocks and per-item shuffled blocks, and the 4B backbone
at revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`. The claim that transfers
is the shape: a panel-2 finish over the block winners is the right second stage,
and per-option judgement over the same shortlist is not. The accuracy levels are
this distribution's.
