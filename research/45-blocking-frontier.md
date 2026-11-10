# Blocking readout frontier at K = 150 (#356) — 2026-11-09

The blocking design of #356, run on CLINC150 at K = 150. **Scheme B — block,
then one panel-2 prompt over the block winners — dominates scheme A and scheme
C at every block size.** The best measured point is `b = 26`: 153/200 = 76.5%
at 7 forwards per item. On the same 50 items full judgement over all 150
options scores 34/50 = 68% at 150 forwards, and scheme B at `b = 32` scores
41/50 = 82% at 6 forwards.

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
manipulation.

## The judgement point, on the same items

Full judgement over all 150 options was measured on the first 50 items, and the
nine blocking cells below are recomputed on that same subset.

| point | correct/50 | forwards | nominal p vs full judgement |
|---|---:|---:|---:|
| full judgement, K = 150 | 34 | 150 | — |
| B, b = 52 | 33 | 4 | 1.00 |
| B, b = 32 | **41** | 6 | 0.039 |
| B, b = 26 | 40 | 7 | 0.11 |
| C, b = 52 | 30 | 6 | 0.22 |
| C, b = 32 | 35 | 10 | 1.00 |
| C, b = 26 | 34 | 12 | 1.00 |
| A, b = 52 | 24 | 3 | 0.013 |
| A, b = 32 | 30 | 5 | 0.42 |
| A, b = 26 | 29 | 6 | 0.36 |

Scheme B is the only scheme above full judgement on the same items: +7 items
at `b = 32` (41 against 34) and +6 at `b = 26` (40 against 34). That comparison
is one of nine against the same reference and does not survive a
multiple-comparison correction (Bonferroni threshold 0.0056). The direction is
suggestive; the level is not established. No scheme C cell is more than 1 item
above full judgement, and every scheme A cell is below it.

Measured wall time per item on the 50-item subset: scheme B totals 0.44 s
(`b = 52`), 0.57 s (`b = 32`) and 0.53 s (`b = 26`); scheme C totals 0.72 s,
0.90 s and 0.90 s; full judgement is 11.83 s. These are from one run on one
machine; the ratio to full judgement (about 20×) is the portable part.

## Pre-registered predictions and what happened

| prediction, written before the run | outcome |
|---|---|
| Scheme C beats B, so the shape is "block, then a judgement finish" | **failed**: B > C at all three sizes, p = 0.019 / 0.0090 / 0.0017 |
| B at `b = 52` ≈ 43% = 0.73 × P(final over 3) | **failed**: 65.0%. Block-win 70.0%, final over 3 = 92.9% |
| End-to-end well below the 62.5% single pass at K = 80 | **failed**: B at `b = 52` is 65.0%, B at `b = 26` is 76.5% |
| Full judgement ≈ 65% | **confirmed**: 68% on 50 items |
| If A is within 5 pp of B, the extra pass is not paid for | **decided**: A is 18.0–22.5 pp below B; the pass is paid for |
| The position split shows whether the block width is effective | **decided**: `b = 32` has a dead tail, `b = 26` does not |

The 45.6% anchor for a 4-option final pass came from stored development cells
and was the loosest number in the model, as the issue said. On these items a
3-option final pass over intent descriptions is 92.9%: a shortlist of the
model's own block winners is a different task from the development items'
two-to-five-option cells, and the anchor does not transfer.

## The frontier

On these items the frontier is scheme B:

| forwards | best point | accuracy |
|---:|---|---:|
| 1 | single pass, K = 52 (1 forward) | 73.0% |
| 1 | single pass, K = 80 (1 forward) | 62.5% |
| 4 | B, b = 52 | 65.0% |
| 6 | B, b = 32 | 73.0% |
| 7 | B, b = 26 | 76.5% |
| 150 | full judgement | 68% (50 items) |

The single-pass cells are on the same 200 items but cover 52 and 80 options;
the blocking cells cover all 150. The frontier statement is therefore: covering
the full 150-class space costs 4 forwards at 65.0% and 6–7 forwards at 73.0–76.5%.
At a 5-forward budget the choice is B at `b = 52`; at 6–7 forwards it is B at
`b = 32` or 26, with no measured difference between them; full judgement spends
150 forwards for a point the blocking cells meet or exceed on the same items.

Blocking's value interval is wider than the issue expected. The issue expected
it to be narrow — 16–43%, below the single pass — on the strength of the
Banking77 cells. On CLINC the second factor is much higher, and the interval
now runs from below the K = 52 single pass (`b = 52`) to above it (`b = 26`).

## What this does not show

- **Block assignment.** The issue declared a sensitivity variant, assignment by
  a seeded shuffle. The run used consecutive slices only; the variant is not
  measured and the class-order confound in the position bins stands.
- **Banking77.** The issue's K = 80 sanity cell was not run. The product model
  was checked on CLINC instead, where the second factor turned out to be the
  mis-estimated one; the Banking77 check would test a different distribution,
  not this model.
- **Generalisation of the 50-item judgement comparison.** Nine scheme cells are
  compared against one 50-item reference; only the scheme B result at `b = 32`
  is nominally significant, and it is not corrected for multiplicity.
- **Code form.** One code form (`space`), one alphabet, one model.

## Scope

The measurements are the CLINC150 in-scope distribution, this 200-item draw,
consecutive class-order blocks, and the 4B backbone at revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`. The claim that transfers is the
shape: a panel-2 finish over the block winners is the right second stage, and
per-option judgement over the same shortlist is not. The accuracy levels are
this distribution's.
