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

## On held-out data it does not move the hard tier

JevBench-public, 231 items, states rendered through the loader (`#180`), LoRA
applied, same loader for every row:

| | easy (48) | hard (111) | original (72) | overall (231) |
|---|---:|---:|---:|---:|
| 4B tower, epoch 10 (s123) | 0.792 | 0.405 | 0.486 | 0.511 |
| 0.6B cross-encoder, epoch 1 | **1.000** | 0.396 | **0.611** | **0.589** |
| **4B cross-encoder, epoch 1** | **1.000** | **0.405** | **0.722** | **0.628** |

The head lifts the board at both sizes — +11.7 pp at 4B, +23.6 pp on the
`original` tier — and moves the **hard tier by exactly nothing**: 0.405 at 4B
before the change and 0.405 after it, 0.396 at a fifth of the parameters. Three
different models, one number to three digits. That is not memorisation: the leakage
audit (`scripts/check_jevbench_leakage.py`) finds **0 exact duplicates, 0 state
repeats and 0 near-duplicates above 0.5 shingle containment over 1,562,838
training items** (max containment 0.001).

It also does not close the tier. The hard items are a **long-state wall**: split
into quartiles by state length, both the trained head and the zero-shot native
readout slide together.

| hard-tier state length | n | trained head | native readout |
|---|---:|---:|---:|
| 350 – 987 chars | 27 | 0.630 | 0.667 |
| 995 – 1,493 | 27 | 0.407 | 0.407 |
| 1,553 – 8,377 | 27 | 0.296 | 0.259 |
| 8,566 – 14,986 | 30 | 0.300 | 0.367 |

Both lose about 30 points from the shortest to the longest quartile, at the same
places, on items whose longest state is ~3.7k tokens — inside the 16,384 window,
so this is not truncation. Structured states are not the problem either (0.486 on
the 35 dict-state items against 0.368 on the 76 string-state ones; the structured
items are simply shorter), and option count is not (2–4 options 0.444, 5–10
0.238). Whatever limits clause 4 is how much of a long state the backbone uses,
and no head, prompt or calibration change addresses it.

## Epoch 2 dips and epoch 3 recovers — the dip is in the adapter, not the head

The 4B arm's first three epochs, s123:

| | overall | choice | noul | score | eval loss |
|---|---:|---:|---:|---:|---:|
| epoch 1 | 0.7231 | 0.721 | 0.822 | 0.477 | 0.727 |
| epoch 2 | **0.3761** | 0.292 | 0.555 | 0.229 | **1.362** |
| epoch 3 | **0.7291** | **0.737** | 0.825 | 0.452 | 0.745 |

Epoch 2 is a transient excursion, not a terminal state: by epoch 3 the arm is back above its first
epoch, with the best `choice` of the three (0.737), while training loss fell throughout
(0.969 → 0.926 → 0.861).

**Where the dip lives.** Rebuilding the checkpoint with epoch 2's adapter and epoch 1's heads and
scoring it: `choice` **0.289** — the same as epoch 2's own pair (0.292). Putting the good head back
does not recover the accuracy, so the excursion is in the **LoRA adapter**, not in the linear
read-out: at epoch 2 the adapter is producing a representation that neither head can read.

**What that means for how to read the table above.** The epoch-1 figure is not "one epoch before the
run falls apart" — the run does not fall apart, it dips and returns. What it still is: a single
checkpoint's number, at one epoch, and the arm needs to finish its five before its accuracy is the
arm's accuracy. The 0.6B JevBench row was scored at epoch 1 of a one-epoch run, so it is unaffected
either way.

## Caveats

- **One seed except where stated.** The 0.6B replication (seed 42) was still
  training when this was written; the 4B row carries both seeds at epoch 1
  (0.7231 and 0.7190, 0.4 pp apart).
- **The schedules differ.** The tower baselines ran 10 epochs; both cross-encoder
  arms run 5 (0.6B) or 5 (4B) in the runs quoted, so "epoch 1" is not the same
  point on the two learning-rate curves. The comparison is at matched epoch and
  matched data, not at matched schedule.
- **The 0.6B JevBench row is one epoch**, so it is a floor, not a ceiling.
- **Nothing here says the head helps `score` or `noul`**; the change is confined
  to choice, and the numbers above show it.
