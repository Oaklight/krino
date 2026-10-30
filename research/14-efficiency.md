# Efficiency: the 0.6B meets the latency bar only when compiled, and no configuration meets both halves of item 5

Goal item 5 asks for two things that pull against each other: ≤100 ms per item at
256 options on one A100, and the 0.6B variant within 3 pp of the 4B variant. This
note measures the first properly for both sizes, on the same host, with the same
script and the same flags, and reports the second from the sweep.

The clause names a model size, a device, an option count and a time — but not a
*state length*, and latency here is dominated by the state, not by the options.
Every number below is at the 186-token state the committed 0.6B artifact used, so
they are comparable to each other and to it. At a 1,000-token state both models
would exceed the bar, which means the clause is only well-defined once a state
length is stated; 186 tokens is a short-ish sentence, and it should be read as
the favourable end of the range.

## Setup

| | |
|---|---|
| Host | one A100-SXM4-40GB |
| State | 186 tokens, batch 1 (one item per timed call) |
| `max_length` | 16,384 |
| Readout | `weighted` on both sizes — `arch_06b_loraAll_weighted`, `arch_4b_loraAll_weighted` |
| Timed op | one `forward_choice(state, instructions, options)`, median of 20-30 repeats after warm-up |
| Compiled | `torch.compile(..., mode="reduce-overhead")` (CUDA graphs); compile time excluded |

## Results

Median milliseconds per item. "—" is a configuration not measured in this round.

| options | 0.6B eager | 0.6B compiled | 4B eager | 4B compiled |
|---:|---:|---:|---:|---:|
| 2 | 78.2 | — | 101.0 | 33.2 |
| 5 | 79.9 | 13.8 | 101.3 | 34.2 |
| 20 | 82.6 | — | 106.7 | 47.6 |
| 64 | 86.9 | — | 124.4 | 68.5 |
| 128 | 92.9 | — | 188.2 | 109.1 |
| **255** | 126.6 | **55.5** | 316.7 | **183.6** |
| peak memory at 255 | 2.35 GB | 1.15 GB | 9.27 GB | 7.63 GB |

The 0.6B compiled column has only two measured points (5 and 255 options); the
k=2, 20, 64 and 128 cells are not measurements, and an earlier revision printed
13.8 in the k=2 cell, which is the k=5 number.

Throughput at 256 options follows from the medians: 17.9 items/s and 5.4 items/s
for the two compiled models. Both fit on one A100 by memory at either size; time
is the constraint, not capacity.

## Finding 1 — the 4B misses the bar by 1.8x even compiled

At 256 options the 4B takes 316.7 ms eager and 183.6 ms compiled. The clause asks
for 100 ms. The 0.6B takes 126.6 ms eager and **55.5 ms compiled** — the only one
of the four configurations that clears the bar, with 1.8x of headroom.

The bar is the right order of magnitude for the product: `research/00-overview.md`
records Jev's own latency as 70-500 ms, so a 55 ms 0.6B is inside that band and a
317 ms 4B is at its slow end.

## Finding 2 — compilation is worth 1.7x to 5.7x, and decides the clause

The speed-up is larger at small option counts: 5.7x at two options for the 0.6B
(78.2 → 13.8) and 3.0x for the 4B (101.0 → 33.2), against 2.3x and 1.7x at 255.
That is the signature of host-side work being removed rather than arithmetic:
CUDA graphs eliminate per-launch overhead, which is what the small-K cases are
mostly made of. It is also why "efficiency" for this model cannot be quoted
without saying whether the graph is captured — the same checkpoint, on the same
card, is 2.3x apart.

## Finding 3 — per option costs 0.17 ms at 0.6B and 0.60 ms at 4B; the rest is fixed

Fitting the compiled columns between 2 and 255 options:

| | fixed cost at K=2 | marginal per option |
|---|---:|---:|
| 0.6B | 13.8 ms | 0.165 ms |
| 4B | 33.2 ms | 0.594 ms |

The marginal cost grows 3.6x for 6.7x the parameters. The fixed cost is the
surprising part: a 186-token forward through either model is a couple of
milliseconds of GPU work, and 13.8 ms — already compiled — is far above that. So
a large part of the small-K latency is not the model at all, and the lever for
the small-K case is the data path (tokenising 256 option texts, building tensors,
the per-call Python), not the architecture. This note does not establish that
split; timing the same call at a 2-token state would, and it is the next
measurement.

### The missing compiled shapes were a harness bug, and the completed curve is on H100

The caveat in *Caveats* used to say that a six-shape compiled run "stalled in
graph capture", which is why only two 0.6B compiled points existed. It did not stall.
`--compile` passes `dynamic=False`, so every option count is its own graph, and
torch's default recompile limit is 8 per frame — past it, dynamo stops compiling
and the remaining shapes run **eager while still being written to the record as
`"compiled": true`**. The five-shape run that produced 91.9 ms at 256 options was
measuring the eager path for its last two rows; an isolated run of the same shape
gives 67.5 ms. `bench_latency.py` now raises `torch._dynamo.config.recompile_limit`
to 256 with the reason in a comment, and the curve completes without a warning.

Measured on one H100 NVL, 186-token state, batch 1, same flags as every other
row in this note — a different host from the A100 table above, so it is reported
separately rather than merged into it:

| options | 0.6B eager | 0.6B compiled | 4B eager |
|---:|---:|---:|---:|
| 2 | 51.9 | 11.0 | 66.7 |
| 5 | 53.0 | 11.8 | 68.4 |
| 20 | 56.7 | 17.0 | 75.8 |
| 64 | 69.4 | 29.6 | 98.6 |
| **256** | **102.9** | **67.4** | **243.8** |

Three things this adds. **The verdict repeats on a second device**: eager misses
the 100 ms bar (102.9 ms) and compiled clears it (67.4 ms), which is the same
ordering as 126.6 against 55.5 on the A100. **The compiled curve is not linear**,
so Finding 3's two-point line overstates the marginal cost above 20 options: the
0.6B pays 0.33 ms per option from 2 to 20 and 0.21 ms per option from 20 to 256,
against the 0.165 a two-point fit reports. And **the two devices do not agree on
which is faster**: the H100 is 1.3× faster than the A100 at 5 options (11.8
against 13.8) and 1.2× *slower* at 255 (67.4 against 55.5). Both clear the bar, so
neither reading changes Finding 1, but the cross-device ratio is not stable and a
number from one card should not be scaled to another.

## Finding 4 — the accuracy half of item 5 fails by 3x the allowed gap

The clause allows the 0.6B to be 3 pp behind the 4B. On the configuration sweep's
own corpus (`research/10-architecture-sweep-r2.md`) the best 0.6B is 56.17 and the
best 4B is 64.89 — **8.7 pp**, and the 4B only reaches 65.81 on the rebuilt
corpus, with no clean 0.6B measurement to pair it with. Either reading is close
to three times the allowance.

So the two halves of item 5 are met by different configurations and by neither
one jointly: the 0.6B compiled clears latency (55.5 ms) and misses the 3 pp
allowance by 5.7-6.6 pp; the 4B clears accuracy and misses latency by 1.8x. That is the
honest state of the clause, and it is a statement about where the project is, not
a measurement artefact of this note.

## Corrections in this revision

The committed `latency_a100_06b_compiled.json` reported 51.0 ms at five options
and 78.1 ms at 255. Re-measured with the same flags as the 4B on the same host it
is 13.8 and 55.5. The earlier run is 3.7x slower at five options while the *eager*
runs agree to within 3.3% across the two rounds (75.7 against 78.2 at two options),
so the discrepancy is in the compiled path of that earlier run, not in the model.
The file is replaced rather than kept beside the new one: two artifacts measuring
the same thing 3.7x apart is worse than one, and the superseded numbers are
recorded here so a reader who saw them can find out what happened.

| | earlier compiled run | this run |
|---|---:|---:|
| options 5 | 51.0 ms | 13.8 ms |
| options 255 | 78.1 ms | 55.5 ms |

The two runs differ in the stack as well as in the result: the superseded file
(`model/experiments/latency_a100_06b_compiled.json`, merged with #160) records
torch 2.11.0+cu128 and CUDA 12.8, this one torch 2.14.0+cu126 and CUDA 12.6, with
the same warmup and repeat counts. So the 78.1 ms to 55.5 ms change is a stack
change on identical model, device, state length and `max_length` — not a
measurement error in either run, and not a model difference. Both clear the
100 ms bar, so nothing in Finding 1 depends on which is quoted; the repository
now carries one artifact for this configuration and this note records what the
other one said.

## Caveats

* **One host, one state length, batch 1.** Batch-1 latency is what "per item"
  means, but a server batching across items would report different numbers; the
  throughput column is the honest one for serving.
* **The state length is the dominant variable and the clause does not name it.**
  These numbers are at 186 tokens. A longer state moves both models toward and
  past the bar, and a 256-option question with a long document is exactly the
  case item 3 also cares about.
* **Compile time is excluded.** Capturing the graph for a new option count takes
  tens of seconds; it is a one-off per shape, but it is real and it is not in the
  table.
* **The 0.6B compiled column in the A100 table is two shapes** (5 and 255). It was
  two shapes because a six-shape run was silently measuring the eager path for its
  last rows — see *The missing compiled shapes were a harness bug*. With the limit
  raised the curve completes; the completed one is on H100 above and the A100 pair
  is still the pair the clause needs.
* **`weighted` readout on both sizes.** The `last` readout pools one hidden state
  instead of 37, so it may be cheaper per option; it has not been measured here.

## What this means for the goal

Item 5 cannot be satisfied today by one model. The 0.6B is the efficiency answer
and it is 8.7 pp behind the 4B on accuracy; the 4B is the accuracy answer and it
is 1.8x over the latency bar even compiled. Since the goal says efficiency counts
equally with accuracy, the practical target is a 4B-scale model that is ~2x
faster than this one, or a 0.6B-scale model that is ~9 pp better — and the fixed
cost measured in Finding 3 suggests the first of those is partly an engineering
problem rather than a modelling one.
