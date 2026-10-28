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
| 2 | 78.2 | 13.8 | 101.0 | 33.2 |
| 5 | 79.9 | 13.8 | 101.3 | 34.2 |
| 20 | 82.6 | — | 106.7 | 47.6 |
| 64 | 86.9 | — | 124.4 | 68.5 |
| 128 | 92.9 | — | 188.2 | 109.1 |
| **255** | 126.6 | **55.5** | 316.7 | **183.6** |
| peak memory at 255 | 2.35 GB | 1.15 GB | 9.27 GB | 7.63 GB |

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
* **The 0.6B compiled run was measured on two shapes only** (5 and 255), after a
  six-shape run stalled in graph capture. The two numbers it reports are the two
  the clause needs, and the eager column covers the shape dependence.
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
