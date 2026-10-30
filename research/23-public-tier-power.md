# The public tier can see a deficit but not a gain

`research/13` and `research/21` both end on the same sentence: the hard tier
cannot resolve the difference being asked about. This note puts a number on
that, and on the more awkward finding underneath it — **the tier is not going to
get bigger**, because 111 is all of it that is published.

The arithmetic needs two inputs and both are already measured: the item count,
and the rate at which two comparable checkpoints disagree on those items. On the
hard tier that rate is 35–37 of 111. With 36 discordant pairs, an exact
two-sided McNemar test at 5% needs 26 of them to land one way before it rejects.
A genuine ten-point accuracy gain, on 36 disagreements, produces about 24.

## What 111 items can detect

`model/evaluation/paired_power.py` computes the exact power (the same binomial
test `paired_mcnemar.py` runs); `model/scripts/tier_power.py` prints these
tables and writes `model/experiments/tier_power_hard111.json`:

| true accuracy difference | power at 111 items | items for 80% power | items for 90% |
|---:|---:|---:|---:|
| 2 pp | 0.040 | 6,357 | 8,635 |
| 5 pp | 0.104 | 1,026 | 1,351 |
| **10 pp** | **0.376** | **255** | **339** |
| 15 pp | 0.758 | 113 | 146 |
| 20 pp | 0.968 | 61 | 76 |

**The smallest difference this tier detects at 80% power is 15.7 points.** That
is the number to put beside every hard-tier claim: not "about ten points" as
`research/21` cautiously wrote, but sixteen, and a ten-point gain has a 38%
chance of being called significant even when it is real.

This is why the paired tests in `research/21` all landed at p ≈ 0.5–1.0. The
models differed on 35–37 items and split them roughly evenly; nothing about that
was surprising, and nothing about it was evidence that the head does not help on
the tier. The measurement could not have said so either way.

Size matters more than any of the levers measured in this project, and the
curve is steep:

| item count | smallest detectable difference (80% power) |
|---:|---:|
| 67 (hard-tier choice only) | 20.9 pp |
| 111 (the hard tier) | 15.7 pp |
| 144 (our own `authored144`) | 13.4 pp |
| 231 (every public item we score) | 10.8 pp |

So even scoring **all** 231 public items — not just the hard tier — buys one
point of resolution over the tier alone: 10.8 versus 15.7. And clause 5's
three-point allowance between the 4B and the 0.6B would need **2,851 items** at
this discordance rate to be decided at 80% power, which is twelve times the
whole public set.

## The set cannot grow, and this is checkable

Our loader reads `datasets/public/{easy,original,hard}.jsonl` from
`fstandhartinger/jevbench`. Those three files are the entire published corpus —
231 items, 139 choice / 74 noul / 18 score, with the hard tier 111 (67 / 38 /
6). The benchmark's own `datasets/manifest.json` (protocol `jevbench::v1.2`,
frozen 2026-09-19) enumerates the rest of the pool and shows where the rest is:

| split | items | published in `datasets/public/` |
|---|---:|---|
| easy | 48 | yes |
| original | 72 | yes |
| hard | 111 | yes — "public half; 109 more held out" |
| easy-heldout | 24 | no |
| heldout | 24 | no |
| router | 78 | no |
| judge | 68 | no |

The tiers the board reports are `easy` 72, `standard` 96, `judge` 146 (314
pooled decisions) plus the 220-item hard tier. The README puts the frozen suite
at **534 decisions**, of which **425 are public** — everything but the 109 held
out hard items — and says plainly that an API-flagged row "received item text
without answer keys". So 425 decisions have published *outcomes* somewhere and
231 items have published *text*. The gap is not obtainable for local scoring:
194 of the public decisions are scored by the board's harness, not by us, and
the 109 held-out hard items are withheld by construction.

**Consequence for clause 4.** The deficit is decidable; progress is not. Jev's
hard-tier choice accuracy on the same 67 items is 0.761 against our 0.373–0.493
(`research/13`), a gap of 27–39 points — far above the 21-point floor the 67
choice items impose, so that comparison stands. A five-point improvement on the
way there would need about a thousand paired items to be believed, and no public
JevBench split provides them.

**Consequence for the next held-out set.** If the aim is to decide a ten-point
gain, the set needs about 255 items at this discordance rate — roughly the size
of the entire public release, and twice the hard tier. `authored144` is 144
items and would resolve 13.4 points. That is the number to size the next
authoring pass against, not a hope that more epochs will show up on the board.

## Caveats

- **The discordance rate is model-pair specific.** Two checkpoints three epochs
  apart disagree on 32% of the hard tier; a weak model against a strong one
  disagrees far more, and higher discordance raises power at the same accuracy
  difference. Every number here uses the measured 32%, which is the right rate
  for the comparisons this project actually makes.
- **These are powers for paired comparisons on identical items.** They say
  nothing about an unpaired comparison to a number another harness published;
  that quantity needs its own argument, and `research/13` shows the board's own
  reranker row does not reproduce even on its documented mapping.
- **Nothing here applies to clause 2's ECE or QWK half.** Those are calibration
  statistics on the 13,444-item in-distribution eval, not paired accuracy
  tests, and the instability `research/21` reports there is a different
  quantity (a 4× seed spread on 1,856 score items).
- **Power is not the only way to fail.** A tier can be underpowered *and* the
  model can genuinely not have improved; this note only removes the option of
  reading a null result as evidence of the second.

## Reproducing

```bash
python model/scripts/tier_power.py --items 111 --discordant 36 \
    --json model/experiments/tier_power_hard111.json
# or straight from the dumps a McNemar run used:
python model/scripts/tier_power.py --tier hard \
    --preds tower=jevbench_tower_ep10_preds.json xenc=jevbench_xenc_ep3_preds.json
```
