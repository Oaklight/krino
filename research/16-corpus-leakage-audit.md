# 100 of 253,701 eval items are exact duplicates of a training item, and the audit finds the STS-B sentinel (#177) on the eval side

Every eval item in the corpus is checked against the training side, and 99.96% of them are
unique: 1,124 state repeats, 100 identical items and 15 label conflicts across 26 benchmark
sources — 0.44%, 0.04% and 0.006% respectively. The title says what was found rather than what
was hoped for: 100 eval items are byte-identical to a training item, which under this note's own
definition is leakage, and the audit exists to size it rather than to certify zero.

Goal item 1 says "leakage-free, held-out data". The synthetic corpus has a checker
(`scripts/check_split_leakage.py`) that clusters near-duplicate states, because the split is
assigned per cluster. The 26 benchmark sources had no audit of any kind: they keep the split
their files carry, so nothing verified that an eval item had not also been seen in training.

`scripts/check_benchmark_leakage.py` now measures it, and separates three counts that are easy
to collapse into one:

- **state repeats** — an eval item whose state also occurs in train, with a different question
  or label. In PIQA the goal sentence legitimately recurs with different option pairs.
- **identical items** — same state, same options, same gold on both sides. The model was
  trained on that item; there is no reading of it other than leakage.
- **label conflicts** — same state and options, different gold on the two sides. Not leakage,
  but a sign the label is unreliable.

Matches are exact after whitespace and case normalisation.

## Result

253,701 eval items across 26 sources:

| | count | share of eval |
|---|---:|---:|
| repeat a train state | 1,124 | 0.44% |
| are the same item with the same label | **100** | **0.04%** |
| conflict on the label | 15 | 0.01% |

The sources that contribute any identical item:

| source | eval | state repeats | identical | conflicts |
|---|---:|---:|---:|---:|
| contractnli | 6,097 | 77 | 70 | 7 |
| agnews | 7,600 | 10 | 10 | 0 |
| banking77 | 3,080 | 7 | 7 | 0 |
| stsb | 2,879 | 13 | 6 | 7 |
| sst5 | 3,311 | 4 | 4 | 0 |
| arc | 4,417 | 18 | 2 | 0 |
| piqa | 1,838 | 192 | 1 | 0 |

**0.04% identical items is small enough to read as a floor, not a confound.** Even if every one
of the 100 were answered correctly because of training, it moves a 13,444-item eval by 0.7 pp
at the very most, and only if they were all otherwise wrong.

The two biggest "state repeat" counts are the two that are not leakage: PIQA's 192 repeats are
goal sentences reused with different option pairs (1 of them is a true duplicate), and
CodeSearchNet's 626 are docstrings reused across challenges (none duplicates). Reporting only
the state count would have condemned both sources.

## The STS-B conflicts are the withheld-label sentinel, on the eval side

All 7 label conflicts in STS-B have the same shape: the eval row says `0.0` and the train row
with the same state and criteria says 2.0, 3.0, 4.0 or 5.0, on pairs that are plainly similar —

```
eval 0.0 / train 5.0  "Iran, six global powers sign landmark nuclear deal"
                      "Iran, world powers ..."
eval 0.0 / train 5.0  "Car bomb at Swedish consulate in Libya's Benghazi, no casualties"
eval 0.0 / train 4.0  "New Zealand set to legalise gay marriage"
                      "New Zealand votes to legalise ..."
```

That is #177's withheld-label sentinel: a removed score written as `0.0` rather than dropped.
The audit finds it independently, and finds that **it is not confined to the train side** —
those rows sit in the eval split, where a correct answer is scored wrong.

The audit cannot count the sentinel rows that have no contradicting duplicate; 1,621 STS-B eval
rows carry `0.0`, some genuine and some withheld. The measurement splits that number exactly, and
it is the same 1,379 #177 reports from the other side: the 2,879 STS-B eval items are 1,500 from
GLUE's validation file and 1,379 from its test file, and **every one of the 1,379 test rows carries
the withheld `0.0`** while 242 of the 1,500 validation rows are genuine "No similarity" items —
1,379 + 242 = 1,621. So #177's 1,379 is the withheld subset and this note's 1,621 is every eval row
whose stored label is zero; the two are counts of different things and both are right.

Contradicted duplicates are still only a lower bound on the damage, and #177 should land — and the
corpus be rebuilt with `python -m data.pipeline stsb` — before any STS-B score is quoted. Every
STS-B number published before that rebuild is measured against 1,379 rows whose gold label is
fabricated, which is roughly half the source.

## Caveats

* **Exact match only.** Near-duplicate leakage on benchmark sources is not covered. On the
  synthetic corpus it is, and #138's cluster split is what makes that the relevant check there.
  On the benchmarks it would need the same LSH clustering over a corpus ten times larger, and
  the 0.04% exact figure is a floor rather than an estimate of the true duplicate rate.
* **One corpus, on one host** (`open-decisions` on rbh101, benchmark files from 2026-10-13).
  That corpus predates #177, which is why the sentinel rows are still present.
* **The gate is opt-in.** `--max-identical-rate` exists so a corpus rebuild can fail on it, but
  the default reports and exits 0: five sources already carry duplicates (contractnli 1.15%,
  banking77 0.23%, stsb 0.21%, agnews 0.13%, sst5 0.12%), and inventing a threshold that
  condemns existing data would be a policy decision, not a measurement.

## What this changes

* Goal item 1's "leakage-free" clause now has a measurement behind it for the benchmark half,
  and the answer is that benchmark leakage is negligible at 0.04%.
* #177 has an independent confirmation, and a sharper one than "some rows look wrong": the
  sentinel reaches the eval split and turns correct answers into errors.
* Nothing here changes the 67.37% result. The harness draws its 13,444 eval items with
  per-source caps (500 for most sources), so the identical items that reach it are at most the
  per-source rate applied to the capped count — about 9 in expectation, 0.07 pp of the eval.
  Even all of them cannot move 67.37 by more than a tenth of a point.
