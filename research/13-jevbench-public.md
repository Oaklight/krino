# JevBench-public: 46%, against a 74% reference

Goal item 4 asks for JevBench-public to be reported next to Jev's reference
numbers, which the goal records as a teacher at 84.9 and the best open student
at 74.1. Until now the only Krino rows on that benchmark came from published
HuggingFace releases, because `eval_jevbench.py` builds a head-only model from a
repo and `DecisionModel.predict` rounds probabilities to two decimals. Neither
fits a LoRA training checkpoint, so the models the rest of the goal is measured
on had never been scored there.

`eval_calibration.py --jevbench` closes that: it scores the public tiers with the
architecture and heads it already loaded, using the same accuracy rule as the
rest of the script. The rule and the tier aggregation moved to
`model/evaluation/jevbench.py` so the two paths cannot drift.

## What is measured

| | |
|---|---|
| Items | 231 public: easy 48, hard 111, original 72 |
| Types | choice 139, noul 74, score 18 |
| Chance | 0.284 easy, 0.336 hard, 0.311 original, **0.318 overall** |
| Reference | teacher 84.9, best open student 74.1 (the goal's figures) |

Chance is computed per item — 0.5 for noul, 1/|options| otherwise — and averaged
over the tier. It is high because most JevBench items have two or three options;
that is the number any accuracy has to beat to mean anything.

## Results

Overall, then the three tiers:

| model | overall | easy | hard | original |
|---|---:|---:|---:|---:|
| raw Qwen3-Reranker-4B | **0.628** | 1.000 | 0.441 | 0.667 |
| raw Qwen3-Reranker-0.6B | 0.476 | 0.854 | 0.306 | 0.486 |
| **Krino 4B** (`arch_4b_loraAll_weighted`) | 0.463 | 0.708 | 0.351 | 0.472 |
| Krino 0.6B (`arch_06b_loraAll_last_s123`) | 0.433 | 0.688 | 0.360 | 0.375 |
| Krino 4B heads (HF release) | 0.429 | 0.667 | 0.360 | 0.375 |
| Krino ettin-150m heads (HF) | 0.368 | 0.556 | 0.324 | 0.319 |
| Krino reranker-0.6B heads (HF) | 0.364 | 0.479 | 0.324 | 0.347 |
| Krino qwen3.5-4b heads (HF) | 0.346 | 0.312 | 0.342 | 0.375 |
| Krino qwen3-0.6b heads (HF) | 0.342 | 0.500 | 0.300 | 0.312 |
| Krino modernbert-base heads (HF) | 0.325 | 0.333 | 0.324 | 0.333 |
| _chance_ | _0.318_ | _0.284_ | _0.336_ | _0.311_ |

"raw" is the base reranker scored through its own yes/no logits, with no Krino
heads and no fine-tuning.

By type, for the four models the comparison rests on:

| model | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|
| raw Qwen3-Reranker-4B | 0.622 | 0.604 | 0.833 |
| Krino 4B | 0.595 | 0.396 | 0.444 |
| raw Qwen3-Reranker-0.6B | 0.459 | 0.482 | 0.500 |
| Krino 0.6B | **0.554** | 0.396 | 0.222 |

## Finding 1 — the best trained model clears chance by 14.5 points and misses the reference by 27.8

The 4B checkpoint that carries the rest of the goal scores 0.463 overall: 14.5
points above chance, 27.8 below the best open student and 38.6 below the teacher.
The smallest Krino model in the table, a 150M encoder, is 5 points above chance.
That spread — 0.325 to 0.463 across every model we have — is what the gap looks
like from the inside.

## Finding 2 — the untouched backbone scores 16.5 points higher than the trained model

The same base model, before any of our training, scores **0.628** through its own
reranker readout against the trained checkpoint's **0.463**. At 0.6B it is 0.476
against 0.433 — the same direction, 4.3 points. On the in-distribution eval the
same training is worth +5.35 to +7.06 points at 0.6B over a frozen backbone
(`research/10-architecture-sweep-r2.md`), so it is not that the training does
nothing; it is that what it learns does not transfer here. On JevBench it is a
net loss at both sizes.

The loss is not spread evenly. Noul is fine — at 0.6B the trained model beats the
raw reranker by 9.5 points — and the deficit is almost entirely choice (−20.8 at
4B, −8.6 at 0.6B) and score (−38.9 and −27.8, on 18 items, so treat those two as
suggestive rather than measured). That matters for where to look: the noul head
takes a single hidden state, while choice and score score an option list, and it
is the option-list path that fails to transfer.

## Finding 3 — on the hard tier we are at chance

The hard tier is the one that separates models: 111 items, chance 0.336. The
trained 4B scores 0.351 — 1.5 points above chance, which on 111 items is
indistinguishable from it. The raw 4B scores 0.441, and the published reference
is 74.1 overall. This is the tier to watch, because easy is nearly saturated for
anything that works (the raw 4B is at 1.000) and the original tier has only 72
items.

## Caveats

* **The raw comparison confounds two things.** It measures what we ship against
  the untouched base model, which is the honest product comparison, but it
  cannot say whether the 16.5 points went to the fine-tuning or to scoring
  options with a head instead of the reranker's own yes/no logit. Separating them
  needs the fine-tuned backbone scored through the native readout — a
  merge-and-rerun, not a different code path — and that is the next measurement.
* **18 score items.** One item is worth 5.6 points on the score column, and the
  0.833 vs 0.444 row is 7 items. Nothing in this note should be read as a
  score-primitive finding, and #179 measures that primitive on 1,856 items
  instead.
* **231 items.** A 3-point difference on the overall column is at the resolution
  of the sample; the choice finding (139 items, 20.8 points) is not.
* **The HF-release rows are a different generation of checkpoint.** They were
  committed in `f997648`, before the current training configs, the
  cluster-disjoint split and the stsb label fix, and they are not re-measured
  here. They are context for how far the releases sat from chance, not an
  ablation against the two rows that matter.
* **The reference numbers are the goal's, not re-derived.** They are quoted as
  the target. Our harness's prompt format and option handling may differ from
  Jev's public one, so the comparison to 74.1 is indicative — the internal
  comparisons (trained vs raw, 4B vs 0.6B, tier by tier) are the measurements
  this note stands behind.
* **Two eval settings differ between the two new rows.** The 4B ran at
  `max_length` 8,192, the 0.6B at 4,096. JevBench items are short enough that
  neither truncates, but the rows are not bit-identical configurations.

## What this means for the goal

Item 4 is the widest gap in the goal, and the one with the most direct evidence
about where it comes from: the capability is partly present in the base model and
the choice/score heads do not preserve it. The next two measurements are named
above — the fine-tuned backbone under the native readout, and the hard tier as the
target rather than the overall column. Until then the honest statement is that
our best JevBench-public result is 0.463, against a reference of 74.1, and that
the model we train is worse at it than the model we started from.
