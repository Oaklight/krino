# JevBench-public: the heads cost 14-21 points against their own backbone

Goal item 4 asks for JevBench-public to be reported next to Jev's reference
numbers, which the goal records as a teacher at 84.9 and the best open student
at 74.1. Measured, the trained models land at 0.463 (4B) and 0.433 (0.6B)
against a 0.318 chance floor. The interesting part is not the level: scored
through the backbone's own reranker readout, the same two checkpoints reach
**0.623 and 0.541**. Almost all of the gap is the decision heads.

Until now the only Krino rows on that benchmark came from published HuggingFace
releases, because `eval_jevbench.py` builds a head-only model from a repo and
`DecisionModel.predict` rounds probabilities to two decimals. Neither fits a LoRA
training checkpoint, so the models the rest of the goal is measured on had never
been scored there.

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

All ten rows below scored the identical 231 items with identical per-tier and
per-type denominators (48/111/72 and 139/74/18), so the comparison between them
is not confounded by differing item sets or by one path silently skipping items.

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

The 4B checkpoint that carries the rest of the goal scores 0.463 overall through
its heads: 14.5 points above chance, 27.8 below the best open student and 38.6
below the teacher. Finding 4 shows the same weights score 0.623 when the backbone
scores options natively, so 0.463 is a statement about the heads and not about
what the model knows.
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

Both trained models land on exactly 55/139 choice items (0.396), which is worth
ruling out as a degenerate answer before reading anything into it. It is not a
constant rule: always taking the first option scores 0.216 and always taking the
last scores 0.266, so 0.396 is 13 to 18 points above any position-only strategy,
and the labels are spread too thinly (the commonest label is 2.9% of items) for a
majority guess to be the explanation. Two independent checkpoints agreeing on the
same count is then a coincidence at the few-percent level, not a shared failure
mode — but it is the kind of number that should be checked rather than trusted.

## Finding 3 — on the hard tier we are at chance

The hard tier is the one that separates models: 111 items, chance 0.336. The
trained 4B scores 0.351 — 1.5 points above chance, which on 111 items is
indistinguishable from it. The raw 4B scores 0.441, and the published reference
is 74.1 overall. This is the tier to watch, because easy is nearly saturated for
anything that works (the raw 4B is at 1.000) and the original tier has only 72
items.

## Finding 4 — the deficit is in the choice head, not the fine-tuning

Finding 2 compares what we ship against the untouched base model, which mixes
two explanations: fine-tuning may have cost the backbone its reranker behaviour,
or the heads may score options worse than the backbone's own yes/no logit. They
are separable — fold the LoRA factors into the base weights, and the fine-tuned
backbone can be scored by the reranker's own readout.

`model/src/lora.py` gains `merge_lora` for that, and the merge is checked rather
than assumed: the reloaded merged model reproduces the LoRA model's choice logits
to **0.0000** on held-out probes, so a merge that dropped or double-applied the
adapter would fail instead of producing a plausible number.

Same 231 items, three ways of scoring the same weights, at both sizes:

| scoring | overall | choice | noul | score |
|---|---:|---:|---:|---:|
| 0.6B, untouched base, native | 0.476 | 0.482 | 0.459 | 0.500 |
| 0.6B, **fine-tuned backbone, native** | **0.541** | **0.540** | **0.554** | 0.500 |
| 0.6B, fine-tuned backbone, Krino heads | 0.433 | 0.396 | 0.554 | 0.222 |
| 4B, untouched base, native | 0.628 | 0.604 | 0.622 | 0.833 |
| 4B, **fine-tuned backbone, native** | **0.623** | **0.604** | **0.635** | 0.722 |
| 4B, fine-tuned backbone, Krino heads | 0.463 | 0.396 | 0.595 | 0.444 |

Three things follow.

* **Fine-tuning does not damage the backbone's reranker capability.** At 0.6B it
  improves it, 0.476 → 0.541 (+6.5 points on the same scorer). At 4B it leaves
  the total where it was — 0.628 → 0.623, one item out of 231 — while moving
  performance around inside it: original 0.667 → 0.736, hard 0.441 → 0.387. So
  the capability Finding 2 read as lost was never lost; at 4B it was partially
  reshuffled, and the hard tier is where the reshuffle went wrong.
* **The choice head loses its own backbone's advantage.** 14.4 points at 0.6B
  (0.540 → 0.396) and 20.8 at 4B (0.604 → 0.396). The trained choice head is the
  same 0.396 at both sizes, which is the number to explain: a 6.7x larger backbone
  changes nothing about how the head scores an option list.
* **The noul head is free.** It matches the native readout at 0.6B exactly (0.554)
  and lands close at 4B (0.595 against 0.635). Whatever the choice head is doing
  wrong, the noul head is not doing it.

So the JevBench deficit is a property of how the choice head scores an option
list, not of the training data or the backbone. That is a much narrower target
than Finding 2 posed, and it is the one the next experiment should attack.

## Caveats

* **The raw comparison is a product comparison, not an attribution.** It
  measures what we ship against the untouched base model. Finding 4 separates
  the two explanations it mixes, by scoring the fine-tuned backbone through the
  reranker's native readout; that is the like-for-like comparison.
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

Item 4 is the widest gap in the goal — 0.463 against a 74.1 reference — and
Finding 4 says where it is. The fine-tuned backbones keep the reranker
capability (0.541 at 0.6B, where it improves on the base, and 0.623 at 4B, where
it holds), and the choice head throws 14 to 21 points of it away. So the sentence
this note would have ended on — that the model we train is worse than the model
we started from — is true of the heads and false of the backbones, and the work
belongs in how the choice head scores an option list. The trained choice head
scores exactly 0.396 at both model sizes, which is the number that says the head,
not the backbone, is the constraint.

That is one measurement, one seed, 231 items. The hard tier is where the
remaining gap sits (0.351 against a 0.336 chance floor), and it is the column to
optimise rather than the overall number, because easy is nearly saturated for
anything that works.
