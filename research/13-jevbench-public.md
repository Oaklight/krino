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

## Finding 5 — the two scorers cross over at about twenty options

Finding 4 leaves the impression that the choice head is simply the wrong tool.
Measured on the in-distribution eval, that is half the story: the two scorers
trade places as the option list grows.

Same checkpoint, clean corpus, 1,500 items per type, choice accuracy by option
count:

| options | n | trained head | native readout |
|---|---:|---:|---:|
| 2 | 224 | 0.6071 | **0.6696** |
| 3-5 | 1,169 | 0.5595 | **0.6758** |
| >20 | 107 | **0.5607** | 0.3178 |
| all choice | 1,500 | 0.5667 | **0.6493** |

The native readout wins where the option list is short — by 6.3 points at two
options and 11.6 at three to five — and loses by 24.3 points above twenty. Both
are far outside sampling noise (the >20 bucket is 107 items, about 4.8 points per
standard error).

That is the mechanism, and it also explains JevBench. Its items nearly all carry
two or three options (chance is 0.318), which is exactly the regime where
scoring each option independently wins — so Finding 4's 14-21 point deficit is
not "the head is bad", it is "the head pays for competition that short lists do
not need". Per-option P(yes) has no term that makes options compete; with two
options that costs nothing, and with twenty plausible ones it stops
discriminating, while the head's attention over the option list is precisely
that competition.

Two consequences for how this should be reported. The head is not replaceable by
the native readout — it is the only one of the two that holds up on long lists,
and it passes goal item 3's option-count clause (0.5607 above twenty against
0.6071 at two, a 4.6 point gap) where the native path fails it by 35 points. And
the native readout is not a usable product even where it wins accuracy: its
choice ECE is 0.2139 against the head's 0.0448, because independent per-option
probabilities are not a distribution over the options.

The obvious next experiment is the one neither scorer does: feed the native
per-option score into the head as a feature, so short lists get the reranker's
judgement and long lists keep the competition.

## The 231 items are held out, measured rather than asserted

Every number above is only a held-out number if the items were not in training.
JevBench is not a training source in any config, so the only way it is not held
out is if its items duplicate items from the corpus we do train on.
`scripts/check_jevbench_leakage.py` tests that against the benchmark files,
which hold **1,562,838 training items** across the 25 non-JevBench sources:

| tier | n | state repeat | identical | label conflict | near-duplicate |
|---|---|---|---|---|---|
| easy | 48 | 0 | 0 | 0 | 0 |
| hard | 111 | 0 | 0 | 0 | 0 |
| original | 72 | 0 | 0 | 0 | 0 |

Three counts, because a match means different things. A **state repeat** is a
JevBench state that also appears in training under a different question — not
leakage on its own. **Identical** is the same state with the same options and the
same gold: a duplicate the model was trained on. A **label conflict** is the same
state and options with a different gold, which is not leakage but makes the item
unreliable. All three are zero.

Exact matching would miss a reworded copy, which is what a corpus assembled from
public sources would actually present, so the check also measures eight-word
shingle containment between each JevBench state and the training corpus. The
largest containment over the 231 items is **0.0011** — a single shared eight-word
shingle inside a long state — and the p99 is 0.0. The three tiers are authored
scenarios, which is what that distribution looks like.

The audit is `scripts/check_jevbench_leakage.py` (20 torch-free tests,
`tests/test_jevbench_leakage.py`) and its output is committed at
`model/experiments/jevbench_leakage.json`, so the claim can be re-run rather
than taken on trust.

## Structured states were reaching the model as Python reprs

The 111 hard items are not all plain text. **35 carry a structured `state`** —
an agent policy plus a conversation, a service table, a set of contact records —
and the loader passed that object through unchanged. Both scorers build their
prompt with an f-string, so those items were scored on `str(dict)`: single-quoted
keys, escaped unicode, one line, no structure. `render_jevbench_state` in
`data/pipeline.py` now serialises a dict or list as indented JSON with the
author's key order and unescaped unicode, and strings pass through untouched.

Measured, it is a small fix. Of the 35 structured items, 18 are choice; rendering
them changes **1 of 18** argmaxes and leaves the hard-tier choice accuracy
unchanged at 7/18. The other 17 are noul and score.

So this is a correctness fix — the evaluation should score the item as authored,
not as a Python literal — and **not an explanation of Finding 3**. The hard tier
is at chance for some other reason, and the repr rendering was not it.

## Two format hypotheses for the hard tier, both measured and both null

Finding 3 says we are at chance on the hard tier, and a prompt-construction bug
is the charitable explanation, so the two most likely candidates were tested
rather than argued about.

**Structured states as Python reprs.** Covered above: 1 of 18 choice items
changes answer, accuracy unchanged.

**The assistant suffix.** `model/src/reranker_prompt.py` ends the prompt with
`<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. Qwen3-Reranker is a
non-thinking model and its published format stops at `<|im_start|>assistant\n`;
the empty think block is what Qwen3's *generative* template emits. Scoring all
139 JevBench choice items through the fine-tuned backbone with the suffix
stripped:

| suffix | easy | hard | original | choice |
|---|---|---|---|---|
| with `<think>` block (shipped) | 36/36 | 25/67 | 30/36 | 91/139 = 0.6547 |
| plain `assistant\n` | 36/36 | 26/67 | 30/36 | 92/139 = 0.6619 |

One item. The hard tier moves 25 → 26 of 67, which is noise.

Between them the two hypotheses are worth two items on 139. **The hard tier is
not a prompt-format artefact**, and neither is the 14-21 point gap between the
head and the native readout. Whatever limits this benchmark at 4B is in the
model's ability to reason over a long composite state, not in how we render the
prompt. That is worth knowing before anyone spends a training run on it.

## The reference numbers measure something other than accuracy

The goal quotes "teacher 84.9, best open student 74.1" as JevBench references, and
this note has been comparing our accuracy against them. That comparison is
wrong, and the published board says why.

The live board (Benchmark Heaven, `benchmarkheaven.com/jev-models`, artifact
`results/v1.2/jevbench-v1.2-results.json`) scores a **composite of four axes** —
Intelligence, Calibration, Speed and Cost at 25% each, geometric mean — not an
accuracy. In that artifact:

| row | JevBench Score | Intelligence | hard-tier accuracy (220 items) |
|---|---|---|---|
| **Jev 1.13.0** | **74.40** | **85.69** | 0.7409 |
| SemIf (Qwen3.5-4B) | 73.09 | 78.96 | 0.5954 |
| djev | 73.03 | 82.67 | 0.6954 |
| openjev-thinking | 59.99 | 87.97 | 0.7818 |
| **Qwen3-Reranker-4B** | 63.83 | 63.98 | 0.5000 |

So **84.9 is Jev's Intelligence axis and 74.1-74.4 is Jev's composite score**, not
a second system's accuracy. Our 0.6475 is a *choice accuracy on 139 public
items*; quoting it against 74.1 compares two different quantities, and in the
direction that makes the gap look smaller than it is on accuracy and larger than
it is on the composite.

### The item-matched comparison

The artifact also publishes `public_tasks`: a per-item outcome code for every
system on the 111 public hard items we score. That makes an exact comparison
possible, on identical items:

| system | public hard, all 111 | public hard, the 67 choice |
|---|---|---|
| openjev-thinking | 0.766 | **0.791** |
| djev-thinking | 0.766 | 0.687 |
| Jev 1.13.0 | 0.730 | 0.761 |
| simplejev-qwen3.8-27b | 0.739 | 0.761 |
| openjev-sglang | 0.730 | 0.746 |
| djev | 0.676 | 0.672 |
| SemIf (Qwen3.5-4B) | 0.613 | 0.597 |
| open-alternative-jev | 0.568 | 0.552 |
| **Qwen3-Reranker-4B, our own backbone, untrained** | 0.495 | **0.507** |
| **our 4B, seed 42 ep5** (native / blend) | — | 0.493 / 0.478 |
| **our 4B, seed 123 ep10** (native / blend) | — | 0.373 / 0.403 |

Two things follow, and neither is comfortable.

**The gap is real and it is on accuracy, not units.** Jev's hard-choice accuracy
is 0.761 against our 0.373-0.493 — 27 to 39 points — and the best open rebuild
reaches 0.791. Nothing in the composite-versus-accuracy confusion rescues that.

**On our own backbone we are at or below the published row.** The board carries an
official `qwen3-reranker-4b` row — our exact base weights, scored through "one
neutral, preregistered option-ranking adapter" with temperature and yes/no
threshold grids fitted on public items — at 0.507 on the same 67 choice items. Our
fine-tuned checkpoint's native readout reaches 0.493 on one seed and 0.373 on the
other. **Fine-tuning has not improved our hard-tier choice scoring above the
untrained public row, and on one seed it is 13 points below it.**

That last comparison has a caveat that cuts both ways: their adapter's prompt and
threshold fitting differ from ours, so part of the difference is protocol rather
than weights. That is why it looked worth acting on. **It is not, and the check
below is why.**

### Their documented mapping does not reproduce their number

The board publishes the adapter (`jevbench/adapters/reranker.py` and
`docs/v1.2-additions-rerankers.md`), so the candidate explanation is testable.
Reading it, two things stand out. Its **query carries the complete answer
rubric** — state, question and every option — where ours carries only the state
and the question; that is an input difference the model actually sees. And its
**temperature grid and noul threshold cannot move accuracy at all**, because both
are monotone maps that leave every argmax where it was — the same argument that
ruled out temperature scaling for clause 2.

So the only accuracy-relevant difference is the prompt, and it is cheap to test
in our own harness. `model/scripts/eval_jevbench_adapter.py` implements their
mapping verbatim — rubric in the query, their exact instruction wording, compact
JSON for structured states — and scores the 111 public hard items:

| base Qwen3-Reranker-4B, 111 public hard | all | 67 choice |
|---|---|---|
| the board's published row | 0.495 | 0.507 |
| our reimplementation of their mapping | 0.4505 | 0.4179 |
| our own prompt, same harness | 0.4595 | 0.4328 |

**Two results, both negative.** The rubric-in-query prompt is worth **minus one
item** — 28/67 against 29/67 — so it is not the explanation and there is nothing
to adopt. And our reimplementation of their published mapping lands 8.9 points
below their published figure on the same weights, so **their number is not
reproducible from their own documentation**: something in the transport, which
the docs do not pin down, accounts for it.

That means the sentence above — that fine-tuning leaves us below the untrained
public row — **is not established**. Comparing our 0.373-0.493 against their
0.507 compares two harnesses whose calibration we cannot measure; our best
reimplementation of theirs sits at 0.418-0.433. The honest statement is that our
hard-tier choice accuracy is 0.373-0.493 in our harness, and that an official row
for the same base weights exists at 0.507 which we cannot reproduce.

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
not the backbone, is the constraint — with Finding 5's qualification that the
native readout wins only on short lists, and that the head is what keeps long
lists and calibrated probabilities working.

That is one measurement, one seed, 231 items. The hard tier is where the
remaining gap sits (0.351 against a 0.336 chance floor), and it is the column to
optimise rather than the overall number, because easy is nearly saturated for
anything that works.
