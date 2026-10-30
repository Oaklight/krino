# JevBench-public: the trained heads lose 14-21 points to our own readout

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
majority guess to be the explanation.

Identical counts across two backbones 6.7x apart are still at least as consistent
with the head being driven by something the backbone does not move — a dominant
option key, a degenerate logit pattern on out-of-distribution items — as with a
real capability, so the predictions themselves were dumped
(`--dump-predictions`, `model.evaluation.jevbench.prediction_rows`). They say it
is not that either:

| | 4B head | 0.6B head |
|---|---:|---:|
| choice items correct | 54 / 139 | 55 / 139 |
| distinct option keys predicted | 96 | 98 |
| modal key's share of the 139 items | 5.8% | 5.8% |
| items where both arms predict the same key | 54 / 139 | |
| both arms correct, against an independence null of 21.4 | 29 | |

**The head does not collapse onto a key.** It spreads its predictions over 96 of
the 139 items' keys, and the key it names most often accounts for 5.8% of items —
about twice the concentration of the commonest gold key (4 items, 2.9%), which is
what any scorer mildly favouring a frequent answer would produce, and nowhere
near a base rate. What the two arms do share is accuracy, not choices: they name
the same key on only 39% of items, and 29 items are right on both against 21.4
expected if they answered independently. So 0.396 is two models that are related
but not the same model landing on the same score, which is the honest description
— and it means the identity is a coincidence of two middling scorers, not
evidence that one number measures both.

That pass reproduces 54 and 55 rather than 55 and 55; the one-item movement is
the same pass-to-pass variation the calibration reports in `research/11` measure
at 0.19% of items. Nothing in the reading depends on which item moved.

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

## The long state is the wall — measured, not asserted

The paragraph above was a hypothesis when it was written. It is now a
measurement: the per-item predictions (`--dump-predictions`) joined to each
item's state length, for the trained head and for the backbone's zero-shot
native readout on the **same checkpoint** (s123, epoch 10).

| hard-tier state length | n | trained head | native readout |
|---|---:|---:|---:|
| 350 – 987 chars | 27 | **0.630** | **0.667** |
| 995 – 1,493 | 27 | 0.407 | 0.407 |
| 1,553 – 8,377 | 27 | 0.296 | 0.259 |
| 8,566 – 14,986 | 30 | **0.300** | **0.367** |

Both scorers lose about 30 points from the shortest quartile to the longest, at
the same places. The longest state in the tier is ~3.7k tokens, inside the
16,384 window, so this is not truncation, and the two scorers build their
prompts differently — so it is not the prompt either.

Two other cuts rule out the explanations that would have been cheaper to fix:

- **Structured states are not the problem.** Hard-tier accuracy is 0.486 on the
  35 dict-state items against 0.368 on the 76 string-state ones. The rendering
  works, and the structured items are simply shorter.
- **Option count is not the problem.** Hard items with 2–4 options score 0.444;
  5–10 score 0.238.

**And the head is not the problem either.** Swapping the tower head for the
cross-encoder raises the board a lot and this tier not at all:

| model | easy (48) | hard (111) | original (72) | overall (231) |
|---|---:|---:|---:|---:|
| 4B tower, epoch 10 | 0.792 | 0.405 | 0.486 | 0.511 |
| 0.6B cross-encoder, epoch 1 | 1.000 | 0.396 | 0.611 | 0.589 |
| 4B cross-encoder, epoch 1 | 1.000 | **0.405** | 0.722 | 0.628 |

Three architectures, a fifth of the parameters in one of them, and the hard tier
lands on the same number to three digits while the rest of the board moves 12 to
24 points. Whatever closes the gap to the reference has to act on how much of a
long state the backbone uses — training on long states, or compressing and
retrieving over them — and no head, prompt or calibration change will.

### What the corpus contains is a long state asked as a noul question, and almost never as a choice

The wall has an explanation one query away, and it is about the input
distribution rather than the model. `scripts/state_length_audit.py` reads the
`state` field of every source the training config names, on the train split and
capped at each source's own `max_train` — the corpus a run actually draws from,
not the union of every file — and splits it by question type, because the
conjunction is what the hard tier is made of.

| | n | p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| corpus, all types | 172,262 | 246 | 2,017 | 14,093 | 54,765 |
| **corpus, choice only** | **69,287** | **117** | **770** | 1,600 | 38,047 |
| **JevBench hard, choice only** | **67** | **2,533** | — | — | 14,986 |

The hard tier's *shortest quarter* starts at 1,274 characters, above the corpus's
90th percentile for choice. Concretely, of the 69,287 choice items in the corpus:

- **3.13 %** (2,167 items) are at least as long as a hard-tier item in its
  shortest quarter;
- **0.036 %** (25 items) reach the tier's median;
- **0.003 %** (2 items) reach its upper quarter.

The corpus does contain long states — `contractnli` at p50 2,774 characters,
`multirc` at 1,661, `tabfact` at 847, `boolq` at 578 — but **every one of those
sources is noul**. The choice sources are uniformly short: `banking77` p50 46
characters, `piqa` 35, `commonsenseqa` 64, `swag` 67, `winogrande` 97,
`hellaswag` 126. The only choice source of any size with real length is
`typed_decisions` (p50 397, max 1,559, 800 items), and its maximum is still well
below the 8,566 characters where the tier's longest quartile begins.

So the model has been trained to answer long states as yes/no questions and short
states as choice questions, and the hard tier asks a third thing — a 2.5 kB state
with a list of options — that appears 25 times in 172,262 training items. That is
also why the three architectures agree on 0.405: they differ in how they score an
option, and this is not a scoring problem. The failure is upstream of every head,
in a region none of them was trained on, which is the same conclusion the
quartile table reached from the other direction.

The implication for what to build is specific and cheap relative to the accuracy
it could recover: **generate long-state choice items**. The synthetic pipeline
already carries a `long_doc` family at 8,000 characters and an 8-domain
"long-horizon" assembly, but the per-domain files are not what training reads —
`data/pipeline.py` assembles synthetic states at load time, so this audit excludes
them rather than measure them wrongly. Whether the assembled corpus closes the
length gap is a question about the loader, and the next thing to measure.

Two limits on this table. Lengths are characters of the `state` field, and what
the model sees is `state + instructions`, so the gap in tokens is a little
smaller than the gap in characters — not enough to matter at 20×. And the 67
hard-tier choice items are 60 % of the 111-item tier; the remaining 44 are noul,
whose accuracy is not split out by state length here, so this explains most of
the tier's deficit but not measurably all of it.

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

**The comparison to the published row cannot be made, and the next section is
why.** The board carries an official `qwen3-reranker-4b` row — our exact base
weights, scored through "one neutral, preregistered option-ranking adapter" with
temperature and yes/no threshold grids fitted on public items — at 0.507 on the
same 67 choice items, where our fine-tuned checkpoint's native readout reaches
0.493 on one seed and 0.373 on the other. Read as a like-for-like, that says
fine-tuning has not cleared its own starting point. It cannot be read that way:
the two numbers come from different harnesses, and when the board's documented
adapter is reimplemented below it lands 8.9 points under their published figure
on the same weights. Their row is not reproducible from their own documentation,
so it is not a yardstick for ours. Our hard-tier choice accuracy is **0.373-0.493
in our harness**, full stop, and no statement about how that compares to the
untrained base on the board's terms is available.

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

That is what removes the only candidate explanation that was available for the
published row: their adapter's calibration is monotone and their prompt is worth
a single item, so the 8.9-point difference sits in the transport rather than in
anything documented. Their row cannot be reproduced, and ours is therefore
reported on its own terms — 0.373-0.493 in this harness — with no claim about
where it sits relative to the untrained base as the board measures it.

### How much of the hard-tier deficit is checkpoint variance?

Every hard-tier number in this note is one checkpoint. The 67 public hard choice
items are few enough that the question is worth asking directly: how much of the
0.373–0.493 spread is the checkpoint rather than the model? Two checkpoints on the
same split at the same training length answer it — seed 123 and seed 42, both at
epoch 5, scored through both readouts. `model/scripts/jevbench_cross_seed.py`
takes their per-item score vectors (the `--dump-records` output of
`blend_scorers.py`, on the PR that introduces it), maps them onto the 67 items by
file order after checking every option count, and writes
`model/experiments/jevbench_cross_seed_hard.json`:

| readout | seed 123 | seed 42 | both right | either right | items they disagree on |
|---|---:|---:|---:|---:|---:|
| native (backbone readout) | 0.358 | 0.493 | 21 | **36 (0.537)** | 15 |
| trained choice head | 0.328 | 0.254 | 8 | 31 (0.463) | 23 |

Three things follow.

**The seed alone moves the hard tier by 13 points.** The matched pair puts the
native readout at 0.358 and 0.493 — nine items, 13.4 pp — which is larger than
every effect this project has measured anywhere else. On 67 items a single
checkpoint's hard-tier accuracy is not a measurement of the model; it is that
checkpoint's draw. Only the paired comparison above is quoted.

**The head is close to independent across checkpoints; the native readout is
not.** If two checkpoints answered independently at their own rates, the expected
number of items both would get right is 5.6 for the head and 11.8 for the native
readout. Observed: 8 and 21. The head's overlap is barely above chance — what it
gets right on a hard JevBench item is largely not a property of the training
recipe — while the native readout shares most of its successes, which is the same
stability that made it the better scorer in the first place.

**Ensembling checkpoints cannot close the gap.** A perfect oracle that took the
right answer from either seed would score 0.537 on the native readout. Jev scores
0.761 on these 67 items. So even with an oracle — not achievable, it uses the
label — two independently trained checkpoints leave 22 points to Jev. The deficit
is not checkpoint variance and no amount of seed averaging reaches it.

The items both seeds miss are 30 of 67: long_policy 7 of 12, multi_hop 8 of 16,
tradeoff 4 of 4, probability 4 of 7, ambiguous 3 of 7, temporal_numeric 4 of 10,
adversarial 1 of 3, trap 0 of 3, routing_hard 0 of 5. That ordering is suggestive
— tradeoff and long_policy look like the worst families — but a family with four
items cannot support a claim, and this note does not make one: the same table run
at different checkpoints moves by one to three items per family, which is the
whole reason the aggregate needed two seeds first.

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
Finding 4 says where part of it is. The fine-tuned backbones keep the reranker
capability (0.541 at 0.6B, where it improves on the base, and 0.623 at 4B, where
it holds), and the choice head throws 14 to 21 points of it away on the board as
a whole. So the sentence this note would have ended on — that the model we train
is worse than the model we started from — is true of the heads and false of the
backbones, and for the easy and original tiers the work belongs in how the choice
head scores an option list.

**The hard tier is a different finding, and the last section changes the
prescription.** There the head is not the constraint: three architectures, one of
them a fifth the size, return 0.405 and the quartile table shows the trained head
and the untouched native readout failing at the same state lengths. The corpus
audit says why — long states reach the model as noul questions, and a long state
asked as a choice appears 25 times in 172,262 training items. So the hard column
is a data problem before it is an architecture problem, and the next experiment on
it is to assemble long-state choice items rather than to build another scorer.

That is one measurement, one seed, 231 items. The hard tier is where the
remaining gap sits (0.351 against a 0.336 chance floor), and it is the column to
optimise rather than the overall number, because easy is nearly saturated for
anything that works.
