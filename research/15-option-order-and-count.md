# Option order and option count: clause 3 measured

Goal item 3 has two halves — an option-order flip rate at or below 5%, and accuracy
on questions with more than twenty options within 5 points of the ones with five
or fewer. Both are measurable from one run, and neither had been reported on the
checkpoint that carries the rest of the goal.

Measured on `arch_4b_loraAll_weighted`, the clean corpus after #177, 1,500 items
per type, seed 42, with order probes enabled
(`calibration_4b_loraAll_weighted_probe.json`).

## The flip rate is exactly zero, and that is structural

| probe | n | flips | flip rate | max total-variation |
|---|---:|---:|---:|---:|
| choice, option permutation | 1,500 | 0 | **0.0000** | 0.00000 |
| score, levels reversed | 1,500 | 0 | **0.0000** | 0.00000 |
| score, levels permuted | 1,500 | 0 | **0.0000** | 0.00000 |

Accuracy is unchanged by every probe (0.5667 before and after in each case), and
the total-variation distance between the original and re-run distributions is
zero to five decimals on every item. Two of the 1,500 choice items have a tied
argmax (duplicate option text, both in the 3-5 bucket); the flip rate excluding
ties is also 0.0. The same measurement on two other
checkpoints — `arch_4b_loraAll_32k` and `arch_06b_loraAll_weighted` — gives 0 of
6,864 choice items and 0 of 1,857 score items.

A number that clean needs an explanation before it is believed, so: the probe
permutes the option *texts*, keeps each option's source label alongside them, and
remaps the returned logits back onto the original order before comparing. It is
the only reordering the model can observe, because the head consumes texts and
never keys. The head itself is an attention over the set of option hidden states
with no positional term for options, so a permutation reorders the outputs and
nothing else — the invariance is a property of the architecture, not a lucky
sample. Exact equality rather than approximate equality suggests the attention
reduction is deterministic under this reordering; the note does not claim more
than that.

So the flip-rate half of clause 3 is met with room, and it will stay met: no seed
draw can move a structural zero.

## The option-count half is met at the point estimate, and the bucket is too small to certify it

Choice accuracy by option count, same run:

| options | n | accuracy | mean chance | normalised |
|---|---:|---:|---:|---:|
| 2 | 224 | 0.6071 | 0.5000 | 0.214 |
| 3-5 | 1,169 | 0.5595 | 0.2543 | 0.409 |
| >20 | 107 | 0.5607 | 0.0125 | 0.555 |

The clause compares the extreme buckets: 0.5607 against 0.6071 is a **4.6 point**
gap, inside the 5 point allowance, and against the 3-5 bucket it is 0.1 points.
On the normalised column the ordering reverses — the >20 bucket is the *best*
relative to chance, because chance there is 1.25% and the model gets 56% of them.

Two cautions. The >20 bucket is 107 items, so its standard error is about 4.8
points, and the *difference* of 4.6 points traces back to two buckets whose
combined standard error is about 5 points: the clause is met on this draw, not
demonstrated to be met on every draw. And it is measured at 1,500 items per type
on one seed; the goal asks for two.

## Why the two halves behave differently

Order invariance comes free with a set-attention head — it is a symmetry of the
architecture, so it is stable across seeds and checkpoints (three checkpoints,
zero flips). Count robustness is a capability, not a symmetry: it depends on how
well the head compares twenty options against each other, and it is exactly where
the alternative scoring path collapses. Scoring options independently with the
reranker's own yes/no logit — which beats the head on JevBench, where items have
two or three options — scores **0.3178** on the same >20 bucket against the head's
0.5607, a 35 point gap that fails this clause outright (#180, Finding 5).

So the clause is a capability test wearing a robustness label, and the head is the
reason it passes.

## Caveats

* **One seed, 1,500 items per type.** The goal wants two seeds on the
  cluster-disjoint split; this run is on the clean corpus but a pre-#138
  checkpoint, so it is the right measurement on the wrong checkpoint to count.
* **The >20 bucket is 107 items.** Its standard error is ~4.8 points and the
  clause's margin is 4.6, so a second seed could flip it. This is the number to
  watch, not the flip rate.
* **The probe tests the model, not the tokenizer or prompt.** A pipeline that
  reorders options before tokenisation in a way that changes the text would not be
  caught here — though none does today, because the harness builds the option list
  straight from the criteria dict.
* **Score probes reorder levels, which for an ordinal head is a stronger
  invariance claim than it looks.** The head has no notion of level position, so a
  reversed rubric list is the same computation with the outputs permuted; a model
  that learned ordinality from position would show flips here and does not.

## What this means for the goal

Clause 3 is the first clause that can be reported as met on both halves — 0.0% flip
rate, 4.6 point count gap — with the caveats that the count margin is smaller than
one standard error of its bucket and that neither half has been run on a
#138-trained checkpoint with two seeds. The flip half is safe; the count half is
the one that needs the second seed before it is called.
