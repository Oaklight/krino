# A set-function head over frozen option states (#340) — 2026-11-09

Panel 2 reads one code token per option and takes the argmax; 3b scores options
with a dot product and 3c with a shared scalar. In all three the options never
interact inside the head — whatever interaction exists happened inside the
backbone before the states were read. This note runs D: one forward, the K
option states, and a small set transformer over them, trained with the
validity gate that B failed.

D is valid and does not beat panel 2. It is above panel 2 on the development
set (+5.67 pp, exact paired p = 0.00061) and below it on JevBench (−10.39 pp,
p = 0.0037). Under the reading rule fixed before the run, that lands in the
issue's second branch: the run is reported as the negative it is. The
order-invariance requirement fails on both instruments as well — D changes
35.6% of its choices under reversal against panel 2's 28.3% on the 812-item
development set and 43.7% against 22.5% on JevBench — which matters more than
the accuracy sign, because it is the one property the design was supposed to
buy.

## Design

The head is the shape #340 pre-registers. Width 192, two pre-LN
self-attention blocks with a feed-forward of width 384 and four heads, one
learned query that pools the set through multi-head attention (PMA), and a dot
product between each option and the pooled vector. Dropout is zero. The loss is
a softmax cross-entropy over the K options, so noul is a two-way choice and
choice is a K-way one under one objective; score items use the same K-way form
over their levels.

The function is permutation-equivariant by construction: no parameter or
operation refers to an option's index, so a permutation of the input permutes
the K logits and leaves the pooled vector unchanged. Tests assert that
directly, and also that masking a slot gives it zero probability mass without
moving the real options.

Two layers have to be kept apart when reading the reversal numbers. The head
is equivariant; **the states are not.** An option shown in a different place
gets a different hidden state because the tower's position encoding changes,
and the head then acts on a different set. Equivariance guarantees only that
the head adds no order dependence of its own — it cannot remove the one in its
input. The measurement below is of the whole pipeline, and every bit of
sensitivity it finds was created before the head.

Training is on the #324 canonical corpus: 177,224 train items (543,572 option
states) and 12,944 eval items (42,541 states). AdamW at 1e-3 with weight decay
0.01, batch 64, five epochs, seed 42; the epoch is chosen by minimum
eval-split NLL, and the validity gate runs on that same eval split. JevBench
and the development set are read once, after selection.

## The state cache

Each item is rendered with the shipped panel-2 prompt and tokenised with
`add_special_tokens=False`, the same input panel 2 reads. One forward through
the frozen tower gives a hidden state at the last token of each option's text;
the K states are stored as the bf16 bit pattern in an `int16` array, so a
reload is exact. A reversed-order run stores its states in natural option
order, so the head always sees natural order and the two orders differ only in
the content of the states.

The cache is keyed on the model revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, the tokenizer's own file digest
`a83517a80bba2185d4aec2719ffc4fcfd1c3a322b9d87e9b673039bdc2f7997e`, the
prompt-template digest and the display order; the composite key is
`f9245673d23ac0ea9a0e6b403e26c7986e00815b92e5fbddd6f58a3e1c01c565`.

| job | split | items | states | GB | s | bit-identical |
|---|---|---:|---:|---:|---:|---|
| corpus shard 0 | train + eval | 47,543 | 146,436 | 0.76 | 964 | yes, 2 batches |
| corpus shard 1 | train + eval | 47,543 | 146,492 | 0.76 | 959 | yes, 2 batches |
| corpus shard 2 | train + eval | 47,541 | 146,614 | 0.77 | 943 | yes, 2 batches |
| corpus shard 3 | train + eval | 47,541 | 146,571 | 0.77 | 957 | yes, 2 batches |
| JevBench natural | jevbench | 231 | 849 | 0.00 | 26 | yes, 4 batches |
| JevBench reversed | jevbench | 231 | 849 | 0.00 | 26 | yes, 4 batches |
| dev natural | dev | 812 | 4,341 | 0.02 | 69 | yes, 2 batches |
| dev reversed | dev | 812 | 4,341 | 0.02 | 69 | yes, 2 batches |

Every job re-ran its leading batches twice and compared the stored arrays with
`torch.equal`: all report `identical: true` and `max_abs_diff: 0.0`. The whole
cache is 192,254 items, 596,493 states, 3.11 GB.

The guarantee is "the same batch twice", not "any batching". A dev re-extraction
at batch size 8 instead of 16 moved two of 788 predictions, because padding
changes the GEMM shapes and bf16 reductions are not associative; the stored
cache is fixed, so every number below is reproducible from it, but a future
extraction should reuse the same batch construction rather than assume the
states are batch-independent to the bit.

Items whose prompt exceeds 8192 tokens are excluded and counted: 38 of 190,206
corpus items (all in the training split, none in JevBench or dev). The first
attempt had no cap and stalled for over 25 minutes on a 29,476-token prompt, so
the cap is a throughput decision, not an accuracy one — the state of an option
at the end of a 30k-token code snippet is not a state worth spending a GPU-hour
on.

The development set is the registered 812 items, all scored. Twenty-four of them
are banking77 items with 80 options, which the shipped 26-letter alphabet cannot
name; those prompts are rendered with the 84-symbol alphabet the #326
development cell used (A–Z, a–z, then single-character symbols, identical to
main's first 26), so the panel-2 reference and the states come from the same
text.

## Results

JevBench 231 and the registered dev 812, panel 2's reference against D on the
same items. Panel 2's JevBench reversal is the committed cell (157/231 on the
display-order tie rule); under the #347 natural-index rule its one exact-tie
item `hard-opus-a-probability-03` is wrong in reversed order, making it 156.
That is noted where it is compared and is not retro-fitted.

| instrument | order | D | panel 2 | difference |
|---|---|---:|---:|---:|
| JevBench 231 | natural | 138/231 = 59.74% | 162/231 = 70.13% | −10.39 pp |
| JevBench 231 | reversed | 130/231 = 56.28% | 157/231 = 67.97% | −11.69 pp |
| dev 812 | natural | 545/812 = 67.12% | 499/812 = 61.45% | +5.67 pp |
| dev 812 | reversed | 438/812 = 53.94% | 466/812 = 57.39% | −3.45 pp |

The dev reference is the committed #326 cell
(`results/jevbench/dev_panel2_Qwen_Qwen3-Reranker-4B.json` at `b68f266`,
sha256 `f952b7e47b98a2d95c25d0ba45f497632d9c11476e9e1a1d099c4c2950562875`),
which predates the #347 tie rule. On the 788 items whose prompts use the shipped
A–Z alphabet, a re-run under the #347 rule gives 490/788 natural and 463/788
reversed against that cell's 490 and 460: three reversed items differ by the
tie convention alone. Nothing is retro-fitted, and the comparison does not move
with them.

By type on JevBench, natural / reversed: noul 45/74 and 43/74 (panel 2 48 and
50), choice 84/139 and 79/139 (panel 2 102 and 94), score 9/18 and 8/18
(panel 2 12 and 13). On dev, natural / reversed: noul 281/391 and 232/391
(panel 2 258 and 241), choice 211/322 and 168/322 (panel 2 186 and 177), score
53/99 and 38/99 (panel 2 55 and 48).

On its own corpus eval split the head reaches 70.72% at the selected epoch
(epoch 4 of 5, eval NLL 0.664).

## The validity gate

On the corpus eval split, 12,944 items / 42,541 option outcomes, never on
JevBench and never on dev.

| check | value | threshold | passes |
|---|---:|---:|---|
| Brier vs constant base rate | 0.1165 | < 0.2117 | yes |
| gold positive rate (P(gold) > 1/K) | 78.90% | ≥ 15.21% | yes |
| paired gold − non-gold | +0.4179 | > 0, p < 0.01 | yes (p ≈ 0) |

The run is valid under the issue's three checks. The Brier comparison is on the
option-level binary outcome, the pooled gold fraction is `mean(1/K) = 0.3043`,
and the positive-rate threshold is half of it; for a two-option item the
positive test is exactly B's `P(yes) > 0.5`. The paired test is the #324
`paired_row_test` over 12,944 rows.

## Order invariance

The four numbers the #346 protocol asks for, plus panel 2's own changed
fraction on the same items:

| instrument | natural | reversed | Δ pp | changed | McNemar W/L | exact p | panel 2 changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| JevBench 231 | 59.74% | 56.28% | +3.46 | 43.72% (101) | 46/38 | 0.445 | 22.51% |
| dev 812 | 67.12% | 53.94% | +13.18 | 35.59% (289) | 182/75 | 1.9e-11 | 28.33% |

**D's order invariance is worse than panel 2's on both instruments, so the
design intent is not met even where accuracy improves.** On dev the head is
5.67 pp better than panel 2 in natural order and 3.45 pp worse in reversed
order, and the reversal costs it 13.18 pp with p = 1.9e-11 — a real effect, not
noise. On JevBench the churn nearly doubles panel 2's (43.7% against 22.5%)
while the accuracy difference between orders is unresolved.

This is the reading the equivariance argument predicts but does not excuse. The
head cannot inject order dependence, and it does not: permuting its input
permutes its output. What it does is read order information that the frozen
tower already wrote into the option states, and it uses it more heavily than
the vocabulary readout does. A set function over states that are not
order-invariant is not an order-invariant decision rule; it only moves the
responsibility for the sensitivity one layer back, into the extraction.

No exact ties occur in any of the four cells (`tie_items` is 0 in both orders
on both instruments), so the #347 tie rule has nothing to resolve here.

## Against panel 2

Paired exact McNemar on the same items, and the project's three-state interval
screen with D as the arm:

| instrument | order | D | panel 2 | Δ raw | screen Δ (cc) | 95% CI (cc) | W/L | exact p | screen |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| JevBench | natural | 138 | 162 | −10.39 pp | −14.32 pp | [−25.17, −3.47] | 20/44 | 0.0037 | unresolved |
| JevBench | reversed | 130 | 157 | −11.69 pp | −17.58 pp | [−26.90, −8.25] | 11/38 | 0.00014 | unresolved |
| dev | natural | 545 | 499 | +5.67 pp | +9.87 pp | [4.62, 15.13] | 110/64 | 0.00061 | improvement supported |
| dev | reversed | 438 | 466 | −3.45 pp | −5.33 pp | [−11.48, 0.83] | 90/118 | 0.061 | unresolved |

Within the 788 dev items the shipped A–Z alphabet can name, D is 538 against
panel 2's 490, +6.09 pp, p = 0.00029, and 435 against 460 reversed, −3.17 pp,
p = 0.092 — the same reading as the full 812, so the 24 extended-alphabet items
do not carry the result.

The two instruments disagree in sign, and unlike #324's B arm the disagreement
is not a pair of nulls: dev supports an improvement and JevBench supports a
regression, both at p < 0.01. The screen calls the JevBench cells "unresolved"
only because the chance-corrected interval crosses the 9.3 pp margin; the exact
paired test says the deficit is real.

## Reading

**The arm lands in branch two: validity passes, and interaction in the head
does not substitute for the vocabulary anchor.** On the acceptance instrument D
is 10.4 pp below panel 2 and 5.2 pp below judgement (59.74% against 64.94%),
with 44 items that panel 2 gets and D does not against 20 the other way. On dev
it is 5.7 pp above panel 2. A 231-item instrument cannot hold both readings at
once, and neither can an 812-item one contradict a 10 pp JevBench deficit
cheaply — but the acceptance gate is JevBench, and there the arm loses.

**Which limitation binds is the anchor, not the interaction.** The head has
more capacity to model the option set than either 3b or 3c (it is a learned set
transformer, not a dot product), and it still does not reach the vocabulary
readout on JevBench. The training corpus is the #324 list, 55% two-option
items, and the JevBench choice items are where the loss is concentrated: 84/139
against panel 2's 102/139. The dev result says the interaction is not worthless
— it is worth about 6 pp there — but it does not buy what the anchor buys.

**The design intent fails independently of accuracy.** The issue's point in
building D was a decision rule that is permutation-equivariant by construction,
with the hope that this would make the answer stable under reordering. The
accuracy did not improve on JevBench, and the stability got worse on both
instruments. A future set-function arm should either extract states that are
themselves order-robust (the #339 content-addressing path, or a position-free
readout) or accept that the sensitivity is a property of the backbone and stop
attributing it to the head.

## What this does not say

It does not say cross-option interaction is useless: D improves on panel 2 on
dev and its corpus eval split is at 70.72%. It does not say a bidirectional or
adapted backbone would fail — the states here are causal and frozen, and the
head sees only their last-token values. It does not price the vocabulary anchor
in isolation, because D differs from panel 2 in more than the anchor (trained
versus zero-shot, set versus code readout). What it does say is that this shape
of set head, on these states, at this corpus, does not beat the shipped readout
on the acceptance instrument and does not make order sensitivity smaller.

## Provenance

Pre-registration: #340 comment (`issuecomment-6744071670`), written before the
training run. Config hash
`debd40bf46a883947db2433bf407d5695b342e3e44a6d609a765090bf690e348`, data hash
`5fd0e6f9cb6b28509247a7dc1475fa5b0c630286a04d9e647086823277421ad1` (sha256 of
the corpus rows file; its item set is the #324 plan's `row_id` set, rebuilt and
compared before extraction), branch `feature/340-set-head`. Model
`Qwen/Qwen3-Reranker-4B` at `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, bf16,
one forward per item, `--code-form space`, prompt template unchanged from main.

**The extraction recipe is part of the run, not an implementation detail.**
Re-extracting the 788-item dev subset at batch size 8 instead of 16 moved two
predictions, so the stored states are deterministic for a given batching and
not across batchings. The recipe used here is corpus batch 32 (four shards),
JevBench 16, dev 8 (the second pass, whose states are the ones scored), with
`max_batch_tokens` 65536 and `max_prompt_tokens` 8192; every manifest records
the batch size, both caps and the skip list, and `head.json` records the
sha256 of the train states it consumed. The registered hash above predates that
field; the same settings under the current code give
`e9094d15622c0e6910815064926e4bd295defac4998361f5924a0287dc16844d`, which is
the hash a re-run now produces. No reported number changes.

The corpus is the canonical source list of `model/configs/r2_reranker4b.yaml`
with its per-source caps and seed 42. Rows are built by
`model/scripts/build_set_head_rows.py`; states by
`model/scripts/extract_panel2_states.py`; the head by `model/src/set_head.py`
and `model/scripts/train_set_head.py`; evaluation by
`model/scripts/eval_set_head.py`.

The dev rows file used for the states is a later build than the corpus rows
(sha256 `91fe63dd1aca1b15ba496f457f7e0c422b6fc2bb1179c6c8ccb534d9f071d317`): it
holds all 812 items, and the 24 banking77 ones are rendered with the #326
84-symbol alphabet. The corpus rows file — the one the config hash covers — is
unchanged. Panel 2 on the JevBench rows is scored by
`model/scripts/score_panel2_rows.py`, which drives the shipped readout and
reproduces the committed JevBench cell exactly in natural order (162/231) and
except for the one #347 tie item in reversed order (156 against 157); the same
script on the 788-item dev subset gives the 490/463 quoted above.

Files: `results/jevbench/set_head_jevbench_Qwen_Qwen3-Reranker-4B.json`,
`results/jevbench/set_head_dev_Qwen_Qwen3-Reranker-4B.json`,
`results/jevbench/set_head_config_340.json`, the panel-2 dev reconstruction
`results/jevbench/panel2_dev_rows_Qwen_Qwen3-Reranker-4B.json` (788 rows, used
only for the tie-rule check, not as the comparison reference), and the trained
weights at `model/experiments/set_head/head.pt` with its run JSON beside it.
