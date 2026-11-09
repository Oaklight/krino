# Judgement architecture: readout shape and direction (#324) — 2026-11-09

Our best trained head scores 42.9% on JevBench 231 while the same backbone,
untrained, scores 64.9% under the judgement readout. This note asks whether the
gap is the causal reading direction or the per-option scope, on one backbone:
`Qwen/Qwen3-Reranker-4B`, frozen.

The trained-head arm B is not reported here. Its first run failed its own
validity check, so it is a process failure rather than a hypothesis test; a
pre-registered rerun is the arm of record and lands separately. The diagnosis
of that first run is kept at the end because it is the reason the rerun exists.

The design also moved before any cell ran, and the withdrawn cells are recorded
so the numbers below are not read against the wrong question: a per-option mask
flip (E, G) was replaced by a per-request layout; a softmax-versus-independent
pair (H2, H4) was dropped because softmax over a shared head is monotone and
cannot change the argmax; and the trained one-read code head (D1, D2) is held
because its frozen-backbone form is a random linear on a frozen backbone, the
configuration already measured at 0.351 against a 0.441 raw readout.

## The cells

| cell | layout | direction | readout | trained |
|---|---|---|---|---|
| A | per-option, one sequence per option | causal | vocabulary `log P(yes) − log P(no)` at `Answer:` | no |
| J-R1 | per-request, one sequence with K `Answer:` spans | causal | vocabulary at each span | no |
| J-R2 | per-request, one sequence with K `Answer:` spans | full | vocabulary at each span | no |

All on the same 231 items, same backbone, same 36-layer full-attention config.

## Results

| cell | overall | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|---:|
| A | 150/231 = 64.94% | 47 | 90 | 13 |
| J-R1 | 137/231 = 59.31% | 46 | 80 | 11 |
| J-R2 | 74/231 = 32.03% | 34 | 33 | 7 |

Exact paired McNemar: J-R1 vs A +39 −26, p = 0.136; J-R1 vs J-R2 +97 −34,
p = 3.3e-8.

## The branches

**The direction is not a free change.** J-R2, the same layout under a full
mask, falls to 32.03%, which is chance (31.8%). It predicts option 0 on 161 of
231 items against 70 gold zeroes, and 54 of those 161 are right: a first-position
prior, not a uniform readout. This is a zero-shot flip on a causally trained
checkpoint, so it says the checkpoint cannot be flipped for free; it is not
evidence that a trained bidirectional arm would fail. J-R2 is killed as a
zero-shot arm.

**The scope difference is smaller than dev resolves.** On the 812-row
development set — the primary instrument here because it resolves about 4.9 pp
— J-R1 against A is 466 against 478, 1.48 pp, +64 −76, exact McNemar p = 0.353.
JevBench says the same thing less sharply: 137 against 150, 5.6 pp, +39 −26,
p = 0.136, against a 9.3 pp floor. The per-request layout is neither better nor
demonstrably worse, and it does not recover the gap.

## B's first run: withdrawn, and the diagnosis

The first B run is withdrawn as invalid. On its own corpus eval split only 17
of 10,911 gold options reached P(yes) > 0.5, so the head carried no usable
probability and its 140/231 was a within-item ranking artefact. It supports
neither side of the projection question.

The diagnosis matters for the rerun. The head initialises with the yes bias at
+2.0 (`TwoWayHead` sets `bias[1]`, and `logits[1]` is yes), so at step zero
P(yes) is about 0.88 on every option; the run cannot have collapsed at
initialisation. The per-epoch count of gold options above P(yes) = 0.5 was 4,
24, 17, 1117, 124: the head was driven down almost immediately, partially
recovered at epoch 4, and fell again. That is drift under the training recipe
(cross-entropy, lr 3e-3, early stopping), not a sign error in the bias.

The pre-registered rerun fixes the recipe instead of reinterpreting the run:
three declared variants (plain BCE with zero bias and lr 1e-3; the same plus a
bias-only calibration pass on the train split; BCE with positive weight
`1 / base_rate`), a validity gate on the eval split only (Brier against the
base-rate predictor, gold positive rate, and a paired gold-minus-non-gold test
at p < 0.01), and one JevBench evaluation of whichever variant passes. If none
passes, the projection hypothesis is reported untested.

## Latency

Single-item mean and median, same 231 items: J-R1 0.330 s / 0.441 s; J-R2
0.335 s / 0.450 s. One forward either way, so the mask shows no single-item
cost. The second question over the same state is the trade: a causal pass can
reuse the state's KV cache, a full-attention pass cannot. Measured on this
layout, the causal state-prefix cache costs 0.116 s mean (0.025 s median) and
the cached suffix 0.554 s mean, against 0.330 s for a full forward. The
mechanism is real; the saving needs a long shared prefix and an efficient cache
path, and this layout has neither. What survives regardless is structural:
causal is cacheable, bidirectional is not.

## Provenance

Cells: `jevbench_per_request_causal.json`, `jevbench_per_request_full.json`,
`dev_per_request_causal.json`. They are named so the dashboard's zero-shot globs
(`jevbench_judgement_*`, `jevbench_logit_*`) do not pick them up as judgement
rows; they are not zero-shot per-option cells.

Commands: `model/scripts/eval_jevbench_judgement_per_request.py --model
Qwen/Qwen3-Reranker-4B --snapshot <cache>/22e683669bc0f0bd69640a1354a6d0aebcfeede5
--direction {causal,full} [--dataset dev] --output <file>`. The mask is applied
as a 4D full-attention mask through the model's prepared-mask path, and a
tiny-Qwen3 unit test pins that a token before a later change moves under the
full mask and not under the causal one.

## What this does not say

It does not say the judgement band is capacity- or direction-bound. It says an
untrained mask flip is not available, the per-request layout does not recover
the gap and is not demonstrably worse on dev, and the projection question waits
on the pre-registered rerun. D1/D2 remain held; the next discriminating arm is
not in this note.
