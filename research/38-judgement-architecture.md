# Judgement architecture: readout shape, direction and a trained head (#324) — 2026-11-09

Our best trained head scores 42.9% on JevBench 231 while the same backbone,
untrained, scores 64.9% under the judgement readout. This note asks whether the
gap is the pretrained yes/no projection, the causal reading direction, or the
per-option scope, on one backbone: `Qwen/Qwen3-Reranker-4B`, frozen.

The design moved before any cell ran, and the withdrawn cells are recorded so
the numbers below are not read against the wrong question: a per-option mask
flip (E, G) was replaced by a per-request layout; a softmax-versus-independent
pair (H2, H4) was dropped because softmax over a shared head is monotone and
cannot change the argmax; and the trained one-read code head (D1, D2) is held
because its frozen-backbone form is a random linear on a frozen backbone, the
configuration already measured at 0.351 against a 0.441 raw readout.

## The cells

| cell | layout | direction | readout | trained |
|---|---|---|---|---|
| A | per-option, one sequence per option | causal | vocabulary `log P(yes) − log P(no)` at `Answer:` | no |
| B | per-option, one sequence per option | causal | two-logit linear head at `Answer:` | head only |
| J-R1 | per-request, one sequence with K `Answer:` spans | causal | vocabulary at each span | no |
| J-R2 | per-request, one sequence with K `Answer:` spans | full | vocabulary at each span | no |

All on the same 231 items, same backbone, same 36-layer full-attention config.
B is the rerun described below, not the first run.

## Results

| cell | overall | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|---:|
| A | 150/231 = 64.94% | 47 | 90 | 13 |
| B | 138/231 = 59.74% | 47 | 82 | 9 |
| J-R1 | 137/231 = 59.31% | 46 | 80 | 11 |
| J-R2 | 74/231 = 32.03% | 34 | 33 | 7 |

Exact paired McNemar: B vs A +11 −23, p = 0.0576; J-R1 vs A +39 −26, p = 0.136;
J-R1 vs J-R2 +97 −34, p = 3.3e-8.

## The branches

**The projection is not the lever.** B, a valid two-logit head trained on the
judgement objective over the canonical corpus, does not beat A: 138 against 150,
a 5.2 pp deficit against the 9.3 pp floor, p = 0.058. Training the readout layer
alone does not recover the 22 pp. The arm is not a demonstration that a
projection cannot work — it is one frozen-backbone recipe — but it rules out the
cheapest version of the projection hypothesis.

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
probability and its 140/231 was a within-item ranking artefact.

The diagnosis matters for the rerun. The head initialises with the yes bias at
+2.0 (`TwoWayHead` sets `bias[1]`, and `logits[1]` is yes), so at step zero
P(yes) is about 0.88 on every option; the run cannot have collapsed at
initialisation. The per-epoch count of gold options above P(yes) = 0.5 was 4,
24, 17, 1117, 124: the head was driven down almost immediately, partially
recovered at epoch 4, and fell again. That is drift under the training recipe
(cross-entropy, lr 3e-3, early stopping), not a sign error in the bias.

## B's rerun

Three variants were declared before the run: `plain` (BCE, zero bias, lr 1e-3,
five epochs, no early stop), `calibrated` (plain plus a bias-only pass fitted on
the train split's P(yes) distribution), and `weighted` (BCE with positive weight
`1 / base_rate`). The validity gate runs on the 13,409-row head eval split and
never on JevBench.

| variant | Brier | gold above P(yes)=0.5 | paired gold−non-gold (p) | passes |
|---|---:|---:|---:|---|
| plain | 0.161 | 10,053/10,911 = 92.1% | 0.252 (≈0) | no |
| calibrated | 0.0747 | 4,637/10,911 = 42.5% | 0.235 (≈0) | **yes** |
| weighted | 0.308 | 10,697/10,911 = 98.0% | 0.170 (≈0) | no |

The constant base-rate predictor's Brier is 0.1207, and the positive-rate
threshold is half the empirical gold fraction, 0.0776. Plain and weighted are
well ranked but badly calibrated — non-gold mean P(yes) 0.32 and 0.50 against a
0.155 base rate — so they lose the Brier check. The calibrated variant is the
only one passing all three checks, and it is the head reported above: bias
offset −2.285, best epoch 4 of 5, seed 42.

JevBench was evaluated once, after selection: 138/231 = 59.74%. Selecting on
JevBench would have contaminated the acceptance instrument, so the choice was
made on the eval split first.

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

Corpus: the canonical source list of `model/configs/r2_reranker4b.yaml` with the
per-source caps, the documented 182,262-row training set; 933,230 judgement
prompts; abstention rows excluded. The plan is built by
`model/scripts/build_judgement_plan.py`, which calls the readout's own
`candidates`/`render` (#299). Frozen-backbone states are cached at both causal
read positions by `precompute_judgement_states.py`; B trains on the `Answer:`
position. Head: `model/experiments/judgement_head/arm_b_selected/head.pt`,
hidden size 2560, one linear layer, BCE, seed 42, lr 1e-3, five epochs, 181,602
train rows, 13,409 eval rows. `head_version: v5` is a proposed label for the
judgement-objective line; the register is the owner's and this note does not
claim it.

Cells: `jevbench_head_arm_b_selected.json`, `jevbench_per_request_causal.json`,
`jevbench_per_request_full.json`, `dev_per_request_causal.json`. They are named
so the dashboard's zero-shot globs (`jevbench_judgement_*`, `jevbench_logit_*`)
do not pick them up as judgement rows; they are not zero-shot per-option cells.

## What this does not say

It does not say the judgement band is capacity- or direction-bound. It says the
cheapest projection recipe does not recover the gap, an untrained mask flip is
not available, and the per-request layout is neither better nor demonstrably
worse on dev. D1/D2 remain held; the next discriminating arm is not in this
note.
