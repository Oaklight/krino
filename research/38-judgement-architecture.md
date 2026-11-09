# Judgement architecture: readout shape, direction and a trained head (#324) — 2026-11-09

Our best trained head scores 42.9% on JevBench 231 while the same backbone,
untrained, scores 64.9% under the judgement readout. This note asks whether the
gap is the pretrained yes/no projection, the causal reading direction, or the
per-option scope, on one backbone: `Qwen/Qwen3-Reranker-4B`, frozen.

The design moved three times before any cell ran, and the final family is
smaller than the first draft. The withdrawn cells are recorded so the numbers
below are not read against the wrong question: a per-option mask flip (E, G) was
replaced by a per-request layout; a softmax-versus-independent pair (H2, H4)
was dropped because softmax over a shared head is monotone and cannot change
the argmax; and the trained one-read code head (D1, D2) is held because its
frozen-backbone form is a random linear on a frozen backbone, the configuration
already measured at 0.351 against a 0.441 raw readout.

## The cells

| cell | layout | direction | readout | trained |
|---|---|---|---|---|
| A | per-option, one sequence per option | causal | vocabulary `log P(yes) − log P(no)` at `Answer:` | no |
| B | per-option, one sequence per option | causal | two-logit linear head at `Answer:` | head only |
| J-R1 | per-request, one sequence with K `Answer:` spans | causal | vocabulary at each span | no |
| J-R2 | per-request, one sequence with K `Answer:` spans | full | vocabulary at each span | no |

All on the same 231 items, same backbone, same 36-layer full-attention config.

## Results

| cell | overall | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|---:|
| A | 150/231 = 64.94% | 47 | 90 | 13 |
| B | 140/231 = 60.61% | 47 | 83 | 10 |
| J-R1 | 137/231 = 59.31% | 46 | 80 | 11 |
| J-R2 | 74/231 = 32.03% | 34 | 33 | 7 |

Exact paired McNemar: B vs A +14 −24, p = 0.143; J-R1 vs A +39 −26, p = 0.136;
B vs J-R1 +33 −30, p = 0.80; J-R1 vs J-R2 +97 −34, p = 3.3e-8.

## The branches

**The projection is not the lever.** B, a trained two-logit head on the same
frozen backbone and the same prompt, does not beat A: 140 against 150, a
4.3 pp deficit against the 9.3 pp floor. Training the readout layer alone buys
nothing.

**The direction is not a free change.** J-R2, the same layout under a full
mask, falls to 32.03%, which is chance (31.8%). It predicts option 0 on 161 of
231 items against 70 gold zeroes, and 54 of those 161 are right: a first-position
prior, not a uniform readout. This is a zero-shot flip on a causally trained
checkpoint, so it says the checkpoint cannot be flipped for free; it is not
evidence that a trained bidirectional arm would fail. J-R2 is killed as a
zero-shot arm.

**The scope question is unresolved on both instruments.** J-R1 against A is
5.6 pp on JevBench (13 items against a 9.3 pp floor, p = 0.136). On the 812-row
development set, which resolves about 4.9 pp, the same pair is 466 against 478,
1.48 pp, +64 −76, p = 0.353. A tighter floor does not resolve it either.

## B's degeneracy

The trained head is degenerate on its own corpus eval split: 17 of 10,911 gold
options reach P(yes) > 0.5 (0.16%), failing #299's positive-rate check. Its
absolute probability is not calibrated, while its ordering still scores
140/231. The number is in `model/experiments/judgement_head/arm_b/config.json`
rather than omitted. A calibrated B would need a bias-only pass or a lower
learning rate; that is a deviation this note does not take.

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
per-source caps, which is the documented 182,262-row training set; 933,230
judgement prompts; abstention rows excluded. The plan is built by
`model/scripts/build_judgement_plan.py`, which calls the readout's own
`candidates`/`render` (#299). The frozen-backbone states are cached at both
causal read positions by `precompute_judgement_states.py`; the head trains on
the `Answer:` position only.

Head: `model/experiments/judgement_head/arm_b/head.pt`, hidden size 2560, one
linear layer, yes bias 2.0, seed 42, lr 3e-3, early stop at epoch 3 of 5,
181,602 train rows, 13,409 eval rows. `head_version: v5` is a proposed label for
the judgement-objective line; the register is append-only and this note does not
claim it until the owner accepts it.

Cells: `jevbench_head_arm_b.json`, `jevbench_per_request_causal.json`,
`jevbench_per_request_full.json`, `dev_per_request_causal.json`. They are named
so the dashboard's zero-shot globs (`jevbench_judgement_*`, `jevbench_logit_*`)
do not pick them up as judgement rows; they are not zero-shot per-option cells.

## What this does not say

It does not say the judgement band is capacity- or direction-bound. It says the
pretrained projection is not the lever, an untrained mask flip is not available,
and the scope difference is smaller than either instrument can resolve. D1/D2
remain held; the next discriminating arm is not in this note.
