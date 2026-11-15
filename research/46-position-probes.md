# Where the long-list tail comes from (#357) — 2026-11-10

Panel 2 at K = 52 loses the last position bucket entirely on Banking77 (0.0%)
and partially on CLINC150 (59.1%). Four probes hold the items fixed and change
one thing about the display. **The two datasets do not have the same mechanism.
On Banking77 the collapse follows the gold's rank in the class order, not where
it is displayed; on CLINC150 it follows the display position.** Neither is
explained by the number of candidates alone, and the issue's "the model ignores
the tail" reading is true of CLINC but not of Banking77.

## Gates

Both K = 52 natural cells reproduce the recorded cells exactly: Banking77
72/200, CLINC150 146/200. Config hashes
`9b4fb09edeb9e7d287399c72f0df9ddb1b6b875c60bffd7f1fa3f2d1b65e01a5` (Banking77)
and `5e8ed18d7c9d8464eaacbf3fcdd668ff5da407f838d41128b650412aed3b8e99`
(CLINC150), printed before the run.

## P1 — natural position against display position

The K = 52 option set is held fixed and only its display order is permuted, four
seeded permutations per item. Accuracy by the gold's natural rank (its rank in
the selected options' class order) and by its display rank:

| dataset | cell | 0–20% | 20–40% | 40–60% | 60–80% | 80–100% | spread |
|---|---|---:|---:|---:|---:|---:|---:|
| Banking77 | natural, natural rank | 64.6% | 69.4% | 33.3% | 2.1% | 0.0% | 64.6 pp |
| Banking77 | permuted, natural rank | 58.3% | 64.6% | 28.3% | 2.7% | 0.0% | 64.6 pp |
| Banking77 | permuted, display rank | 43.6% | 35.0% | 34.1% | 23.4% | 27.3% | 20.2 pp |
| CLINC150 | natural, natural rank | 82.6% | 86.5% | 68.8% | 68.3% | 59.1% | 13.8 pp |
| CLINC150 | permuted, natural rank | 73.9% | 70.3% | 64.1% | 76.8% | 63.1% | 13.8 pp |
| CLINC150 | permuted, display rank | 86.2% | 84.8% | 69.6% | 53.2% | 52.0% | 34.2 pp |

The pre-registered rule (positional if display spread ≥ 15 pp and natural spread
< 10 pp, item-driven if the reverse) fires **mixed on both datasets**, because
each has a real effect on both axes. The informative fact is the ordering:

- **Banking77: the natural-rank effect is 6× the display effect** (64.6 against
  20.2 pp). The recorded collapse reproduces under permutation almost exactly
  (0.0% in the last natural bucket either way), so it is not produced by where
  the options are shown.
- **CLINC150: the display effect is 2.5× the natural effect** (34.2 against
  13.8 pp). Under permutation the profile turns monotone and steep
  (86.2% to 52.0%), while the natural profile flattens.

Within-item tests, which hold the item fixed: display early (buckets 0–1)
against late (3–4) gives 39.5% against 28.7% on Banking77 (+10.8 pp, item sign
test 29 wins and 8 losses, p = 7.5e-4) and 85.5% against 53.0% on CLINC150
(+32.5 pp, 68 against 5, p = 3.4e-15). Natural first against last bucket gives
64.6% against 0.0% on Banking77 (Fisher p = 2.1e-8) and 82.6% against 59.1% on
CLINC150 (p = 0.020).

So a display-position effect exists on both datasets, but on Banking77 it is
small next to the class-rank effect, and on CLINC150 it is the larger one. The
natural rank is the gold's rank in the class order with the option set held
fixed; it is not a display quantity.

## P2 — the K ladder

Nested draws: K = 7/14/28 keep the gold plus the first k−1 distractors of the
K = 52 draw, so only the list length changes. The K = 52 cell is the recorded
subsample. "Dead" is the pre-registered rule: last bucket below half the first.

| dataset | K | natural | permuted | natural last/first | permuted last/first | dead (natural / permuted) |
|---|---:|---:|---:|---:|---:|---|
| Banking77 | 7 | 122/200 | 247/400 | 57.1% / 100.0% | 56.2% / 55.4% | no / no |
| Banking77 | 14 | 102/200 | 201/400 | 39.0% / 90.0% | 50.0% / 55.8% | **yes** / no |
| Banking77 | 28 | 79/200 | 152/400 | 8.4% / 87.0% | 37.3% / 42.9% | **yes** / no |
| Banking77 | 52 | 72/200 | 261/800 | 0.0% / 64.6% | 27.3% / 43.6% | **yes** / no |
| CLINC150 | 7 | 195/200 | 387/400 | 97.2% / 100.0% | 98.1% / 96.6% | no / no |
| CLINC150 | 14 | 183/200 | 370/400 | 91.3% / 100.0% | 96.8% / 88.0% | no / no |
| CLINC150 | 28 | 152/200 | 337/400 | 67.3% / 90.9% | 82.7% / 85.7% | no / no |
| CLINC150 | 52 | 146/200 | 559/800 | 59.1% / 82.6% | 52.0% / 86.2% | no / no |

Two things fall out. The Banking77 dead tail appears in the natural order from
K = 14 onward and **never under permutation at any K**: the class-rank effect
grows with the list length because a longer list gives the rank more room to
matter, but it is not a display tail. On CLINC150 there is no dead tail by the
rule at any K; the display gradient appears only at K = 52 (86.2% to 52.0%) and
is absent at K = 28 (85.7% to 82.7%). Total accuracy falls with K on the
position-controlled cells on both datasets (Banking77 61.8%, 50.3%, 38.0%,
32.6%; CLINC150 96.8%, 92.5%, 84.3%, 69.9%): list length itself costs accuracy
even when position is neutralised.

## P3 — alphabet arrangement at K = 80

The gold carries a letter at the same rank among the letters in both cells;
only the letter block's absolute position and the surrounding codes change.

| dataset | letters early | letters late | difference | 95% CI | exact p | reading |
|---|---:|---:|---:|---|---:|---|
| Banking77 | 67/200 | 68/200 | +0.5 pp | [−4.5, +5.5] | 1.00 | code-driven (equivalence) |
| CLINC150 | 141/200 | 137/200 | −2.0 pp | [−7.5, +3.5] | 0.61 | code-driven (equivalence) |

Moving the whole letter block to the end of the list changes nothing on either
dataset. The K = 80 collapse is therefore **not** the symbol tier's position and
not the position of the letter block; the symbol codes are dead wherever they
sit, and the gold's letter is read equally well at the front and at the back.

## P4 — distractor text at K = 52

The gold text, state, instructions and gold position are fixed; the 51
distractor texts are drawn from the same dataset (control) or from the other
dataset.

| dataset | own | cross | difference | 95% CI | exact p | last bucket, own / cross |
|---|---:|---:|---:|---|---:|---|
| Banking77 | 71/200 | 100/200 | +14.5 pp | [+9.5, +20.0] | 1.3e-7 | 0/24 / 0/24 |
| CLINC150 | 139/200 | 152/200 | +6.5 pp | [+1.0, +12.5] | 0.041 | 47.7% / 63.6% |

On Banking77 the cross-domain distractors raise overall accuracy by 14.5 pp but
leave the last bucket at 0/24: the tail items are not rescued by changing the
distractors, which fits a gold/class property rather than a distracting
context. On CLINC150 the overall gain is 6.5 pp, below the 10 pp threshold, but
the last bucket rises 15.9 pp (21/44 to 28/44; paired McNemar 8 cross-only
against 1 own-only, p = 0.039). Part of CLINC's display tail is therefore about
in-domain distractors being confusable, not about position alone.

The own-swap control sits close to the natural cell (Banking77 71 against 72,
CLINC150 139 against 146), so the replacement itself is not the effect.

## What the dead tail is made of

The natural K=52 cell grouped by natural bucket, with the gold's class index in
Banking77's label order and the label's character length:

| bucket | correct | distinct classes | class index | mean label chars |
|---|---:|---:|---|---:|
| 0 | 31/48 | 20 | 0–20 | 23.5 |
| 1 | 25/36 | 18 | 13–39 | 22.3 |
| 2 | 15/45 | 21 | 30–53 | 17.9 |
| 3 | 1/47 | 18 | 43–64 | 18.1 |
| 4 | 0/24 | 12 | 61–76 | 20.6 |

Buckets 3 and 4 together hold 71 items over **28 distinct classes, all at class
indices 43–76** — the alphabetical suffix of the label space (`pending_*`,
`top_up_*`, `transaction_*`, `transfer_*`, `verify_*`, `virtual_card_*`,
`visa_or_mastercard`, and their neighbours). None of those 28 classes appears in
the head bucket, whose 20 classes sit at indices 0–20.

**The labels are not longer.** The dead classes average 18.1 and 20.6
characters against 23.5 and 22.3 in the head buckets, so the "longer or vaguer
description" hypothesis is not supported by label length.

Three of the four candidates are now excluded by measurement — display position
(P1), the letter block's position (P3), distractor text (P4) — and length does
not separate the tail. What remains is the gold class itself. That is a
description, not a mechanism: the next steps are a per-class error analysis of
these 28 classes and a gold-text replacement probe, not another display probe.

### Where the tail's wrong answers land

The stored natural K=52 rows give the predicted display position, and the
option list is reconstructible from the recorded draw, so the predicted class
of every wrong tail answer is known without another run. Of the 71 tail items,
70 are wrong. Their predictions land:

| zone | observed | uniform null |
|---|---:|---:|
| tail (class index 43–76) | 42 (60.0%) | 31.7 (45.2%) |
| middle (21–42) | 12 (17.1%) | 19.9 (28.5%) |
| head (0–20) | 16 (22.9%) | 18.4 (26.3%) |

The null is the zone composition of each item's own 51 distractors, so it
already accounts for the fact that tail items draw mostly tail distractors.
Two things stand out. **The wrong answers are enriched within five class
indices of the gold: 34/70 = 48.6% against a 12.9% null, a 3.8× enrichment.**
And the head is *not* over-selected: 22.9% observed against 26.3% expected, so
there is no evidence that the model prefers the head labels.

The most-selected wrong classes are `extra_charge_on_statement` (5),
`automatic_top_up` (4), `receiving_money` (4),
`wrong_exchange_rate_for_cash_withdrawal` (4), `transaction_charged_twice` (4),
`Refund_not_showing_up` (3), `transfer_fee_charged` (3),
`top_up_by_cash_or_cheque` (3). They are the same money-movement vocabulary as
the tail, and some carry middle or head indices, so the class-order suffix is a
proxy for a semantic cluster rather than the cause itself.

**This is the dataset-property branch.** The 28-class suffix is a mutually
confusable cluster: the readout's errors stay inside it and next to the gold,
not at the head. The Banking77 aggregates (36.0% at K=52, 33.5% at K=80) are
therefore depressed by the class table's confusability and are not a measure of
panel 2's capability at those option counts.

The only cells not depressed by a label-space pathology are on the CLINC150
distribution: 73.0% at K=52 and 62.5% at K=80. Even those are panel 2 on that
**one distribution**, not panel 2's capability. A third intent space would be
needed before either number could be quoted as a capability figure, and none of
the blocking cells has been run outside CLINC150.

## What this says, and what it does not

- **The mechanism differs by dataset.** Banking77's tail is the gold's rank in
  the class order; CLINC150's is the display position, moderated by distractor
  text. A statement about "panel 2 and long lists" has to name the dataset.
- **The K = 80 alphabet collapse is not positional.** Moving the letter block
  changes nothing (P3), so the collapse is the symbol tier plus whatever makes
  the late-class golds hard, not the place where the symbols sit.
- **The natural-rank effect is not yet explained.** With the option set fixed,
  accuracy follows the gold's class rank. The dead region is the alphabetical
  suffix of the label space (28 classes at indices 43–76), and their labels are
  shorter, not longer, so label length does not separate them. The rank remains
  confounded with the gold class's text and with the distractor pool the draw
  can offer, and this design does not separate them. Calling it "the item" is a
  description, not a cause.
- **The cross-domain salience confound is named, not removed.** In P4 the
  in-domain gold among out-of-domain distractors is the odd one out; the
  own-swap controls for text replacement but not for that salience.
- **Nested ladder.** K = 7/14/28 are prefixes of the K = 52 draw, so the ladder
  changes length only; the K = 52 draw itself is the recorded subsample,
  verified to match `subsample` on 200/200 synthetic items.

## Scope

CLINC150 in-scope and Banking77, 200 items each, the 4B backbone at revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, one code form, one template. The
blocking frontier's b = 26 versus b = 52 difference is consistent with the
CLINC display effect — a 26-option block escapes it, a 52-option block does not
— but that comparison is on CLINC only and is not evidence about Banking77.
