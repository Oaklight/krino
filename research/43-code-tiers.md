# Which code tiers anchor an answer (#354) — 2026-11-09

Panel 2 can name at most 84 options, and the stored K=80 development items show
the symbol half of that alphabet is dead: 9/15 correct when the gold option
carries a letter code, 0/9 when it carries a symbol (exact Fisher p = 0.0038).
So the working alphabet is 52, not 84, and anything above 52 options is drawn
from a block the model does not answer.

The tokenizer is not the limit: 17,526 characters are one token bare and 604 with
a leading space. This experiment asked whether a *tier* of characters can extend
the alphabet, holding items, option order and the gold's position fixed and
changing only which tier labels the gold.

**No tier can. All four cells are rejected on a 200-item main instrument, the
positive control included, and the positive control repeats on the 24-item
replica.** The usable alphabet stays at 52. Supporting more than 52 options
without dropping items therefore needs a blocking readout or a multi-token code,
not a wider alphabet.

## Design

K = 58 = 52 letters (`A`–`Z`, `a`–`z`) + 6 codes from the tier under test. The
six tier positions always include the gold option's position; the other five are
the last five positions that are not the gold. Each item is scored twice per
tier, paired:

* **gold_in_tier** — the gold carries a code from the tier;
* **letter_gold** — the codes at the gold position and at the smallest position
  that carries a letter are exchanged, so the gold carries a letter.

Option order, K, the gold's position and the multiset of codes are identical
between the two conditions; two labels are exchanged. A tier that cannot hold
its own against a letter on the same item cannot extend the alphabet, whatever
its other merits.

Four tiers, on every item:

| tier | codes |
|---|---|
| `shipped_symbols` | `! # $ % ^ *` — positive control; `@` is replaced by `*` because it is also in the clean-symbol tier |
| `accented_latin` | `Ä È Ü İ ž Ấ` — Latin-script letters, seeded sample of the `other_letter` tier |
| `hangul` | `계 고 김 들 여 전` — seeded sample of the Hangul tier |
| `clean_symbols` | `& < @ [ { ~` — all six symbols that do not glue to the separator |

The samples are a deterministic function of (tier, count, seed 342, tokenizer)
and were written into the pre-registration before the run.

**Instruments.** The main set is 200 real Banking77 customer messages outside the
development split, each with the gold intent plus 57 distractors drawn uniformly
from the same 80-intent space and kept in their original criteria order. This is
the same construction as the real K=80 items with 22 fewer distractors. The
replica is the 24 real K=80 development items, subsampled to 58 the same way.

**The abstention item.** The development set also holds one 79-option
abstention item, `banking77-train-05906-abstain-none_of_these`, whose correct
label has been removed from its options on purpose. The replica selector
requires the gold to be among the options, which excludes it and leaves exactly
the 24 items the pre-registration names. It is not a scorer failure; the item
has no gold option to label.

## Results

Main instrument, 200 items, 1,600 forwards. Two-sided exact McNemar on the
discordant items; 95% percentile bootstrap interval over 10,000 item-level
resamples (seed 342); reading fixed before the run.

| tier | gold_in_tier | letter_gold | difference | exact p | 95% CI | reading |
|---|---:|---:|---:|---:|---|---|
| `shipped_symbols` | 26/200 | 78/200 | −26.0 pp | 6.1e-15 | [−32.5, −20.0] | rejected |
| `accented_latin` | 25/200 | 70/200 | −22.5 pp | 7.1e-11 | [−29.0, −16.0] | rejected |
| `hangul` | 2/200 | 78/200 | −38.0 pp | 2.7e-23 | [−45.0, −31.5] | rejected |
| `clean_symbols` | 6/200 | 79/200 | −36.5 pp | 4.0e-21 | [−43.5, −30.0] | rejected |

Replica, the 24 K=80 development items, 192 forwards:

| tier | gold_in_tier | letter_gold | difference | exact p | 95% CI | reading |
|---|---:|---:|---:|---:|---|---|
| `shipped_symbols` | 2/24 | 10/24 | −33.3 pp | 7.8e-3 | [−54.2, −16.7] | rejected |
| `accented_latin` | 3/24 | 8/24 | −20.8 pp | 0.125 | [−41.7, 0.0] | undecided |
| `hangul` | 0/24 | 10/24 | −41.7 pp | 2.0e-3 | [−62.5, −20.8] | rejected |
| `clean_symbols` | 1/24 | 10/24 | −37.5 pp | 3.9e-3 | [−58.3, −20.8] | rejected |

The one undecided cell is the replica's accented-Latin cell: n = 24 can only
reject, and the same tier is rejected at −22.5 pp with p = 7e-11 on 200 items.
It is a power limitation of the replica, not a conflicting result.

**The method self-check passes.** `shipped_symbols` is rejected on both
instruments, so the positive control reproduces the failure the stored K=80
items show and the other cells are readable. No paired contrast was skipped: the
guard that drops a pair whose two conditions give the gold the same code fired
0 times on either instrument.

## What the 60% prior was worth

The design was motivated by 9/15 = 60% on the K=80 items whose gold carried a
letter. At K=58 on 200 items, the letter-coded cell is 70–79/200 = 35–39.5%
depending on the tier's swap position. The 60% figure came from 15 items, and
its 95% interval runs from 32% to 84%, so 39% is not a contradiction — but the
larger estimate is the one to carry forward: **panel 2 answers a 58-option item
with a letter-coded gold about 39% of the time, against a 1.7% chance level.**

That does not weaken the tier result, which is a paired contrast of 20–38 pp at
p ≤ 1e-10, but it is the number a blocking or multi-token design should assume
for the per-block readout.

## Reading

**No extension is adopted.** Under the pre-registered three-way rule every
candidate tier is `rejected` on the main instrument: accented Latin is the
closest and is still 22.5 pp below the letter condition, with an interval that
excludes any loss under 16 pp. Hangul and clean symbols are further away.

**What an adoption would have covered, and what it does not.** Even had a cell
been adoptable, the evidence would have concerned the six characters tested, not
the whole tier; the pre-registration says so explicitly. No cell was, so the
question does not arise here, but it is the rule any later extension has to
carry.

**The symbol half of the shipped alphabet is dead, and not because of gluing.**
The clean symbols — the six that never merge with the option line's period —
score 6/200 and 1/24 against 79/200 and 10/24 for a letter. Two of the nine
symbol-gold items in the stored K=80 cell used clean symbols and both were
wrong. Gluing is a separate, real defect (26 of the 84 codes never appear in the
prompt as independent tokens) but it is not the mechanism.

**What the mechanism looks like.** A code works as an answer anchor when it is a
token the model has seen in answer position, and the letters are. Accented
Latin is closer to that than Hangul or symbols — the ordering of the three
departures from `A` is the ordering you would predict if distance from the
letter distribution mattered — but close is not enough: 22.5 pp is far outside
the adoption bound.

## What this does not say

It does not say no alphabet extension exists: bare-form codes, multi-character
codes and codes drawn from the answer-position distribution of real text were
not tested, and the tokenizer has 17,526 bare-form characters. It does not say
the K=80 dev items were scored wrong; the 9/24 cell is reproduced. It does not
price a blocking readout, which is the recommended path for K above 52 and is
not measured here. It does say that among the space-form tiers the tokenizer
offers, none behaves like a letter.

## Provenance

Pre-registration: #354, written before the run. Config hash
`63f5181fca89383f8bb2ec5ff57950fabb606086f4e01f042899dab3fefde5f1` (64
characters), printed before the run, identical in the issue text and in the
result file's `_provenance.config_hash` after it. Branch `feature/code-tiers`
from main at `3f8078d`. Model `Qwen/Qwen3-Reranker-4B` at
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, bf16, code form `space`, tie rule
first maximum by natural option index, 58 options per prompt.

Two runs were made and the second is reported. The first completed the 200-item
instrument and crashed on the replica with `ValueError: 'declined_card_payment'
is not in list`, the abstention item above; the selector then required the gold
to be among the options. The second run recomputed the main instrument
identically, cell for cell, which is also the check that the run is
deterministic.

Design and enumeration: `model/evaluation/code_space.py`,
`model/evaluation/code_tiers.py`. Run: `model/scripts/probe_code_tiers.py`.
Result: `results/jevbench/code_tiers_Qwen_Qwen3-Reranker-4B.json`.
