# Does a code's lexical content matter? (#354 lexical test) — 2026-11-09

Step 1b left 505 two-letter and 550 three-letter codes that pass the panel-2
format, but the survivors are not neutral: 25 doubles are common English words
and the stable triples are dominated by words and abbreviations. A word carries
meaning into the option list, the opposite failure from the symbol tier, which
had no prior rather than a wrong one.

**The answer to the pre-registered question is no, and the decision it drove is
taken: the alphabet stays at the 52 letters, and K above 52 needs a blocking or
multi-token readout.** No lexical tier is adoptable. But the experiment found
something it was not looking for: **word codes degrade the option list even when
the gold option is labelled by a letter**, by 8 pp on the same items.

## Design

The #354 paired arm, unchanged: K = 58 = 52 letters + 6 tier codes, the six tier
positions always include the gold's position, and each item is scored with the
gold labelled by the tier and with that label exchanged for a letter at the
smallest letter position. Same 200 Banking77 items, same 57 distractors from the
80-intent space, same item seed, plus the 24-item K=80 replica.

Declared before the run: `WORD2` (26 two-letter words), `WORD3` (50 three-letter
words), and the exclusion of `YES` and `NO` from every cell because the noul
option texts are literally `no` and `yes`.

| cell | codes | pool |
|---|---|---|
| `double_listed_word` | AS AT DO ON SO TO | 25 listed doubles, NO excluded |
| `double_unlisted` | BP BR BS CU EU OL | 479 stable doubles outside `WORD2` |
| `triple_listed_word` | ALL AND FOR ONE THE YOU | declared |
| `triple_unlisted` | BLE BMW BOX CSI GNU USA | 515 stable triples outside `WORD3` |

"Unlisted" means outside the declared list, not meaningless: the sample holds
abbreviations (EU, USA, BMW, CSI, GNU) and BOX, an English word outside `WORD3`.
That contamination biases the lexical contrast toward the null.

## Results

Main instrument, 200 items, 1,600 forwards. Same reading rule as #354.

| cell | gold_in_tier | letter_gold | difference | exact p | 95% CI | reading |
|---|---:|---:|---:|---:|---|---|
| `double_listed_word` | 52/200 | 63/200 | −5.5 pp | 0.027 | [−10.0, −1.0] | undecided |
| `double_unlisted` | 50/200 | 79/200 | −14.5 pp | 3.7e-9 | [−19.5, −10.0] | rejected |
| `triple_listed_word` | 20/200 | 76/200 | −28.0 pp | 4.1e-16 | [−34.5, −21.5] | rejected |
| `triple_unlisted` | 29/200 | 79/200 | −25.0 pp | 2.4e-14 | [−31.0, −19.0] | rejected |

Replica, 24 items: `triple_listed_word` rejected (−33.3 pp, p = 7.8e-3); the
other three undecided, which n = 24 can only be.

**The lexical contrast is null.** On the same items, the listed-word doubles
score 52 and the unlisted doubles 50 — 11 items correct in one only against 9 in
the other, exact McNemar p = 0.82. Wordness as the gold's label neither helps
nor hurts. The triple cells agree: 20 against 29 is not a lexical effect either
(−28.0 against −25.0 pp).

## The finding the design did not ask for

The letter-gold baseline is not the same across cells: 63/200 in the
listed-word cell against 79/200 in the unlisted cell and 79/200 in the
triple-unlisted cell, on identical items with identical gold positions and
option order. Paired, that is 18 items lost against 2 gained, exact p = 4.0e-4.

The only thing that differs between those prompts is the six codes labelling the
other tier slots. When they are words (`AS`, `AT`, `DO`, `ON`, `SO`, `TO`), the
model is 8 pp less accurate on an option that is labelled by a plain letter.

So the −5.5 pp of the listed-word cell understates the damage: it is measured
against a baseline the word codes have already degraded. Against the unpoisoned
79/200, the listed tier code is 52/200, −13.5 pp. That cross-cell figure is a
post-hoc observation and is not used for any decision; the pre-registered rule
uses the within-cell difference, and it decides the case on `double_unlisted`
anyway.

The mechanism is a hypothesis: an option list whose labels read as prose
(`AS.`, `ON.`, `TO.`) shifts the model's next-token distribution in a way a list
of single letters does not, and the shift applies to every option, not only to
the one carrying the word.

## Across the three experiments

Put the cells from both paired experiments on one scale, each against its own
letter condition:

| code tier | difference from the letter condition |
|---|---:|
| accented Latin (single character) | −22.5 pp |
| two-letter unlisted | −14.5 pp |
| two-letter listed word | −5.5 pp, against a degraded baseline |
| three-letter | −25 to −28 pp |
| clean symbols (single character) | −36.5 pp |
| shipped symbols | −26.0 pp |
| Hangul (single character) | −38.0 pp |

The ordering is not "single characters beat strings" — two-letter codes beat
accented Latin and Hangul. It is closer to distance from what the model expects
after `Answer:`: letters first, then short strings, then foreign scripts and
symbols. Nothing except the 52 letters is certifiable.

## Reading

**Single letters, blocking above 52.** `double_unlisted` is rejected at
−14.5 pp, so the pre-registered decision is that the alphabet does not extend
and larger K needs the blocking or multi-token scheme. The census figure of 505
doubles is not a usable alphabet.

**Lexical neutrality is not the discriminator.** The hypothesis that motivated
the run — that words are a wrong prior, worse than arbitrary pairs — is not
supported. It is also not refuted for the whole space: only six codes per cell
were tested, and the pre-registration says an adoption covers the tested codes
only. What is supported is narrower: these six common words are not worse than
these six arbitrary pairs as labels.

**Word codes are still a bad idea, for a different reason.** They cost accuracy
on the options they do not label. A serving design that used wordlike codes
would pay for them everywhere, which the paired design understates.

## What this does not say

It does not say all two-letter codes fail: `double_listed_word` is undecided,
and its interval only excludes losses under 10 pp against a degraded baseline.
It does not identify the mechanism behind the baseline shift. It does not test
two-letter codes at other K, and it does not test bare-form codes (the CJK
path). It does say that no tested extension is certifiable and that the code
space census is an encoding ceiling, not a capability.

## Provenance

Pre-registration: #354 comment, posted before the run. Config hash
`d1b85a2eefbc85fbfafbcd880d97d6fd3b6ba575151bd80794de88ae0d73cf4a` (64
characters), printed before the run and identical in the result file's
`_provenance.config_hash`. Branch `feature/code-strings` (stacked on
`feature/code-tiers`, #355). Model `Qwen/Qwen3-Reranker-4B` at
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, bf16, code form `space`, K = 58,
0 skipped pairs on either instrument.

Run: `model/scripts/probe_code_lexical.py` (design shared with
`model/scripts/probe_code_tiers.py` through `model/evaluation/code_tiers.py`).
Result: `results/jevbench/code_lexical_Qwen_Qwen3-Reranker-4B.json`.
