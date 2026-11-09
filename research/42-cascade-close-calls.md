# The cascade: a panel-2 read, then a judgement re-read of the close calls (#342) — 2026-11-09

Arm F. Panel 2 over all K; when the top-two probability margin falls below a
threshold, the per-option judgement readout re-scores just those two candidates
and the better score wins. The issue's question is not whether the second
readout is better, it is whether it is better *per forward*.

**It is not. Under the pre-registered objective the dev sweep selects θ = 0:
no item triggers, and the cascade never spends the second pass.** JevBench at
that threshold is the base cell, 162/231. The second pass does move dev accuracy
— 499 to 511 at a 10.8% trigger rate — but that gain costs 21.7% more forwards,
is not significant on its own (exact p = 0.058), and has disappeared by a 22%
trigger rate. The issue's second reading applies: the gain is below the floor
and the hard-tier failure is not recovered.

## Design

The first pass is the shipped panel-2 readout: one forward per item, natural
option order, shipped option order, default attention implementation, one code
token per option, argmax through `choose_option` (first maximum by natural
index). The second pass is the per-option judgement of
`model/evaluation/judgement_prompt.py` — `render(state, instructions, option)`,
`LogitScorer(norm="mean", strategy="description")`, reading log P(" yes") at
`Answer:`, the path `eval_jevbench_judgement.py` uses — applied to the top two
candidates, one forward each. Higher log P(" yes") wins; an exact tie goes to
the smaller natural option index.

The panel-2 natural index maps to the judgement candidate index by identity for
choice and score and by inversion for noul, because panel 2 shows `no`/`yes`
while the judgement shows the item's own answer wording first. For a non-noul
item the run stops unless the mapped candidate text matches the panel-2 text
exactly; on all 578 such items it did.

The margin is `p1 − p2` from a softmax over the K code logits, never a
difference of logits. An item triggers when `margin < θ`. The cost of the rule
is `1 + 2f` forwards, where `f` is the fraction of items that trigger.

θ is chosen on dev 812 by maximising `accuracy(θ) / (1 + 2f(θ))`, ties to the
largest θ, over the complete grid of trigger sets: 0, every distinct dev margin,
1 — 397 points. JevBench is evaluated once, at the selected θ.

**The eligibility gap.** All 391 dev noul items carry no criteria, so
`candidates()` has no option text to judge and returns `None`; those items keep
the first-pass answer and can never trigger. All 74 JevBench noul items carry
criteria and are eligible. The set statistics are not the same on the two
instruments:

| instrument | items | eligible | choice | score | noul | eligible noul |
|---|---:|---:|---:|---:|---:|---:|
| dev 812 | 812 | 421 (51.8%) | 322 | 99 | 391 | 0 |
| JevBench 231 | 231 | 231 (100%) | 139 | 18 | 74 | 74 |

The dev sweep therefore tunes on a set with no noul at all and a 47% share of
score items, while JevBench hands the same rule 32% noul and 7.8% score. This
was written into the pre-registration before the sweep and is not a
reinterpretation after the fact. A polarity readout for the criteria-less noul
items would remove the gap but is a different second readout and was not used.

## The base pass

Before any cascade decision, the reimplementation of panel 2 had to reproduce
the stored cells; it does, on both instruments and in both orders:

| instrument | natural | reversed |
|---|---:|---:|
| JevBench 231 | 162/231 = 70.13% | 156/231 = 67.53% |
| dev 812 | 499/812 = 61.45% | 469/812 = 57.76% |

JevBench natural is the committed cell. JevBench reversed is 156, the #347
natural-index tie rule (the committed display-order cell is 157). Dev natural is
the #326 cell's 499/812; dev reversed is 469, against that cell's 466, which
predates the #347 rule and the 84-symbol alphabet on the 24 banking77 items.

## The dev cost curve

The second pass is a real signal on dev; it just does not pay for itself.

| θ | triggered | f | correct | accuracy | objective |
|---|---:|---:|---:|---:|---:|
| 0.0000 (selected) | 0 | 0.0% | 499 | 61.45% | **0.6145** |
| 0.1863 | 58 | 7.14% | 507 | 62.44% | 0.5463 |
| 0.3023 | 88 | 10.84% | 511 | **62.93%** | 0.5172 |
| 0.4810 | 148 | 18.23% | 507 | 62.44% | 0.4576 |
| 0.5810 | 179 | 22.04% | 505 | 62.19% | 0.4316 |
| 0.9908 | 393 | 48.40% | 485 | 59.73% | 0.3035 |
| 1.0000 | 421 | 51.85% | 485 | 59.73% | 0.2932 |

Maximum accuracy alone is 62.93% (+1.48 pp) at θ = 0.3023, 88 triggers, 10.84%
of all items and 20.9% of eligible ones. Maximising accuracy is exactly the
choice the issue rules out. Accuracy per forward at that θ is 0.5172 against
0.6145 for doing nothing, and no point on the curve has an objective above the
base. The raw-accuracy gain is real but small: it survives up to a 26.6%
trigger rate (216 items) and turns into a loss above it, while the per-forward
objective never reaches the base value again. Of the 88 items triggered at the
accuracy maximum, 23 flip from wrong to right, 11 from right to wrong and 48
keep the answer they had.

The flips are where the design expected them. At θ = 0.3023 the dev cells are
choice 186 → 193 (+7, 55 triggered) and score 55 → 60 (+5, 33 triggered), with
noul unchanged at 258 because no dev noul item is eligible. Twenty-three gained
and 11 lost is exact McNemar p = 0.0576 — not significant even before the cost
is counted.

The margin distributions differ between the instruments. Median tolerance is
0.671 on dev against 0.893 on JevBench; the tenth percentiles are 0.146 and
0.214. JevBench's first pass is more confident, so a dev-chosen θ would trigger
less often on JevBench than it did on dev — one more reason the selection cannot
be transferred by inspection.

## Selection and stability

The dev optimum is θ = 0, and the two declared split halves agree (they both
select 0):

| split | left θ | right θ | agree | left on right | right on left |
|---|---:|---:|---|---:|---:|
| even/odd | 0.0 | 0.0 | yes | 260/406 = 64.04% | 239/406 = 58.87% |
| first/second | 0.0 | 0.0 | yes | 269/406 = 66.26% | 230/406 = 56.65% |

The agreement is degenerate: the halves agree on "never cascade", not on a
positive threshold. The issue's third reading — an unstable threshold — does
not fire, because there is no threshold to be unstable.

## JevBench at the selected threshold

One evaluation, at θ = 0:

| cell | correct | accuracy |
|---|---:|---:|
| overall | 162/231 | 70.13% |
| choice | 102/139 | 73.38% |
| score | 12/18 | 66.67% |
| noul | 48/74 | 64.86% |
| easy | 48/48 | 100.00% |
| original | 62/72 | 86.11% |
| hard | 52/111 | 46.85% |

Gain against the base readout: 0 items, 0 pp, paired McNemar with 0 discordant
items. Trigger fraction 0, mean forwards 1.000.

The hard cell is 46.85%, the issue's 46.8% starting point, **unchanged**. Under
the selected rule it has to be: no item triggers anywhere, so re-reading two
candidates cannot recover a single hard item. Whether the second pass would
recover hard items at a positive θ is not answerable from this run — the
pre-registration allows JevBench one evaluation, at the dev-selected θ, and
that θ has no hard item to re-read.

## Cost

Measured on the same items, bf16, one H100 per pass, batch size 1:

| instrument | first pass | second pass per item | implied mean at θ=0 |
|---|---:|---:|---:|
| dev 812 | 0.206 s | 0.437 s | 0.206 s |
| JevBench 231 | 0.322 s | 0.891 s | 0.322 s |

The second pass is 2.1× the first on dev and 2.8× on JevBench — the judgement
prompt repeats the state and the option, so it is a longer sequence than the
code readout's. At the accuracy-maximising θ = 0.3023 the dev rule would cost
`1 + 2(0.1084) = 1.217` forwards and about 0.253 s per item, 23% more wall time
for 1.5 pp of accuracy that is not significant.

## Reading

**Branch two.** A JevBench gain above the 9.3 pp floor at a mean cost under 1.5
forwards is branch one; nothing above the base appears at any cost, so branch
one is not reached. The threshold is stable across both splits, so branch three
does not fire. What is left is branch two: report the null with the cost curve,
and record that the hard-tier failure is not recovered by re-reading two
candidates — here because the selection never lets the re-read happen.

**The second readout is not worthless; the trigger is.** The judgement does find
signal among the close calls: +7 choice and +5 score on dev at a 10.8% trigger
rate. Every point where it fires is a point where the first pass was confused,
and it is right more often than the first pass there. The failure is in the
rule — a fixed 2-forward charge against an accuracy gain of 1.5 pp cannot win a
per-forward objective, and the issue's own criterion says so before the data
does.

**A positive-θ cascade is not excluded, only unselected.** If the objective were
accuracy alone, or if the second pass were cheaper than the first, θ = 0.3 would
be a live arm. It is also possible that the eligibility gap biases the dev
selection against triggering — the 391 excluded noul items are a large part of
the set, and JevBench would hand the same rule 74 eligible noul items. That
would be a reason to re-run the selection under a different second readout for
noul, not to adopt this one after seeing JevBench.

## What this does not say

It does not say the judgement readout is worse than panel 2 — on the items where
it fires it is better, and the arm is not a comparison of the two readouts but
of the scheduling rule over them. It does not say a positive threshold would
fail on JevBench; that run was not made, deliberately. It does not say the
cascade cannot help a different backbone, margin definition or cost model. It
does say that under the issue's own criterion, on this backbone and these
instruments, the answer is to keep the first pass and not pay for the second.

## Provenance

Pre-registration: #342 comment (`issuecomment-6750455345`), written before the
sweep, with config hash
`62f58bdcfc1665a0d928dc1177a079e3923abfa39f94f01a23f19d37f5f2d5dc` and input
digests dev pairs `c0cf0712…`, JevBench pairs `213a1e4b…`, dev panel 2
`a92031c4…`, JevBench panel 2 `90e9ac8b…`. Branch `feature/342-cascade` from
main at `f1deb25`. Model `Qwen/Qwen3-Reranker-4B` at
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, bf16.

Rows: dev 812 (`sha256 91fe63dd1aca1b15ba496f457f7e0c422b6fc2bb1179c6c8ccb534d9f071d317`)
and JevBench 231 (`sha256 3a36e83106d0d9ee448262ec735924b5861b3077b62417c50041ee8a7df04d80`),
built by `model/scripts/build_set_head_rows.py` and rendered with the 84-symbol
alphabet. Base pass `model/scripts/score_panel2_rows.py`, pairs pass
`model/scripts/score_cascade_pairs.py` (184.0 s on dev, 205.8 s on JevBench),
rule `model/evaluation/cascade.py`, sweep `model/scripts/sweep_cascade.py`.

Result: `results/jevbench/cascade_F_Qwen_Qwen3-Reranker-4B.json`
(`sha256 766cada168e5b9cb850e4f5ed2de983a1023fa56ef2941ab160e5e2e9bf2df7b`).
