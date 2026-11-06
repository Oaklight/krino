# Score-question generation, round two (#156) — 2026-11-06

Second pilot, after the rubric rule in `research/31-score-question-pilot.md`.
Same budget shape: 2,000 accepted rows, one call per candidate, stop and report.
This is still a diagnostic set, not training data.

## Changes from round one

- **Redesigned rubrics.** The families that collapsed in round one got a
  three-level scale over a dimension the state carries:
  hellaswag, piqa, swag, winogrande — "How constrained is the next step by the
  context?"; commonsenseqa — "How much background knowledge is needed to answer
  this question?"; hotpotqa, 2wiki, longbench — "How closely related are the
  passages in this state?"; contractnli — "How restrictive is this contract
  clause?". The peaked families from round one keep their rubrics.
- **Running level-0 cap.** A source may add a level-0 row only while the running
  share stays at or below 50% (`--level0-cap 0.5`). The check is running, not a
  fixed quota, so a source that stops early cannot end far above the cap.
  The first attempt applied the cap to the selection quota instead; commonsenseqa
  still reached 96% level 0 because a source that stops early never spends its
  quota. A cap on the selection quota does not control the accepted rows.
- Everything else is unchanged: same candidate rule, same 3,000-candidate
  stratified selection, same contamination rule, same row shape (`label` =
  argmax, distribution under `teacher_probs.jev`, `provenance = teacher`).

## Yield

| quantity | value |
|---|---:|
| Jev calls | 2,640 |
| accepted rows | 2,000 |
| rejected | 633 (118 degenerate, 515 level-0 cap) |
| candidates per accepted row | 1.32 |
| contamination hits | 0 (id 0, state 0, shingles 0) |
| estimated input tokens | 1,607,928 |
| estimated cost at $0.042/M | $0.068 |
| wall time, 8 workers | 100 s |

The cap costs calls: 515 otherwise valid rows were dropped to hold the per-source
share. That is the intended trade and is reported per source below.

## Result against round one

| statistic | round 1 | round 2 |
|---|---:|---:|
| argmax differs from expected level | 44.9% | **26.6%** |
| mean absolute difference | 0.58 levels | 0.38 levels |
| level-0 share | 48.6% | **32.1%** |
| label distribution (levels 0..4) | 972 / 168 / 373 / 205 / 282 | 641 / 374 / 669 / 147 / 169 |
| mean normalised entropy | 0.564 | 0.557 |
| contamination hits | 0 | 0 |

The aggregate entropy barely moves, but the argmax becomes much more stable: the
distribution still carries uncertainty, and the most likely level now agrees
with the expected level on 73.4% of rows instead of 55.1%. The level-0 base-rate
problem is materially smaller.

## The redesigned families

| source | rows | level shares | argmax≠expected | norm. entropy |
|---|---:|---|---:|---:|
| hellaswag | 108 | 25 / 69 / 6% | 24% | 0.864 |
| piqa | 53 | 51 / 25 / 25% | 55% | 0.850 |
| swag | 57 | 51 / 49 / 0% | 4% | 0.693 |
| winogrande | 63 | 0 / 24 / 76% | 21% | 0.708 |
| commonsenseqa | 13 | 54 / 46 / 0% | 0% | 0.564 |
| hotpotqa | 40 | 35 / 25 / 40% | 30% | 0.668 |
| 2wiki | 23 | 22 / 26 / 52% | 22% | 0.635 |
| longbench | 25 | 0 / 4 / 96% | 4% | 0.580 |
| contractnli | 143 | 11 / 10 / 78% | 10% | 0.514 |

Round one disagreement for the same families was 99%, 94%, 81%, 49%, 80%, 100%,
100%, 96%, and 77%. Every redesigned family improves. The weakest is piqa at
55%; its states are short goals and the "constrained next step" dimension is
still partly inferred rather than stated. longbench and contractnli still
concentrate (96% at "Interlinked", 78% at "Restrictive"), but those are now
readings of the state rather than a collapsed "Unanswerable" or "Minimal" level.

The cap also changes the sample sizes: sources that chose level 0 often
(commonsenseqa, codesearchnet, apigen) contribute fewer rows because the cap
rejects the excess. That is visible in `by_source_levels` in the report.

## Licence

Every row inherits the source state's terms. Round-two totals: 738 restrictive,
322 unresolved, 940 permissive. No exclusion was applied; the research-use
policy in `research/30` covers all three.

## Scaling threshold

A family is eligible for a larger run only when both hold:

1. fewer than **35%** of its rows have argmax ≠ expected level;
2. at least **25 accepted rows** (the per-source selection floor).

Condition 2 is not a formality. The level-0 cap removes most of what some
families produce: commonsenseqa has 0% disagreement on 13 rows, which is not
evidence of quality at volume. Applying both conditions to round two:

| verdict | families |
|---|---|
| eligible (18) | agnews, anli, apigen policy_compliance, apigen tool_action, banking77, boolq, contractnli, fever, hellaswag, hotpotqa, longbench, mednli, mmlu, mmlu_pro, mnli, sst2, swag, winogrande |
| quality fail (≥35%) | apigen tool_argument (47%), apigen tool_next_action (52%), logiqa (36%), multirc (36%), openjev (50%), piqa (55%), race (54%), race_short (65%), synthetic (80%), tabfact (38%) |
| yield fail (<25 rows) | 2wiki (23), arc (22), codesearchnet (13), commonsenseqa (13), apigen tool_choice (19; also 68% disagreement) |

arc and 2wiki clear the quality bar and miss the yield bar; neither is a winner.
longbench is exactly at 25 rows and is eligible only by the letter of the
threshold.

## What round two says

1. **The rubric rule works.** Changing the dimension to one the state carries
   cut argmax/expected disagreement roughly in half for every redesigned family,
   without changing the model or the prompt mechanics.
2. **The cap works.** No source ends above 54% level 0, and the aggregate
   level-0 share falls from 48.6% to 32.1% at a cost of 515 rejected rows.
3. **Still not training data.** The redesigned set is much better behaved, but
   piqa remains at 55% disagreement and longbench/contractnli are concentrated.
   Scale only the families that stay below a stated disagreement threshold, and
   treat the rest as rubric evidence.
4. **Recommended next step:** scale the 18 eligible families, keep the cap, and
   judge the next run against the same two conditions. Residual concentration
   (contractnli 78% at level 2, longbench 96% at level 2) is a separate balance
   issue and should be capped or reweighted before training. The three pilot
   runs so far cost about $0.17.
