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
- **Running level-0 cap.** A source may add a level-0 row only while the result
  keeps `level0 <= ceil(0.5 * n)` over its accepted rows (`--level0-cap 0.5`).
  The check is running, not a fixed quota, so a source that stops early cannot
  end far above the cap. At small n the ceiling itself is above 50% - 66.7% at
  n=3 and 53.8% at n=13 - so 50% is the share the rule drives towards, not a
  bound it enforces at every n.
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

2,640 calls against 2,000 accepted + 633 rejected leaves seven calls unaccounted.
They are post-target skips: a concurrent chunk returned valid rows after the
2,000-row target was reached and the loop kept the first ones it saw. The
report's `post_target_skips` field is null because it postdates this run; the
counters file records it as `round2.calls_unaccounted = 7`, and the current
script counts the field directly.

The two level-0 statistics from the pilot recur here: 19 of the 2,000 rows carry
a different level under the stored-argmax and the `label` definitions, for the
same rounding-tie reason described in `research/31`, so a source's level-0 share
can move by a point depending on which one a table uses. Every table in this
note uses the `label` field, and the per-source label counters are in
`data/score_generation_counters.json`.

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
rejects the excess. That is visible in the per-source `cap` column of the table
below and in `data/score_generation_counters.json`.

## Licence

Every row inherits the source state's terms. Round-two totals: 738 restrictive,
322 unresolved, 940 permissive. No exclusion was applied; the research-use
policy in `research/30` covers all three.

## Scaling threshold

A family is eligible for a larger run only when both hold at the 95% confidence
bound, not at the point estimate:

1. the **upper** Wilson bound of its disagreement rate (argmax ≠ expected level,
   over accepted rows) is at most **35%**;
2. the **lower** Wilson bound of its accepted rate is at least **30%**, where the
   rate is accepted rows over calls — candidates drawn **and labelled**, not over
   the whole candidate pool. The pool denominator would measure sampling coverage
   and fails every family; the labelled denominator measures whether the family
   produces usable rows at scale, which is the question this rule asks.

The bounds replace both an absolute row floor and the point-estimate comparison.
Twenty-five accepted rows is a property of this pilot's draw, not of the family,
so a row floor would not travel to another draw. At n≈25 the standard error of
such a rate is near 10 points, so a family that lands on the right side of 35% by
one row has shown nothing: codesearchnet is 2 of 13 (15.4%) on the point estimate
and 42% at its upper bound. The rule now says no.

The `cap` and `degenerate` columns are the rejected calls; `accepted + cap +
degenerate` plus any post-target skip is the call count. Both columns are from
`data/score_generation_counters.json`.

| source | accepted | cap | degenerate | calls | disagreement | upper 95% | rate | lower 95% | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2wiki_long_choice | 23 | 1 | 1 | 25 | 21.7% | 41.9% | 92.0% | 75.0% | no |
| agnews | 45 | 0 | 0 | 45 | 6.7% | 17.9% | 100.0% | 92.1% | yes |
| anli | 46 | 0 | 5 | 51 | 28.3% | 42.5% | 90.2% | 79.0% | no |
| apigen_policy_compliance | 27 | 9 | 0 | 36 | 33.3% | 52.2% | 75.0% | 58.9% | no |
| apigen_tool_action | 53 | 0 | 0 | 53 | 30.2% | 43.5% | 100.0% | 93.2% | no |
| apigen_tool_argument | 53 | 97 | 0 | 150 | 47.2% | 60.3% | 35.3% | 28.1% | no |
| apigen_tool_choice | 19 | 6 | 0 | 25 | 68.4% | 84.6% | 76.0% | 56.6% | no |
| apigen_tool_next_action | 29 | 133 | 0 | 163 | 51.7% | 68.6% | 17.8% | 12.7% | no |
| arc | 22 | 19 | 3 | 44 | 4.5% | 21.8% | 50.0% | 35.8% | yes |
| banking77 | 112 | 1 | 12 | 125 | 21.4% | 29.9% | 89.6% | 83.0% | yes |
| boolq | 54 | 0 | 3 | 57 | 24.1% | 36.9% | 94.7% | 85.6% | no |
| codesearchnet | 13 | 12 | 0 | 25 | 15.4% | 42.2% | 52.0% | 33.5% | no |
| commonsenseqa | 13 | 96 | 30 | 139 | 0.0% | 22.8% | 9.4% | 5.5% | no |
| contractnli | 143 | 0 | 19 | 163 | 11.2% | 17.4% | 87.7% | 81.8% | yes |
| fever | 68 | 0 | 5 | 73 | 16.2% | 26.7% | 93.2% | 84.9% | yes |
| hellaswag | 108 | 0 | 0 | 108 | 22.2% | 30.9% | 100.0% | 96.6% | yes |
| hotpotqa_long_choice | 40 | 0 | 1 | 41 | 30.0% | 45.4% | 97.6% | 87.4% | no |
| logiqa | 118 | 43 | 0 | 163 | 36.4% | 45.4% | 72.4% | 65.1% | no |
| longbench_long_choice | 25 | 0 | 0 | 25 | 4.0% | 19.5% | 100.0% | 86.7% | yes |
| mednli | 92 | 0 | 12 | 104 | 27.2% | 37.0% | 88.5% | 80.9% | no |
| mmlu | 70 | 15 | 5 | 90 | 31.4% | 43.0% | 77.8% | 68.2% | no |
| mmlu_pro_short_reasoning_choice | 158 | 0 | 3 | 163 | 9.5% | 15.1% | 96.9% | 93.0% | yes |
| mnli | 80 | 0 | 1 | 81 | 20.0% | 30.0% | 98.8% | 93.3% | yes |
| multirc | 154 | 6 | 1 | 162 | 37.0% | 44.9% | 95.1% | 90.6% | no |
| openjev_authored144 | 22 | 2 | 1 | 25 | 50.0% | 69.3% | 88.0% | 70.0% | no |
| piqa | 53 | 11 | 0 | 64 | 54.7% | 67.3% | 82.8% | 71.8% | no |
| race | 57 | 27 | 0 | 84 | 56.1% | 68.2% | 67.9% | 57.3% | no |
| race_short_passage_choice | 20 | 5 | 0 | 25 | 65.0% | 81.9% | 80.0% | 60.9% | no |
| sst2 | 60 | 0 | 16 | 76 | 8.3% | 18.1% | 78.9% | 68.5% | yes |
| swag | 57 | 22 | 0 | 79 | 3.5% | 11.9% | 72.2% | 61.4% | yes |
| synthetic | 15 | 10 | 0 | 25 | 80.0% | 93.0% | 60.0% | 40.7% | no |
| tabfact | 88 | 0 | 0 | 88 | 37.5% | 47.9% | 100.0% | 95.8% | no |
| winogrande | 63 | 0 | 0 | 63 | 22.2% | 33.9% | 100.0% | 94.3% | yes |

Twelve families clear both bounds: agnews, arc, banking77, contractnli, fever,
hellaswag, longbench_long_choice, mmlu_pro_short_reasoning_choice, mnli, sst2,
swag, winogrande. The point-estimate rule admitted 21. The nine it drops — anli,
apigen_policy_compliance, apigen_tool_action, boolq, codesearchnet,
hotpotqa_long_choice, mednli, mmlu, 2wiki_long_choice — are not shown to be bad;
they are not shown to be good enough at their sample size. Scaling one of them is
a change to this rule, not a reading of it.

On the point estimates the families that miss quality are logiqa 36, multirc 36,
tabfact 38, apigen tool_argument 47, apigen tool_next_action 52, openjev 50,
piqa 55, race 54, race_short 65, synthetic 80, and apigen tool_choice 68. The
families that miss yield are commonsenseqa 9% and apigen tool_next_action 18%.
Every family in those lists also fails the bound test, so the lists describe the
same exclusions the rule makes; the rule adds the ones the point estimates
cannot resolve.

## The constraint: state property, not solution property

The quality failures are not spread evenly. A score question works when the
rubric asks for a property of the state text: sentiment intensity, disclosure
exposure, contract restrictiveness, support urgency, policy risk, question
difficulty, passage relatedness, how constrained a continuation is. It fails when
the rubric asks for a property of the solution or the reasoning: whether an
answer is right (logiqa), how strong the evidence for an answer is (multirc), how
hard a puzzle is to solve (piqa, race), whether a table supports a claim
(tabfact), or what the next agent action should be (apigen action/argument).

The rule is **state property, not solution property**. "Documents rather than
problems" is a shorthand for the common case, not the rule itself: a puzzle can
carry a text property (arc's question difficulty), and a document can ask about
its own solution (a clause question about who wins). What decides is whether the
evidence for the level is in the state.

Arc is the case that shows why the distinction is about the evidence, not the
genre: its rubric asks how difficult the question text is, which is a property of
the text, not of the solution, so it passes at 9% despite being a reasoning
benchmark. That is where the remaining score data can come from: states with a
text-property rubric. It is not worth a week redesigning logiqa or race rubrics.

The synthetic rows in this pilot are 15 synthetic states re-scored with the
default decision-clarity rubric, because the rubric map has no synthetic-domain
entry. Fifteen states scored under a default rubric that has no domain entry is
a result about a missing entry in the rubric map, not a result about synthetic
data. It says nothing about the corpus's existing score rows, which use the
per-domain severity/quality/difficulty templates.

## What round two says

1. **The rubric rule works.** Changing the dimension to one the state carries
   cut argmax/expected disagreement roughly in half for every redesigned family,
   without changing the model or the prompt mechanics.

   The rubric and the cap are confounded in this run, and this note does not
   claim otherwise. The cap rejects a row because of its label, so it lowers the
   disagreement statistic mechanically by removing level-0 rows from the sample;
   the two changes cannot be separated from this run alone. The control that can
   be recomputed from the published per-family rates: the 24 families whose
   rubric did not change moved from 30.0% to 28.8% weighted disagreement (1,442
   to 1,475 accepted rows), while the nine redesigned families moved from 84.3%
   to 19.6% (479 to 525 rows; `swag` has no round-one row).
   `apigen_tool_next_action` lost 133 of its 163 candidates to the cap (one more
   was a post-target skip) and still sits at 52%. The recomputation uses the
   rounded per-family percentages, so it carries about a point of error; a
   two-family run with the redesigned rubric at `--level0-cap 0` would settle it
   outright. It is queued, with one caveat that belongs in the record: at n≈25
   accepted rows the level-0 share has a standard error near 10 points, so a
   single control run reports a point estimate with a wide interval, not a
   verdict. The eligibility rule above, which uses the interval rather than the
   point estimate, is what decides; the run is a check on direction.
2. **The cap works.** Every accepted row satisfies the running ceiling
   `level0 <= ceil(0.5 * n)` for its source; in this run the smallest source is
   large enough that no source ends above 54%, but that maximum is a fact about
   these sample sizes, not a property of the rule. The aggregate level-0 share
   falls from 48.6% to 32.1% at a cost of 515 rejected rows.
3. **Still not training data, and the failures are structural.** The redesigned
   set is much better behaved, but the quality failures are the reasoning
   families and the apigen action scales. That is the state-property constraint,
   not a tuning problem. Scale only families whose rubric asks about the state
   text; treat the rest as rubric evidence.
4. **Recommended next step:** scale the 12 families that clear both bounds,
   keep the cap, and judge the next run against the same two conditions. Residual
   concentration (contractnli 78% at level 2, longbench 96% at level 2) is a
   separate balance issue and should be capped or reweighted before training. The
   three pilot runs so far cost about $0.17.
