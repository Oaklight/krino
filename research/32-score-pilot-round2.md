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
2. an accepted rate of at least **30%**, where the rate is accepted rows over
   candidates drawn **and labelled** — not over the whole candidate pool. The
   pool denominator would measure sampling coverage and fails every family; the
   labelled denominator measures whether the family produces usable rows at
   scale, which is the question this rule asks.

The rate replaces an absolute row floor: 25 accepted rows is a property of this
pilot's draw, not of the family, and it means nothing at a different draw size.
On round two:

| source | accepted | drawn | rate | argmax≠exp | eligible |
|---|---:|---:|---:|---:|---|
| agnews | 45 | 45 | 100% | 7% | yes |
| anli | 46 | 51 | 90% | 28% | yes |
| apigen_policy_compliance | 27 | 36 | 75% | 33% | yes |
| apigen_tool_action | 53 | 53 | 100% | 30% | yes |
| apigen_tool_argument | 53 | 150 | 35% | 47% | no |
| apigen_tool_choice | 19 | 25 | 76% | 68% | no |
| apigen_tool_next_action | 29 | 163 | 18% | 52% | no |
| arc | 22 | 44 | 50% | 9% | yes |
| banking77 | 112 | 125 | 90% | 21% | yes |
| boolq | 54 | 57 | 95% | 24% | yes |
| codesearchnet | 13 | 25 | 52% | 15% | yes |
| commonsenseqa | 13 | 139 | 9% | 0% | no |
| contractnli | 143 | 163 | 88% | 10% | yes |
| fever | 68 | 73 | 93% | 16% | yes |
| hellaswag | 108 | 108 | 100% | 24% | yes |
| hotpotqa_long_choice | 40 | 41 | 98% | 30% | yes |
| logiqa | 118 | 163 | 72% | 36% | no |
| longbench_long_choice | 25 | 25 | 100% | 4% | yes |
| mednli | 92 | 104 | 89% | 27% | yes |
| mmlu | 70 | 90 | 78% | 29% | yes |
| mmlu_pro_short_reasoning_choice | 158 | 163 | 97% | 10% | yes |
| mnli | 80 | 81 | 99% | 20% | yes |
| multirc | 154 | 162 | 95% | 36% | no |
| openjev_authored144 | 22 | 25 | 88% | 50% | no |
| piqa | 53 | 64 | 83% | 55% | no |
| race | 57 | 84 | 68% | 54% | no |
| race_short_passage_choice | 20 | 25 | 80% | 65% | no |
| sst2 | 60 | 76 | 79% | 8% | yes |
| swag | 57 | 79 | 72% | 4% | yes |
| synthetic | 15 | 25 | 60% | 80% | no |
| tabfact | 88 | 88 | 100% | 38% | no |
| winogrande | 63 | 63 | 100% | 21% | yes |
| 2wiki_long_choice | 23 | 25 | 92% | 22% | yes |

Twenty-one families clear both conditions. The rate changes the earlier result:
arc (50%) and 2wiki (92%) now pass, because they were rejected for a small draw,
not for a low hit rate. codesearchnet also passes at 52% on 13 rows. **The
eligibility set changed from 18 to 21 because the metric changed, not because the
data changed.**

The families that fail quality are logiqa 36, multirc 36, tabfact 38, apigen
tool_argument 47, apigen tool_next_action 52, openjev 50, piqa 55, race 54,
race_short 65, synthetic 80, and apigen tool_choice 68. The families that fail
yield are commonsenseqa 9% and apigen tool_next_action 18%.

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
   `apigen_tool_next_action` lost 134 of its 163 candidates to the cap and still
   sits at 52%. The recomputation uses the rounded per-family percentages, so it
   carries about a point of error; a two-family run with the redesigned rubric at
   `--level0-cap 0` would settle it outright.
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
4. **Recommended next step:** scale the 21 families that clear both conditions,
   keep the cap, and judge the next run against the same two conditions. Residual
   concentration (contractnli 78% at level 2, longbench 96% at level 2) is a
   separate balance issue and should be capped or reweighted before training. The
   three pilot runs so far cost about $0.17.
