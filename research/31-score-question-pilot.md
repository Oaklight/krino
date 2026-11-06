# Score-question generation pilot (#156) — 2026-11-06

Pilot, not a corpus. 2,000 score questions were generated over existing
canonical states and labelled by Jev. This note reports yield, label quality,
and the source mix; it does not authorise scaling.

## Method

- Input: the canonical artifact, 243,366 records with 243,366 distinct ids
  (`sha256 07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53`).
  The pilot file itself stays on the compute host; the script and this note are
  the repository artefacts.
- Candidates: the first record for each normalised state that carries no score
  question. 19,027 distinct score states were excluded (19,033 raw; six pairs
  collapse under normalisation); 65,604 candidate states remained across 33
  source names.
- Selection: deterministic seeded sample, floor 25 per source, cap 350, then
  round-robin interleaving so early stopping does not bias the mix. 3,000
  candidates were ordered; labelling stopped at 2,000 accepted.
- Rubric: one ordinal scale per source family from
  `data/score_rubrics.json`, resolved against the 23 domain templates in
  `data/synthetic_templates.py`; custom scales for banking77 (support urgency),
  the five apigen files (policy risk), openjev (decision clarity), and sst2
  (3-level sentiment).
- Row shape: `label` is the argmax level; `teacher_probs["jev"]` holds the full
  distribution keyed `"0"`..`"k-1"`; `provenance = teacher`. Score arms consume
  the distribution through the soft-target loss. The row is not eligible for a
  hard-label loss, so `label` is for evaluation only.
- Contamination: the #235 rule (id match, normalised state match, or at least
  three shared 8-word shingles) ran over every accepted row before it was
  written.

## Rubric rule

A rubric may be assigned to a source family only when someone states which field
of the state carries the evidence for it. If the evidence is not in the state —
confidence in an answer whose options are absent, a hop count when the passages
being asked about are not the ones present — the rubric is malformed, not hard.
The model can then only return a flat or collapsed distribution, and a flat
number is evidence about the rubric, not about the model.

## Yield

| quantity | value |
|---|---:|
| Jev calls | 2,064 |
| accepted rows | 2,000 |
| rejected rows | 63 |
| rejection reason | 63 degenerate (one level ≥ 0.99, or fewer than two levels > 0.01) |
| candidates per accepted row | 1.032 |
| contamination hits | 0 (id 0, state 0, shingles 0) |
| estimated input tokens | 1,212,959 |
| estimated cost at $0.042/M | $0.051 |
| wall time, 8 workers | 88 s |

Yield is not the constraint. Degenerate labels are 3.0% and none was off-scale;
every accepted row covers all levels and sums to 1.

## Label quality

| statistic | value |
|---|---:|
| label levels | 972 / 168 / 373 / 205 / 282 (level 0..4) |
| argmax differs from expected level | 898 / 2,000 (44.9%) |
| mean absolute difference | 0.58 levels |
| mean entropy | 0.907 nats (max 1.609 for five levels) |
| rows with 5 levels / 3 levels | 1,941 / 59 |

The level distribution is concentrated at level 0 (48.6%). That is not a Jev
failure by itself, but it means a corpus built this way would be badly
imbalanced toward one level unless scaled per source or rebalanced.

The argmax-versus-expected disagreement is the important result. On 45% of rows
the most likely level is not the level a regression on the distribution would
predict. Mean entropy is 56% of the five-level maximum. Jev is returning broad
distributions on these rubrics, so a hard `label` is a weak form of truth even
when it is well defined. The soft target is the usable signal; the hard label
should be described as an evaluation convenience, not gold.

Per source (accepted rows; level-0 share; argmax≠expected; mean entropy):

| source | rows | level 0 | argmax≠expected | entropy | licence |
|---|---:|---:|---:|---:|---|
| hellaswag | 87 | 69% | 99% | 1.343 | unresolved |
| winogrande | 63 | 10% | 49% | 1.267 | permissive |
| piqa | 64 | 73% | 94% | 1.258 | permissive |
| commonsenseqa | 87 | 59% | 80% | 1.247 | permissive |
| longbench_long_choice | 25 | 0% | 96% | 1.233 | permissive |
| contractnli | 87 | 83% | 77% | 1.185 | restrictive |
| 2wiki_long_choice | 25 | 0% | 100% | 1.124 | permissive |
| synthetic | 25 | 76% | 68% | 1.126 | permissive (project) |
| race_short_passage_choice | 25 | 64% | 64% | 0.969 | restrictive |
| hotpotqa_long_choice | 41 | 0% | 100% | 0.967 | permissive |
| logiqa | 87 | 70% | 39% | 0.912 | unresolved |
| openjev_authored144 | 24 | 42% | 58% | 0.901 | permissive |
| tabfact | 86 | 34% | 41% | 0.889 | permissive |
| apigen_tool_choice | 25 | 64% | 68% | 0.881 | restrictive |
| anli | 46 | 20% | 28% | 0.876 | restrictive |
| race | 84 | 65% | 49% | 0.867 | restrictive |
| boolq | 54 | 9% | 33% | 0.855 | permissive |
| mednli | 76 | 33% | 30% | 0.855 | restrictive |
| mnli | 80 | 15% | 21% | 0.848 | permissive |
| mmlu | 82 | 56% | 27% | 0.823 | permissive |
| codesearchnet | 25 | 72% | 12% | 0.821 | unresolved |
| multirc | 86 | 51% | 35% | 0.820 | restrictive |
| banking77 | 78 | 41% | 21% | 0.806 | permissive |
| mmlu_pro_short_reasoning_choice | 83 | 30% | 7% | 0.804 | permissive |
| agnews | 45 | 2% | 13% | 0.733 | restrictive |
| apigen_tool_action | 53 | 8% | 32% | 0.728 | restrictive |
| apigen_tool_next_action | 87 | 92% | 31% | 0.702 | restrictive |
| apigen_tool_argument | 87 | 87% | 45% | 0.684 | restrictive |
| fever | 68 | 38% | 12% | 0.652 | permissive |
| apigen_policy_compliance | 36 | 61% | 19% | 0.639 | restrictive |
| arc | 41 | 73% | 5% | 0.541 | permissive |
| sst2 | 59 | 12% | 8% | 0.511 | unresolved |

Per-source level shares (0..4):

| source | rows | l0 | l1 | l2 | l3 | l4 |
|---|---:|---:|---:|---:|---:|---:|
| 2wiki_long_choice | 25 | 0% | 0% | 0% | 0% | 100% |
| agnews | 45 | 2% | 29% | 51% | 18% | 0% |
| anli | 46 | 20% | 13% | 9% | 22% | 37% |
| apigen_policy_compliance | 36 | 61% | 0% | 36% | 3% | 0% |
| apigen_tool_action | 53 | 8% | 0% | 92% | 0% | 0% |
| apigen_tool_argument | 87 | 87% | 0% | 13% | 0% | 0% |
| apigen_tool_choice | 25 | 60% | 0% | 40% | 0% | 0% |
| apigen_tool_next_action | 87 | 92% | 0% | 6% | 2% | 0% |
| arc | 41 | 71% | 29% | 0% | 0% | 0% |
| banking77 | 78 | 41% | 8% | 35% | 13% | 4% |
| boolq | 54 | 9% | 0% | 11% | 20% | 59% |
| codesearchnet | 25 | 72% | 28% | 0% | 0% | 0% |
| commonsenseqa | 87 | 56% | 0% | 2% | 36% | 6% |
| contractnli | 87 | 82% | 2% | 14% | 2% | 0% |
| fever | 68 | 38% | 16% | 3% | 10% | 32% |
| hellaswag | 87 | 69% | 0% | 0% | 1% | 30% |
| hotpotqa_long_choice | 41 | 0% | 0% | 0% | 0% | 100% |
| logiqa | 87 | 69% | 11% | 18% | 0% | 1% |
| longbench_long_choice | 25 | 0% | 0% | 0% | 8% | 92% |
| mednli | 76 | 33% | 4% | 30% | 25% | 8% |
| mmlu | 82 | 56% | 27% | 17% | 0% | 0% |
| mmlu_pro_short_reasoning_choice | 83 | 30% | 33% | 36% | 1% | 0% |
| mnli | 80 | 14% | 22% | 10% | 26% | 28% |
| multirc | 86 | 50% | 1% | 5% | 16% | 28% |
| openjev_authored144 | 24 | 38% | 0% | 0% | 8% | 54% |
| piqa | 64 | 73% | 0% | 0% | 19% | 8% |
| race | 84 | 65% | 8% | 26% | 0% | 0% |
| race_short_passage_choice | 25 | 60% | 12% | 28% | 0% | 0% |
| sst2 | 59 | 12% | 32% | 56% | 0% | 0% |
| swag | 79 | 99% | 0% | 0% | 0% | 1% |
| synthetic | 25 | 76% | 0% | 4% | 12% | 8% |
| tabfact | 86 | 34% | 1% | 58% | 7% | 0% |
| winogrande | 63 | 10% | 0% | 2% | 67% | 22% |

The multi-hop families collapse at level 4 ("Unanswerable"), not level 0: 2wiki
and hotpotqa are 100% level 4 and longbench 92%. That is the same malformed-
rubric failure as the level-0 collapse, and the next round must fix the
dimension rather than the counts.

Licence totals: 737 restrictive, 258 unresolved, 1,005 permissive. Each row
inherits the licence of the source state; the generated question text adds no
new third-party terms. No exclusion was applied in the pilot, and the owner's
research-use policy in `research/30` covers the restrictive and unresolved
groups.

## What the pilot says

1. **The generating loop works.** Yield is 1.03, the API is fast, and the
   contamination rule comes back clean. Volume is not the risk.
2. **The hard label is weak, and this is bigger than #156.** On 45% of rows the
   argmax and the expected level disagree, and the mean entropy is 0.907 nats
   against a 1.609 ceiling. A hard score label is a lossy summary of the
   teacher's distribution by construction. This is a plausible part of why the
   score head sits at 0.3083 corrected against 0.5991 for choice in the current
   gate: some of that gap is label ambiguity, not model failure. The score-head
   discussion in `research/12-score-calibration-ceiling.md` links here.
3. **Rubric families split cleanly.** The peaked, well-posed scales are sst2,
   arc, fever, banking77, mmlu_pro, mmlu, mednli, and the apigen action/argument
   scales. The flat scales are hellaswag, winogrande, piqa, commonsenseqa,
   longbench, 2wiki, hotpotqa, and contractnli: there the rubric asks for a
   judgment whose evidence is not in the state (for example "confidence in the
   most plausible answer" when the options are not part of the state), and Jev
   spreads probability across the whole scale.
4. **Level 0 dominates.** 48.6% of accepted rows sit at level 0, driven by the
   apigen policy-risk scale (87-92% level 0) and contractnli (83%). Some of that
   is real, but a head trained on a corpus that is half level 0 will learn the
   base rate. The next round caps the level-0 share per source and reports the
   level distribution per source, so the cap's effect is visible.
5. **Recommended next step, if the owner wants to scale:** keep the peaked
   families, replace the flat rubrics with coarser 3-level scales or with a
   dimension the state actually carries, and cap the level-0 share per source.
   Scaling the current mix would add volume without adding usable label
   variation.
