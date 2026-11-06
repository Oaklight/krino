# Corpus dedup and JevBench contamination audit — 2026-11-06

Issue #235. The rule was stated on the issue before the run and is not tuned here. The canonical source list and id-dedup order come from #234 / PR #241.

## Rule

Target: the 231 public JevBench items (48 easy / 111 hard / 72 original). Structured JevBench states are rendered with `render_jevbench_state`, the same text the model reads.

Normalisation: Unicode NFC, casefold, every run of non-word characters replaced by one space, strip.

A record is contaminated when any of these holds:

1. Its `id` equals a JevBench `id`.
2. Its normalised `state` equals a JevBench normalised `state`.
3. Its `state` shares at least three distinct 8-word shingles with one JevBench `state`.

The comparison text is the state. Instructions and criteria are task scaffolding shared across sources; including them would flag records that never saw the JevBench content.

Near-duplicate threshold: MinHash over character 5-grams of the normalised state, 128 permutations, 32 bands, confirmed at a MinHash Jaccard estimate of at least 0.7. The threshold is the one `data/synthetic_dedup` already uses for state text; it was fixed before the counts were seen.

## Coverage

| item | count |
|---|---:|
| raw rows loaded | 246,305 |
| canonical records | 243,366 |
| passage ids | 243,012 |
| dropped duplicate rows | 2,939 |
| same-source id reuse kept | 354 |
| distinct normalised states | 84,632 |

The canonical order is target-30K, residual-20K, typed-decisions, synthetic, then the remaining corpora. The canonical item is the question; a passage id is the split group. The 2,939 dropped rows are exact duplicates of a higher-priority record: 2,274 synthetic rows (1,141 against residual, 1,133 against target) and 665 typed-decisions rows (415 against target, 250 against residual). The 354 kept rows are RACE questions that share a passage id.

## Contamination

0 of 243,366 canonical records (243,012 passage ids). There is no hit list because there are no hits.

## Exact duplicates

No audited source contains an exact duplicate record. Counts by source:

| source | raw rows | audited rows | distinct ids | distinct states | exact duplicate rows |
|---|---:|---:|---:|---:|---:|
| label_target_30k_jev | 30,000 | 30,000 | 30,000 | 29,775 | 0 |
| residual_train_20000_jev | 20,000 | 20,000 | 20,000 | 19,895 | 0 |
| typed_decisions | 8,000 | 7,335 | 7,335 | 1,600 | 0 |
| synthetic (23 domains) | 154,091 | 151,817 | 151,817 | 13,191 | 0 |
| apigen (5 files) | 16,081 | 16,081 | 16,081 | 13,477 | 0 |
| abstention_train_20k | 3,000 | 3,000 | 3,000 | 2,820 | 0 |
| abstention_train_30k | 3,000 | 3,000 | 3,000 | 2,879 | 0 |
| mmlu_pro_short_reasoning_choice | 9,965 | 9,965 | 9,965 | 9,714 | 0 |
| long-context (4 files) | 1,670 | 1,670 | 1,670 | 1,481 | 0 |
| race_short_passage_choice | 498 | 498 | 144 | 144 | 0 |

`race_short_passage_choice` has 498 rows over 144 ids and 144 states. All 498 full records are distinct: each passage id carries 3-5 different questions. Exact dedup by id would drop 354 distinct questions. The audit keeps them and treats a repeated id with a different question as a distinct record. The canonical item is the question, so the canonical count is 243,366 records; 243,012 is the passage-level id count, and a passage id is the split group.

8,498 normalised states appear in more than one source. The largest exact overlaps are abstention_30k with target-30K (2,001 states), abstention_20k with residual-20K (2,000), apigen_tool_argument with apigen_tool_choice (1,676), typed-decisions with target-30K (1,456), and the seeded synthetic domains with their donor corpora. These are state reuse across questions, not duplicate records: exact duplicate records are 0.

## Near-duplicates

5,734 clusters at the stated threshold. They cover 28,651 of 84,632 distinct states and 118,162 of 243,366 canonical records. 3,040 clusters span more than one source. The largest cluster has 6,307 states.

The full cluster list, a `cluster_id` for every cluster, and a state-to-cluster map are published in `235-near-duplicate-clusters.json` for downstream group-closing work.

The clusters are groups of near-identical states, not duplicate records. Exact duplicate records are still 0. Three patterns explain almost all of the coverage.

- Tool-call (apigen): two clusters cover all 16,081 rows and 10,277 states. The five files reuse one scenario template, so the state text is near-identical while the question and label differ. The effective state diversity of that corpus is far below its row count.
- Typed-decisions: one workflow state carries several questions. Clusters of 170, 113, 44 and 40 states cover the shared context across task types. They overlap target-30K, residual-20K and abstention because those corpora contain the same donor states.
- Synthetic: 1,918 clusters stay inside one synthetic family, which is the paraphrase and counterfactual design. Other clusters connect seeded synthetic domains to their donor corpora (target-30K and residual-20K) and to each other.

The near-duplicate pass is separate from the contamination rule. No canonical record matched a JevBench id, normalised state, or three 8-word shingles.

## CI

`.github/workflows/corpus-audit.yml` restores the pinned data lock, runs the rule tests, then `python -m data.corpus_audit --check --skip-near-dup`. The job fails on any contamination hit. The near-duplicate pass is a one-off audit: the pure-Python MinHash over 84,632 states is too slow for a pull-request gate.

The workflow can only restore files that are in the pinned data lock. The two host-local corpora (`label_target_30k_jev.jsonl`, `residual_train_20000_jev.jsonl`) are not in the lock, so CI reports them as missing and audits the remaining canonical sources. The full 243,366-record result above comes from the complete corpus.
