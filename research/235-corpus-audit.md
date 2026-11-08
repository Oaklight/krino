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

The shingle branch needs at least three 8-word shingles. 48 of the 231 JevBench states have fewer than 10 words, so they cannot produce three shingles and that branch cannot fire against them; 14 of those have fewer than 8 words and produce no shingle at all. Those states can still match by id or by exact normalised state.

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

Of the 118,162 rows inside clusters there are 28,651 distinct normalised states, 33.9% of the corpus's 84,632 distinct states. Those states collapse to 22,917 near-duplicate groups, so clustered rows hold 37.1% of the corpus's 61,715 state groups while making up 48.6% of its records. The other 55,981 states are singletons.

The clusters split by where their states come from:

| cluster kind | clusters | rows | states |
|---|---:|---:|---:|
| apigen, across files | 2 | 16,081 | 10,277 |
| synthetic only, one domain | 1,940 | 50,498 | 3,910 |
| cross-source, not apigen | 3,038 | 49,812 | 12,781 |
| one non-synthetic source | 754 | 1,771 | 1,683 |
| total | 5,734 | 118,162 | 28,651 |

The full cluster list, a `cluster_id` for every cluster, and a state-to-cluster map keyed by blake2b-128 of the normalised state are published in `235-near-duplicate-clusters.json` for downstream group-closing work.

The clusters are groups of near-identical states, not duplicate records. Exact duplicate records are still 0. Three patterns explain almost all of the coverage.

- Tool-call (apigen): two clusters cover all 16,081 rows and 10,277 states. The five files reuse one scenario template, so the state text is near-identical while the question and label differ. The effective state diversity of that corpus is far below its row count.
- Typed-decisions: 32 clusters touch it. The four largest have 170, 113, 44 and 40 states and connect it to target-30K, residual-20K and the abstention corpora, because those sources contain the same donor states under other questions.
- Synthetic: 1,940 clusters stay inside one synthetic domain, which is the paraphrase and counterfactual design. Other clusters connect seeded synthetic domains to their donor corpora (target-30K and residual-20K) and to each other.

The near-duplicate pass is separate from the contamination rule. No canonical record matched a JevBench id, normalised state, or three 8-word shingles.

## Deviations from the #235 checklist

Two checklist items are implemented differently from the first wording in the issue. Both deviations are visible in the output, and neither changes the hit count, which is 0.

- Exact dedup by id. Id-only dedup drops 354 RACE questions that share a passage id with another question. The canonical item is the question, so a repeated id with a different question is kept. #242 records the same correction.
- State-level dedup. The checklist asked for dedup by normalised state. 8,498 normalised states appear in more than one source, mostly a donor passage reused under a different question: abstention rows, typed-decisions across task types, and the seeded synthetic domains. Deleting a record because another source reuses its state would delete distinct questions. The audit reports state reuse and does not delete it. Exact duplicate records are 0.

## Supersession

`research/25-corpus-inventory.md` records the contamination status as "exact-match audit only; near-duplicate audit pending #235" and the canonical count as 243,012 ids. #242 corrects the count to 243,366 and the RACE treatment; this note replaces the contamination status. Nothing about the near-duplicate audit is pending after this run.

## Reproducibility

    python -m data.corpus_audit \
        --artifact research/235-corpus-audit.json \
        --clusters-artifact research/235-near-duplicate-clusters.json \
        --workers <n> --progress

The run took 407 s wall on the compute host, 280.8 s of it in the near-duplicate pass, with a peak working set of 1.4 GB. The corpus files are not in the repository; the audit reads them under `data/benchmarks/`. `--workers 1` gives the same output and is slower.

## CI

`.github/workflows/corpus-audit.yml` restores the pinned data lock, runs the rule tests, then:

    python -m data.corpus_audit --check --skip-near-dup \
        --typed-decisions-from-hf --allow-missing --min-records 190000 \
        --artifact corpus-audit.json

The job exits 1 on any contamination hit and 2 when a canonical source is missing or fewer than 190,000 records are audited. `--allow-missing` exists because two host-local corpora (`label_target_30k_jev.jsonl`, `residual_train_20000_jev.jsonl`) are not in the pinned lock; the floor is what fails the job if a source that should be there is empty or absent from a directory glob. The gate covers 195,404 of 243,366 records (80.3%) and 195,050 of 243,012 passage ids. The 243,366-record result above comes from the complete corpus.

The near-duplicate pass is a one-off audit: the pure-Python MinHash over 84,632 states is too slow for a pull-request gate.
