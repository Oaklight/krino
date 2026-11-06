# Canonical corpus data card — 2026-11-06

One page for the training arm. Definitions and per-file provenance are in
`research/25-corpus-inventory.md`; the numbers here come from
`scripts/corpus_datacard.py`, which assembles the primary corpora in the
recorded precedence order and deduplicates by id.

## Size

**243,012 distinct ids** after id-level dedup. No near-duplicate dedup has run
yet (see Contamination and dedup).

| corpus | ids kept |
|---|---:|
| synthetic | 151,817 |
| target 30K | 30,000 |
| residual 20K | 20,000 |
| tool-call (`apigen`) | 16,081 |
| mmlu-pro | 9,965 |
| typed-decisions | 7,335 |
| abstention | 6,000 |
| long-context | 1,670 |
| race-short | 144 |

The table shows ids kept after dedup. Before dedup the components sum to 245,951:
2,274 synthetic ids sampled into the 20K/30K keep their sampled copy, and 665
typed-decisions ids already in the 20K/30K are dropped from the full
typed-decisions copy. `race_short_passage_choice` collapses 498 rows to 144 ids.

## Provenance tiers

| tier | ids | share |
|---|---:|---:|
| `gold` | 220,931 | 90.9% |
| `execution` | 22,081 | 9.1% |
| `teacher` | 0 | 0% |

`gold` does not mean human-verified. It breaks down as:

- 154,091 model-authored synthetic labels (151,817 in the synthetic corpus plus
  2,274 sampled into the 20K/30K);
- 58,803 upstream dataset labels;
- 8,037 community-authored labels (`typed_decisions`, `openjev_authored144`).

`execution` is 16,081 apigen trajectory labels plus 6,000 constructed
abstention labels. No hard label is a Jev prediction; `teacher_probs` is always
an additional soft target.

**Hard-label usability:** 237,012 records are eligible for option/type CE
(CE/BCE/ordinal). 6,000 abstention records drive the abstain/answerable target
and must not drive option CE. Zero records fail eligibility after label
normalisation; 124/70 noul labels and 85/108 score labels in the two sampled
corpora are strings that require coercion.

## Type mix

| type | ids | share |
|---|---:|---:|
| noul | 113,594 | 46.7% |
| choice | 99,903 | 41.1% |
| score | 29,515 | 12.1% |

## Choice K

`K` is the number of criteria keys. Counts cover the 99,903 choice records.

| statistic | value |
|---|---:|
| median | 4 |
| p90 | 10 |
| max | 80 |

| K bucket | ids |
|---|---:|
| 2 | 2,967 |
| 3-4 | 52,104 |
| 5-8 | 28,743 |
| 9-16 | 13,110 |
| 65-255 | 2,979 |

The 65-255 bucket is entirely the tool-call files at K=79/80 (2,979 records).
mmlu-pro contributes 9,965 records at K=10 and falls in the 9-16 bucket. No
canonical file exercises K > 80, so the documented K=255 head is not covered by
training data.

Score items have 3-6 levels: 146 at 3, 9,819 at 4, 18,120 at 5, 1,430 at 6. There
is no continuous-score regression data.

## State length

Characters of `state`, all 243,012 records.

| statistic | chars |
|---|---:|
| p10 | 89 |
| median | 646 |
| p90 | 5,891 |
| p99 | 10,228 |
| max | 54,765 |

| bucket | ids |
|---|---:|
| <512 | 105,240 |
| 512-2K | 93,398 |
| 2K-8K | 40,743 |
| 8K-32K | 3,584 |
| >=32K | 47 |

Median length by type: choice 574, noul 695, score 626. The long tail is the
long-context and long-document sources; 47 records exceed 32K characters.

## Contamination and dedup

**Exact-match audit only; near-duplicate audit pending #235.** The exact check
found 0 hits against the 231 public items in the 97,355 rows it covered
(`random 54,214 + label_target 30k/38.9k + mmlu_pro + race`). That result does not
cover the full 243,012-record canonical union, and the MinHash 8-word-shingle
check has not run. This card is not a clean bill of health.

Id-level dedup is applied. State-level and near-duplicate dedup is #235's deliverable;
until it lands, paraphrase and near-duplicate pairs may cross the canonical set.

## Known gaps

The corpus covers classification/choice, boolean judgment (noul), and ordinal
scoring, plus tool selection, long-context choice, and constructed abstention.
It does not contain first-class training data for:

- ranking/reranking and retrieval pairs (no pairwise margin data);
- structured extraction and per-field verification;
- multi-question states sharing one state (batch equivalence);
- hierarchy / taxonomy selection;
- self-consistency and multi-sample agreement targets;
- continuous-score regression (scores are 3-6 levels);
- large-K selection above K=80 (the head supports 255).

Vision and image inputs are out of scope for the current arm. Date and count
arithmetic are handled in code, not in the corpus.

## Reproduction

```bash
python scripts/corpus_datacard.py \
  --root data/benchmarks --repo-root . --json /tmp/datacard.json
```

If `typed_decisions.jsonl` sits outside `--root`, pass
`--typed-decisions <path>`. The script is single-process, stdlib-only apart from
`data.provenance` and `data.synthetic.family_to_typed_questions`. It exits if a
primary file is missing, so a partial mirror cannot silently produce a smaller
card.
