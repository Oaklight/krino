# Ability labels and coverage — 2026-11-06

Issue #247. The taxonomy is fixed in `data/abilities.py`; the labeler is `scripts/label_abilities.py`; the committed coverage report is `247-ability-coverage.json`.

Input: `krino-canonical-243366.jsonl`, SHA-256 `07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53`, 243,366 records and 243,366 distinct ids (the artifact built after the RACE id correction). That file is the output of the canonical builder in #249; this branch is based on main, so the reproduction below uses #249's builder once it lands. The labels are published as a sidecar beside the artifact, `krino-canonical-243366.ability.jsonl`, SHA-256 `5933f899a3660d529bdad39d7226a6bad1d7605e0fabfa872db7a8bc1c606670`, one row per record with `id`, `ability`, `label_method`, `ability_rule`. The sidecar id set equals the artifact id set; the labeler rejects duplicate ids.

## Taxonomy

The ten names and definitions are the decision shapes from the TypeSafe use-case map, the list #207 is organised around. They are not redefined anywhere else.

| ability | definition |
|---|---|
| Classification | One known category should win. |
| Detection | A probability that one property is present. |
| Scoring | The answer belongs on an ordered rubric. |
| Routing | A category selects the next code path. |
| Search | Find items that match a natural-language query. |
| Retrieval | Fetch the most relevant context or records for a workflow. |
| Ranking | Order items by semantic relevance or quality. |
| Verification | Check an artifact for specific failure modes. |
| ML Feature Extraction | Extract probabilistic signals for a downstream classical model. |
| Structured Data Extraction | Recover known fields from unstructured input. |

`label_method` is one of `source_rule`, `heuristic` or `model`. `ability_rule` names the exact rule that fired and is carried in the sidecar for audit.

## How a record is labelled

The ability is the decision the question asks for, not the dataset. Rules run in this order:

1. A `score` question is Scoring: the answer is an ordered rubric by construction (29,515 records).
2. An exact source table: 27 upstream source names, the five apigen files, and the twenty typed-decisions instruction families. These are `source_rule`.
3. An exact table for the dominant generated template in each synthetic domain. These are `source_rule`.
4. Ordered heuristic patterns for free-form instructions: value extraction, routing, evidence sufficiency, verification, detection. These are `heuristic`.
5. A type default: Classification for a choice or a generic yes/no, Scoring for a score item. These are `heuristic`.

An unmapped source fails the run; the labeler does not default a new source.

The patterns were narrowed after a first pass: bare "retrieve", "threat", "which step" and "is there" were labelling incidental scenario vocabulary, so they were removed or tied to the question's decision. Evidence-sufficiency questions ("is there evidence", "is the information sufficient to determine") are Verification, not Detection.

No record is model-labelled. `model_share` is 0.0: the residue after rules and heuristics is empty, so no ability stands on a model label alone.

## Coverage

| ability | noul | choice | score | total | share | source_rule | heuristic | after exclude_restrictive | after exclude_restrictive_and_unresolved | listwise gradient | flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Classification | 81,125 | 69,446 | 0 | 150,571 | 61.9% | 45,007 | 105,564 | 131,700 | 110,675 | 69,446 | |
| Detection | 1,782 | 2,868 | 0 | 4,650 | 1.9% | 3,592 | 1,058 | 4,554 | 3,684 | 2,868 | |
| Scoring | 0 | 0 | 29,515 | 29,515 | 12.1% | 29,515 | 0 | 25,563 | 20,059 | 29,515 | |
| Routing | 909 | 18,124 | 0 | 19,033 | 7.8% | 18,130 | 903 | 9,542 | 9,539 | 18,124 | |
| Search | 0 | 479 | 0 | 479 | 0.2% | 479 | 0 | 479 | 0 | 479 | under_2000 |
| Retrieval | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | unsupported |
| Ranking | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | unsupported |
| Verification | 29,770 | 2,762 | 0 | 32,532 | 13.4% | 22,617 | 9,915 | 15,730 | 14,871 | 2,762 | |
| ML Feature Extraction | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | unsupported |
| Structured Data Extraction | 8 | 6,578 | 0 | 6,586 | 2.7% | 6,575 | 11 | 11 | 9 | 6,578 | |
| **total** | 113,594 | 100,257 | 29,515 | 243,366 | 100% | 125,915 | 117,451 | 187,579 | 158,837 | 129,772 | |

The two exclusion columns use the builds in `data/licence_manifest.json`; their totals match the licence audit's 243,366 / 187,579 / 158,837.

Record counts by task form: 113,594 noul, 100,257 choice, 29,515 score. 6,000 records are abstention constructions (`answerable = false` with a reason); they carry the ability of their donor question and a nonzero `selective_prediction` count in the JSON.

## What the matrix says

- **The corpus is a reading-comprehension corpus.** Classification is 61.9% of records, and 70% of classification is heuristic-labelled synthetic free-form questions. The next largest abilities are Verification (13.4%), Scoring (12.1%) and Routing (7.8%).
- **Three abilities have no records: Retrieval, Ranking, ML Feature Extraction.** The corpus cannot currently support them at all. No source asks to fetch context, order candidates, or produce features for a downstream model.
- **Search is a licence artifact.** All 479 records are CodeSearchNet, and CodeSearchNet is one of the unresolved licence groups, so Search falls to 0 under the full exclusion build. It is flagged `under_2000` even before that.
- **Structured Data Extraction is effectively apigen-only.** 6,575 of 6,586 records are `apigen_tool_argument`, which is a restrictive group. The ability falls to 11 records after excluding restrictive sources. A mixture that relies on it must accept that licence position or generate new data.
- **Detection is thin.** 4,650 records (1.9%), 3,592 of them source-ruled (the synthetic issue/error templates and three typed-decisions families). It clears the 2,000 threshold but not by much.
- **Routing loses half under exclusion.** 18,124 choice rows, mostly apigen tool choice/next action; 9,542 remain after excluding restrictive groups.

## Gradient eligibility, and why noul coverage is misleading

The listwise and pairwise trainers filter the training set with `TRAINED_TYPES = ("choice", "score")` (`model/scripts/train_listwise.py:305`, `model/scripts/train_pairwise.py:187`). Every noul row is dropped before the first step, so the listwise arm takes **zero gradient from 113,594 noul records**, 46.7% of the corpus. The coverage matrix reports that directly: `types.noul.gradient_eligible_listwise = 0`, while 129,772 choice and score records are eligible.

The per-option residual trainer does train noul with its fixed two-option `["no", "yes"]` set, and all 243,366 records are eligible there. The caveat for any future head that builds noul options from `criteria`: a question with a single criterion has a one-element softmax, whose log is identically zero, and that row would contribute no gradient. No such question exists in this corpus today, but a coverage report that counts rows without this check would miss it.

A row count for noul therefore says nothing about what the current listwise arm learns. The matrix separates the two columns for that reason.

## Hand-check

Stratified sample: 10 records per ability that has any, fixed seed 247. Seven abilities have records, so 70 records; Retrieval, Ranking and ML Feature Extraction have none to sample. The sample is 55 `source_rule` and 15 `heuristic` rows.

Agreement: **69 of 70, 98.6%**. Every ability scored 10/10 except Classification (9/10). The single disagreement is a heuristic row:

- "Does the evidence suggest that Rob Lowe's less prominent film roles directly caused him to transition to high-profile television roles?" — classifier: Classification; hand label: Verification (the question checks a causal claim against evidence).

The filled sample is committed as `247-handcheck-sample.json` and the scored report as `247-handcheck.json`. The sample is small: 10 per ability gives a wide interval around each per-ability rate, so the matrix is a coarse map, not a certified label set. The heuristic share (48.3% of the corpus) is the main uncertainty.

## Reproduction

    python scripts/build_canonical_corpus.py --build full \
        --root data/benchmarks \
        --typed-decisions data/benchmarks/typed_decisions.jsonl \
        --out research/krino-canonical-243366.jsonl \
        --json research/krino-canonical-243366.build.json

    python scripts/label_abilities.py \
        --canonical research/krino-canonical-243366.jsonl \
        --manifest data/licence_manifest.json \
        --out research/canonical_labeled.jsonl \
        --map-out research/krino-canonical-243366.ability.jsonl \
        --coverage research/247-ability-coverage.json

    python scripts/label_abilities.py --canonical research/krino-canonical-243366.jsonl \
        --sample-out research/247-handcheck-sample.json --sample-size 10 --seed 247

    python scripts/label_abilities.py --handcheck research/247-handcheck-sample.json \
        --handcheck-out research/247-handcheck.json

Build: about 21 s; label: about 22 s; peak working set 15 MB for the label pass. The labelled corpus (`--out`) is large and stays beside the canonical artifact, not in the repository. The sidecar is the id-to-ability map other work should consume; the coverage JSON is the aggregate.

## Out of scope

The mixture weights and augmentation targets in workstream D are a separate decision and are not proposed here. This note and its artifacts are the classification fact the decision needs.
