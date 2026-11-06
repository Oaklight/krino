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

| ability | noul | choice | score | total | share | source_rule | heuristic | after exclude_restrictive | after exclude_restrictive_and_unresolved | listwise gradient | JevBench acc (n) | flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Classification | 81,125 | 69,446 | 0 | 150,571 | 61.9% | 45,007 | 105,564 | 131,700 | 110,675 | 69,446 | 67.2% (198) | |
| Detection | 1,782 | 2,868 | 0 | 4,650 | 1.9% | 3,592 | 1,058 | 4,554 | 3,684 | 2,868 | — (0) | |
| Scoring | 0 | 0 | 29,515 | 29,515 | 12.1% | 29,515 | 0 | 25,563 | 20,059 | 29,515 | 77.8% (18) | |
| Routing | 909 | 18,124 | 0 | 19,033 | 7.8% | 18,130 | 903 | 9,542 | 9,539 | 18,124 | 100% (1) | |
| Search | 0 | 479 | 0 | 479 | 0.2% | 479 | 0 | 479 | 0 | 479 | — (0) | under_2000, single_source, licence_single_point |
| Retrieval | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | — (0) | unsupported |
| Ranking | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | — (0) | unsupported |
| Verification | 29,770 | 2,762 | 0 | 32,532 | 13.4% | 22,617 | 9,915 | 15,730 | 14,871 | 2,762 | 0% (2) | |
| ML Feature Extraction | 0 | 0 | 0 | 0 | 0% | 0 | 0 | 0 | 0 | 0 | — (0) | unsupported |
| Structured Data Extraction | 8 | 6,578 | 0 | 6,586 | 2.7% | 6,575 | 11 | 11 | 9 | 6,578 | 91.7% (12) | single_source, licence_single_point |
| **total** | 113,594 | 100,257 | 29,515 | 243,366 | 100% | 125,915 | 117,451 | 187,579 | 158,837 | 129,772 | 68.8% (231) | |

The two exclusion columns use the builds in `data/licence_manifest.json`; their totals match the licence audit's 243,366 / 187,579 / 158,837.

Record counts by task form: 113,594 noul, 100,257 choice, 29,515 score. 6,000 records are abstention constructions (`answerable = false` with a reason); they carry the ability of their donor question and a nonzero `selective_prediction` count in the JSON.

## What the matrix says

- **The corpus is a reading-comprehension corpus.** Classification is 61.9% of records, and 70% of classification is heuristic-labelled synthetic free-form questions. The next largest abilities are Verification (13.4%), Scoring (12.1%) and Routing (7.8%).
- **Ranking and ML Feature Extraction are real gaps.** Both have zero records, and the public cookbooks demonstrate both as the model's job: Re-ranking and Skill suggestion for Ranking, Autoresearch feature discovery for ML Feature Extraction. These two lines say the corpus cannot teach an ability the product asks for. They are the most important rows in the table.
- **Retrieval is not a gap.** It has zero records, but the cookbooks keep retrieval in code — BM25 shortlists, RAG candidate fetch — and ask Jev only to score or rank the candidates. The use-case map lists the ability, so the row stays; zero corpus coverage is expected rather than missing capability.
- **Search is a single point of failure.** All 479 records are CodeSearchNet, which is one unresolved licence group. One licence decision takes Search to zero, and it is already flagged `under_2000`.
- **Structured Data Extraction is a single point of failure.** 6,575 of 6,586 records are `apigen_tool_argument`, a restrictive group. One licence decision takes the ability to 11 records. The row is not six thousand records of coverage; it is one generator plus one licence position.
- **Detection is thin.** 4,650 records (1.9%), 3,592 of them source-ruled (the synthetic issue/error templates and three typed-decisions families). It clears the 2,000 threshold but not by much.
- **Routing loses half under exclusion.** 18,124 choice rows, mostly apigen tool choice/next action; 9,542 remain after excluding restrictive groups.

Two flags make the single-source cases explicit in the JSON. `single_source` means one source supplies at least 90% of the ability. `licence_single_point` means that source's group is not permissive, so one licence decision can zero the ability.

## Demand side

The acceptance set is the 231 public JevBench items, classified by question form with the same heuristic. It is not the corpus, and what it can see bounds what any JevBench number means.

| ability | JevBench items | JevBench share | corpus share |
|---|---:|---:|---:|
| Classification | 198 | 85.7% | 61.9% |
| Scoring | 18 | 7.8% | 12.1% |
| Structured Data Extraction | 12 | 5.2% | 2.7% |
| Verification | 2 | 0.9% | 13.4% |
| Routing | 1 | 0.4% | 7.8% |
| Detection, Search, Retrieval, Ranking, ML Feature Extraction | 0 | 0% | 1.9% / 0.2% / 0 / 0 / 0 |

Four consequences:

- The acceptance set is 85.7% Classification while the corpus is 61.9%. A JevBench score measures the part of the corpus that overlaps with the set, not the corpus composition.
- Scoring has 18 items. A binomial 95% interval at p=0.5 is about +/-23 points on 18 items, so a change confined to Scoring cannot move the 231-item total beyond noise by itself. A held-out split is what sees a Scoring regression.
- Ranking and ML Feature Extraction have zero items in the acceptance set as well as zero corpus records. The metric cannot see those gaps by construction; the coverage table is the only place they show up.
- Verification is the corpus-side mirror: it is 13.4% of the corpus but 0.9% of the acceptance set, and its two JevBench items both miss. Much of the corpus trains an ability the acceptance set barely measures, and where it does measure it the sample cannot tell us anything.

The coverage JSON carries this as `demand`, with the per-ability counts, both share columns, and these four notes.

## Gradient eligibility, and why noul coverage is misleading

The listwise and pairwise trainers filter the training set with `TRAINED_TYPES = ("choice", "score")` (`model/scripts/train_listwise.py:305`, `model/scripts/train_pairwise.py:187`). Every noul row is dropped before the first step. This is an exclusion, not a measured zero gradient: the trainer never sees the row, so the row cannot contribute. The ablation's null result is separate evidence that the exclusion matters.

The size is 113,594 noul records, 46.7% of the corpus, and the coverage matrix reports it directly: `types.noul.gradient_eligible_listwise = 0`, while 129,772 choice and score records are eligible. Every percentage in the coverage table is therefore "records labelled", not "records trained".

The ability that loses most is Verification: 32,532 records (13.4% of the corpus), but only 2,762 listwise-eligible rows, 8.5% of its own records and 1.1% of the corpus, because 29,770 of its records are noul. The per-ability `gradient_share` in the JSON makes the difference explicit:

| ability | records | listwise-eligible | share |
|---|---:|---:|---:|
| Verification | 32,532 | 2,762 | 8.5% |
| Classification | 150,571 | 69,446 | 46.1% |
| Detection | 4,650 | 2,868 | 61.7% |
| Routing | 19,033 | 18,124 | 95.2% |
| Structured Data Extraction | 6,586 | 6,578 | 99.9% |
| Scoring | 29,515 | 29,515 | 100% |
| Search | 479 | 479 | 100% |

The per-option residual trainer does train noul with its fixed two-option `["no", "yes"]` set, and all 243,366 records are eligible there. The caveat for any future head that builds noul options from `criteria`: a question with a single criterion has a one-element softmax, whose log is identically zero, and that row would contribute no gradient. No such question exists in this corpus today, which is why `small_option_items` is empty.

## Accuracy join

The labeler joins per-item predictions to these labels with `--jevbench-predictions`, which classifies the 231 public JevBench items by question form and groups the predictions by ability. The input is `control_pred_231.jsonl`, SHA-256 `e55ae4dc919ccb906d99308129ae9702c9bed373f168d85f4875a153ff118661`, 231 rows, 0 unmatched.

| ability | corpus records | JevBench n | accuracy | chance-corrected |
|---|---:|---:|---:|---:|
| Classification | 150,571 | 198 | 67.2% | 0.492 |
| Verification | 32,532 | 2 | 0% | -1.000 |
| Scoring | 29,515 | 18 | 77.8% | 0.708 |
| Routing | 19,033 | 1 | 100% | 1.000 |
| Structured Data Extraction | 6,586 | 12 | 91.7% | 0.889 |
| Detection | 4,650 | 0 | — | — |
| Search | 479 | 0 | — | — |
| Retrieval, Ranking, ML Feature Extraction | 0 | 0 | — | — |
| **total** | **243,366** | **231** | **68.8%** | — |

The type totals reproduce the control report exactly: choice 139 at 68.3%, noul 74 at 67.6%, score 18 at 77.8%. Only Classification has enough items to read as a rate. Verification (2), Routing (1) and the four abilities with no items are cells, not measurements; SDE at 12 items is suggestive at best. The table is committed as `eval` in the coverage JSON with `n`, `accuracy`, `chance_corrected` and the chance-corrected count for every cell.

The Scoring comparison is asymmetric and is the finding. On the split, chance-corrected accuracy is 0.3083 at n=474 (3,999 split items: noul 1,908, choice 1,617, score 474), a real measurement with a standard error of about 2.1 points. On JevBench it is 0.708 at n=18; 18 items carry a binomial 95% interval of roughly +/-23 points and are not a measurement of anything. The split says Scoring is the weakest of the three types; JevBench says it is the best. Same model, same head, opposite direction. The asymmetry is not symmetric: trust the split's magnitude, not JevBench's rank. A plausible partial explanation for the split number comes from the pilot: Jev score labels are broad (mean entropy 0.907 of a 1.609 ceiling; 44.9% argmax-versus-expected disagreement), which 18 JevBench items cannot show. This is the third worked example of why standard C needs both instruments.

The residual split numbers in the issue thread (noul 0.4874, choice 0.5991, score 0.3083) are a different, harder population and are never mixed into this table. A canonical-id `--predictions` join exists for split diagnostics.

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

The mixture weights and augmentation targets in workstream D are a separate decision and are not proposed here. They still need two inputs: the ability-level accuracy join (once the prediction file is supplied) and a demand-side count of what the acceptance set asks for, ability by ability. This classification does not block either; it is the coverage half of the decision.
