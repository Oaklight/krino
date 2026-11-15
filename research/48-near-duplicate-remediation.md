# Canonical corpus v2: near-duplicate policy and freeze — 2026-11-15

Issue #235 reported the near-duplicate clusters and did not adjudicate them. This
note fixes the per-row policy, applies it, and freezes the corpus that training
will consume. It does not train a model and does not change the dev or held-out
selection.

The owner approved the per-row rule (option C) on 2026-11-15. No model was
trained or evaluated on v2 before that decision. The only numbers seen before it
were corpus counts, coverage counts and held-out overlap counts.

| item | value |
|---|---:|
| canonical v1 records | 243,366 |
| near-duplicate clusters at Jaccard >= 0.7 | 5,734 |
| rows in those clusters | 118,162 |
| canonical v2 records | 234,285 |
| v2 records removed | 9,081 |
| v2 training rows (v2 minus held-out ids) | 227,861 |
| held-out ids retained | 6,424 |

## Rule

A non-held-out row is removed if either condition holds:

1. its exact normalized state equals a held-out row's state; or
2. its MinHash Jaccard to at least one held-out row is at least 0.7.

The MinHash is the #235 one: character 5-grams of the normalized state, 128
permutations, 32 bands, seed 42, threshold 0.7. Cluster membership is the
transitive closure of the 0.7 edge, so a row at or above 0.7 to a held-out row is
in that row's cluster. The build compares within the cluster and therefore does
not depend on a second LSH pass.

Cluster mates below the threshold are kept. Held-out rows are never removed and
never re-split.

A held-out row's cluster can still contain a training row that is below the
threshold. That row is not a near-duplicate under the rule and stays. This is the
difference between the per-row rule and the earlier cluster-level rule.

## Why per-row

The #235 clusters group near-identical `state` text. Exact duplicate records are
0. The rows inside a cluster are distinct questions and labels, and the 1,940
synthetic one-domain clusters are the paraphrase, counterfactual, negation,
shuffle and fill-variant design.

Three readings were compared.

| option | removed | kept | training | apigen kept | Routing | SDE | Verification | held-out exposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A, strict cluster quarantine | 9,950 | 233,416 | 226,992 | 9,487 | 14,052 | 5,220 | 31,163 | 0 |
| B, keep cluster-touching rows | 702 | 242,664 | 236,240 | 16,081 | 19,017 | 6,586 | 32,343 | 718 |
| **C, per-row (chosen)** | **9,081** | **234,285** | **227,861** | **9,501** | **14,396** | **5,221** | **31,173** | **0 direct; 20 cluster-only** |

A is the strict cluster quarantine. It removes every row whose cluster contains a
held-out row, including 869 rows that are only cluster mates and are below the
threshold. B removes only the 702 exact-state rows outside clusters and keeps all
cluster-touching rows; it preserves coverage but leaves 718 held-out rows with a
direct near-duplicate training row. C removes every row that crosses the rule and
keeps the below-threshold cluster mates.

C was chosen because it is the minimal set that leaves no training row at or above
the threshold against any held-out row, and it does not delete a distinct question
only because a transitive chain connects it to held-out. The 869 rows C keeps that
A removes are 823 `typed_decisions`, 18 `synthetic` and 14 apigen rows, plus 14
rows from other sources.

The apigen cost is intrinsic, not an artifact of transitive closure. C still
removes 6,580 of 16,081 apigen rows and keeps 9,501. The five apigen files reuse
one scenario template, so against a held-out apigen row most remaining apigen rows
are direct near-duplicates. Routing falls 24.4% and Structured Data Extraction
20.7% because the removed rows are genuinely at or above the threshold. Only 14
apigen rows are recovered relative to A.

## v2 definition and counts

| removal branch | rows |
|---|---:|
| exact normalized state | 1,063 |
| — inside a held-out cluster | 361 |
| — outside every cluster | 702 |
| Jaccard >= 0.7 only (not exact) | 8,018 |
| **total removed** | **9,081** |

The 702 exact-state removals outside clusters stand in every variant. The 361
inside clusters are also exact-state matches; A removed them through the cluster,
and C removes them through the exact branch.

v2 = v1 minus the 9,081 ids. The 6,424 held-out ids stay in the artifact for
evaluation; the trainer excludes them by id, as it did for v1. Training rows are
227,861.

Of the 9,248 non-held-out rows in clusters that touch held-out, 869 stay: they are
below the threshold to every held-out row. The other 8,379 are removed. After the
rule, no training row is at or above 0.7 to any held-out row.

### By type

| type | removed | kept |
|---|---:|---:|
| noul | 2,280 | 111,314 |
| choice | 6,572 | 93,685 |
| score | 229 | 29,286 |
| **total** | **9,081** | **234,285** |

### By label kind

| label_kind | removed | kept |
|---|---:|---:|
| option | 6,551 | 91,706 |
| withheld_gold | 21 | 1,979 |
| level | 227 | 27,383 |
| fractional | 2 | 1,903 |
| none (noul) | 2,280 | 111,314 |

### By source

The `source` field is the item's own source. The bundled licence-group view
(`research/30`) changes by 7,651 restrictive-group rows, 341 unresolved-group rows
and 1,089 permissive-group rows. Full counts are in the manifest.

| source | removed | kept | v1 |
|---|---:|---:|---:|
| `apigen_tool_action` | 1,569 | 411 | 1,980 |
| `synthetic` | 1,562 | 152,807 | 154,369 |
| `apigen_tool_choice` | 1,525 | 412 | 1,937 |
| `apigen_tool_next_action` | 1,456 | 3,330 | 4,786 |
| `apigen_tool_argument` | 1,365 | 5,210 | 6,575 |
| `apigen_policy_compliance` | 665 | 138 | 803 |
| `multirc` | 202 | 5,982 | 6,184 |
| `contractnli` | 186 | 4,603 | 4,789 |
| `mednli` | 103 | 2,449 | 2,552 |
| `mnli` | 47 | 1,908 | 1,955 |
| `sst2` | 47 | 1,829 | 1,876 |
| `typed_decisions` | 42 | 7,995 | 8,037 |
| `logiqa` | 38 | 3,960 | 3,998 |
| `boolq` | 30 | 1,374 | 1,404 |
| `fever` | 25 | 1,725 | 1,750 |
| `codesearchnet` | 23 | 456 | 479 |
| `sst5` | 23 | 1,718 | 1,741 |
| `tabfact` | 23 | 2,079 | 2,102 |
| `commonsenseqa` | 22 | 3,386 | 3,408 |
| `banking77` | 17 | 2,962 | 2,979 |
| `anli` | 16 | 1,223 | 1,239 |
| `race` | 14 | 2,034 | 2,048 |
| `piqa` | 13 | 1,412 | 1,425 |
| `swag` | 13 | 1,877 | 1,890 |
| `hellaswag` | 12 | 2,559 | 2,571 |
| `stsb` | 12 | 1,418 | 1,430 |
| `winogrande` | 10 | 1,383 | 1,393 |
| `agnews` | 5 | 1,113 | 1,118 |
| `arc` | 5 | 1,115 | 1,120 |
| `mmlu` | 4 | 2,180 | 2,184 |
| `mmlu_pro_short_reasoning_choice` | 4 | 9,961 | 9,965 |
| `yelp` | 3 | 1,067 | 1,070 |
| `2wiki_long_choice` | 0 | 345 | 345 |
| `hotpotqa_long_choice` | 0 | 899 | 899 |
| `longbench_long_choice` | 0 | 426 | 426 |
| `openjev_authored144` | 0 | 41 | 41 |
| `race_short_passage_choice` | 0 | 498 | 498 |
| **total** | **9,081** | **234,285** | **243,366** |

## Hashes

The portable identity is `kept_ids_sha256`, not the file hash. Apply `remove_ids`
from the manifest to the v1 artifact and verify `kept_ids_sha256`. The v2 file
hash is materialization-local: it identifies this build, not the v2 id set. A
materialization from different v1 row order, or a republish from the canonical v1
bytes, has the same ids and a different file hash.

| artifact | SHA-256 |
|---|---|
| canonical v1 (published) | `8eb3b2a490a04c600bd652ebeed738c02976bb77e731b337efa513805c62e4f6` |
| v2 removed ids (sorted, newline-joined) | `984d0355bfc1932de8b9ff7f8fe995e974dd52f959a65421db95f5f74241e98c` |
| **v2 kept ids (portable identity)** | **`85096d0e3dd1acfa754b3766aba38e60c3194f644c2f018912f580c77528e9f2`** |
| v2 JSONL materialized here (local) | `768c4e537092f9f27bde67d20344ed5a4ee765ee23c7fbc3e15a34147c61a6c7` |
| held-out ids (unchanged) | `492ee0afd47bfe819dee690127b8cfc2fab8c39fe10c4ebd6293754a53dcf87b` |
| v2 ability sidecar | `18133beb3607cef9a60f42e39e09451c85899de375c2f6d338dd02a688924a9f` |

The manifest is `research/48-near-duplicate-remove-manifest.json`. It carries the
9,081 `remove_ids`, the counts above, the near-duplicate parameters, the two
removal branches and both id hashes. The v2 JSONL (234,285 rows), its `.sha256`,
the manifest and the ability files sit beside the canonical artifact under the
data root; they are not in the repository.

### Rebuild caveat

The published v1 bytes were not present in the working tree. v2 was computed
against a rebuild from the pinned sources. The rebuild agrees with every published
v1 aggregate: 243,366 records and ids; noul 113,594, choice 100,257, score 29,515;
provenance gold 221,285 and execution 22,081; 84,632 distinct normalized states;
118,162 rows in 5,734 near-duplicate clusters with the same four-way split; and
every licence-group count in `research/30`. All 6,424 held-out ids are present. The
manifest is id-keyed, so applying it to the v1 artifact does not depend on row
order. The file hash `768c4e53...` identifies this materialization only.

## Checks on v2

- **JevBench contamination: 0 of 234,285 rows.** The three #235 rules (id,
  normalized state, three shared 8-word shingles) were re-run against the 231
  public items. Exact duplicate records: 0.
- **Held-out freeze.** The recomputed held-out id hash is
  `492ee0af...` and matches. All 6,424 ids are in v2, none is missing, none is in
  the training side and none is in `remove_ids`. All 928 `(source, group)` pairs
  are complete with 6,424 rows, and no pair has a non-held-out row.
- **Threshold exposure.** No training row is at or above 0.7 to any held-out row;
  the build asserts this over every surviving row in a held-out cluster. Direct
  exposure is 0. Twenty held-out rows share a cluster with a surviving training
  row that is below the threshold; those rows are not near-duplicates under the
  rule.
- **Ability coverage.** v2 coverage was recomputed with the same labeler and
  licence manifest. The 6,424 held-out rows are included in this table, as they
  are in v1.

| ability | v1 | v2 | delta | share |
|---|---:|---:|---:|---:|
| Classification | 150,571 | 149,111 | -1,460 | -1.0% |
| Detection | 4,650 | 4,642 | -8 | -0.2% |
| Scoring | 29,515 | 29,286 | -229 | -0.8% |
| Routing | 19,033 | 14,396 | -4,637 | -24.4% |
| Search | 479 | 456 | -23 | -4.8% |
| Retrieval | 0 | 0 | 0 | 0% |
| Ranking | 0 | 0 | 0 | 0% |
| Verification | 32,532 | 31,173 | -1,359 | -4.2% |
| ML Feature Extraction | 0 | 0 | 0 | 0% |
| Structured Data Extraction | 6,586 | 5,221 | -1,365 | -20.7% |
| **total** | **243,366** | **234,285** | **-9,081** | **-3.7%** |

  The listwise/pairwise eligible count falls from 129,772 to 122,971. The residual
  trainer sees all 234,285 rows. Label methods are 118,102 `source_rule` and
  116,183 `heuristic`; `model_share` stays 0. Abstention rows fall from 6,000 to
  5,765.

## Residual risk

1. **Within-training near-duplicates remain.** 109,783 v2 rows are still inside a
   near-duplicate cluster. That is intentional: they are distinct questions and
   the synthetic augmentation. The internal train/val split must use the
   near-duplicate cluster, merged with exact state and `group`, as the split unit.
   The current `split_items` in `model/scripts/train_residual.py:152` is a
   row-level stratified split. The trainer-side change is assigned to the trainer;
   this note does not modify trainer code.
2. **Sub-threshold cluster mates remain.** 869 training rows in held-out-touching
   clusters are below 0.7. Twenty held-out rows share a cluster with one of them.
   The rule treats them as not near-duplicate; a stricter threshold would remove
   them and cost more coverage.
3. **The apigen cost is intrinsic.** Routing loses 24.4% and Structured Data
   Extraction 20.7%. The five apigen files reuse one scenario template, so 6,580 of
   their 16,081 rows are direct near-duplicates of held-out apigen rows. Option B
   would keep them but leave 718 held-out rows exposed.
4. **The abstention instrument moves.** 235 rows with `answerable=false` are
   removed. The committed abstention eval selection is unchanged, but a future
   rebuild on v2 will differ from one on v1.
5. **The byte hash is materialization-local.** Pin `kept_ids_sha256`. A peer that
   materializes v2 from the v1 artifact must verify the id set and row count, not
   the file hash.

## Reproduction

```bash
# v1 rebuild from the pinned raw sources (root layout as in the data mirror)
python scripts/build_canonical_corpus.py --build full \
  --manifest data/licence_manifest.json \
  --root data/benchmarks --synthetic-dir data/benchmarks/synthetic \
  --typed-decisions data/benchmarks/typed_decisions.jsonl \
  --repo-root . \
  --out data/benchmarks/krino-canonical-243366.jsonl \
  --json data/benchmarks/canonical-v1.build.json

# v2 artifact and manifest (per-row rule)
PYTHONPATH=. python scripts/build_canonical_v2.py \
  data/benchmarks/krino-canonical-243366.jsonl \
  research/235-near-duplicate-clusters.json \
  model/experiments/canonical/heldout_6400.json \
  data/benchmarks/krino-canonical-234285.jsonl \
  data/benchmarks/krino-canonical-234285.manifest.json

# checks
PYTHONPATH=. python scripts/check_canonical_v2.py \
  data/benchmarks/krino-canonical-234285.jsonl \
  data/benchmarks/jevbench /tmp/v2-contamination.json
python scripts/label_abilities.py \
  --canonical data/benchmarks/krino-canonical-234285.jsonl \
  --manifest data/licence_manifest.json \
  --map-out data/benchmarks/krino-canonical-234285.ability.jsonl \
  --coverage data/benchmarks/krino-canonical-234285.ability-coverage.json \
  --jevbench-root data/benchmarks/jevbench
python scripts/check_canonical_v2_heldout.py \
  data/benchmarks/krino-canonical-234285.jsonl \
  model/experiments/canonical/heldout_6400.json \
  data/benchmarks/krino-canonical-234285.manifest.json /tmp/v2-heldout.json
```
