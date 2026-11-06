# Matched held-out set v1: specification

**Status: spec only. Do not build until #234 fixes the canonical corpus.** This document
is sufficient for another session to implement the set without asking questions: it fixes
the size, the quotas, the group key, the split algorithm, the freeze procedure and the
analysis.

## What the set is

One `TypedQuestion` JSONL file, the same interface as production (`data/format.py`), with
`split="heldout"` on every row. It is drawn from the canonical corpus defined by #234, so
it is matched to the training distribution by construction rather than by sampling a
different benchmark. It is never trained on, never used for model selection, and never
rebuilt after the first model sees it.

JevBench public is a hard non-regression constraint, not the promotion target. The public
set is too small to resolve the effects we act on: two checkpoints disagree on about a
third of its 231 items. This set is the promotion signal; JevBench can only veto, by
showing a regression on the public items.

## Size and power

Target: **6,400 items**.

The figure comes from the paired exact McNemar calculation in
`model/evaluation/paired_power.py`, at the discordance rate measured on JevBench public
between comparable checkpoints (32.4%):

| items | power at 2 pp | smallest detectable at 80% |
|---:|---:|---:|
| 6,357 | 0.800 | 2.00 pp |
| 6,400 | 0.792 | 2.00 pp |

The rounding is why 6,400 is the target and not 6,357. The calculation assumes paired
comparisons on the same items, two-sided alpha 0.05, and a discordance rate that
transfers from JevBench to this set. The last assumption is the weak one: a set on which
models disagree more has more power, and one on which they agree more has less. Record
the realized discordance after the first two runs and recompute the detectable effect.
Do not rebuild the set to chase a number.

**Instrument limit.** 2.0 pp at 6,400 items is the same size as the project's effect
floor. The set detects that floor at 80% power and nothing smaller at the planning
discordance; a smaller true effect will usually not reach significance here. That is a
property of the instrument (6,400 paired items at 32.4% discordance), not of any result.
Keep the set fixed and recompute the detectable effect at the realized discordance after
the first two runs.

Per-type power at the quotas below, at the same 32.4% discordance:

| type | items | smallest detectable at 80% |
|---|---:|---:|
| noul | 2,200 | 3.50 pp |
| choice | 2,900 | 3.00 pp |
| score | 1,300 | 4.50 pp |

The overall 2 pp is powered. A per-type 2 pp is not: it would need about 6,357 items of
each type, or roughly 19,000 in total. State per-type effects at their own resolution.

### Paired standard error and what 6,400 resolves

For a paired binary comparison on a fixed item set, let `b` be the items the arm gets
right and the control wrong, `c` the reverse, and `pi = (b + c) / n` the realized
discordance. The accuracy difference is `d = (b - c) / n`; its standard error is
`sqrt((pi - d^2) / n)`. Under the null (`b = c`), `SE0 = sqrt(pi / n)`. At the planning
discordance 0.324 and n = 6,400, `SE0 = 0.0071`, or **0.71 pp**. The exact two-sided
McNemar test reaches 80% power at a true difference of **2.0 pp**
(`power(6400, 0.324, 0.02) = 0.792`); the normal approximation gives
`(1.96 + 0.84) * 0.71 = 2.0 pp`.

| realized discordance | SE0 at 6,400 | MDE at 80% power |
|---:|---:|---:|
| 0.20 | 0.56 pp | 1.60 pp |
| 0.30 | 0.68 pp | 1.90 pp |
| 0.324 | 0.71 pp | 2.00 pp |
| 0.40 | 0.79 pp | 2.20 pp |
| 0.50 | 0.88 pp | 2.50 pp |

Per-type SE0 at the quotas is 1.21 pp (noul, 2,200), 1.06 pp (choice, 2,900) and 1.58 pp
(score, 1,300), which gives the MDEs in the table above.

The SE is a planning value, not a result. After the first control and arm run, compute
`b`, `c` and the realized `pi`, report `SE(d)` and the exact McNemar p, and recompute the
MDE at the realized discordance. If the realized `pi` is much below 0.324, the set may
not resolve 2 pp; record that before promoting. Do not rebuild the set to chase power.

Reproduce:

```bash
python - <<'PY'
from model.evaluation.paired_power import items_for_power, detectable_delta
print(items_for_power(0.324, 0.02, 0.80))          # 6357
for n in (2200, 2900, 1300):
    print(n, detectable_delta(n, 0.324, 0.80))
PY
```

## Composition

### Type quotas

| type | target | reason |
|---|---:|---|
| noul | 2,200 | binary decisions are the largest production primitive |
| choice | 2,900 | the largest training supply and the type with the most option counts |
| score | 1,300 | the least-supplied primitive (#142); keep it above 1,000 so MAE/QWK are not noise |
| **total** | **6,400** | |

These are targets, not exact counts. Group atomicity can leave a type a few items short;
the walk continues until no remaining group fits. The achieved counts go in the manifest.

### Source allocation

Within each type:

1. **Cap.** No single source contributes more than 25% of the type's quota in the first
   pass. This stops one large source (for example the choice pool) from filling a type.
2. **Floor.** Every source with at least 50 eligible groups and at least 20 eligible items
   contributes at least one group, if the type's quota allows it.

The selection itself is the hash-order walk in "Split assignment"; the cap is relaxed
only when a type would otherwise be short. The builder records the achieved per-source
counts. This keeps domain coverage without a proportional-allocation step that would make
the result depend on source sizes.

### Eligibility

Include a row only when all of these hold:

- `provenance` is `gold` or `execution`. Exclude `teacher` soft labels: a held-out set
  scored against teacher probabilities measures agreement with the teacher, not accuracy.
- The label is present, is in the option set (choice) or in the declared scale (score),
  and is not a withheld sentinel. Exclude the STS-B withheld `0.0` rows (#177) and any
  other score row whose loader marks the label as withheld.
- The options are non-empty, unique, and (for choice) contain the label.
- The id is unique after exact dedup (#235).
- The row is not a JevBench public item, not a Decision Index suite item, and not a hit
  under #235's contamination rule (id match, normalized-state match, or ≥3 shared 8-word
  shingles with any of the 231 public JevBench items).

## Group key: the leakage unit

The split is assigned to **groups**, never to rows. A group is a connected component of
the graph below, built over the whole canonical corpus, including training rows.

### Edges

1. **Same source record.** All questions generated from one source row share a
   `record_group` edge. This covers multi-question records (`typed_decisions` workflows,
   DROP passages, HotpotQA contexts, multi-label rows). The loader's item id carries the
   row index for these sources (for example `drop-validation-00042-score-1`); the builder
   uses the longest id prefix shared by the items emitted from one row. When the loader
   emits one item per row, the item id is the record group.
2. **Same native entity.** When the source has a real entity id, rows with the same id are
   merged: `table_id` (TabFact), `video-id` (SWAG), `set_unique_id` (Quality),
   `workflow` (typed-decisions), `para_idx` (MultiRC, prefixed by split).
3. **Exact normalized state.** Equal after NFKC, casefold and whitespace collapse.
4. **Near-duplicate state.** MinHash over 8-word shingles, Jaccard ≥ 0.7. This is the
   threshold and the parameter set already used by the synthetic split cluster file
   (`.split_clusters.json`: 128 permutations, 32 bands, threshold 0.7), not a value tuned
   on this set. Build the signature with `MinHash.update()` on the 8-word shingle set:
   `LSHIndex.make_minhash()` is character-5-gram-only and is not what this step needs.

Edges are transitive: the group is the union-find component. Merge across sources, so a
state that appears in two datasets cannot land on both sides.

### Group key per source

Use the first applicable rule. The existing `group` field is not sufficient for the
sources marked **replace**: several loaders currently assign arbitrary shards
(`f"{source}-{i % 100}"`) or a coarse field (`genre`, `config`, `lp`), which either
leaks near-duplicates across the boundary or puts an entire config on one side.

| source | group key | note |
|---|---|---|
| banking77 | LSH cluster | no entity id |
| sst2 | LSH cluster | no entity id |
| agnews | LSH cluster | article text; **replace** `agnews-{i % 200}` |
| mnli | LSH cluster on premise + hypothesis | **replace** `genre` |
| typed_decisions | `workflow` | native; keep |
| stsb | LSH cluster on the sentence pair | **replace** `stsb-{i % 50}` |
| sst5 | LSH cluster | **replace** `sst5-{i % 50}` |
| arc | LSH cluster on question + options | **replace** `arc-{config}` |
| race | `sha256(article)` merged by LSH | article is the shared passage; **replace** `race-{config}` |
| hellaswag | LSH cluster on `ctx` | `activity_label` is a topic, not an entity; **replace** |
| tabfact | `table_id` | native; keep |
| fever | LSH cluster on evidence + claim | **replace** `fever-{i % 200}` |
| yelp | LSH cluster | `yelp_review_full` has no business id; **replace** |
| swag | `video-id` | native; keep |
| multirc | `multirc:{split}:{para_idx}` | paragraph; keep with prefix |
| mednli | LSH cluster on premise + hypothesis | **replace** `mednli-{i % 50}` |
| contractnli | LSH cluster on the contract clause | **replace** `contractnli-{config}`: `config` is two values and would split the whole dataset at once |
| codesearchnet | `repo` + `path` when present, else LSH | **replace** `codesearchnet-{lang}` |
| mmlu | LSH cluster on the question | **replace** `mmlu-{subject}` |
| winogrande, piqa, commonsenseqa, logiqa, boolq | LSH cluster | **replace** the `-{i % 100}` shard |
| anli | LSH cluster on premise | **replace** `anli-r{round}` |
| quality | `set_unique_id` | native; keep |
| hotpotqa | source record id merged by context LSH | **replace** `hotpotqa-{type}` |
| drop | `drop:{split}:{row_index}` merged by passage LSH | keep the record group and merge |
| helpsteer | response/prompt id when present, else LSH | **replace** `helpsteer-{i % 100}` |
| wmt-da | document id when present, else LSH | `lp` is a language pair, not an entity; **replace** |
| ultrafeedback | record id when present, else LSH | |
| synthetic | `synthetic-{domain}-{family_idx:04d}` | native family; all variants share it |
| residual_train_20000 | its `group` when present, else LSH | |

For every LSH group the id is `lsh:{sha256(sorted member ids)[:16]}`, so it does not
depend on which source happened to be scanned first.

## Construction rule per question type

### noul

One item per `(state, instructions, criteria)` with a boolean label. Keep `criteria` when
the source supplies `true`/`false` descriptions; do not invent option descriptions. A
noul item is eligible only with a hard gold bool. Teacher probabilities do not qualify.

### choice

One item per `(state, instructions, criteria map, label)`. Keep the exact option set and
the order the loader emits; do not add, remove or shuffle options at build time. If the
same state yields several questions, they are separate items in the same group. Store the
option count in the manifest's distribution.

### score

One item per `(state, instructions, ordered criteria list, float label)`. Preserve the
production loader's ordinal scale and legend. Do not resample, re-bin or re-weight: a
held-out score measured on a different scale is not comparable to training. Record the
loader and the scale mapping version in the manifest. Exclude withheld sentinels.

### synthetic

One item per `(family, question type)`: the base variant only. Counterfactual, paraphrase,
negation, shuffle and fill variants of the same family are not independent observations
and would overstate the effective sample size. If a family lacks a base item of a type,
take the lowest-numbered variant and record which. All variants remain in the group, so
the family can never straddle the boundary.

## Split assignment

Deterministic, seed `heldout-v1`, no random module state:

1. Sort every eligible group by `sha256("heldout-v1|" + group_id)`, ascending. A group
   can contain more than one type (a synthetic family with noul and choice items); it is
   kept whole.
2. Coverage pass: for each type, take one group from each source that has an eligible
   group of that type, in the same hash order, when it fits and is not already selected.
   This guarantees a source with a single eligible group is not crowded out by a large
   one.
3. Keep a remaining quota per type (2,200 / 2,900 / 1,300). Walk the sorted list and
   accept a group when, for every type it contains, the group's count of that type is at
   most the remaining quota. Otherwise skip it. A group is never split, and no type's
   quota is exceeded.
4. If a type is still short after the walk, continue down the sorted list with a relaxed
   source cap but the same per-type fit check. If it remains short, record the deficit;
   do not relax group atomicity and do not pad with partial groups.
5. Record the seed, the selected group ids, and the per-type and per-source achieved
   counts. The target is 6,400 within group granularity: 6,380 or 6,430 is acceptable
   and is recorded, not hidden.

## Freeze procedure

Build once, hash it, commit the hash, never rebuild after a model has seen it.

### Inputs are pinned

The manifest records:

- the canonical corpus manifest hash from #234;
- the SHA-256 of every input JSONL;
- the git commit of the builder script;
- the exact dependency lock and the Python version used to build;
- the LSH parameters (shingle size 8, 128 permutations, 32 bands, threshold 0.7, seed 42);
- the assignment seed and the list of selected group ids;
- the command that replays the build from a clean checkout.

### Outputs

```
heldout/v1/items.jsonl.gz     # canonical JSON, one TypedQuestion per line, sorted by id
heldout/v1/groups.json        # group id -> item ids, source, type
heldout/v1/manifest.json      # counts, hashes, seed, command, git sha, corpus hashes
```

Canonical line: `json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`
followed by a newline. `items_sha256` is the SHA-256 of the uncompressed bytes of that
file, so a gzip header change does not change the identity.

Do **not** commit the data. Store it in the private Hugging Face dataset
`oaklight/krino-heldout-v1`, or on the compute host, and record the path in the manifest.
Commit the manifest and the builder script.

### Freeze rule

After the first model has been scored on v1, v1 is immutable. A correction is v2: a new
manifest, a new hash, a new dataset path. Results are always reported with the version.
Never regenerate v1 in place, and never "fix" a leaked item by rebuilding the same
version.

### Replay, not re-hash

The freeze is reproducible only if the build replays. CI rebuilds `items.jsonl.gz` from
the recorded tuple in a clean environment and compares `items_sha256`, the per-source
counts and the per-group counts. A mismatch fails. The response is a version bump and an
updated tuple, never a silent re-hash of the committed artifact.

The interim split being carved before #234 is not this set, but it should be frozen with
the same tuple: dependency lock, Python version, builder commit, and a replay check.
Retrofitting a tuple after gate rows exist is more expensive than recording it at birth.

### Training exclusion

The training loaders drop every id in the manifest and every item whose group is in the
selected group list. Excluding the group, not only the id, covers a variant added to a
selected family after the build. The manifest is the source of truth for both lists, so a
training file cannot be built without reading it.

### Guards

- `tests/test_heldout_freeze.py`: reads the committed manifest; when the data file is
  present, recomputes `items_sha256` and fails on a mismatch; when it is absent, checks
  the manifest's internal counts and the committed hash only.
- `scripts/check_heldout_contamination.py` (or the #235 audit wired to the manifest):
  fails if any training file contains an id, normalized state, or group listed in the
  manifest. Run it in CI so a future data drop cannot silently reintroduce a member.

## Scoring and analysis

Run every model on all 6,400 items with the production inference path. Report:

- **Primary:** paired McNemar on per-item exact match (two-sided, alpha 0.05), overall on
  the fixed type mix and per type. Report the two discordant counts, the exact p, and the
  realized discordance rate.
- **Calibration:** ECE on the chosen-answer probability (10 bins) for noul and choice;
  Brier for noul; RPS, MAE and QWK for score.
- **Breakdowns:** per source, per option-count bucket, per tier if the source has one.
- **No composite.** Do not pool the three primitives into one index. A single number
  would hide which primitive moved and is not comparable to any board.

A result is confirmed only by the pre-registered 6,400-item evaluation. An interim look at
a subset is labelled interim and does not count as confirmation.

## Implementation checklist

1. Wait for #234 to publish the canonical corpus definition and its manifest hash.
2. Run the #235 contamination audit on that corpus; exclude every hit by source and id.
3. Build the group graph with `data/lsh.py` (8-word shingles, 128 permutations, 32 bands,
   threshold 0.7, seed 42), merging record groups and native entity ids first.
4. Select groups in the `heldout-v1` hash order with the per-type fit check.
5. Write `items.jsonl.gz`, `groups.json` and `manifest.json`; compute `items_sha256`.
6. Commit the builder script and the manifest; upload the data to the private dataset.
7. Add `tests/test_heldout_freeze.py` and the contamination guard to CI.
8. Run the first two models; record the realized discordance and recompute the detectable
   effect. Do not rebuild the set.

## Out of scope

- Building the set before #234 lands.
- Training on it, using it for model selection, or tuning a threshold to make it "clean".
- Treating JevBench public as the promotion target; under the dual standard it can only
  veto a regression.
- A single composite score.
