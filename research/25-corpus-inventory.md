# Corpus inventory — 2026-11-06

Scope: every candidate training file in `data/benchmarks/` across the compute
hosts, plus the synthetic corpus assembled from `synthetic/`. Host identities and
paths are deliberately not recorded in this repository. The machine-readable
numbers come from `scripts/corpus_inventory.py`; the commands are at the end.

## First pass, one host only (earlier on 2026-11-06)

Scope: `data/benchmarks/*.jsonl` on one host. `residual_train_20000.jsonl` and
`data/benchmarks/synthetic/` were on another host and are not in this count.

### Candidate training sets

| file | rows | distinct ids | types (noul/choice/score) | teacher probs | hard label |
|---|---|---|---|---|---|
| residual_train_20000_jev.jsonl | 20,000 | 20,000 | 9,542 / 8,085 / 2,373 | 19,995 | 20,000 |
| label_target_30k_jev.jsonl | 30,000 | 30,000 | 9,600 / 18,000 / 2,400 | 30,000 | 30,000 |
| label_target_30k_jev_noul_agree.jsonl | 28,013 | 28,013 | 7,613 / 18,000 / 2,400 | 28,013 | 28,013 |
| label_target_30k_jev_noul_union_mix50.jsonl | 38,892 | 38,892 | — / 18,000 / 2,400 | 38,892 | 38,892 |
| apigen_* (5 files) | 16,081 | 16,081 | 803 noul / 15,278 choice | 0 | 16,081 |
| mmlu_pro_short_reasoning_choice.jsonl | 9,965 | 9,965 | all choice | 0 | 9,965 |
| long-context (hotpotqa/musique/longbench/twowiki) | 1,670 | 1,670 | all choice | 0 | 1,670 |
| abstention_train_20k / _30k | 3,000 / 3,000 | 3,000 / 3,000 | 2,000 noul / 1,000 choice | 0 | 3,000 |
| race_short_passage_choice.jsonl | 498 | 144 passage ids | all choice | 0 | 498 |

### Findings

1. `residual_train_20000_jev` and `label_target_30k_jev` share **0** ids. They are
   two disjoint corpora, not two views of one. Total teacher-labelled material is
   therefore 50,000 items, not 30,000.
2. The six `label_target_30k_*` files are variants of one 30,000-item corpus;
   `_noul_agree` is a strict 28,013-id subset. Only one should be the primary set.
3. `race_short_passage_choice` has 498 distinct questions over **144 passage
   ids** (2-5 questions per passage). The `id` is a passage id, not an item id.
   No questions are duplicates: keep all 498. The canonical builder gives each
   one the unique id `<passage id>-q<N>` and keeps `<passage id>` as the group.
4. Exact contamination against the 231 public JevBench items (id and normalised
   state): **0 hits** in 97,355 rows checked
   (`random 54,214 + label_target 30k/38.9k + mmlu_pro + race`).
   The 8-word shingle near-duplicate check has not run yet.

### Not covered here

- `residual_train_20000.jsonl` (20,000) and `synthetic/` (230 files) are on another host.
- Near-duplicate audit (>=3 shared 8-grams) against the 231 items.
- Provenance split by source dataset (`gold` vs `teacher`).

---

# Second pass: both machines

The first-pass numbers reproduce. This pass adds the other host, expands the synthetic
corpus the way `load_synthetic()` reads it, and fixes the first-pass omissions.

## Method

- Read-only. The script counts physical JSONL rows, parses every line, and
  validates the hard label per question type.
- Synthetic is not a flat item file. `synthetic/<domain>_families.jsonl` holds
  families; `_variants.jsonl` holds counterfactual/negation/paraphrase variants;
  the item set is produced by `data.synthetic.family_to_typed_questions`, the
  same function `load_synthetic()` calls. The inventory reports both families
  (rows) and assembled items.
- Teacher coverage counts rows with a `teacher_probs` object. For assembled
  synthetic items it counts ids present in any `*_labels.jsonl` sidecar.
- Coercion counts are labels usable after normalisation (`"true"` -> `True`,
  `"2"` -> `2.0`). They are not counted as invalid.
- Machine totals:

| machine | flat `*.jsonl` | synthetic files | physical rows | distinct ids |
|---|---:|---:|---:|---:|
| host A | 20 | 230 | 803,053 | 235,677 |
| host B | 78 | 230 | 3,830,546 | 1,969,395 |
| CPU mirror | 21 | 230 | 823,053 | 235,677 |
| union | 94 | 230 | — | — |

The CPU mirror holds the dataset plus `residual_train_20000.jsonl`, so its row
count is host A's count plus 20,000 and its distinct-id count is unchanged (the
file repeats the `residual_train_20000_jev` ids). Wall time and resource use are
recorded in the PR, not here.

These are id-level counts. RACE's 144 passage ids carry 498 questions; the
canonical builder suffixes them, so the artifact has 243,366 items and 243,366
distinct ids. The training host has no RACE short file, so its id count is
unchanged.

The four files present on both machines (`label_target_30k_jev.jsonl`,
`residual_train_20000_jev.jsonl`, and the two `noul_union_*` variants) have
identical md5 sums. The 230 synthetic filenames are identical on both machines;
their contents were not checksummed end to end.

## Synthetic corpus

| quantity | value |
|---|---:|
| domains | 23 |
| files | 230 (23 x 10) |
| families | 4,476 |
| assembled items | 154,091 |
| base / counterfactual / negation / paraphrase | 53,301 / 25,943 / 17,904 / 56,943 |
| noul / choice / score | 88,096 / 43,921 / 22,074 |
| distinct ids | 154,091 (no duplicates) |
| items with `jev` soft labels | 152,956 (99.26%) |
| items with `gpt_5_6_luna` soft labels | 152,956 |
| items with `jev_augmented` soft labels | 67,649 |
| items with `gpt_5_6_luna_augmented` soft labels | 87,087 |

Defects found:

1. **1,135 items have no `jev` soft label.** 1,066 are in
   `education_assessment` (families 0200+), 69 in `game_strategy`. The rest of the
   corpus is fully labelled.
2. **Two augmented label sets never join.** `*_jev_augmented_llm_labels.jsonl`
   (17,904 rows, 17,772 ids) and `*_gpt_5_6_luna_augmented_llm_labels.jsonl`
   (17,885 rows, 17,753 ids) end their ids with `-<n>-llm`. Assembled item ids do
   not contain that suffix, so `load_synthetic()` cannot find them. These 35,789
   rows are dead weight: no loader reads them, they add no coverage, and they
   should be regenerated with the joined id or deleted. The
   `*_llm_confounders.jsonl` files (17,904 rows) use a third id namespace
   (`<domain>-<family>-<question>`) and are sidecars, not labels.
3. **51 family sub-questions are dropped at expansion time.** The family expander
   logs and skips them: counterfactual choice labels not in the base criteria
   (mainly `game_strategy` and `multi_agent_coordination`), a score label out of
   range, and a criteria list where a dict was required. Assembled counts already
   exclude them; the families are still on disk.
4. `long_document` is the size outlier: 108 families / 2,374 items against ~200
   families / ~6,900 items in the other 22 domains.
5. **The `jev` label sidecars contain 1,126 duplicate rows.** 154,092 rows cover
   152,966 distinct ids; 931 ids repeat, in `education_assessment` (1,056 rows)
   and `game_strategy` (70), across 160 families. `load_synthetic()` merges by
   id, so the duplicate rows do not change a model input, but they inflate any
   count taken over the label files. `research/09-score-data-sources.md` quotes
   154,092 as "existing synthetic states"; that is the label-row count, not the
   state count. The assembled item count is 154,091, and the 51 expansion drops
   above are a separate quantity.

Per-domain item counts:

| domain | families | items | noul | choice | score | jev-labelled |
|---|---:|---:|---:|---:|---:|---:|
| academic_reasoning | 198 | 6,917 | 3,960 | 1,967 | 990 | 6,917 |
| adversarial_inference | 199 | 6,941 | 3,980 | 1,966 | 995 | 6,941 |
| code_review | 200 | 6,980 | 4,000 | 1,980 | 1,000 | 6,980 |
| commonsense_decision | 200 | 6,984 | 4,000 | 1,984 | 1,000 | 6,984 |
| content_analysis | 200 | 6,969 | 4,000 | 1,969 | 1,000 | 6,969 |
| education_assessment | 198 | 6,899 | 3,944 | 1,965 | 990 | 5,833 |
| financial_analysis | 197 | 6,860 | 3,928 | 1,950 | 982 | 6,860 |
| game_strategy | 194 | 6,721 | 3,844 | 1,908 | 969 | 6,652 |
| legal_judgment | 194 | 6,382 | 3,628 | 1,847 | 907 | 6,382 |
| logical_inference | 198 | 6,916 | 3,960 | 1,966 | 990 | 6,916 |
| long_document | 108 | 2,374 | 1,208 | 823 | 343 | 2,374 |
| medical_triage | 196 | 6,849 | 3,920 | 1,949 | 980 | 6,849 |
| multi_agent_coordination | 200 | 6,957 | 4,000 | 1,957 | 1,000 | 6,957 |
| multi_hop_reasoning | 199 | 6,704 | 3,824 | 1,924 | 956 | 6,704 |
| navigation_planning | 200 | 6,972 | 4,000 | 1,972 | 1,000 | 6,972 |
| numerical_reasoning | 195 | 6,806 | 3,900 | 1,933 | 973 | 6,806 |
| passage_decision | 200 | 6,978 | 4,000 | 1,978 | 1,000 | 6,978 |
| product_categorization | 200 | 6,985 | 4,000 | 1,985 | 1,000 | 6,985 |
| resource_management | 200 | 6,988 | 4,000 | 1,988 | 1,000 | 6,988 |
| safety_moderation | 200 | 6,980 | 4,000 | 1,981 | 999 | 6,980 |
| scientific_reasoning | 200 | 6,983 | 4,000 | 1,983 | 1,000 | 6,983 |
| sequential_action | 200 | 6,978 | 4,000 | 1,978 | 1,000 | 6,978 |
| spatial_reasoning | 200 | 6,968 | 4,000 | 1,968 | 1,000 | 6,968 |

## Master table

One row per flat JSONL file on either machine, plus the synthetic corpus as one
row below. `rows` is physical rows; for synthetic files the training unit is the
assembled item. `both` means the two source hosts; the CPU mirror holds a copy of
host A's set plus `residual_train_20000.jsonl`. Provenance is defined in the next
section.

| file | machines | kind | rows | distinct ids | noul | choice | score | teacher | provenance | disposition |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `abstention_train_20k.jsonl` | host A | typed | 3,000 | 3,000 | 2,000 | 1,000 | 0 | 0 | execution | primary abstention |
| `abstention_train_30k.jsonl` | host A | typed | 3,000 | 3,000 | 2,000 | 1,000 | 0 | 0 | execution | primary abstention |
| `apigen_policy_compliance.jsonl` | host A | typed | 803 | 803 | 803 | 0 | 0 | 0 | execution | primary tool-call |
| `apigen_tool_action.jsonl` | host A | typed | 1,980 | 1,980 | 0 | 1,980 | 0 | 0 | execution | primary tool-call |
| `apigen_tool_argument.jsonl` | host A | typed | 6,575 | 6,575 | 0 | 6,575 | 0 | 0 | execution | primary tool-call |
| `apigen_tool_choice.jsonl` | host A | typed | 1,937 | 1,937 | 0 | 1,937 | 0 | 0 | execution | primary tool-call |
| `apigen_tool_next_action.jsonl` | host A | typed | 4,786 | 4,786 | 0 | 4,786 | 0 | 0 | execution | primary tool-call |
| `hotpotqa_long_choice.jsonl` | host A | typed | 423 | 423 | 0 | 423 | 0 | 0 | gold | primary long-context |
| `label_target_30k_jev.jsonl` | both | typed | 30,000 | 30,000 | 9,600 | 18,000 | 2,400 | 30,000 | gold | primary target-30K |
| `longbench_long_choice.jsonl` | host A | typed | 426 | 426 | 0 | 426 | 0 | 0 | gold | primary long-context |
| `mmlu_pro_short_reasoning_choice.jsonl` | host A | typed | 9,965 | 9,965 | 0 | 9,965 | 0 | 0 | gold | primary mmlu-pro |
| `musique_long_choice.jsonl` | host A | typed | 476 | 476 | 0 | 476 | 0 | 0 | gold | primary long-context |
| `race_short_passage_choice.jsonl` | host A | typed | 498 | 498 (144 passage ids in the source) | 0 | 498 | 0 | 0 | gold | primary, unique id `<passage id>-q<N>`, group = passage id |
| `residual_train_20000_jev.jsonl` | both | typed | 20,000 | 20,000 | 9,542 | 8,085 | 2,373 | 19,995 | gold | primary residual-20K |
| `twowiki_long_choice.jsonl` | host A | typed | 345 | 345 | 0 | 345 | 0 | 0 | gold | primary long-context |
| `typed_decisions.jsonl` | host B | typed | 8,000 | 8,000 | 2,400 | 2,400 | 3,200 | 0 | gold | primary typed-decisions |
| `label_target_30k.jsonl` | host B | typed | 30,000 | 30,000 | 9,600 | 18,000 | 2,400 | 1,128 | gold | variant: unlabelled base |
| `label_target_30k_jev_noul_agree.jsonl` | host A | typed | 28,013 | 28,013 | 7,613 | 18,000 | 2,400 | 28,013 | gold | ablation: drop disagreeing noul |
| `label_target_30k_jev_noul_agree_mix50.jsonl` | host A | typed | 28,013 | 28,013 | 7,613 | 18,000 | 2,400 | 28,013 | gold | ablation: agree + gold-mixed soft target |
| `label_target_30k_jev_noul_mix50.jsonl` | host A | typed | 30,000 | 30,000 | 9,600 | 18,000 | 2,400 | 30,000 | gold | ablation: gold-mixed soft target |
| `label_target_30k_jev_noul_union_agree_mix50.jsonl` | both | typed | 36,014 | 36,014 | 15,614 | 18,000 | 2,400 | 36,014 | gold | ablation: union + agree + mix50 |
| `label_target_30k_jev_noul_union_mix50.jsonl` | both | typed | 38,892 | 38,892 | 18,492 | 18,000 | 2,400 | 38,892 | gold | ablation: union + mix50 |
| `residual_band20k_thr050.jsonl` | host B | typed | 4,374 | 4,374 | 0 | 3,010 | 1,364 | 287 | gold | variant: band-selected subset |
| `residual_long.jsonl` | host B | typed | 3,000 | 3,000 | 2,977 | 19 | 4 | 45 | gold | variant: long-context noul subset |
| `residual_train_15000_extra.jsonl` | host B | typed | 15,000 | 15,000 | 7,152 | 6,062 | 1,786 | 847 | gold | variant: 15K subset |
| `residual_train_20000.jsonl` | host B | typed | 20,000 | 20,000 | 9,542 | 8,085 | 2,373 | 1,133 | gold | variant: unlabelled base |
| `residual_train_5000.jsonl` | host B | typed | 5,000 | 5,000 | 2,390 | 2,023 | 587 | 286 | gold | variant: 5K subset |
| `agnews.jsonl` | host B | typed | 127,600 | 127,600 | 0 | 127,600 | 0 | 0 | gold | excluded: raw source conversion |
| `anli.jsonl` | host B | typed | 166,065 | 166,065 | 166,065 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `arc.jsonl` | host B | typed | 7,787 | 7,787 | 0 | 7,787 | 0 | 0 | gold | excluded: raw source conversion |
| `banking77.jsonl` | host B | typed | 13,083 | 13,083 | 0 | 13,083 | 0 | 0 | gold | excluded: raw source conversion |
| `boolq.jsonl` | host B | typed | 12,697 | 12,697 | 12,697 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `codesearchnet.jsonl` | host B | typed | 55,688 | 55,688 | 0 | 55,688 | 0 | 0 | gold | excluded: raw source conversion |
| `commonsenseqa.jsonl` | host B | typed | 10,962 | 10,962 | 0 | 10,962 | 0 | 0 | gold | excluded: raw source conversion |
| `contractnli.jsonl` | host B | typed | 20,107 | 20,107 | 20,107 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `corpus_full.jsonl` | host B | typed | 195,706 | 195,706 | 93,479 | 78,839 | 23,388 | 10,916 | gold | excluded: unlabelled source pool |
| `fever.jsonl` | host B | typed | 244,212 | 244,212 | 0 | 244,212 | 0 | 0 | gold | excluded: raw source conversion |
| `hellaswag.jsonl` | host B | typed | 49,947 | 49,947 | 0 | 49,947 | 0 | 0 | gold | excluded: raw source conversion |
| `logiqa.jsonl` | host B | typed | 8,678 | 8,678 | 0 | 8,678 | 0 | 0 | gold | excluded: raw source conversion |
| `mednli.jsonl` | host B | typed | 14,049 | 14,049 | 14,049 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `mmlu.jsonl` | host B | typed | 115,415 | 115,415 | 0 | 115,415 | 0 | 0 | gold | excluded: raw source conversion |
| `mnli.jsonl` | host B | typed | 402,517 | 402,517 | 402,517 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `multirc.jsonl` | host B | typed | 32,088 | 32,088 | 32,088 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `piqa.jsonl` | host B | typed | 17,951 | 17,951 | 0 | 17,951 | 0 | 0 | gold | excluded: raw source conversion |
| `race.jsonl` | host B | typed | 97,687 | 97,687 | 0 | 97,687 | 0 | 0 | gold | excluded: raw source conversion |
| `sst2.jsonl` | host B | typed | 68,221 | 68,221 | 68,221 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `sst5.jsonl` | host B | typed | 11,855 | 11,855 | 0 | 0 | 11,855 | 0 | gold | excluded: raw source conversion |
| `stsb.jsonl` | host B | typed | 7,249 | 7,249 | 0 | 0 | 7,249 | 0 | gold | excluded: raw source conversion |
| `swag.jsonl` | host B | typed | 93,552 | 93,552 | 0 | 93,552 | 0 | 0 | gold | excluded: raw source conversion |
| `tabfact.jsonl` | host B | typed | 117,854 | 117,854 | 117,854 | 0 | 0 | 0 | gold | excluded: raw source conversion |
| `winogrande.jsonl` | host B | typed | 41,665 | 41,665 | 0 | 41,665 | 0 | 0 | gold | excluded: raw source conversion |
| `yelp.jsonl` | host B | typed | 70,000 | 70,000 | 0 | 0 | 70,000 | 0 | gold | excluded: raw source conversion |
| `band_full_u50.jsonl` | host B | typed | 88,215 | 88,215 | 26,985 | 42,670 | 18,560 | 6,195 | — | excluded: residual selection band |
| `band_lo0.0.jsonl` | host B | typed | 16,074 | 16,074 | 7,448 | 6,386 | 2,240 | 1,012 | — | excluded: residual selection band |
| `band_lo0.01.jsonl` | host B | typed | 15,789 | 15,789 | 7,448 | 6,108 | 2,233 | 1,007 | — | excluded: residual selection band |
| `band_lo0.05.jsonl` | host B | typed | 15,322 | 15,322 | 7,438 | 5,799 | 2,085 | 974 | — | excluded: residual selection band |
| `band_up0.5.jsonl` | host B | typed | 8,968 | 8,968 | 2,750 | 4,340 | 1,878 | 651 | — | excluded: residual selection band |
| `band_up0.5_jev_pilot.jsonl` | host B | typed | 8,968 | 8,968 | 2,750 | 4,340 | 1,878 | 921 | — | excluded: residual selection band |
| `band_up0.6.jsonl` | host B | typed | 12,081 | 12,081 | 4,716 | 5,370 | 1,995 | 780 | — | excluded: residual selection band |
| `base_probs_20000.jsonl` | host B | aux | 20,000 | 20,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen35_4b.jsonl` | host B | aux | 20,000 | 20,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen35_4b_shard0.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen35_4b_shard1.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen35_4b_shard2.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen35_4b_shard3.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_base.jsonl` | host B | aux | 20,000 | 20,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_base_shard0.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_base_shard1.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_base_shard2.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_base_shard3.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_instruct.jsonl` | host B | aux | 20,000 | 20,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_instruct_shard0.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_instruct_shard1.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_instruct_shard2.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_20000_qwen3_4b_instruct_shard3.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_5000.jsonl` | host B | aux | 5,000 | 5,000 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_extra_shard0.jsonl` | host B | aux | 3,750 | 3,750 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_extra_shard1.jsonl` | host B | aux | 3,750 | 3,750 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_extra_shard2.jsonl` | host B | aux | 3,750 | 3,750 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_extra_shard3.jsonl` | host B | aux | 3,750 | 3,750 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full.jsonl` | host B | aux | 195,706 | 195,706 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full5_shard0.jsonl` | host B | aux | 39,279 | 39,279 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full5_shard1.jsonl` | host B | aux | 39,254 | 39,254 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full5_shard2.jsonl` | host B | aux | 39,261 | 39,261 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full5_shard3.jsonl` | host B | aux | 39,241 | 39,241 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full5_shard4.jsonl` | host B | aux | 39,270 | 39,270 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full_shard0.jsonl` | host B | aux | 48,086 | 48,086 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full_shard1.jsonl` | host B | aux | 48,063 | 48,063 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_full_shard2.jsonl` | host B | aux | 48,044 | 48,044 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_jevbench_qwen35_4b.jsonl` | host B | aux | 231 | 231 | — | — | — | — | — | excluded: base-model probabilities |
| `base_probs_union30k20k.jsonl` | host B | aux | 196,410 | 196,410 | — | — | — | — | — | excluded: base-model probabilities |
| `internal_eval_1000.jsonl` | host B | typed | 1,000 | 1,000 | 385 | 472 | 143 | 74 | — | excluded: internal eval |
| `jevbench.jsonl` | host B | typed | 231 | 231 | 74 | 139 | 18 | 0 | — | excluded: JevBench eval |
| `jevbench_231_typed.jsonl` | host B | typed | 231 | 231 | 74 | 139 | 18 | 0 | — | excluded: JevBench eval |


## Synthetic corpus row

| file | machines | kind | families (rows) | items | noul/choice/score | teacher | provenance | disposition |
|---|---|---|---:|---:|---|---:|---|---|
| `synthetic/` (23 domains) | both | assembled | 4,476 | 154,091 | 88,096 / 43,921 / 22,074 | 152,956 | gold (model-authored) | primary synthetic |

## Cross-file id relations

| relation | count |
|---|---:|
| `label_target_30k_jev` ∩ `residual_train_20000_jev` | 0 |
| `label_target_30k_jev` ⊆ `corpus_full` | 30,000 |
| `residual_train_20000_jev` ∩ `corpus_full` | 17,848 (2,152 residual ids are not in `corpus_full`) |
| synthetic (assembled) ∩ `corpus_full` | 11,000 |
| synthetic ∩ `label_target_30k_jev` | 1,133 |
| synthetic ∩ `residual_train_20000_jev` | 1,141 |
| `typed_decisions` ∩ `label_target_30k_jev` | 415 |
| `typed_decisions` ∩ `residual_train_20000_jev` | 250 |
| `race_short_passage_choice` rows / ids | 498 rows / 144 source ids; 498 artifact ids |
| `abstention_train_20k` ∩ `abstention_train_30k` | 0 |
| `corpus_full` ∪ `residual` ∪ `label_target` ∪ `typed_decisions` ∪ synthetic | 209,152 |

`label_target_30k.jsonl` and `label_target_30k_jev.jsonl` have the same 30,000
ids; the first is the unlabelled base, the second adds teacher probabilities.
`residual_train_20000.jsonl` and `residual_train_20000_jev.jsonl` are the same
20,000 ids. `residual_train_15000_extra` is a strict subset of the residual 20K.

One source-field defect: every row in `musique_long_choice.jsonl` (476) carries
`source: hotpotqa_long_choice`. The filename is the only reliable attribution.
Provenance is unaffected (both are gold), but a report grouped by `source` will
fold musique into hotpotqa.

## Canonical corpus definition

A **primary** file is the single canonical representation of its source and may
be read directly by training. A **variant** is derived from a primary file and is
used only for a named ablation. **Excluded** files are inputs, intermediates, or
evaluation data.

### Primary

| corpus | file | items | provenance |
|---|---|---:|---|
| residual-20K | `residual_train_20000_jev.jsonl` | 20,000 | gold |
| target-30K | `label_target_30k_jev.jsonl` | 30,000 | gold |
| synthetic | assembled from 23 domains | 154,091 | gold (model-authored) |
| tool-call | `apigen_*.jsonl` (5 files) | 16,081 | execution |
| abstention | `abstention_train_20k.jsonl`, `abstention_train_30k.jsonl` | 6,000 | execution |
| mmlu-pro | `mmlu_pro_short_reasoning_choice.jsonl` | 9,965 | gold |
| long-context | `hotpotqa_long_choice.jsonl`, `musique_long_choice.jsonl`, `longbench_long_choice.jsonl`, `twowiki_long_choice.jsonl` | 1,670 | gold |
| race-short | `race_short_passage_choice.jsonl` | 498 (144 source passage ids; unique `<id>-q<N>`) | gold |
| typed-decisions | `typed_decisions.jsonl` | 8,000 | gold |

Canonical union after item dedup: **243,366 items**. Dedup order is
target-30K, residual-20K, typed-decisions, synthetic, then the remaining
corpora. The 2,274 synthetic ids that already sit in the 20K/30K keep their
sampled copy; the 665 typed-decisions ids that already sit there are dropped from
the full `typed_decisions.jsonl` copy. The RACE records get unique ids of the
form `<passage id>-q<N>`; the passage id stays as the group, so its questions
stay in one split. The pre-dedup
component sum is 246,305 (245,951 before the RACE correction); after the
2,939 drops the total is 243,366 items with 243,366 distinct ids. The
`abstention_*` choice rows whose gold option is withheld are kept for the
abstain target, not for option CE (see below).

`abstention_train_20k` and `abstention_train_30k` are disjoint 3,000-item
constructions. Both are primary; `_30k` matches the primary 30K corpus if only one
is wanted. Their states are borrowed from the donor corpora, so id dedup does not
remove state-level duplication; the #235 near-duplicate audit covers that.

`race_short_passage_choice` reuses 144 passage ids for 498 questions, so an id
alone drops 354 distinct questions. The builder keeps all 498 with unique
`<passage id>-q<N>` ids and sets `group` to the passage id.

### Variants and ablations

- `label_target_30k.jsonl` — unlabelled base of the primary 30K.
- `label_target_30k_jev_noul_agree.jsonl`, `_noul_agree_mix50`,
  `_noul_mix50`, `_noul_union_agree_mix50`, `_noul_union_mix50` — label-repair
  ablations; see the next section.
- `residual_train_20000.jsonl` — unlabelled base of the primary 20K.
- `residual_train_5000.jsonl`, `residual_train_15000_extra.jsonl` — subsets.
- `residual_long.jsonl` — 3,000-item long-context noul subset.
- `residual_band20k_thr050.jsonl`, `band_full_u50`, `band_lo0.0`, `band_lo0.01`,
  `band_lo0.05`, `band_up0.5`, `band_up0.5_jev_pilot`, `band_up0.6` — base-model
  confidence-band selections over the residual pool.

### Excluded

- `corpus_full.jsonl` — the 195,706-item source pool the 20K/30K were sampled
  from. Only 10,916 rows carry teacher probabilities (all synthetic); it shares
  47,848 ids with the two primary corpora and 11,000 ids with the assembled
  synthetic corpus. It is a parent pool, not a training file.
- The 24 raw source conversions (`agnews`, `anli`, `mmlu`, `race`, ...) — the
  inputs to `corpus_full`, before capping. `typed_decisions.jsonl` is the one
  converted source promoted to primary because #207 makes its task shape a
  first-class target.
- `base_probs_*` (32 files) — base-model probability dumps (`{id, keys, probs}`),
  not training targets.
- `jevbench.jsonl`, `jevbench_231_typed.jsonl`, `internal_eval_1000.jsonl` —
  evaluation.
- `openjev/` — evaluation and authored-provenance pool; 37 of its ids appear in
  the 20K/30K as `openjev_authored144`.

## Licence position

`research/30-canonical-source-licences.md` audits every source group with quoted
terms and URLs, and records the owner's 2026-11-06 decision to keep every group.
Of 243,366 items, 55,787 are in 8 restrictive groups (ANLI, ContractNLI, APIGen,
RACE, AG News, MedNLI, Yelp, MultiRC), 28,742 are in 8 unresolved groups, and
158,837 are in permissive groups. Synthetic items inherit their seed dataset's
terms; an ANLI-, ContractNLI- or MedNLI-seeded domain is as restricted as its
seed. The corpus is research use only; `data/licence_manifest.json` and
`scripts/build_canonical_corpus.py` exist as the contingency if the policy is
revisited.

## Provenance

The field is per record: `provenance ∈ {gold, execution, teacher}`, assigned by
the corpus that produced the record. `TypedQuestion.provenance` defaults to
`None`; `None` means untagged and is ineligible for a hard-label loss. The
executable definition is `data/provenance.py`
(`classify`, `tag`, `may_drive_hard_label`). A per-record `source` is not enough
on its own: the abstention files carry the donor dataset in `source` but their
labels are constructed by a program, so the classifier also uses the corpus
name, the `-abstain-` id suffix, and `answerable`/`reason`.

| provenance | sources | reason |
|---|---|---|
| `gold` | the 24 benchmark conversions, `typed_decisions`, `openjev_authored144`, synthetic families | Label supplied by the source dataset, or authored into the synthetic dataset. None is a Jev prediction. |
| `execution` | `apigen_*` | Label is read from a recorded agent trajectory: the first `function_call` (tool choice/argument), or a program classification of the first assistant turn (action), or `action == call_tool` (policy). |
| `execution` | `abstention_*` | Labels are constructed by `model/scripts/build_abstention_sets.py`: the gold option is withheld (`none_of_these`), a foreign state is attached (`out_of_scope`, label False), or evidence is cut (`insufficient`, label False). |
| `teacher` | none today | No hard label in either corpus comes from a Jev prediction. `label_target_*_mix50` mixes gold into the **soft** target only; `label` is unchanged. |

Tier mix of the canonical union: **221,285 gold, 22,081 execution, 0 teacher**
(243,366 items before the label-usability exclusions below). Read the split
precisely:

- **No canonical record's hard label is a Jev prediction.** `teacher_probs` is
  always an additional soft target attached to a gold or execution label. No
  record is teacher-tier, so nothing was folded into another tier.
- **`gold` does not mean human-verified.** 154,091 gold records are
  model-authored synthetic labels (151,817 in the synthetic corpus and 2,274
  sampled into the 20K/30K). The other 67,194 are upstream dataset labels or
  community-authored labels; they were not re-annotated here.
- **Two thousand execution records cannot drive option CE.** The `none_of_these`
  choice rows (1,000 per abstention file) have a provenance-only label and
  belong to the abstain target; the other 20,081 execution records are usable.

Synthetic labels are `gold` in the epic's sense (a dataset label, not a Jev
prediction), but they are model-authored rather than human-annotated. This tier
does not imply human verification. `label_target_30k_jev` is 28,426 dataset labels,
441 community-authored (`typed_decisions` + `openjev`), and 1,133 synthetic.
`residual_train_20000_jev` is 18,598 dataset, 261 community-authored, 1,141
synthetic.

## Hard-label rule

A record may drive a hard-label loss (CE / BCE / ordinal) only when all hold:

1. `provenance ∈ {gold, execution}`. A `teacher` record is a soft target only
   (KL / distillation), never CE. No such record exists today; the rule is
   prospective for any future file whose label is the Jev argmax.
2. The label validates as encoded, or normalises without ambiguity:
   - noul: `bool`, or the string `"true"`/`"false"` (124 in `label_target_30k_jev`,
     70 in `residual_train_20000_jev`, from `typed_decisions`);
   - choice: the label is a key in `criteria`;
   - score: numeric and in `[0, len(criteria) - 1]` (string numerals are
     coercible: 85 in `label_target_30k_jev`, 108 in `residual_train_20000_jev`).
3. Abstention records (`answerable = false`) may drive the abstain/answerable
   target only. In `none_of_these` the gold option is deliberately absent from
   `criteria`; the `label` is provenance, not a selectable option, so these 1,000
   choice rows per file must not drive option CE. The 2,000 noul rows
   (`out_of_scope`, `insufficient`) are constructed negatives and may drive the
   answerable target; they are not evidence that the proposition is false.
4. Items are deduplicated by id. RACE records get unique `<passage id>-q<N>`
   ids before dedup; the original passage id is the group, so all 498 stay in
   one split per passage.

Records that fail (2) are excluded from hard-label loss but may still supply a
soft target. Recommend fixing the `typed_decisions` conversion to emit `bool` and
`float` labels rather than strings.

## Primary `label_target_30k_*` variant

**Selected: `label_target_30k_jev.jsonl`.** Criterion: maximise gold hard-label
coverage. Teacher agreement is not a filter on gold labels; dropping the records
where the teacher disagrees selects on the teacher's errors and changes the noul
target distribution.

| variant | rows | dropped vs primary | added vs primary |
|---|---:|---:|---:|
| `label_target_30k_jev` (selected) | 30,000 | 0 | 0 |
| `_noul_agree` | 28,013 | 1,987 noul | 0 |
| `_noul_agree_mix50` | 28,013 | 1,987 noul | 0 |
| `_noul_mix50` | 30,000 | 0 | 0 |
| `_noul_union_agree_mix50` | 36,014 | 1,987 noul | 8,001 residual noul |
| `_noul_union_mix50` | 38,892 | 0 | 8,892 residual noul |

The 1,987 ids dropped by any `_noul_agree` variant are all noul, concentrated in
`multirc` (561), `contractnli` (410), `mednli` (305), `sst2` (194), `mnli` (188),
`tabfact` (113), `boolq` (112), `anli` (81). The full id lists, and the ids each
union variant adds, are in `research/corpus-234-label-target-deltas.json`
(regenerate with `--deltas`).

Rejected alternatives:

- `_noul_agree` — drops 1,987 gold noul items on a teacher-correctness filter.
- `_mix50` — same ids as the primary; changes the soft target only. Keep as a KL
  ablation, not as a corpus definition.
- `_union_*` — add 8,001/8,892 residual noul ids that already exist in
  `residual_train_20000_jev`. Using a union file together with the residual 20K
  double-counts those ids; the two-primary composition gets the same coverage
  without a baked-in merge.

## Reproduction

```bash
# On the CPU host, from a code checkout
python scripts/corpus_inventory.py \
  --root data/benchmarks --repo-root . \
  --json /tmp/corpus-inventory.json \
  --deltas research/corpus-234-label-target-deltas.json

# On the training host, read-only: stream the script over stdin
ssh <training-host> 'cd <repo> && python3 - \
  --root data/benchmarks --repo-root .' < scripts/corpus_inventory.py \
  > /tmp/corpus-inventory-training-host.json
```

The script is single-process and stdlib-only apart from
`data.synthetic.family_to_typed_questions`, which it imports only when
`*_families.jsonl` is present. It classifies `base_probs*` as auxiliary
probability files rather than typed items.
