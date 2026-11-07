# Abstention evaluation instrument — 2026-11-06

Issue #237 (Scope B reporting). The instrument is built by `scripts/build_abstention_eval.py`; the committed selection is `237-abstention-eval-selection.json`, SHA-256 of the ids `d3158efb28c04ebcda932270b78aebaac3f8274d98c87ab027692632f7b3b3e7`.

Status: the set is built and pre-registered, and the frozen-base reference line is run. The head line does not exist yet; the comparison is ready for it.

## What this instrument measures, and what it does not

A head can detect that a gold was removed from surface cues, so this measures detection of three specific constructions, not real-world abstention.

The three constructions are the existing canonical abstention edits (`none_of_these`, `out_of_scope`, `insufficient`) plus the paired builder's two evidence edits (`support_removed`, `entity_corrupted`). A score that separates them can be a template detector. The instrument is diagnostic, not a gate: no promote, kill or branch decision may cite it, and it does not replace the acceptance set.

## Population

The instrument is 5,800 pairs over 5,462 distinct donors, 11,262 rows. Each pair is one unanswerable variant and its answerable donor, with the donor's question, source and group kept.

- 5,287 pairs are existing canonical abstention rows paired with their `-abstain-` donors: `none_of_these` 1,830, `out_of_scope` 1,725, `insufficient` 1,732.
- 513 pairs are constructed to reach the target: `support_removed` 253, `entity_corrupted` 260.
- By type: 3,814 noul, 1,986 choice. Twenty-two sources; the largest are `multirc` (1,082 pairs), `contractnli` (859) and `mednli` (348).
- 1,744 `(source, group)` clusters. Every record in a listed cluster must be excluded from training, not only the emitted rows.

A donor is written once even when two variants use it, so the row count is pairs plus distinct donors, not twice the pairs. 338 pairs share their donor with another pair: 200 are the canonical donors that carry both an `insufficient` and an `out_of_scope` variant, and 138 are constructed donors that yield both evidence edits. Both variants are kept because they are different constructs; the duplicate-donor dependence is handled in the analysis, where variants that share a donor are one cluster (see the power section).

Selection is by whole cluster. Clusters are ordered by `sha256(source + NUL + group)`; eligible existing pairs are taken first and construction fills the remainder. The canonical artifact is the post-#260 build, SHA-256 `8eb3b2a4...`.

Disjointness is the binding constraint. The 928 clusters of the frozen held-out v1 (`model/experiments/canonical/heldout_6400.json`, ids SHA-256 `492ee0af...`, 6,424 ids) are removed before selection, as are their ids. The builder fails if the held-out ids are not in the canonical artifact, and it re-checks after selection that no emitted id or cluster is in the held-out set. 713 existing pairs fell to that exclusion (`none_of_these` 170, `out_of_scope` 275, `insufficient` 268). Construction is what replaces them.

## Pre-registered metrics

The primary comparison follows the existing convention in `model/experiments/residual/abstention_20k/` (`eval_mix.jsonl`, 885 rows): `AURC(p_answerable) < AURC(max_prob)`. AURC is mean risk over the coverage curve in `model/scripts/eval_abstention.py`, with strict counting: answering an unanswerable item scores it wrong even when the task head emits the gold label.

1. **Primary**: AURC of `p_answerable` and AURC of `max_prob`, reported together with their difference. Lower is better. The three borrowed strata (`none_of_these`, `out_of_scope`, `insufficient`) are the powered comparison and lead the report. The constructed pair (`support_removed`, `entity_corrupted`) is a different phenomenon and is reported descriptively, never blended into the primary number.
2. **Secondary, reported separately**: AUROC of each score, and threshold accuracy with a fixed 0.5 threshold (answer when the score is at least 0.5; abstaining on an answerable item and answering an unanswerable item both count as wrong). AUROC and threshold accuracy are never folded into the primary result.
3. **Coverage**: risk at 90% coverage, from the same risk-coverage curve.
4. **Per reason**: AURC and abstention recall at 90% coverage for all five reasons, with the constructed reasons marked descriptive. Recall is the share of a reason's items that fall below the global cutoff.

No composite score is defined. A result is reported as the four blocks above or not at all.

## Size and smallest detectable effect

At 5,800 pairs the instrument detects:

| comparison | method | smallest detectable effect |
|---|---|---:|
| accuracy, paired | McNemar, 32.4% discordance, alpha 0.05 two-sided, power 0.8 | 2.09 pp |
| AUROC difference | Hanley–McNeil at A=0.75, 5,800 positives and 5,800 negatives, two independent estimates | 0.018 |

The proposal's figures assumed independent rows; the pair is the unit, because donors and variants share a question. Those figures were 1.5 pp and 0.013 at 11,600 independent observations. These are the corrected figures, and they are the ones pre-registered.

The 338 shared-donor pairs make 5,800 pairs look like 5,462 independent units. The pre-registered sensitivity check reruns the paired test with one pair per donor (the deterministic first by variant id); at 5,462 units the MDE is 2.16 pp, so the conclusion does not move.

Per reason, the three borrowed strata are powered and the two constructed strata are descriptive:

| reason | pairs | accuracy MDE | AUROC MDE |
|---|---:|---:|---:|
| none_of_these | 1,830 | 3.73 pp | 0.032 |
| out_of_scope | 1,725 | 3.84 pp | 0.033 |
| insufficient | 1,732 | 3.83 pp | 0.033 |
| support_removed | 253 | 10.03 pp | 0.086 |
| entity_corrupted | 260 | 9.89 pp | 0.084 |

The 8,000-row floor is the minimum build; the builder fails below it and below the pair target.

## Frozen-base baseline

The frozen base has no answerable head, so `p_answerable` is undefined and a base score in that metric does not exist. The reference is the base's own confidence, `max_prob`, which is defined for any model. It is the external line, not a trained arm.

`max_prob` comes from the same production path as the arms' probabilities, `model/scripts/eval_gate.py --raw-reranker --shared-prefix --no-order-sensitivity --dtype float32`, on the backbone the evaluated arm uses. For this line that is `Qwen/Qwen3.5-4B` (manifest row 78, the control every gate cites; evidence `results/gate/qwen35_shared/s2_qwen35_4b_instruct.json`, predictions `shared_f32_qwen35_4b_instruct.pred.jsonl`). It is not the `-Base` checkpoint: that is a different model, and its matching accuracy is a coincidence. It is also not a separate scoring script: the AURC comparison must not compare two code paths as well as two quantities. `p_answerable` is null on that output because the frozen base has no answerable head. The row file is passed with `--data`.

A baseline is only a baseline relative to the model the arm was trained on, so the backbone is read from the arm's own config. If a future answerable arm is trained on another backbone, its baseline is that backbone and this line changes.

The baseline run is one forward pass over the 11,262 rows through that path, written beside the row file. The committed result is `237-abstention-eval-base.json`, summarised from `abstention-eval-11262-base.pred.jsonl` (SHA-256 `61f81eb9...`, 11,262 rows).

| metric | baseline |
|---|---:|
| AURC(`max_prob`) | 0.5447 |
| AUROC(`max_prob`) | 0.5285 |
| answered at threshold 0.5 | 9,071 (80.5%) |
| threshold accuracy | 0.3247 |
| answered accuracy at 90% coverage | 0.3860 |
| abstention capture at 90% coverage, `none_of_these` | 0.221 |
| `out_of_scope` / `insufficient` / `support_removed` / `entity_corrupted` | 0.000 / 0.000 / 0.004 / 0.015 |

The zeros are not a tie artifact. The 90% cutoff falls at 0.37980, one row sits exactly on it, and `max_prob` takes 11,038 distinct values; all 1,725 `out_of_scope` rows and all 1,732 `insufficient` rows sit at or above the cutoff. The mechanism is a small confidence shift that does not reach the bottom decile: `out_of_scope` has median `max_prob` 0.6428 against 0.6748 for its donors (1,026 of 1,725 below their donor), and `insufficient` 0.6589 against 0.6875 (953 of 1,732 below). `support_removed` and `entity_corrupted` show the same shape. The base is slightly less confident on borrowed evidence, but not enough to abstain. The check is committed as `tie_check` in the baseline JSON.

The pre-registered aggregate is the headline: AURC(`max_prob`) 0.5447 over all 11,262 rows. It is diluted by `none_of_these`, where removing the gold already moves confidence. On the 9,432 rows excluding that stratum, where the baseline has no usable signal, AURC is 0.4726, AUROC 0.458 (below chance), and only 0.38% of the 3,970 negatives fall below the 90% cutoff. That subset is the informative comparison, reported under the pre-registered per-reason block, not a replacement for the aggregate. Per-reason AURC is degenerate (a reason-only group has no positive rows and scores 1.0), so the per-reason numbers that carry information are capture and the paired distribution above.

The eval log's accuracy figures are not abstention numbers and must not be read as any part of this result. The 11,262 rows are 5,800 abstain variants plus 5,462 distinct donors. The 1,830 rows the accuracy metric skips are exactly the `none_of_these` stratum, which has no label inside its options; the other 3,970 abstain variants keep a valid label because that is the construction. The skip is structural, not a filter: the AURC, AUROC and threshold blocks above use every row, including the 1,830.

    python model/scripts/eval_abstention.py \
        --predictions abstention-eval-11262-base.pred.jsonl \
        --score-field max_prob --out research/237-abstention-eval-base.json

## Relationship to `eval_mix.jsonl`

The instrument **sits alongside** `model/experiments/residual/abstention_20k/eval_mix.jsonl`. It does not supersede it.

The two have different populations and different grouping: `eval_mix` is 885 rows (434 answerable, 451 unanswerable) drawn from the 20K residual corpus with base-id grouping; this instrument is 11,262 rows from whole canonical clusters, disjoint from the held-out v1. The answerable-head arms were trained and evaluated against the 885-row set, and no arm has been rerun here. A number from one set is therefore not comparable to a number from the other, and the two are not put in one table. When the answerable head is next reported, this instrument is the set to use, with the frozen-base `max_prob` line above; the old numbers stay with their own file and their own reference.

## Reproduction

    python scripts/build_abstention_eval.py \
        --canonical research/krino-canonical-243366.jsonl \
        --heldout model/experiments/canonical/heldout_6400.json \
        --out research/237-abstention-eval-selection.json \
        --jsonl-out <rows beside the canonical artifact>

Build: about 14 s on the compute host; the paired rows are large and stay beside the canonical artifact. The committed selection records the ids, the ids hash, the 1,744 clusters, and the composition counts; the rows are regenerated from the canonical artifact by the command above.

## Out of scope

No synthesis job is run here: the 513 constructed pairs are the minimum needed to reach the target, not a new corpus. The answerable-head loss design (weight, warm-up, forward pass) is #237 Scope B and is not proposed here.
