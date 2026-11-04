# Krino v3: Architecture, Data, Pipeline, and Infrastructure Redesign

**Date:** 2026-11-02
**Status:** Proposal for review. No code changes yet.
**Supersedes for planning purposes:** `19`, `73`, `106` roadmaps; `93`; `97`; `146` scope.

## 0. Why v3

Three measured results block the current line.

1. **Choice collapses with option count.** Same 300 items, gold kept, distractors added,
   frozen 0.6B pooled readout, epoch 5: 0.677 at K=4, 0.217 at K=256 (`#166`).
   Training only ever sees K in 2..80, and 92.7% of eval items have K <= 5.
   This is an architecture plus data problem, not a tuning problem.
2. **Calibration is arm-specific, not method-specific.** Same accuracy to 0.13 pp,
   score ECE after per-type temperature is 0.0250 on `arch_4b_loraAll_weighted` and
   0.0811 on `arch_4b_loraAll_32k` (`#172` comments). A single per-type temperature
   cannot be the protocol.
3. **The stated goal is undefined.** "Brier >= 0.85 on the score primitive" has no
   reading under which it is meaningful (`#178`). The standard quantity is the Brier
   skill score against the label marginal; measured BSS is 0.286.

A fourth, structural problem: parameter spend is not buying accuracy. On the same
19 benchmarks and 20 epochs, 150M Ettin scores 57.9, 0.6B Qwen3 57.4, Reranker-0.6B
58.1, and Reranker-4B 65.5 (`06-head-size-sweep.md`). A 27x parameter increase buys
7.6 pp. The 4B/8B line is the wrong place to spend the next month.

## 1. Current state

### Measured

| Item | Value | Source |
|---|---:|---|
| Best overall accuracy | 65.94 | `#172` comment, weighted 4B arm |
| Same, 32k last arm | 65.81 | `#172` comment |
| Score ECE after T (weighted) | 0.0250 | `#172` comment |
| Score ECE after T (32k) | 0.0811 | `#172` comment |
| 0.6B frozen / LoRA weighted | 49.11 / 56.17 | `09-architecture-exploration.md` |
| 4B frozen / LoRA last | 55.59 / 61.87 (ep4) | `09-architecture-exploration.md` |
| Train items / eval items | 182,262 / 13,444 | `09-architecture-exploration.md` |
| Synthetic corpus | ~154K, 23 domains | `r2_reranker0.6b.yaml` |
| Score supply | 27K of 167K available | `#142` (closed) |
| Branches in flight | 20+ | `git branch -a` |

### Implemented

- `DecisionModel`: frozen backbone, LoRA, `LayerMixer` (`last`, `mean_last_n`,
  `weighted`, `layer`, `fuse_pair`), optional second backbone, optional MLP projector.
- Heads: `NoulHead`, `AttentionHead`, `ChoiceHead`, `ScoreHead`, `UnifiedHead`.
  Option scoring reads **pooled** option vectors; options interact only through the
  shared context and the optional `rival_aware` path.
- Losses: Brier, MMCE, focal, label smoothing, RPS.
- Calibration: metrics module and `#167` harness; per-type temperature.
- Data: 26 benchmark loaders, synthetic pipeline, `TypedQuestion`, `teacher_probs`.

### Not implemented on the trained path

- Option-marker readout. `#144` is closed and `feature/option-marker-readout` exists,
  but `AttentionHead` still takes `option_hidden` pooled vectors. Branch state needs
  confirmation before this is called available.
- Abstention. No `answerable` head, no `none` option, no insufficient-evidence set.
- Schema-level holdout. Splits are item-level.
- K-augmentation. No training items above K=80.

## 2. Architecture (v3-A)

### Shape

```
state + instructions + option_i  ->  encoder  ->  marker_i hidden  ->  option stack
option stack  ->  2 cross-option layers  ->  heads
```

### Decisions

**A1. Encoder-first, 150-600M.**
Default `Ettin-400m`. `Ettin-150m` as the cost floor. `Qwen3-Reranker-0.6B` as the
decoder reference arm. Retire 4B and 8B from the main line; keep 4B as an upper-bound
reference only. Rationale: 150M matches 0.6B; 4B buys 7.6 pp for 27x parameters;
the causal path also pays a vocabulary-logit cost that the task never uses
(`09`, the 19.9 GB allocation).

**A2. Per-option marker readout.**
Each option is encoded with the same state and instructions, followed by a marker
token. Read the marker position. Options do not share a pooled vector. This is the
cross-encoder form and is well-defined for any K.
Train with per-option sequences; pack at inference.
Exit criterion: K=256 accuracy within 10 pp of K=4 on the K-sweep.

**A3. Cross-option attention, 2 layers.**
Option embeddings attend to each other after readout. This replaces the bi-encoder
weakness named in `#144` and makes `rival_aware` the default rather than a flag.
Parameter cost is O(K^2) attention only, not O(K^2) backbone passes.

**A4. Four heads.**

| Head | Output | Loss |
|---|---|---|
| `answerable` | Bernoulli | BCE or KL to teacher sufficiency |
| `noul` | Bernoulli | BCE + KL |
| `choice` | softmax over K + `none` | KL to teacher distribution |
| `score` | cumulative-link over L levels | ordinal KL, full distribution |

The `choice` distribution and `answerable` combine as a mixture:
`p(abstain) = 1 - p(answerable)`, `p(option_i) = p(answerable) * p(option_i | answerable)`.
Do not put `none` in the same softmax as the options.

**A5. Calibration protocol.**
Temperature per `(question type, option-count bucket, level count)`, not per type.
Buckets: K in {2, 3-5, 6-20, 21-80, 81-256}; levels in {2, 3-5, 6-10}.
Fit on one half of a distribution-shifted split, score on the other.
Fallback to a monotone logit map only if the bucketed temperature misses the bar.
Report the arm name with every calibration number.

**A6. Drop from the default.**
MLP projector (loses 3.2-5.4 pp at matched LR), second backbone (no evidence),
`fuse_pair` (untested after the LR correction). Keep all three as ablation flags.

## 3. Data (v3-D)

### P0 fixes, before any new data

| Issue | Problem | Action |
|---|---|---|
| `#147` | `load_drop` emits constant labels and prints gold in the instruction | fix before retraining |
| `#140` | duplicated `_family_idx` collides ids across families | fix |
| `#121` | `sampling_temperature` is the reciprocal of the cited alpha | fix |
| `#176` | stsb withheld test labels enter as gold level 0 | fixed on `fix/stsb-withheld-labels`; verify merged |

### D1. Schema-level split

Hold out schemas, not items. A schema is `(instructions, label set)` after
normalisation. Report:
- seen-schema accuracy,
- held-out-schema accuracy,
- held-out-domain accuracy.

Without this, "general decision model" cannot be measured.

### D2. K-augmentation

Draw K from log-uniform 2..256. Keep the gold option; fill distractors from other
items in the same domain. No new labels. This is the proposed fix in `#166`.
Also cap per-item option memory: the `fix/k-augment-long-option-memory` branch
already targets this.

### D3. Counterfactual suite

| Variant | Construction | Expected label |
|---|---|---|
| evidence dropped | remove the supporting span | `insufficient` |
| gold removed | drop the correct option | `none` |
| distractor swap | replace support with a near-miss span | gold or `insufficient` |
| negation | negate the condition | flipped or `insufficient` |
| paraphrase | rewrite state, keep label | unchanged |
| option permutation | reorder criteria | unchanged distribution |

### D4. Abstention sets

Three explicit sets:
- `insufficient`: state present, evidence removed or weakened.
- `out_of_scope`: state from a domain not in the schema's domain.
- `none_of_these`: gold option removed from the set.

Each item carries `answerable: bool` and a `reason` field.

### D5. Distillation

- Primary target: teacher soft distribution.
- Selective distillation: query the teacher only where it beats the current student
  or where the student is uncertain. The Banking77 result (150M 95.2 vs Jev 75.0)
  shows uniform distillation would pull the student down on that distribution.
- Multi-teacher: keep `teacher_probs` namespaced by teacher. Teacher disagreement
  is a hard-example score and an abstention candidate.
- Recalibrate the teacher before distilling if its ECE is above 0.05.

### D6. Sampling

Stratify by `(option-count bucket, evidence-length bucket, domain)`. Random sampling
erases the tail, which is the part a general model is judged on.

### D7. Target size

| Axis | Current | v3 target |
|---|---:|---:|
| items | 182K | 1-3M |
| distinct schemas | unknown, low | 3K-10K |
| tokens | ~0.3B | ~6B |
| long-evidence items (>8K) | few | 300K-1M |

## 4. Training pipeline (v3-P)

| Stage | Objective | Data | Exit gate |
|---|---|---|---|
| P0 backbone | frozen probe | 19 benchmarks + K-sweep | best 150-600M encoder by acc/FLOP |
| P1 interface | distribution matching | short, schema-diverse, K-aug, counterfactuals | K=256 within 10 pp of K=4; held-out-schema acc reported |
| P2 length | same, longer | 2K -> 8K -> 32K + selector | long-evidence within 5 pp of short |
| P3 precision | hard negatives, abstention, calibration | insufficient/out-of-scope sets, drift split | score ECE < 0.05 per bucket; abstention beats max-prob |
| P4 policy | escalation | deployment distribution + utility labels | utility gain over single model |

Rules:
- Do not start P2 before P1 clears the K gate. Length amplifies a broken readout.
- Do not start P3 before P2. Calibration on truncated evidence is not calibration.
- Freeze the eval suite at P0. Changing the suite mid-stage makes the gates meaningless.

## 5. Infrastructure (v3-I)

**I1. Encoder training path.**
No vocabulary logits. Memory scales with sequence length, not `seq x vocab`.
This removes the 19.9 GB / 72 GB allocations in `09` and the failure mode behind `#161`.

**I2. Config schema.**
Each config names `stage`, `backbone`, `readout`, `k_distribution`, `loss_weights`,
`calibration_buckets`, and `data_mix`. No implicit defaults that differ from the
committed config; `#125` and the 4B LR incident both came from that.

**I3. Artifact naming.**
`{stage}_{backbone}_{readout}_{length}_{seed}`. Every eval and calibration report
carries the arm name. `#172` needed three rounds to establish that two arms with the
same accuracy differ 3x on score ECE.

**I4. Eval harness as a gate.**
One command, one JSON, these fields: accuracy, ECE, BSS, QWK, RPS, MAE, order
sensitivity, per-K-bucket accuracy, risk-coverage, per-type and per-schema-split
breakdowns. `#143` is the spec; `#178` decides the BSS threshold.

**I5. Splits.**
Seeded, cluster-disjoint (source group disjoint), shared across arms. Store the split
hash in the run artifact.

**I6. Branch consolidation.**
Twenty open branches is an infrastructure problem. Merge or delete before P1:
`feature/option-marker-readout`, `fix/k-augment-long-option-memory`,
`fix/per-scale-temperature`, `fix/stsb-withheld-labels`, `feature/eval-score-metrics-rework`
are on the v3 path; the rest should be triaged against `main`.

## 6. Issue triage

Legend: **K** keep, **U** update body, **R** rewrite scope, **X** close not planned.

| # | Title (short) | Action | Reason |
|---|---|---|---|
| 178 | Brier >= 0.85 undefined | U | decide BSS; restate threshold |
| 172 | score calibration fails at 4B | U | both arms resolved; per-scale is the open question |
| 166 | choice collapses with option count | K | core v3 work item |
| 161 | 8B LoRA OOM | X | 8B retired from v3 |
| 156 | generate score questions | K | score supply is the thin primitive |
| 155 | TREC-COVID score source | K | score diversity |
| 154 | DICES score source | K | score diversity |
| 153 | ASAP 2.0 score source | K | score diversity |
| 152 | Open-Jev CC0 score items | K | score diversity |
| 151 | FineWeb-Edu score loader | K | score diversity |
| 147 | load_drop constant labels | K | P0 data bug |
| 146 | layer sweep / fusion / two-backbone | R | narrow to encoder last vs weighted; drop two-backbone |
| 145 | cumulative-link score head | K | promoted to v3 architecture |
| 143 | calibration/order metrics | K | v3 eval gate |
| 140 | duplicate `_family_idx` | K | P0 data bug |
| 121 | sampling temperature reciprocal | K | P0 data bug |
| 106 | epic: data pipeline | R | rewrite as v3 data plan |
| 104 | noul-to-choice augmentation | K | feeds K-aug |
| 97 | data mixing ratios | R | rewrite as stratified schema/K/length sampling |
| 95 | multi-task optimization | K | keep, low priority; uniform LR won so far |
| 93 | MLP projector + per-component LR | X | superseded by the 2026-10-14 correction |
| 76 | CLM-8B literature | X | literature note, done |
| 74 | routing/calibration literature | X | literature note, done |
| 73 | epic: training pipeline | R | rewrite as v3 stage gates |
| 63 | web dashboard | X | not on the research path |
| 60 | hybrid encoder-decoder | U | reframe as the P4 escalation model |
| 59 | compact <10M model | U | floor is 150M for general; <10M only fixed schema |
| 23 | calibration training / RLCD | X | losses shipped; superseded by the P3 design |
| 19 | epic: decision model replication | R | rewrite as the v3 master roadmap |

Counts: keep 14, update 5, rewrite 4, close 6.

## 7. Execution: fast path

The staged plan above assumed compute was the constraint. It is not. With
gpu-host-a at our discretion, the constraints are:

1. data quality and throughput,
2. implementation of marker readout, K-augmentation, abstention, and the
   cumulative-link head,
3. decision latency between runs.

GPU hours are a rounding error at this scale. A 12-arm screen on 150M-0.6B
frozen backbones is roughly 12-15 GPU-hours. A 4-arm LoRA confirmation is
roughly 30 GPU-hours. A 400M run at 1M items and 2K tokens is roughly
30 GPU-hours for 3 epochs. None of these need scheduling.

### Principles

1. Freeze the eval suite and every gate on day 0. No gate changes after that.
2. Screen wide and short; confirm narrow and long.
3. Every run logs effective LR, config hash, data hash, and arm name. No CLI
   overrides. The 2026-10-14 LR confound cost weeks; parallel runs multiply
   that risk.
4. Pre-register kill criteria. A run that misses its gate is killed, not tuned.
5. Data generation runs from day 0, in parallel with training.

### Parallel lanes

| Lane | Work | Starts | Blocking? |
|---|---|---|---|
| L1 data QA | `#147`, `#140`, `#121`, stsb merge, schema inventory and split | day 0 | yes, for all training |
| L2 harness | eval gate, BSS, per-bucket calibration, K-sweep, risk-coverage | day 0 | yes, for all decisions |
| L3 screen | 12 arms: 3 backbones x 2 readouts x K-aug on/off | day 0 | no |
| L4 generation | counterfactuals, abstention sets, teacher labels | day 0 | no, feeds day 5 |
| L5 engineering | marker readout, K-aug, `answerable`, cumulative-link | day 0 | partly; screen can start pooled |
| L6 reference | one 4B arm, one 0.6B decoder arm | day 2 | no |

L1 and L2 are the real critical path. L3 can start on the existing 182K corpus
with the pooled readout while L5 finishes the marker path.

### Screen matrix (day 0-2)

| | pooled | marker |
|---|---|---|
| Ettin-150M, K-aug off | S1 | S2 |
| Ettin-150M, K-aug on | S3 | S4 |
| Ettin-400M, K-aug off | S5 | S6 |
| Ettin-400M, K-aug on | S7 | S8 |
| Reranker-0.6B, K-aug off | S9 | S10 |
| Reranker-0.6B, K-aug on | S11 | S12 |

5 epochs, frozen, 2K tokens, seed 42. Report: accuracy, K-sweep at
{4, 16, 64, 256}, seen vs held-out schema, ECE, seconds per epoch.

Kill rule: an arm is out if its K=256 gap exceeds 15 pp and a sibling cell with
the same backbone closes it. Backbone is out if it loses to a smaller backbone
on accuracy per FLOP.

### Confirm (day 3-4)

Four arms: winner readout x {frozen, LoRA}, 2 seeds, 20 epochs. This is the
first point where a number is allowed into the goal table.

### Scale (day 5-8)

| Day | Action |
|---|---|
| 5 | 500K items ready; K-aug at 2..256; counterfactual suite |
| 5-6 | 400M winner, 1M items, 2K, 3 epochs, 2 seeds |
| 6-7 | length arm at 8K, same data, selector stub |
| 7-8 | abstention sets and `answerable` head; cumulative-link score head |

### Precision and report (day 9-12)

| Day | Action |
|---|---|
| 9 | hard negatives; per-bucket temperature; BSS and risk-coverage |
| 10 | held-out-schema and held-out-domain evaluation |
| 11 | 3-seed final arm; 4B reference on the same suite |
| 12 | goal table, eval JSON, decision on the 4B line |

### Cautions

- Do not run more arms than the harness can score. A run without a gate result
  is wasted compute.
- Do not compare arms across different data hashes. L4 changes the data while
  L3 is running; record the hash.
- Do not skip the 3-seed final check. Seed noise is 0.4 pp at 0.6B and the
  differences we care about are 1-3 pp.
- Do not let branch sprawl grow. L5 has four code changes in flight; merge to
  one integration branch before day 5.

## 8. Open questions

- Does marker readout close the K gap at 150M, or does it need 400M?
- Is held-out-schema accuracy driven by backbone size or by schema count in training?
- Does selective distillation beat uniform distillation at equal teacher budget?
- Is `answerable` learnable from constructed negatives, or does it need teacher
  sufficiency labels?
- What is the per-decision FLOP budget that makes the 150-600M range the right one?
