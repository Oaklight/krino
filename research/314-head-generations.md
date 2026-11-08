# Trained-head generations

**Date:** 2026-11-08
**Issue:** [#314](https://github.com/Oaklight/krino/issues/314)
**Status:** Mapping approved; recorded here so every `head_version` label has one source.

`v3` and `v4` are the project's own words in `research/24-v3-closeout.md`; `v1` and `v2` are derived labels for the project's Round 1 and Round 2 lines and do not appear literally anywhere in the repository at the commit that introduced this note.

## Derivation

`research/04-model-replication.md` line 424 contrasts "Round 1 (benchmarks only) vs Round 2 (+ synthetic)", line 390 calls Round 2 the in-progress multi-task line, and `research/10-architecture-sweep-r2.md` is titled "Round 2 architecture sweep". `research/24-v3-closeout.md` names v3 as the residual line and says Qwen3.5-4B "starts v4". The mapping Round 1 -> v1 and Round 2 -> v2 is the assignment that puts four generations in the order the later notes use.

| generation | line | first documented | evidence |
|---|---|---|---|
| v1 | Round 1: single-task Banking77 heads, then multi-task over 19 benchmarks; 130K train / 8.9K eval, 20 epochs, frozen backbone, head ranks 64/128 | Sep 19, 2026 | `research/04-model-replication.md:11` (Sep 19 initial), `:200-217` (Step 2 single-task protocol), `:315-319` (Round 1 setup), `:437-452` (R1 results and six HF links), `:424` (Round 1 vs Round 2); the six result files were added 2026-09-25 in `17c4826` |
| v2 | Round 2: 182,262 train / 13,444 eval, 26 benchmarks + synthetic; `r2_reranker0.6b.yaml` / `r2_reranker4b.yaml`; readout last/mean4/weighted; LoRA none/last8/all | documented 2026-10-28; the runs precede the writeup | `research/04-model-replication.md:390` and `:424`, `research/10-architecture-sweep-r2.md:1-30`, `research/09-architecture-exploration.md:3,9-25` (dated 2026-10-28, "The Round 2 model", `r2_reranker0.6b.yaml`), `research/19-native-head-blend.md:14` (4B epoch-10 `arch_4b_loraAll_32k_s123`) |
| v3 | Residual line on `Qwen/Qwen3-Reranker-4B`: zero-gated `log p_base + alpha*head`, marker readout, 5K -> 20K -> 30K items; stable cc 0.43-0.46 | 2026-11-03 | `research/24-v3-closeout.md:1-12` (v3 is the residual line; the 0.50 cc bar was missed), `model/experiments/residual/manifest.jsonl` rows 1-74 dated 2026-11-03..2026-11-04; rows 76-77 are #207's noul-union arms and are explicitly not v3 (`research/24-v3-closeout.md:50-53`); row 75 is v4 baseline data |
| v4 | `Qwen/Qwen3.5-4B` line: raw control cc 0.5187, then `v4_listwise`, `v4_two_stage_pairwise`, `v4_combined_top4`, LR/KL scans, `fullarm` | 2026-11-04 (raw baselines) / 2026-11-05 (control registered) | `research/24-v3-closeout.md:10-12` ("starts v4"), `:150-155` ("The v4 starting point"), manifest row 75 `qwen35_raw_baselines` (2026-11-04), row 78 `v4_control_qwen35_4b_raw` (2026-11-05), rows 81-96 and 116-119 |

## The six September head files

All six are v1. `research/04-model-replication.md:443-452` lists exactly these six HuggingFace models under "Round 1: Multi-Task Training Results (Sep 24-25, 2026)", and `17c4826` added the six JevBench evals on 2026-09-25.

| file | `run_id` (the HF artifact the eval loaded) | `backbone.model_id` | `head_version` |
|---|---|---|---|
| `results/jevbench/krino-ettin-150m-heads.json` | `oaklight/krino-ettin-150m-heads` | `cross-encoder/ettin-reranker-150m-v1` | v1 |
| `results/jevbench/krino-modernbert-base-heads.json` | `oaklight/krino-modernbert-base-heads` | `answerdotai/ModernBERT-base` | v1 |
| `results/jevbench/krino-qwen3-0.6b-heads.json` | `oaklight/krino-qwen3-0.6b-heads` | `Qwen/Qwen3-0.6B` | v1 |
| `results/jevbench/krino-qwen3-reranker-0.6b-heads.json` | `oaklight/krino-qwen3-reranker-0.6b-heads` | `Qwen/Qwen3-Reranker-0.6B` | v1 |
| `results/jevbench/krino-qwen3-reranker-4b-heads.json` | `oaklight/krino-qwen3-reranker-4b-heads` | `Qwen/Qwen3-Reranker-4B` | v1 |
| `results/jevbench/krino-qwen3.5-4b-heads.json` | `oaklight/krino-qwen3.5-4b-heads` | `Qwen/Qwen3.5-4B-Base` | v1, not v4 |

`krino-qwen3.5-4b-heads.json` is a September Round-1 head on a Qwen3.5-4B-Base backbone. v4 is the line that *starts* at Qwen3.5-4B on 2026-11-04/05 with the raw baselines and control; it does not inherit this file. Labelling the file v4 because of its backbone would repeat the error #314 exists to fix.

## What the fields mean

`run_id` in these six files is the published model artifact that `model/scripts/eval_jevbench.py` loaded, because that is the only run identifier recorded in the repository. The training run that produced each artifact is not recorded here, and `code_commit` is therefore `null` for all six. The file's own git history is not used as a substitute: it records when the result JSON was committed, not when the head was trained.

`backbone.model_id` comes from the committed config or README that names the backbone (`model/configs/multitask_ettin150m.yaml:10`, `model/configs/multitask_qwen35_4b.yaml:12`, `README_en.md:55`, `model/configs/r2_reranker4b.yaml:16`).

## Worked example

`results/jevbench/krino-qwen3-reranker-4b-heads.json` is 99/231 = 0.4286 raw accuracy, added 2026-09-25, and is v1. `research/24-v3-closeout.md` B2 is cc 0.4603 and is v3. `head_version` separates the generations; it does not make the numbers comparable, because one is raw accuracy and the other is chance-corrected cc.

## Register convention

`model/experiments/residual/manifest.jsonl` is append-only. Rows that predate this note do not carry `head_version` and are not rewritten. Every new row appended to the register carries `"head_version"` set to the generation of the line that produced it. The field is always spelled `head_version`; it is never shortened to `version`, because the manifest already uses `(v1, ...)` for a **corpus** hash in `fullarm_seed0/1/2` (rows 116-118), which is a different thing.

## Caveats

- `v1`/`v2` are derived labels, not literals. `v3`/`v4` are literal in `research/24-v3-closeout.md`.
- `research/12-marker-readout-result.md` (2026-11-03) is a screen (S1b/S2 marker readout) that informs v3, not a separate generation.
- Generations overlap in time: v3 residual arms and v4 arms were both live around 2026-11-04..06; the boundary is the line and backbone, not the calendar.
