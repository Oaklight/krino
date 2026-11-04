# Shared-prefix parity on JevBench public

Scope: 231 items (sha `dcb90f87`), K ≤ 20 (`by_k_bucket` 2: 74, 3-5: 141,
6-20: 16), one H100 on rbh101, env `jev-fla` (fla GDN kernels), batch 1, warm.
Latency is per item over this suite. It is not a serving number.

| file | produced by | `config.raw_scoring` |
|---|---|---|
| `s2_*.json` | `eval_gate.py --raw-reranker --shared-prefix --no-order-sensitivity` | `shared_prefix`, `float32_two_token` |
| `o32_*.json` | `model/scripts/raw_per_option_f32.py --raw-reranker --no-order-sensitivity` | `per_option`, `float32_two_token` |
| `order/os_*.json`, `order/oo_*.json` | same two paths with order sensitivity on | as above |
| `predictions/*.pred.jsonl` | per-item predictions for the three paths below | — |
| `parity.txt` | `python model/scripts/compare_raw_parity.py` (re-runnable from `predictions/`) | — |

`po_bf16` is the stock per-option path (bf16 lm_head) from
`results/gate/qwen35_raw/paired`; its predictions are copied into `predictions/`.

## Accuracy by path

| model | per-option bf16 | per-option f32 | shared f32 | latency per-option f32 → shared (mean / median) |
|---|---|---|---|---|
| Qwen3.5-4B | 0.6883 | 0.6840 | 0.6883 | 1.08 s → 89 ms / 78 ms |
| Qwen3.5-4B-Base | 0.6840 | 0.6883 | 0.6883 | 1.08 s → 87 ms / 76 ms |
| Qwen3-Reranker-4B | 0.6407 | 0.6320 | 0.6364 | 0.69 s → 76 ms / 51 ms |

Reading it:

- per-option bf16 → per-option f32 changes only the readout precision. It
  flips 2-10 items; this is the noise floor.
- per-option f32 → shared f32 changes only the cache path. It moves accuracy by
  +1 choice item (Qwen3.5-4B, Qwen3-Reranker-4B) or 0 (Base), and flips 1-4
  predictions in total. With at most 4 discordant items McNemar cannot reach
  significance in any comparison (the exact two-sided p is discrete: 0.5 for a
  2-0 split, 1.0 otherwise here); the mean |Δp| of 0.003-0.007 is the quantity
  that carries the claim.
- The latency column compares the shipped path with `o32`, the per-option
  control with the same readout. The frozen `po_bf16` runs measured the same
  1.08 s (Qwen3.5) per item.
- Against the stock bf16 path, the shipped path (shared + f32 readout) differs
  in readout too: Qwen3-Reranker-4B moves 0.6407 → 0.6364 (−1 item).

## Other differences checked

- Option order: with order sensitivity on (157 choice/score items, Qwen3.5-4B),
  reversing the options changes 0 predictions on both paths, `mean_tv` is 0.0,
  and `accuracy_before == accuracy_after` (0.6943 shared, 0.6879 per-option).
  Raw scoring rates each option independently, and the shared path keeps that.
  The stock bf16 runs in `results/gate/qwen35_raw/` show 1-36 reversal flips
  (`changed_fraction` 0.013-0.229); those come from argmax ties broken by
  position under the bf16 readout.
- `accuracy_after` before this PR: `Prediction.label_index` read a score label
  as a position, so a reversed score item that kept its answer was counted
  wrong. 37 older artifacts carry that field (`results/gate/`,
  `model/experiments/residual/`); their `accuracy_after` is low by the number
  of score items answered correctly, while `changed_fraction` and `mean_tv` are
  correct. No gate used `accuracy_after`, and no research note cites it. Their
  reversed per-item predictions were not saved, so they are not regenerated.
- Clipping: `score_many` clips the document to its budget and keeps the
  reranker suffix; `_score_option` clips the whole body. The longest JevBench
  prompt body is 3,719 tokens against a budget of 8,144, so neither path clips
  any item here.
- Exact-0.5 tie: on Qwen3.5-4B-Base, both options of `original-adequacy-04-0`
  read (no, yes) logits (20.375, 21.375) through the bf16 lm_head (step 0.125
  at that magnitude) and tie at 0.5. Float32 gives (20.3755, 21.3550) and
  (20.3928, 21.3403): yes−no gaps of 0.980 and 0.948.
- Reproducibility: the six `s2`/`o32` JSONs were re-run to add `raw_scoring`.
  All 1,386 per-item predictions and every accuracy match the first run
  exactly; latency moved by up to 1.5% run to run (instruct 88 → 89 ms mean).
