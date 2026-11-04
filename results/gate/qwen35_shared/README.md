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
  predictions in total. Three of these comparisons involve 1-3 discordant
  items, so McNemar cannot reach significance; the mean |Δp| of 0.003-0.007 is
  the quantity that carries the claim.
- Against the stock bf16 path, the shipped path (shared + f32 readout) differs
  in readout too: Qwen3-Reranker-4B moves 0.6407 → 0.6364 (−1 item).

## Other differences checked

- Option order: with order sensitivity on (157 choice/score items, Qwen3.5-4B),
  reversing the options changes 0 predictions on both paths (`changed_fraction`
  0.0, `mean_tv` 0.0). Raw scoring rates each option independently, and the
  shared path keeps that property.
- Clipping: `score_many` clips the document to its budget and keeps the
  reranker suffix; `_score_option` clips the whole body. The longest JevBench
  prompt body is 3,719 tokens against a budget of 8,144, so neither path clips
  any item here.
- Exact-0.5 tie: on Qwen3.5-4B-Base, both options of `original-adequacy-04-0`
  read (no, yes) logits (20.375, 21.375) through the bf16 lm_head (step 0.125
  at that magnitude) and tie at 0.5. Float32 gives (20.3755, 21.3550) and
  (20.3928, 21.3403): yes−no gaps of 0.980 and 0.948.
- Reproducibility: the six `s2`/`o32` JSONs were regenerated after adding
  `raw_scoring`; all 1,386 per-item predictions match the first run exactly.
