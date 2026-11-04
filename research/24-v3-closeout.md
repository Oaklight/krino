# v3 closeout: the residual line on Qwen3-Reranker-4B

v3 asked one question: does a zero-gated residual head on a frozen
Qwen3-Reranker-4B reach the local acceptance proxy (JevBench public 231,
chance-corrected accuracy ≥ 0.50) inside 300 ms? It does not. The stable level
is 0.43–0.46, and none of the levers tried moved it outside seed noise. The
latency half was met (133 ms). The next backbone, Qwen3.5-4B, scores 0.5187
zero-shot, above every trained v3 arm; it changes the tokenizer, the cache
layout and every baseline, so it starts v4 rather than continuing v3.

Every number below is JevBench public 231 (items sha `dcb90f87…`) unless it says
otherwise. One standard error of accuracy on 231 items is about 3.2 pp; the
hard tier alone (111 items) needs about 16 pp for 80% power (`research/23`).
Arm rows, commands and pre-registered gates are in
`model/experiments/residual/manifest.jsonl`; per-arm results are the
`jevbench*.json` and `history.json` files beside it and in `results/gate/`.

## What was measured

| arm | change against its control | cc | hard | verdict |
|---|---|---:|---:|---|
| raw Qwen3-Reranker-4B | baseline | 0.4459 | 0.4595 | control |
| B1 | pooled residual head | 0.4395 | 0.423 | kill |
| B2 | marker readout residual | 0.4603 | 0.4595 | promote |
| confirm_frozen | B2 at seeds 43/44 | 0.4690 / 0.4545 | 0.468 / 0.450 | confirmed, 3-seed mean 0.4613 |
| confirm_lora | LoRA last 8 blocks, seeds 43/44 | 0.4459 / 0.4394 | 0.459 / 0.450 | reject |
| D20K | 5K → 20K items, same distribution | 0.4603 | 0.4775 | kill (Δ 0.0000 vs B2) |
| H_alpha_lr / H_ce / H_rank | alpha lr 0.03 / CE 0.5 / rank 64 | 0.4369 / 0.4214 / 0.4517 | — | kill |
| H_kl → confirm_kl_5k | KL 0.5 | 0.4747 → 0.4538 / 0.4545 | 0.5045 → 0.4505 | reject (seed-42 false positive) |
| band20k_u50 | train only where p(gold) ≤ 0.5 | 0.4549 | 0.4685 | kill |
| teacher20k → confirm | Jev soft targets | 0.4892 → 0.4488 / 0.4747 | 0.4865 → 0.460 / 0.478 | reject (seed-42 false positive) |
| teacher30k | 30K Jev, band-selected | 0.4300 | 0.4324 | kill |
| backbone_instruct / _base | Qwen3-4B / Qwen3-4B-Base under the same recipe | mean 0.4152 / 0.4270 | 0.4459 / 0.4369 | reject, keep the reranker |
| noul_marker_30k (diagnostic) | noul through the shared marker head | Δ −2.13 pp | Δ −3.60 pp | negative |
| noul_marker_d20k | same at 20K, paired bucketed control | 0.3990 vs 0.4336 | 0.4324 vs 0.4414 | reject |
| length_diag | eval at 4096 instead of 2048 | D20K +1.06 pp, teacher20k −0.32 pp | +2.70 / 0.00 pp | ambiguous, no arm |
| abstention20k | answerable Bernoulli head | AURC 0.4978 / 0.4611 / 0.4948 vs max-prob 0.6315 / 0.5872 / 0.6401 | — | confirmed (3 seeds) |

The two noul label-repair arms (`noul_union_marker`, `noul_union_pooled`) were
started by the data line and are gated in their own manifest rows; they belong
to #207, not to this note.

## Why it stopped at 0.46

**Not under-training.** Choice validation cc is flat at 0.51–0.53 from epoch 1
in every arm (D20K 0.514/0.515/0.515/0.513; teacher20k 0.519/0.528/0.522/0.519),
while noul keeps rising. The 20-epoch confirm arms stopped at 3–4 epochs on
patience; there was nothing left to fit.

**The head only helps noul.** On D20K, base / head-alone / fused accuracy is
0.612 / 0.129 / 0.597 for choice, 0.622 / 0.500 / 0.676 for noul and
0.778 / 0.167 / 0.778 for score. The trained α is mis-signed for choice and
score (choice α −0.234 scores 0.604; α = 0 scores 0.612; the best α, +0.35,
scores 0.619). A pointwise marker head scores each option alone and cannot
compare options; the base reranker already does pointwise relevance better.

**Volume and labels are not the lever on this backbone.** 5K → 20K moved cc by
0.0000; Jev soft targets gave one seed at 0.4892 and two at 0.4488 / 0.4747.
The 30K Jev labels agree with gold on only 74% of items, and least on the hard
band where they would matter.

**Single seeds lied twice.** KL 0.5 (0.4747) and teacher20k (0.4892) both
cleared a promote line at seed 42 and both failed confirmation. Any v4 promote
needs two seeds before it is read.

**Path drift is about 1 SE.** Bucketed and unbucketed training on identical
data land about 3 pp apart on JevBench (scorer cosine 0.12, α signs flipped);
the bucketed eval path moves a checkpoint by 1–2 items. Comparisons are only
read against a same-code control.

## What carries forward

- Marker readout (`research/12`), the zero-gated residual
  `log p_base + α·head` that starts exactly at the base, and the
  answerable head (confirmed on three seeds).
- `option_batch` along the batch dimension and fixed shape buckets: 1.29 s →
  133 ms mean, 106 ms median (`research/13-throughput-levers.md`); the buckets
  are also a 7.5× training speedup. `--train-safe-buckets` is kept off: 9.65
  against 12.68 items/s at the same validation cc.
- `eval_gate.py --raw-reranker` (any checkpoint's own yes/no logits) and the
  Qwen3.5 text/multimodal loaders.
- The data work under #207 (open-dataset long-context choice, tool-call
  selection, abstention sets, label repair) is backbone-independent. Only the
  `base_probs_*.jsonl` caches are tied to Qwen3-Reranker-4B and must be
  recomputed for a new backbone.

## The v4 starting point

Zero-shot, raw yes/no readout, no training, same eval code; the Qwen3-Reranker
control reproduces 0.6407 / 0.4459 exactly. Files:
`results/gate/qwen35_raw/` and `results/gate/qwen35_raw/paired/`.

| model | acc | cc | hard | original | choice | noul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Reranker-4B | 0.6407 | 0.4459 | 0.4595 | 0.6806 | 0.6187 | 0.6351 |
| Qwen3.5-4B (instruct) | 0.6883 | 0.5187 | 0.5045 | 0.7639 | 0.6835 | 0.6757 |
| Qwen3.5-4B-Base | 0.6840 | 0.5069 | 0.5045 | 0.7500 | 0.6906 | 0.6486 |
| Prism-Qwen3.5-Reranker-4B (community) | 0.6753 | 0.4973 | 0.4775 | 0.7639 | 0.6763 | 0.6622 |

2B and 0.8B (base, instruct and Prism) all score below the Qwen3 control.

Paired exact McNemar against Qwen3-Reranker-4B: instruct 32 vs 21 discordant,
p = 0.169; Base 33 vs 23, p = 0.229; Prism 33 vs 25, p = 0.358; instruct vs
Base 17 vs 16, p = 1.0. The gain is consistent in direction (hard choice net
+5, original choice net +4) but not significant on 231 items.

Latency is open. With `flash-linear-attention` and `causal-conv1d` installed,
a warm Qwen3.5-4B forward is 60 ms at 1×2048 and 117 ms at 8×512 (133 and
217 ms on the reference kernels), but the raw path is still 1.2 s per item
because it runs one unbatched forward per option. The hybrid GDN cache skips
the shared-prefix option batching the 133 ms path depends on; that port comes
before any Qwen3.5 latency claim.
