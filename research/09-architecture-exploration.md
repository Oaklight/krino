# Architecture Exploration: Readout Layer, Layer Fusion, Multi-Backbone, Head Width

**Date:** 2026-10-28
**Branch:** `experiment/arch-explore-readout`
**Status:** Exploratory sweep on the 0.6B backbone; results below are single-seed unless noted.

## Question

The Round 2 model reads the decision heads from the final hidden state of a
frozen or LoRA-adapted Qwen3-Reranker. Four architecture questions were open:

1. Is the final layer the best place to read a classification-style decision?
2. Does fusing two layers beat a learned softmax mixture over all layers?
3. Does a second backbone add anything over a single 0.6B model?
4. Are the decision heads undersized for this data volume?

## Setup

All arms use `model/configs/r2_reranker0.6b.yaml` unless stated, overridden
with `--epochs 5` to match the existing architecture table. The config trains
type-balanced heads on 26 benchmarks plus the synthetic corpus, with
`epoch_size=15000`, `accumulation_steps=8`, `lr=1e-3`, `rank=64`,
`shared_attention=true`, `seed=42`. Every new arm uses `--max-length 2048`;
the 4B arms use `model/configs/r2_reranker4b_bb4.yaml` (`lr=3e-4`, `rank=128`,
`batch_backbone=4`). The 4B config is a copy of `r2_reranker4b.yaml` that
differs only in batch size: a 16-context noul batch through both a 4B and a
0.6B backbone peaks at 36.7 GB at 2,048 tokens, which is too close to the
40 GB card once training overhead is added. Four-item batching keeps the
sequence length identical across the 4B group; all 4B arms, including the
4B-alone control, use it.

The first batch evaluates every epoch and therefore contributes a full
per-epoch curve. Shared-cluster contention stretched later arms, so the
remaining arms use `--eval-every 5`: epochs 1-4 log only training, and epoch
5 (always evaluated) reports the final `train_loss`, `eval_loss` and
`eval_acc` in the same shape. Final-epoch comparisons are unaffected; only
the intermediate curve is missing for those arms.

The 2,048-token cap is not the config default. The first launch left
`max_length` at the auto-detected 16,384 and OOMed on the 40 GB A100s. The
immediate cause is that the causal backbone still computes vocabulary logits
over the full padded sequence although training only needs hidden states: a
four-option encode of long documents asks for `4 x 16,384 x 151,936 x 2`
bytes, or 19.9 GB, which is exactly the allocation that failed
(`arch_explore_layer16.m16384_oom.log`). The batched noul path
(`_compute_standard_noul_batch`, up to `batch_backbone=16` contexts in one
forward) is worse: a controlled replay of 16 long contexts at 16,384 tried to
allocate 72 GB for the same reason. Capping the sequence length shrinks both
by 8x and is applied identically to every arm so the sweep remains internally
comparable. Truncation can change absolute accuracy, so the anchor for the
new arms is the `layer=28` arm (which is exactly the `last` readout at 2,048
tokens), not the 49.11 from the 16k reference run.

Evaluation is the same 13,444-item set used by the reference table: 6,864
choice items, 4,723 noul items and 1,857 score items across the benchmark and
synthetic eval splits. The reported accuracy is the micro-average over all
items; per-type accuracies come from the same pass. One epoch of a frozen
0.6B arm takes about 13 minutes on an A100-40GB, so all arms below stayed
inside the four-GPU-hour budget; the LoRA arms from the reference table are
included as fixed reference points, not rerun.

Seed noise at 0.6B final epoch is about 0.40 pp, so differences under 0.8 pp
are treated as noise unless a replicate confirms them. A useful internal
control is the `layer=28` arm: Qwen3-Reranker-0.6B has 28 blocks, so
`hidden_states[28]` is exactly the tensor the `last` readout returns. At the
same 2,048-token cap it measures run noise and the truncation shift together,
which is why it is the anchor.

### Code changes

The sweep required three additive changes, all behind new flags:

- `LayerMixer` gained `layer` (one middle layer) and `fuse_pair` (two layers,
  concatenated or summed). A concatenation doubles the hidden width, so
  `DecisionModel` projects it back to the backbone width before the heads.
- `DecisionModel` accepts an optional `second_backbone`. Both backbones encode
  the same token ids, their readouts are concatenated along the hidden
  dimension, and a learned projection maps the result to the head width
  (`fusion="sum"` is available when the hidden sizes match). The second
  backbone is frozen like the first and can carry its own LoRA scope.
- `train_multitask.py` exposes `--readout-layer`, `--readout-layers`,
  `--readout-fusion`, `--second-model`, the second-backbone readout/LoRA flags,
  `--fusion` and `--mlp-dim`.

The same tokenizer feeds both backbones; the training script refuses a pair
whose vocabularies differ, because a mismatch would score different token ids
against each representation without any visible error.

## Literature

The hypothesis that a middle layer can beat the final layer is not folklore.
Alain and Bengio introduced linear classifier probes at intermediate layers
and observed that probe accuracy is not monotone in depth
([arXiv:1610.01644](https://arxiv.org/abs/1610.01644)). Tenney et al. showed a
layer-to-linguistic-pipeline correspondence in BERT
([arXiv:1905.05950](https://arxiv.org/abs/1905.05950)), and Rogers et al.
survey the resulting probing literature
([arXiv:2002.12327](https://arxiv.org/abs/2002.12327)). The strongest recent
statement is Skean et al., "Layer by Layer: Uncovering Hidden Representations
in Language Models": across 32 text-embedding tasks and several architectures,
"in nearly every task, some intermediate layer outperforms the final layer",
and they argue this follows from a compression/signal trade-off along depth
([arXiv:2502.02013](https://arxiv.org/abs/2502.02013), ICML 2025). Belinkov's
review is the standard caution that probe results depend on probe capacity and
selection ([Computational Linguistics 48(1), 2022](https://direct.mit.edu/coli/article/48/1/207/107783/Probing-Classifiers-Promises-Shortcomings-and)).

Fusing layers has precedent in vision architectures: DenseNet concatenates
earlier feature maps into later blocks
([arXiv:1608.06993](https://arxiv.org/abs/1608.06993)), and Deep Layer
Aggregation argues that iterative, hierarchical fusion of a feature hierarchy
beats one-step skip connections
([arXiv:1707.06484](https://arxiv.org/abs/1707.06484)). The existing
`weighted` readout is a soft mixture; `fuse_pair` is the simpler concatenate
or add operation those papers use.

For two backbones, the relevant prior is ensemble diversity. Snapshot
Ensembles shows that averaging predictions from diverse members helps
([arXiv:1704.00109](https://arxiv.org/abs/1704.00109)), and Model Soups shows
the same for weight-space averaging of fine-tuned models
([arXiv:2203.05482](https://arxiv.org/abs/2203.05482)). Those methods average
outputs or weights; here we concatenate hidden features and learn a projection,
which is closer to multi-view or feature-level fusion. A negative result is
therefore informative: at 0.6B, a second pretrained model may simply not be
diverse enough to pay for its compute.

For head capacity, the probing literature warns that a small probe can
understate what a representation contains (Belinkov, above), and Michel et
al. show that many attention heads are prunable but that head importance is
task-dependent ([arXiv:1905.10650](https://arxiv.org/abs/1905.10650)).
The practical question here is narrower: with roughly 75k training items seen
per run (5 epochs of 15k), do wider attention heads or an MLP projector help?

## Results

### Reference points (existing sweep, not rerun)

| Config | Readout | Backbone(s) | Eval acc | choice | noul | score |
|---|---|---|---|---|---|---|
| Frozen | last | Reranker-0.6B | 49.11 | 40.0 | 65.4 | 41.3 |
| Full LoRA | last | Reranker-0.6B | 54.46 | 42.8 | 74.5 | 46.7 |
| Full LoRA | mean_last_4 | Reranker-0.6B | 53.03 | 41.2 | 73.7 | 44.2 |
| Full LoRA | weighted | Reranker-0.6B | 56.17 | 46.1 | 74.8 | 46.0 |
| LoRA last 8 | weighted | Reranker-0.6B | 50.15 | 39.8 | 68.3 | 42.4 |
| Frozen | last | Reranker-4B | 55.59 | 45.6 | 75.1 | 42.7 |
| Full LoRA | last | Reranker-4B | 61.87 (ep4) | — | — | — |

The per-type columns for the 4B full-LoRA row are unavailable because the
surviving log contains only epoch 1 of that run; the overall figure is the
one reported in the architecture table.

<!-- RESULTS -->

### New arms

Every row is 5 epochs, seed 42, `max_length=2048`, frozen backbone(s) unless
stated. `acc` is the epoch-5 micro-average over 13,444 eval items; the last
three columns are the per-type accuracies.

| Arm | Config | Readout | Backbone(s) | acc | choice | noul | score | Log |
|---|---|---|---|---|---|---|---|---|
| layer8 | r2-0.6b | layer 8 | Reranker-0.6B | 41.70 | 31.8 | 60.5 | 30.5 | `arch_explore_layer08.log` |
| layer12 | r2-0.6b | layer 12 | Reranker-0.6B | 43.10 | 32.0 | 62.2 | 35.5 | `arch_explore_layer12.log` |
| layer16 | r2-0.6b | layer 16 | Reranker-0.6B | 43.13 | 30.4 | 65.8 | 32.5 | `arch_explore_layer16.log` |
| layer20 | r2-0.6b | layer 20 | Reranker-0.6B | 41.84 | 30.5 | 66.0 | 22.1 | `arch_explore_layer20.log` |
| layer24 | r2-0.6b | layer 24 | Reranker-0.6B | 47.49 | 36.1 | 66.8 | 40.4 | `arch_explore_layer24.log` |
| layer28 | r2-0.6b | layer 28 (= last) | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_layer28.log` |
| frozen weighted | r2-0.6b | weighted | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_frozen_weighted.log` |
| fuse concat 16+28 | r2-0.6b | fuse_pair, concat | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_fuse_concat16_28.log` |
| fuse sum 16+28 | r2-0.6b | fuse_pair, sum | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_fuse_sum16_28.log` |
| head r128 | r2-0.6b | last | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_head_r128.log` |
| head r256 | r2-0.6b | last | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_head_r256.log` |
| head MLP 2x | r2-0.6b | last + MLP 1024→2048→1024 | Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_head_mlp2x.log` |
| two 0.6B concat | r2-0.6b | last | Reranker-0.6B + Qwen3-0.6B | TBD | TBD | TBD | TBD | `arch_explore_two06_concat.log` |
| 4B alone (control) | r2-4b bb4 | last | Reranker-4B | TBD | TBD | TBD | TBD | `arch_explore_fourb_control.log` |
| 4B + 0.6B concat | r2-4b bb4 | last | Reranker-4B + Reranker-0.6B | TBD | TBD | TBD | TBD | `arch_explore_fourb06_concat.log` |
| 4B + 0.6B, LoRA on 0.6B | r2-4b bb4 | last | Reranker-4B (frozen) + Reranker-0.6B (LoRA) | TBD | TBD | TBD | TBD | `arch_explore_fourb06_lora06.log` |

## Analysis

<!-- ANALYSIS -->

## Threats to validity

- **Single seed.** All new arms use seed 42. The frozen `last` curve is
  non-monotone (43.4, 41.7, 41.1, 47.3, 49.1), so the final epoch is not a
  stable estimate of the configuration. Differences below 0.8 pp are not
  interpreted.
- **Data and eval set.** The arms see 15,000 sampled items per epoch from
  182,262 training items and are scored on 13,444 eval items. The reference
  head-size sweep used 19 benchmarks; its absolute numbers are therefore not
  comparable, only its ordering is.
- **Sequence length.** Every new arm uses 2,048 tokens, including the 4B
  arms. The reference arms used 16,384 (0.6B) and 8,192 (4B), so absolute
  numbers are not directly comparable across the two groups; the `layer=28`
  anchor and the `frozen weighted`/`head r64` controls supply the comparison
  inside the new group. The cap was forced by the noul-batch OOM described
  above.
- **Frozen backbones.** Every new arm freezes the backbone(s). The LoRA
  reference points come from a different compute budget (about seven GPU-hours
  per arm at 0.6B), so a frozen comparison answers whether a readout helps the
  representation that is actually being read, not whether it composes with
  LoRA adaptation.
- **Second backbone is loaded, not trained.** The two-backbone arms keep both
  backbones frozen and train only the fusion projection and heads. A result
  says nothing about whether joint fine-tuning of both models would help.

## References

- Alain, G. and Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. https://arxiv.org/abs/1610.01644
- Tenney, I., Das, D. and Pavlick, E. (2019). BERT Rediscovers the Classical NLP Pipeline. https://arxiv.org/abs/1905.05950
- Rogers, A., Kovaleva, O. and Rumshisky, A. (2020). A Primer in BERTology. https://arxiv.org/abs/2002.12327
- Skean, O. et al. (2025). Layer by Layer: Uncovering Hidden Representations in Language Models. https://arxiv.org/abs/2502.02013
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. https://direct.mit.edu/coli/article/48/1/207/107783/Probing-Classifiers-Promises-Shortcomings-and
- Huang, G. et al. (2017). DenseNet. https://arxiv.org/abs/1608.06993
- Yu, F. et al. (2018). Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
- Huang, G. et al. (2017). Snapshot Ensembles. https://arxiv.org/abs/1704.00109
- Wortsman, M. et al. (2022). Model Soups. https://arxiv.org/abs/2203.05482
- Michel, P. et al. (2019). Are Sixteen Heads Really Better than One? https://arxiv.org/abs/1905.10650
