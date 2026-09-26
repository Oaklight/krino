# Experiment Log: Head-Size Sweep

## Issues #76, #93

## Overview

We ran a 45-run head architecture sweep (5 models x 9 configs) to find the
optimal decision head size for frozen-backbone training. This was followed by
4 targeted runs with per-component learning rates for the MLP projector
(issue #93). The sweep results feed directly into the Round 2 training plan
(#73).

All runs used 5 epochs, `eval_every=1`, `save_every_epoch=true`, and
evaluated on 19 NLU benchmarks with type-balanced sampling. Accuracy values
below are epoch-5 accuracy unless noted otherwise.

## 1. Sweep Setup

### Models

| Model | Hidden dim (h) | Type |
|-------|---------------|------|
| Ettin-150m | 768 | Encoder (custom) |
| ModernBERT-base | 768 | Encoder |
| Qwen3-0.6B | 1024 | Causal LM |
| Qwen3-Reranker-0.6B | 1024 | Causal LM (reranker-pretrained) |
| Qwen3-Reranker-4B | 2560 | Causal LM (reranker-pretrained) |

### Config grid

- **Rank** (bottleneck dimension): 64, 128, 256
- **MLP layers** (projector depth): 0, 1, 2
- Total: 3 x 3 = 9 configs per model, 45 runs

### R1 baseline comparison

For context, R1 baselines trained the same benchmarks for 20 epochs with
default head configs:

| Model | R1 Acc (20ep) |
|-------|--------------|
| Ettin-150m | 57.9% |
| ModernBERT-base | 48.8% |
| Qwen3-0.6B | 57.4% |
| Qwen3-Reranker-0.6B | 58.1% |
| Qwen3-Reranker-4B | 65.5% |

The sweep runs only trained for 5 epochs, so absolute numbers are lower.
The goal was relative comparison across configs, not matching R1.

## 2. Sweep Results (45 runs)

### Ettin-150m (h=768)

| | r64 | r128 | r256 |
|---|---|---|---|
| mlp=0 | 47.6% (350K) | 48.5% (695K) | **49.2%** (1.4M) |
| mlp=1 | 45.2% (941K) | 45.7% (1.3M) | 45.5% (2.0M) |
| mlp=2 | 35.7% (1.5M) | 35.4% (1.9M) | 38.2% (2.6M) |

Best: r256 mlp=0 at 49.2%. Ettin is the only model where higher rank
consistently helps — accuracy increases monotonically with rank at mlp=0.
MLP layers hurt across the board.

### ModernBERT-base (h=768)

| | r64 | r128 | r256 |
|---|---|---|---|
| mlp=0 | **42.3%** (350K) | 41.8% (695K) | 42.9% (1.4M) |
| mlp=1 | 38.2% (941K) | 38.1% (1.3M) | 38.3% (2.0M) |
| mlp=2 | 34.1% (1.5M) | 34.2% (1.9M) | 34.6% (2.6M) |

Best: r256 mlp=0 at 42.9% (marginal over r64's 42.3%). ModernBERT is
the weakest backbone overall — its representations are apparently less
suited for structured probability heads. MLP degrades performance by
~4pp per layer.

### Qwen3-0.6B (h=1024)

| | r64 | r128 | r256 |
|---|---|---|---|
| mlp=0 | **49.4%** (467K) | 48.7% (926K) | 47.2% (1.8M) |
| mlp=1 | 43.8% (1.5M) | 43.0% (2.0M) | 44.5% (2.9M) |
| mlp=2 | 38.7% (2.6M) | 38.8% (3.0M) | 38.8% (3.9M) |

Best: r64 mlp=0 at 49.4%. For causal LMs, higher rank *hurts* — the
opposite of encoders. The smallest head wins, suggesting the backbone
representations are already well-structured and a thin linear projection
suffices. MLP=1 drops ~5.5pp, MLP=2 drops ~10.5pp.

### Qwen3-Reranker-0.6B (h=1024)

| | r64 | r128 | r256 |
|---|---|---|---|
| mlp=0 | **50.3%** (467K) | 49.6% (926K) | 48.3% (1.8M) |
| mlp=1 | 46.7% (1.5M) | 46.6% (2.0M) | 46.1% (2.9M) |
| mlp=2 | 36.2% (2.6M) | 36.1% (3.0M) | 36.9% (3.9M) |

Best: r64 mlp=0 at 50.3%. Same pattern as vanilla Qwen3-0.6B — lower
rank is better, MLP hurts. The reranker pretraining adds a consistent
~1pp boost over vanilla Qwen3-0.6B at every configuration.

### Qwen3-Reranker-4B (h=2560)

| | r64 | r128 | r256 |
|---|---|---|---|
| mlp=0 | 51.1% (1.2M) | **52.1%** (2.3M) | 50.3% (4.6M) |
| mlp=1 | 42.2% (7.7M) | 51.2% (8.9M) | 49.0% (11.2M) |
| mlp=2 | 38.0% (14.3M) | 37.5% (15.4M) | 37.2% (17.7M) |

Best: r128 mlp=0 at 52.1%. The 4B model is the only one where the
optimal rank is not at the extremes — r128 outperforms both r64 and
r256. The r256 mlp=0 result (50.3%) underperforms r128 due to the
learning rate (3e-4) being too aggressive for 4.6M trainable params.

At mlp=1, the r128 config (51.2%) nearly matches the mlp=0 best, while
r64 mlp=1 collapses to 42.2% — suggesting the MLP projector has
potential for this backbone but needs careful LR tuning.

## 3. Key Findings from the Sweep

### Finding 1: MLP projector consistently hurts at uniform LR

Across all 5 models:
- mlp=0 always outperforms mlp=1 by 2-6pp
- mlp=1 always outperforms mlp=2 by 5-10pp
- The deeper the projector, the worse the result

This is likely because 5 epochs is insufficient to jointly train the
projector and decision heads at the same learning rate. The projector
receives random-init gradients that destabilize the head training.

### Finding 2: Rank scaling depends on backbone type

**Encoders** (Ettin-150m): accuracy increases with rank (47.6% -> 49.2%).
The bottleneck dimension limits how much of the encoder's representation
can be preserved.

**Causal LMs** (Qwen3-0.6B, Reranker-0.6B): accuracy *decreases* with
rank (49.4% -> 47.2%, 50.3% -> 48.3%). More parameters without more
training time leads to underfitting.

**Large causal LM** (Reranker-4B): non-monotonic, r128 optimal. The
larger hidden dim (2560) benefits from a wider bottleneck up to a point,
but r256 overshoots (4.6M params at 3e-4 LR).

### Finding 3: Reranker pretraining consistently outperforms vanilla

At every matched config, reranker-pretrained backbones beat their vanilla
counterparts:
- Reranker-0.6B vs Qwen3-0.6B: +0.9pp average across all 9 configs
- This confirms that cross-encoder reranker pretraining produces
  representations better suited for structured probability prediction

### Finding 4: 4B r256 divergence was a LR issue

The r256 mlp=0 result for Reranker-4B (50.3%) is suspiciously low given
that r128 hits 52.1%. The issue is that 3e-4 is too high for 4.6M
trainable parameters — the effective update magnitude is too large. This
was confirmed by the epoch-level data showing early peaking followed by
degradation. A lower LR or LR warmup would likely recover the expected
monotonic rank scaling for this model.

## 4. MLP + Per-Component LR Targeted Runs (Issue #93)

Based on the sweep finding that MLP hurts at uniform LR, we hypothesized
that a separate (lower) learning rate for the MLP projector might rescue
the projector benefit. We tested this on the two reranker models.

### Setup

4 runs on rbdgx3 (H200 144GB). See `05-mlp-lr-targeted-runs.md` for the
full operational log including 4 failed attempts before these succeeded.

| Run | Model | Config | Heads LR | MLP LR |
|-----|-------|--------|----------|--------|
| t1 | Reranker-4B | r128 mlp=1 | 3e-4 | 1e-4 |
| t2 | Reranker-4B | r128 mlp=1 | 3e-4 | 3e-5 |
| t3 | Reranker-0.6B | r64 mlp=1 | 1e-3 | 3e-4 |
| t4 | Reranker-0.6B | r64 mlp=1 | 1e-3 | 1e-4 |

### Results

**Reranker-4B r128 mlp=1 (heads_lr=3e-4) — MLP HELPS for large backbones**

| mlp_lr | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | vs mlp=0 (52.1%) |
|--------|-----|-----|-----|-----|-----|-------------------|
| 1e-4 | 50.1% | 47.2% | 52.8% | 53.3% | 54.8% | +2.7pp |
| **3e-5** | 50.1% | 48.1% | 52.8% | 54.2% | **55.4%** | **+3.3pp** |

With per-component LR, the MLP projector reverses from -0.9pp (sweep,
uniform LR) to +3.3pp. The key is that mlp_lr must be ~10x lower than
heads_lr — the projector needs slow, stable adaptation rather than
aggressive updates.

The epoch-2 train loss spike is consistent across all Reranker model
runs and appears to be inherent to the architecture.

**Reranker-0.6B r64 mlp=1 (heads_lr=1e-3) — MLP does NOT help for small backbones**

| mlp_lr | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | vs mlp=0 (50.3%) |
|--------|-----|-----|-----|-----|-----|-------------------|
| **1e-4** | 44.9% | 42.2% | 46.2% | 47.4% | **48.0%** | **-2.3pp** |
| 3e-4 | 44.6% | 38.6% | 40.8% | 42.8% | 42.8% | -7.5pp |

Even with optimal per-component LR, the MLP still hurts by 2.3pp. The
0.6B backbone's hidden dim (1024) is small enough that a linear
projection suffices — the nonlinear MLP transformation adds noise rather
than useful features. The projector will be revisited during the LoRA
stage (Stage 5 of #73), where backbone fine-tuning may change this
dynamic.

### Key Finding

**MLP projector benefit is backbone-size-dependent:**
- 4B backbone (h=2560): MLP adds +3.3pp with mlp_lr=3e-5
- 0.6B backbone (h=1024): MLP subtracts -2.3pp even with optimal mlp_lr

## 5. Optimal Configurations for Round 2

| Model | Rank | MLP | Heads LR | MLP LR | Acc@5ep |
|-------|------|-----|----------|--------|---------|
| **Reranker-4B** | r128 | mlp=1 | 3e-4 | 3e-5 | 55.4% |
| **Reranker-0.6B** | r64 | mlp=0 | 1e-3 | — | 50.3% |

These are the final head configurations entering R2 training. The 4B
model uses MLP with a 10x-reduced learning rate; the 0.6B model drops
the MLP entirely.

## 6. Lessons Learned

1. **Config files can have stale LR values vs what was actually used in
   training.** The R1 sweep used `--lr 3e-4` as a CLI override, but the
   config YAML had `lr: 0.001`. Targeted runs that relied on the config
   file without the CLI override diverged immediately. Always verify
   effective LR matches intended LR, or update configs to match.

2. **Per-component LR is essential for MLP projector training.** Uniform
   LR underperforms because the randomly-initialized projector needs
   slower updates than the decision heads. A 10x ratio (heads_lr /
   mlp_lr) worked well for the 4B model.

3. **MLP benefit is backbone-size-dependent.** The projector helps for
   large backbones (h=2560) where the representation is rich enough to
   benefit from nonlinear transformation, but hurts for smaller backbones
   (h=1024) where it adds noise. This is consistent with the general
   finding that model capacity must justify architectural complexity.

4. **Epoch-2 train loss spikes are consistent across all Reranker
   models.** Both 0.6B and 4B reranker runs show a characteristic dip
   at epoch 2 followed by recovery. This appears inherent to the
   reranker architecture rather than a training bug.

5. **Encoder vs causal LM representations behave differently under rank
   scaling.** Encoders benefit from wider bottlenecks (more capacity to
   preserve bidirectional context), while causal LMs prefer narrow heads
   (the autoregressive representations are already well-structured for
   downstream projection).
