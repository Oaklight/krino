# Experiment Log: Multi-Task Optimization Verification

## Issue #95, Parent: #73

## Overview

Verified two multi-task optimization techniques across 4 backbones (2 reranker + 2 base, at 4B and 0.6B scales). 12 runs total, 5 epochs each, on full 26-benchmark + synthetic dataset.

## Setup

### Backbones tested

| Model | Type | Hidden dim | Baseline LR |
|-------|------|-----------|-------------|
| Qwen3-Reranker-4B | Reranker | 2560 | 3e-4 |
| Qwen3.5-4B-Base | Base | 2560 | 3e-4 |
| Qwen3-Reranker-0.6B | Reranker | 1024 | 1e-3 |
| Qwen3-0.6B | Base | 1024 | 1e-3 |

### Configurations

1. **Baseline**: Uniform LR across all head types
2. **Per-type LR**: noul=0.5x, choice=1x, score=2x
3. **Uncertainty weighting** (Kendall et al. 2018): Learnable log-variance per type

## Results

### Final accuracy at epoch 5

| Model | Baseline | Per-type LR | Delta | Uncertainty | Delta |
|-------|----------|-------------|-------|-------------|-------|
| Reranker-4B | **59.5%** | 58.4% | -1.1pp | **59.2%** | -0.3pp |
| Qwen3.5-4B | **55.2%** | 52.4% | -2.8pp | **54.8%** | -0.4pp |
| Reranker-0.6B | **51.6%** | 50.5% | -1.1pp | 50.9% | -0.7pp |
| Qwen3-0.6B | **50.5%** | 48.8% | -1.7pp | **50.2%** | -0.3pp |

### Epoch-by-epoch: Reranker-4B

| Config | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 |
|--------|-----|-----|-----|-----|-----|
| Baseline | 51.2% | 53.6% | 53.5% | 55.5% | **59.5%** |
| Per-type | 51.2% | 49.9% | 53.4% | 54.5% | 58.4% |
| Uncertainty | 51.2% | 50.1% | 54.6% | 56.3% | 59.2% |

### Epoch-by-epoch: Qwen3.5-4B

| Config | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 |
|--------|-----|-----|-----|-----|-----|
| Baseline | 47.1% | 49.0% | 50.0% | 51.5% | **55.2%** |
| Per-type | 46.3% | 40.9% | 47.5% | 47.8% | 52.4% |
| Uncertainty | 47.1% | 49.2% | 48.4% | 51.3% | 54.8% |

### Epoch-by-epoch: Reranker-0.6B

| Config | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 |
|--------|-----|-----|-----|-----|-----|
| Baseline | 44.2% | 44.0% | 45.0% | 46.6% | **51.6%** |
| Per-type | 43.9% | 41.4% | 45.6% | 47.7% | 50.5% |
| Uncertainty | 44.1% | 46.2% | 46.0% | 46.2% | 50.9% |

### Epoch-by-epoch: Qwen3-0.6B

| Config | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 |
|--------|-----|-----|-----|-----|-----|
| Baseline | 42.8% | 41.5% | 43.8% | 46.7% | **50.5%** |
| Per-type | — | 40.9% | 44.2% | 46.3% | 48.8% |
| Uncertainty | 42.8% | — | 45.5% | 47.8% | 50.2% |

## Key Findings

### 1. Uniform LR baseline wins across all 4 backbones

No optimization technique beat the simple uniform LR in this 5-epoch setup.

### 2. Per-type LR with score=2x is universally harmful (-1.1 to -2.8pp)

Every run had epoch-2 train loss explosion from the aggressive score_lr. The hypothesis "hard tasks need higher LR" is wrong — hard tasks need stability, not aggression.

### 3. Uncertainty weighting nearly matches baseline (-0.3 to -0.7pp)

Kendall et al. 2018 is the strongest automatic balancing method — consistently within 0.3-0.7pp of baseline with zero hyperparameter tuning.

### 4. Reranker pretraining advantage confirmed

| Tier | Reranker | Base | Delta |
|------|----------|------|-------|
| 4B | 59.5% | 55.2% | +4.3pp |
| 0.6B | 51.6% | 50.5% | +1.1pp |

### 5. Full dataset >> 19-benchmark sweep

| Model | 19 benchmarks (sweep) | 26 benchmarks + synthetic | Delta |
|-------|----------------------|--------------------------|-------|
| Reranker-4B | 52.1% | 59.5% | +7.4pp |
| Reranker-0.6B | 50.3% | 51.6% | +1.3pp |

## Decision for R2

- **Use uniform LR** — simplest and best
- Uncertainty weighting preserved for future experiments
- Per-type LR needs inverted ratios (score=0.5x) before retesting

## Lessons Learned

1. Hard tasks need lower LR, not higher — stability over aggression
2. Uncertainty weighting is a strong zero-tuning baseline
3. Reranker pretraining advantage is robust across all optimization methods
4. Full dataset matters more than optimization tricks (+7.4pp vs -1.1pp)
5. Epoch-2 instability is universal across all Reranker backbones
