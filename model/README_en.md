# Decision Model

[English](README_en.md) | [中文](README_zh.md)

Open typed decision model replication. Implements the operational pipeline identified through black-box probing of TypeSafe's Jev:

```
State → shared encoding → per-option scoring → softmax → calibrated probabilities
```

## Results

### Best trained-head accuracy (Banking77, 77-class intent classification)

| Backbone | Type | Params | Accuracy |
|----------|------|--------|----------|
| **Ettin-150m r128** | Reranker-pretrained encoder | 150M + ~800K | **95.2%** |
| Ettin-150m r64 | Reranker-pretrained encoder | 150M + ~400K | 93.8% |
| Qwen3-0.6B r64 | Causal LM | 596M + 402K | 93.2% |
| Ettin-400m r64 | Reranker-pretrained encoder | 400M + ~400K | 91.6% |
| ModernBERT-base r64 | Vanilla encoder | 149M + 301K | 89.0% |
| Jev (reference) | — | — | 75.0% |

### Key findings

- **Reranker pretraining transfers:** Ettin-150m beats vanilla ModernBERT-base by +4.8pp at identical architecture and param count
- **Training > scaling:** 0.6B + trained heads beats 7B zero-shot logit readout
- **Heads don't generalize:** Banking77-trained heads score ~35% on other benchmarks — multi-task training needed
- **Surface-form bias persists:** Architecture reduces but doesn't eliminate it — calibration with invariance loss is required

See [research/04-model-replication.md](../research/04-model-replication.md) for full analysis.

## Status

Step 2 (trained heads) complete. Step 3 (calibration training) infrastructure ready, sweep pending.
See epic [#19](https://github.com/Oaklight/krino/issues/19) for details.

## Structure

```
model/
├── src/            # model code (backbone, heads, inference)
├── data/           # data pipeline and benchmark conversion
├── training/       # training loops and loss functions
├── evaluation/     # metrics and comparison tools
├── configs/        # experiment configurations
├── scripts/        # entry points (train, eval, serve)
└── experiments/    # logs, checkpoints, results (gitignored)
```

## Tracks

1. **Logit readout** (#21) — frozen Qwen3-0.6B, direct log-prob scoring
2. **Trained heads** (#22) — frozen backbone + lightweight decision heads
3. **Calibration** (#23) — Brier/MMCE/focal + RLCD-style RL
4. **Fresh architecture** (#24) — bidirectional encoder comparison
