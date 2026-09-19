# Decision Model

Open typed decision model replication. Implements the operational pipeline identified through black-box probing of TypeSafe's Jev:

```
State → shared encoding → per-option scoring → softmax → calibrated probabilities
```

## Status

See epic issue [#19](https://github.com/Oaklight/jev-explore/issues/19) for progress.

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
