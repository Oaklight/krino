# Replication

Open replication of the operational pipeline identified through black-box probing: `State → shared encoding → per-option scoring → softmax → calibrated probabilities`.

We evaluated 10 backbone models across 19 benchmarks, trained lightweight decision heads, and benchmarked against the Jev API on identical evaluation data. Headline result: a 150M reranker-pretrained encoder achieves **95.2% on Banking77** (vs Jev's 75.0%), though Jev dominates on reasoning-heavy tasks (ARC 99.0%, RACE 95.5%).

## Sections

- [**Results**](results.md) — Full findings: zero-training baselines, Jev API benchmarks, trained heads, surface-form sensitivity, and infrastructure
- [**Open Decision Model**](model.md) — Architecture description, model structure, and development tracks
