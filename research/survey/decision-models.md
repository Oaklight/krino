# Alternative Decision Models

Models and approaches for structured decision-making beyond direct Jev reproductions.

## CUA-S1: Computer Use Agent — System One

- **URL:** [github.com/trycua/cua](https://github.com/trycua/cua) `libs/cua-s1/` (23.7k★ parent repo)
- **HuggingFace:** Model [`cua-ai/cua-s1-forms`](https://huggingface.co/cua-ai/cua-s1-forms)
- **Merged:** 2026-09-18
- **Lineage:** Adapted from jevlike (MIT)

First practical application of Jev-style architecture for a real task — GUI form filling. A 706K-parameter one-pass option scorer (2.8 MB checkpoint).

| Component | Detail |
|---|---|
| Embedding | Byte-level, context 224 bytes, options 96 bytes |
| Encoder | 2-layer Transformer, width 128, 4 heads |
| Head | jevlike-derived AttentionHead: option queries context → dot product → softmax |
| Params | 706,048 (~2.8 MB) |
| Training | Cross-entropy, 6 epochs, synthetic data (10K episodes with hard confusers) |
| Primitives | Choice only (no Noul or Score) |
| Calibration | Not RLCD — pure CE |

**Results:** 99.7% vs Jev 83.6% on form-filling (asymmetric: CUA-S1 was trained for this task, Jev was not).

**Key insight:** Narrow specialist models at ~700K params can outperform general-purpose models on well-defined structured decision tasks with synthetic-only training.

## Qwen-2.5-1B-RLCD (Harsha Gundala)

*Awaiting detailed research. Referenced in jev-rerank-bench as a self-hosted RLCD reproduction scoring nDCG@10 = 0.255-0.471 — significantly below Jev.*
