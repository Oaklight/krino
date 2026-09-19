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

## Laya (convaiinnovations) — Only True RLCD Reproduction

- **URL:** [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- **Architecture:** ModernBERT-large (395M) + 2-layer decision head = 421M total
- **Training:** REINFORCE with proper scoring rules (log score + spherical score + ranked probability score for ordinal). GRPO-style group-mean baseline. 7,313 updates, ~2 hours
- **Calibration:** Temperature-fitted per type. ECE 0.466 → 0.081
- **Results:** AG News 0.947 vs Jev 0.910; typed-decisions 0.766 vs Jev 0.727
- **Caveat:** Near chance zero-shot (0.362 vs 0.318 random). English-centric. Jev figures from third parties
- **Key insight:** Uses a **bidirectional encoder** (ModernBERT), not causal LM. Our Probe 3 showed causal LMs leak surface-form biases; encoder-based scoring may avoid this

## Qwen-2.5-1B-RLCD (Harsha Gundala) — NOT Actually RLCD

- **URL:** [huggingface.co/harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
- **What it actually is:** Zero training. Inference-time parallel constrained decoding on unmodified Qwen2.5-1.5B-Instruct-4bit
- **Technique:** Single KV-cache prefill → sub-vocabulary logit slicing → calibrated softmax → programmatic JSON assembly
- **Results:** 5.6-7× speedup over autoregressive. 100% schema validity. But uncalibrated logits (nDCG@10 = 0.255-0.471)
- **Key insight:** Demonstrates that speed and type-safety are achievable through inference engineering alone — the gap is in calibration quality
