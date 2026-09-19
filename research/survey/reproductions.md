# Jev Reproduction Projects

Detailed survey of open-source efforts to reproduce TypeSafe's Jev System One model.

## Tier 1: Substantial Research Efforts

### TianyuCodings/NanoJev (452★)

- **URL:** https://github.com/TianyuCodings/NanoJev
- **Approach:** End-to-end pipeline training a 0.6B decision model from scratch
- **Backbone:** Qwen3-0.6B, full fine-tuning
- **Heads:** Custom decision heads — set attention for Choice (2-255 candidates), sigmoid for Boolean/Noul, level evaluator for Score
- **Training:** CE on gold distributions + experimental RLCD-inspired proper-reward policy gradient
- **Data:** Procedurally generated mazes and Snake games with atomic local questions
- **Results:** NanoJev 75% vs Jev 76.5% on fixed-route diagnostic. Navigation: 95%/90% test/OOD
- **Key insight:** Clean separation of atomic perception (model) from route composition (code planning)
- **HuggingFace:** Model `C-Tianyu/NanoJev`, Dataset `C-Tianyu/NanoJev-Data`

### Bespoke Labs / Nimble (81★)

- **URL:** https://github.com/bespokelabsai/nimble
- **Approach:** LoRA fine-tuning with contrastive data curation
- **Backbone:** Qwen3.5-9B
- **Training:** 2,676 curated contrastive examples across 10 categories. No distillation from Jev
- **Results:** 90.12% reference match vs Jev 93.21% (untuned 9B: 66.36%)
- **Notable:** First established AI lab effort. Published model on HuggingFace

### TheoLeeCJ/openjev (varies)

- **URL:** https://github.com/TheoLeeCJ/openjev
- **Approach:** Zero-shot logit scoring (Route B) with prefix reuse
- **Backbone:** Qwen3.5-4B (frozen)
- **Results:** 0.845 agreement with Jev on 102-row subset (vs Jev 0.883). Direct logits 5.2× faster than autoregressive JSON
- **Browser demo:** WebGPU with quantized models

### daseinlabs/open-jev

- **URL:** https://github.com/daseinlabs/open-jev
- **Approach:** Zero-shot logit scoring with MLX on Apple Silicon
- **Backbone:** Gemma 3 4B
- **Notable:** Full `/v1/systemone` contract with all three primitives. Doom demo

### vinnylarouge/jevlike

- **URL:** https://github.com/vinnylarouge/jevlike
- **Approach:** Trained scorer head (Route A) — byte encoder or frozen HF backbone
- **Training:** Supervised on labeled option sets with ranking loss
- **Results:** ~98% on synthetic menus, 26% on Wikispeedia next-click (frozen Qwen 0.5B + head)
- **Notable:** Doom and chess vision demos with shared option-attention head

### olanotolu/jevbetter

- **URL:** https://github.com/olanotolu/jevbetter
- **Approach:** Improved trained head with rival-aware attention
- **Improvements over jevlike:** Hashed n-gram encoder, gated 2-layer MLP, temperature scaling, near-miss negatives
- **Results:** +4.3pp top-1 over jevlike (91.6% vs 87.3%), 2× better ECE (0.018 vs 0.037)

## Tier 2: Alternative Approaches

### razorback16/openjev (47★)

- **URL:** https://github.com/razorback16/openjev
- **Approach:** Discrete diffusion canvas read
- **Backbone:** DiffusionGemma 26B-A4B (NVFP4)
- **Notable:** Non-autoregressive. Jev-compatible API. Supports images (multimodal). Free hosted on Codiv

### novvoo/nanojev (new)

- **URL:** https://github.com/novvoo/nanojev
- **Approach:** Educational single-file implementation with RLCD training
- **Backbone:** Small bidirectional transformer from scratch (~100K-5M params)
- **Training:** CE + GaussianNLL + ECE + REINFORCE policy gradient
- **Results:** 73.2% on 210-intent unified task, ECE 0.172. RLCD ablation: ECE 0.140→0.118
- **Notable:** Best pedagogical resource for RLCD concepts. OOV-aware abstention. Full web UI

### ikermoel/open-alternative-jev

- **URL:** https://github.com/ikermoel/open-alternative-jev
- **Approach:** Qwen3.6-27B packed scoring with temperature scaling
- **Results:** RACE-H 92.9%, MMLU 84.2%. Temperature scaling: ECE 5.4%→2.1%
- **Notable:** Demonstrated that packing changes 6-9% of individual answers vs 2.7% kernel noise

### zhengxuyu/litjev (11★)

- **URL:** https://github.com/zhengxuyu/litjev
- **Approach:** Zero-training logit readout from any Qwen model
- **Notable:** Includes Doom + chess benchmarks, MMLU-Pro scoring

## Community Resources

- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe) — official community list
- [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev) — 100+ projects
- [thevibeworks/awesome-typesafe-jev](https://github.com/thevibeworks/awesome-typesafe-jev)
- [Archer Hume probing study](https://archerhume.com/posts/jevs-architecture-unmasked/) — 1,600+ probes
- [SamuelSacco/jev-exploration](https://github.com/SamuelSacco/jev-exploration) — calibration audit
- [FirasSX914/calibre](https://github.com/FirasSX914/calibre) — confidence-based routing
- [anessbelbati/jev-rerank-bench](https://github.com/anessbelbati/jev-rerank-bench) — reranking benchmark
