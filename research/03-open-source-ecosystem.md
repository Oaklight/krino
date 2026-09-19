# Open-Source Ecosystem

**Date:** 2026-09-19

## Official TypeSafe Repositories

GitHub organization: [github.com/typesafe-ai](https://github.com/typesafe-ai)

| Repo | Description | Stars | Notes |
|------|-------------|-------|-------|
| [skills](https://github.com/typesafe-ai/skills) | Agent skills for Claude Code, Codex, etc. | 152 | Drop-in skill for building with Jev |
| [typesafe-sdk-js](https://github.com/typesafe-ai/typesafe-sdk-js) | Official TypeScript/JavaScript SDK | 98 | Typed questions and answers |
| [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) | Routes System One calls through OpenAI/Anthropic | 96 | Key reference for reproduction |
| [typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python) | Official Python SDK | 67 | Sync and async clients |
| [LLaDA](https://github.com/typesafe-ai/LLaDA) | Fork of ML-GSAI/LLaDA (diffusion LLM) | 4 | Architecture clue |
| [vllm](https://github.com/typesafe-ai/vllm) | Fork of vllm-project/vllm | 1 | Inference engine clue |

### System One Adapter (Most Relevant)

The [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) is particularly important — it's TypeSafe's own tool for routing the exact same typed-question API through standard LLMs. This is how they do their comparison benchmarks:

- Supports OpenAI (Responses API + Chat Completions) and Anthropic
- Two answer modes: `probabilities` (per-label distribution) or `discrete` (one value per question)
- Can use provider-native structured output or prompt-based JSON validation
- Response includes retry diagnostics, attempt history, and probability normalization info
- Acts as a drop-in replacement for `typesafe_sdk`'s `system_one()` method

## Open-Source Reproduction Projects

### Tier 1: Direct Jev Interface Reproduction

#### Strategy A: Logit Readout from Frozen LLMs (No Training)

| Project | Model | Stars | Accuracy vs Jev | Notes |
|---------|-------|-------|-----------------|-------|
| [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) | Qwen3.5-4B | — | 0.845 vs 0.883 (102-row subset) | Browser WebGPU demo, CUDA backend, prefix sharing |
| [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) | Gemma 3 4B | — | Agrees on routing, disagrees on judgment calls | MLX on Apple Silicon, full `/v1/systemone` contract |
| [LitJev](https://github.com/) | Any Qwen model | 11 | — | Zero-training logit readout, includes Doom benchmark |
| [zhihz/openjev](https://github.com/) | Qwen3-4B (frozen) | 9 | — | Bilingual EN/ZH, Apple Silicon MLX |

#### Strategy B: Trained Heads and Adapters

| Project | Model | Stars | Accuracy vs Jev | Notes |
|---------|-------|-------|-----------------|-------|
| [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) | Qwen3-0.6B (full fine-tune) | 452 | 75% vs 76.5% (fixed-route diagnostic) | Most ambitious open reproduction; RLCD experiment; HF: `C-Tianyu/NanoJev` |
| [Laya](https://github.com/convaiinnovations/laya) (convaiinnovations) | ModernBERT-large (395M + 26M head) | — | AG News 0.947 vs Jev 0.910; typed-decisions 0.766 vs Jev 0.727 | **Only true RLCD reproduction** — REINFORCE + proper scoring rules; HF: `convaiinnovations/laya` |
| [Bespoke Nimble](https://github.com/) (Bespoke Labs) | Qwen3.5-9B (LoRA) | 81 | 90.1% ref match vs Jev 93.2% | First established lab entry; contrastive data curation; 2,676 curated examples |
| [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) | Custom byte encoder or frozen HF backbone | — | ~98% on synthetic, 26% on Wikispeedia | Doom and chess demos with vision scorer; cross-entropy only (no RL) |
| [olanotolu/jevbetter](https://github.com/olanotolu/jevbetter) | Hashed n-gram encoder + rival-aware attention | — | +4.3pp over jevlike, 2x better calibration | Gated MLP, temperature scaling |
| [novvoo/nanojev](https://github.com/novvoo/nanojev) | Small bidirectional transformer (100K-5M params) | 0 | 73.2% on unified test (210 intents), ECE 0.172 | Educational single-file `nanoJEV.py` (~1500 lines); nanoGPT style; RLCD loss ablation |

#### Strategy C: Non-Autoregressive / Diffusion

| Project | Model | Stars | Notes |
|---------|-------|-------|-------|
| [razorback16/openjev](https://github.com/razorback16/openjev) | DiffusionGemma 26B | 47 | Multimodal (supports images); `/v1/systemone` API; free hosted; architecturally closest to Jev's "parallel single-pass" claim |

### Tier 2: Integrations and Tooling

| Project | Description |
|---------|-------------|
| [Jev MCP](https://github.com/itsmostafa/typesafe-mcp) | Python MCP server exposing classify/score/check tools to MCP-compatible agents |
| [TypeSafe MCP](https://github.com/typesafe-ai/skills) | Go CLI and single-binary MCP server for Claude Desktop/Code/Codex |
| [pi-typesafe](https://github.com/) | Pi extension with batched `typesafe_evaluate` tool and offline-testable transport |
| [s1-rs](https://github.com/) | Rust derive layer for typed question sets, confidence gates, and network-free testing |
| [TypeSafeAI.Net](https://github.com/) | .NET client with Microsoft.Extensions.AI adapters |

### Tier 3: Applications and Demos

| Project | Description |
|---------|-------------|
| [CUA-S1](https://github.com/trycua/cua) (trycua/cua, `libs/cua-s1/`) | **First practical application of Jev-style architecture for a real task.** GUI form filling — 706K-param one-pass option scorer (2.8 MB), byte-level encoder, jevlike-derived AttentionHead. 100% on 196 real-form decisions. Trained on synthetic data only. 23.7k★ parent repo. |
| [Jev Browser](https://github.com/) | Agent skill for Jev-selected browser actions in observation-action-verification loops |
| [Jev Ultrafast](https://github.com/) | Browser Use agent with dynamic indexed action space and batched decisions |
| [Crowdcheck](https://github.com/) | Tests posts on 10,000 synthetic personas with batched Jev calls |
| [HEIST//ONE](https://github.com/) | Observable browser stealth game where Jev evaluates guard reactions |
| [Jev Plays Pokemon](https://github.com/) | FireRed/Showdown harness where Jev picks moves from current battle state |
| [Jev Plays StarCraft](https://github.com/) | Structured-state harness for StarCraft shareware campaign |
| [TypeSafe Mario](https://github.com/) | NES controller experiment with emulator telemetry |
| [Supercov](https://github.com/) | Code quality scorer — Jev ranks source files so agents know what to fix first |
| [Every](https://github.com/) | Semantic code search CLI: ask a yes/no question of every function, rank by probability |
| [Jev Review](https://github.com/) | Staged code-review workflow with local dashboard |

### Tier 4: Independent Evaluations

| Project | Description |
|---------|-------------|
| [calibre](https://github.com/) | Independent calibration measurement on Banking77 and Web of Science datasets |
| [Jev Judge vs Dimension Scores](https://github.com/) | 3-task measurement: direct Jev question vs 12-14 Jev-scored dimensions with fitted weights |
| [Jev Rerank Bench](https://github.com/) | Reranking comparison with raw provider responses and uncertainty intervals |
| [Jev Spam Eval](https://github.com/) | Zero-shot spam detection study against trained TF-IDF baselines |
| [OpenJev benchmark](https://github.com/TheoLeeCJ/openjev) | Head-to-head comparison of direct logit scoring vs Jev on shared test set |

## RLCD Reproduction Landscape

A critical finding from the community: most projects claiming "RLCD" do not actually implement reinforcement learning from calibrated distributions. The table below clarifies the status of each.

### RLCD Status of Key Projects

| Reproduction | Training? | RL? | Calibration Method | Key Finding |
|---|---|---|---|---|
| **Laya** (convaiinnovations) | Yes (RL) | REINFORCE + proper scoring rules | Temperature-fitted ECE 0.466 -> 0.081 | **Only open project that actually implements RLCD-style training** |
| NanoJev (TianyuCodings) | Yes | Experimental RLCD arm | Early-stage | Paired proper-reward policy gradient showed slightly lower distribution error than CE baseline; one seed only |
| novvoo/nanojev | Yes | REINFORCE with calibration reward | ECE 0.140 (CE) -> 0.118 (RLCD) | Ablation confirms RLCD reduces ECE, but small model and dataset |
| Qwen-2.5-1B-RLCD (Gundala) | **No** | **No** | None | **Name is misleading** — pure inference trick (parallel constrained decoding on pretrained logits). Built in ~2 hours. Uncalibrated. |
| jevlike (Wang-Mascianica) | Yes (CE) | No | ECE measured, not optimized | Standard cross-entropy only |

### Laya: The Reference RLCD Implementation

Laya is the most complete open RLCD reproduction and deserves detailed treatment:

- **Architecture:** ModernBERT-large (395M) + 2-layer decision head = 421M total params
- **Training:** Policy-gradient RL with strictly proper scoring rules (log score + spherical score + ranked probability score for ordinal). REINFORCE with GRPO-style group-mean baseline. 7,313 updates, ~2 hours training time.
- **Calibration:** Fitted temperatures per question type. Mean ECE reduced from 0.466 to 0.081.
- **Results:** AG News 0.947 vs Jev 0.910; typed-decisions benchmark 0.766 vs Jev 0.727.
- **Caveat:** Near chance zero-shot (0.362 vs 0.318 random) — calibration comes primarily from temperature fitting, not from the RL training itself.
- **Uses bidirectional encoder** (ModernBERT), not causal LM. This avoids the log-probability surface-form leakage found in our Probe 3 but diverges architecturally from what our probes suggest Jev uses.
- **HuggingFace:** [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)

### Gundala's "RLCD" Clarification

Despite the name "Qwen-2.5-1B-RLCD", this is zero training. The technique is: single KV-cache prefill, sub-vocabulary logit slicing, calibrated softmax over candidate tokens, programmatic JSON assembly. Achieved 5.6-7x speedup over autoregressive baseline with 100% schema validity by construction. However, probabilities are uncalibrated pretrained logits, which explains poor nDCG@10 (0.255-0.471) in jev-rerank-bench. Useful as an inference-time baseline showing the speed/type-safety floor.

### jevlike Architecture Confirmed

From source code inspection (Vincent Wang-Mascianica, Senior Research Associate at HAILab, Oxford; PhD in DisCoCat):

- **Loss:** Standard `F.cross_entropy` — no RL, no proper scoring rules
- **AttentionHead:** LayerNorm -> query/key/value projections -> masked attention -> element-wise dot product -> softmax
- **Byte encoder:** 257 tokens (256 bytes + padding), 192-byte context, 32-byte options, default width=64
- **HF path:** Frozen `AutoModel` + trainable AttentionHead only (head stored in checkpoint, encoder name referenced)
- **Theoretical basis:** arXiv:2407.02423 ("On the Anatomy of Attention")
- **No paper planned** on jevlike itself

## NanoJev Deep Dive

TianyuCodings/NanoJev (452 stars) is the most ambitious open reproduction — an end-to-end pipeline training a 0.6B parallel decision model from scratch.

- **Backbone:** Qwen3-0.6B, full fine-tuning (not LoRA)
- **Heads:** Custom decision heads — shared scalar + set attention for Choice (2-255 dynamic candidates), sigmoid for Boolean/Noul, level evaluator for Score (2-10 levels). Each candidate tokenized independently, no cross-question attention
- **Training:** CE on gold distributions + experimental RLCD-inspired proper-reward policy gradient. Mathematical gradient checks pass (max error 1.39e-16)
- **Data:** Procedurally generated mazes (8x8 to 50x50) and Snake games. Atomic local questions (is north/east/south/west traversable?)
- **Results vs Jev:** Fixed-route diagnostic: NanoJev 75% vs Jev 76.5%. Navigation benchmark: NanoJev 95%/90% (test/OOD), Jev 100%/95%. 50x50 maze: NanoJev reaches goal in 244 attempts vs Jev's 2,738 (different exploration strategies)
- **Key insight:** Clean separation of atomic perception (model) from route planning (code). The game results reflect the joint system, not the model alone
- **HuggingFace:** Model at `C-Tianyu/NanoJev`, Dataset at `C-Tianyu/NanoJev-Data`

## CUA-S1: First Practical Jev-Style Application

[CUA-S1](https://github.com/trycua/cua) (Computer Use Agent — System One) is the first practical deployment of a Jev-style architecture for a real task: GUI form filling.

- **Parent repo:** trycua/cua (23.7k stars), component at `libs/cua-s1/`
- **Lineage:** Adapted from [jevlike](https://github.com/vinnylarouge/jevlike) (MIT)
- **Architecture:** Byte-level encoder (no tokenizer), 2-layer Transformer (width 128, 4 heads), jevlike-derived AttentionHead. All options scored in parallel in one forward pass. 706K parameters, 2.8 MB checkpoint.
- **Training:** Pure cross-entropy on synthetic data only (10K episodes with hard confuser pairs). No RLCD.
- **Results:** 100% on 196 real-form decisions; 99.95% on 15K synthetic test decisions; 99.7% vs Jev's 83.6% head-to-head on the same task (but Jev received zero task-specific fine-tuning — asymmetric comparison).
- **Significance:** Validates that one-pass attention-based option-scorer paradigm works at ~700K params for narrow structured tasks. Only implements Choice primitive (not Noul or Score).
- **HuggingFace:** Model [`cua-ai/cua-s1-forms`](https://huggingface.co/cua-ai/cua-s1-forms), Dataset [`cua-ai/cua-s1-forms`](https://huggingface.co/datasets/cua-ai/cua-s1-forms)

## Three Reproduction Strategies

Three architectural strategies have crystallized across ~100+ community projects. These correspond to the original "Routes A/B/C" from the open-jev design doc but have been refined by community experience.

### Strategy 1: Logit Readout from Frozen LLMs (No Training)

Load with `AutoModelForCausalLM` (with LM head). Concatenate context and option. One forward pass. Read log-probability of each option token. Sum or average over option tokens. That's the score.

**Best for:** quick baseline, no labeled data needed. Handle length bias by normalizing.

**Community representatives:** TheoLeeCJ/openjev, daseinlabs/open-jev, LitJev, zhihz/openjev, SemIf, Simple Jev

**Known weakness:** Uncalibrated. Our Probe 3 found that surface form (length, grammaticality, vocabulary) predicts 43% of assigned mass — this is the fundamental limitation of the logit readout approach.

### Strategy 2: Trained Heads and Adapters

Load a pretrained model, freeze or LoRA it, run one forward pass over context + options, train a scoring head with ranking or calibration loss. The head trains in seconds per epoch.

**Best for:** when you have labeled data and the encoder's representations already contain the signal.

**Community representatives:** NanoJev (full fine-tune + RLCD experiment), Laya (REINFORCE + proper scoring rules), Bespoke Nimble (LoRA + contrastive data curation), jevlike (frozen encoder + CE head), jevbetter (improved head), novvoo/nanojev (from-scratch RLCD)

**Key insight from Bespoke Nimble:** 2,676 contrastive-curated examples achieved 90% reference match — data quality matters more than data quantity.

### Strategy 3: Non-Autoregressive / Diffusion

Write answer template onto a denoising canvas; one read-only step. Architecturally closest to Jev's "parallel single-pass" claim.

**Best for:** maximum parallelism, multimodal support.

**Community representative:** razorback16/openjev (DiffusionGemma 26B). Supports images (multimodal), provides `/v1/systemone` API, free hosted instance.

**Our probes' relationship:** Probes 1 and 5 disfavor visible autoregressive decoding but do not exclude diffusion as the internal mechanism.

## Community Resources

### Curated Lists and Trackers

| Resource | Description |
|----------|-------------|
| [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe) | Comprehensive, well-maintained, reviewed 2026-09-17 |
| [awesome-jev](https://github.com/AnotiaWang/awesome-jev) | Alternative curated list |
| [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev) | Catalogs 100+ projects |
| [thevibeworks/awesome-typesafe-jev](https://github.com/thevibeworks/awesome-typesafe-jev) | Catalogs 100+ projects |
| [HF Reproductions Tracker](https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker) | 38 artifacts cataloged by Apolinario (multimodalart). Searchable/filterable. "too many open jev claims and reproductions — which one works? which one can you run on your laptop?" |

### Benchmark Datasets

| Dataset | Description |
|---------|-------------|
| [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 2,000 decisions across 4 domains (Agent Trace, Customer Service, Invoice Processing, Security). Teacher self-agreement ceiling: 0.735. Jev 1.13.0 scored 0.727. Laya scored 0.766. Ready-made benchmark for evaluating reproductions. |
| [`C-Tianyu/NanoJev-Data`](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) | Procedurally generated mazes and Snake games for NanoJev training |
| [`cua-ai/cua-s1-forms`](https://huggingface.co/datasets/cua-ai/cua-s1-forms) | Synthetic GUI form-filling episodes for CUA-S1 training |

## Implications for Our Work

1. **Laya's RL recipe is the reference for RLCD training** — REINFORCE + proper scoring rules + temperature fitting. Their training code is the closest open analog to what Jev might use internally.
2. **ModernBERT vs causal LM backbone** — Laya uses a bidirectional encoder, not causal LM. Our Probe 3 found LM-logit leakage, which Laya's encoder approach avoids. Worth comparing both architectures.
3. **The `typed-decisions` dataset** is a ready-made benchmark for evaluating our model against both Jev and Laya.
4. **Bespoke Nimble's contrastive data curation** is a practical insight — 2,676 curated examples beat much larger unfiltered datasets.
5. **Gundala's constrained decoding** is a useful inference-time baseline achievable in ~2 hours, giving us the speed/type-safety floor to beat with training.
6. **CUA-S1 validates narrow specialization** — 706K params achieving 100% on real forms shows that the architecture works at small scale for well-defined tasks.
7. **NanoJev's RLCD experiment** is the closest published approximation to calibration-aware RL training on a causal backbone. Their proper-reward policy gradient showed slightly lower distribution error than CE baseline, but one seed is not enough to establish ranking.
8. **The diffusion approach (razorback16/openjev)** remains the main alternative to causal-LM readout. Our probes do not exclude it.
