# Open-Source Ecosystem

**Date:** 2026-09-18

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

| Project | Approach | Model | Accuracy vs Jev | Notes |
|---------|----------|-------|-----------------|-------|
| [TheoLeeCJ/openjev](https://github.com/TheoLeeCJ/openjev) | Zero-shot logit scoring (Route B) | Qwen3.5-4B | 0.845 vs 0.883 (102-row subset) | Browser WebGPU demo, CUDA backend, prefix sharing |
| [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) | Zero-shot logit scoring (Route B) | Gemma 3 4B | Agrees on routing, disagrees on judgment calls | MLX on Apple Silicon, implements full `/v1/systemone` contract |
| [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) | Trained scorer head (Route A) | Custom byte encoder or frozen HF backbone | ~98% on synthetic, 26% on Wikispeedia | Doom and chess demos with vision scorer |
| [olanotolu/jevbetter](https://github.com/olanotolu/jevbetter) | Improved trained head (Route A) | Hashed n-gram encoder + rival-aware attention | +4.3pp over jevlike, 2× better calibration | Gated MLP, temperature scaling |

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
| [Jev Browser](https://github.com/) | Agent skill for Jev-selected browser actions in observation-action-verification loops |
| [Jev Ultrafast](https://github.com/) | Browser Use agent with dynamic indexed action space and batched decisions |
| [Crowdcheck](https://github.com/) | Tests posts on 10,000 synthetic personas with batched Jev calls |
| [HEIST//ONE](https://github.com/) | Observable browser stealth game where Jev evaluates guard reactions |
| [Jev Plays Pokémon](https://github.com/) | FireRed/Showdown harness where Jev picks moves from current battle state |
| [Jev Plays StarCraft](https://github.com/) | Structured-state harness for StarCraft shareware campaign |
| [TypeSafe Mario](https://github.com/) | NES controller experiment with emulator telemetry |
| [Supercov](https://github.com/) | Code quality scorer — Jev ranks source files so agents know what to fix first |
| [Every](https://github.com/) | Semantic code search CLI: ask a yes/no question of every function, rank by probability |
| [Jev Review](https://github.com/) | Staged code-review workflow with local dashboard |

### Tier 4: Independent Evaluations

| Project | Description |
|---------|-------------|
| [calibre](https://github.com/) | Independent calibration measurement on Banking77 and Web of Science datasets |
| [Jev Judge vs Dimension Scores](https://github.com/) | 3-task measurement: direct Jev question vs 12–14 Jev-scored dimensions with fitted weights |
| [Jev Rerank Bench](https://github.com/) | Reranking comparison with raw provider responses and uncertainty intervals |
| [Jev Spam Eval](https://github.com/) | Zero-shot spam detection study against trained TF-IDF baselines |
| [OpenJev benchmark](https://github.com/TheoLeeCJ/openjev) | Head-to-head comparison of direct logit scoring vs Jev on shared test set |

## Three Reproduction Routes (from open-jev design doc)

### Route A: Frozen Encoder + Trained Head

Load a pretrained model (no LM head), freeze it, run one forward pass over context + options, pool hidden states, train a small scoring head with a ranking loss. Cheap and stable — the head trains in seconds per epoch.

**Best for:** when you have labeled data and the encoder's representations already contain the signal.

### Route B: Zero-Shot Likelihood Scoring (No Training)

Load with `AutoModelForCausalLM` (with LM head). Concatenate context and option. One forward pass. Read log-probability of each option token. Sum or average over option tokens. That's the score.

**Best for:** quick baseline, no labeled data needed. Handle length bias by normalizing.

### Route C: Fine-Tune the Backbone

Attach head as in Route A but unfreeze the backbone with LoRA. Train end-to-end. Only worth it once Route A plateaus and you have enough data.

**Best for:** maximum quality when Routes A and B aren't enough.

## Curated Lists

- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe) — comprehensive, well-maintained, reviewed 2026-09-17
- [awesome-jev](https://github.com/AnotiaWang/awesome-jev) — alternative curated list
