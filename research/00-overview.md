# Jev & TypeSafe.ai — Research Overview

**Date:** 2026-09-18
**Status:** Initial exploration

## What is TypeSafe.ai?

[TypeSafe AI](https://typesafe.ai) is a San Francisco-based AI startup that came out of stealth on September 15, 2026, with $40M in funding. Their mission: "pave the shortest path to an AI-based economic revolution by making intelligence composable." Tagline: **"Build Prod, Not God"** — explicitly not chasing AGI.

### Founding Team

- **Diogo Almeida (CEO)** — Former OpenAI and Google Brain researcher. Co-author of the InstructGPT paper (the research behind ChatGPT), credited as a co-inventor of RLHF. Left OpenAI in 2024 to found TypeSafe.
- **Sasha Sheng (COO)** — Ex-Meta/FAIR research engineer. Published at NeurIPS and ECCV.
- **Erik Gafni (CTO)** — Repeat founder (Ravel, multimodal AI for DNA sequencing), early employee at two unicorns (Invitae, Freenome).

Team includes alumni from OpenAI, Google Brain, Meta/FAIR, Stripe, Airbnb, Plaid, and Docker.

### Core Thesis

LLMs are optimized for human-facing chat (via RLHF), but large-scale automation will be dominated by AI-to-AI and AI-to-software interactions. The text generation interface is the wrong abstraction for decisions that software needs to consume. TypeSafe built a new model class that outputs typed, structured decisions instead of text.

## What is Jev?

Jev is TypeSafe's flagship model and the first **"System One" model** (named after Kahneman's *Thinking, Fast and Slow* — fast, intuitive judgments). Current version: `jev-1.13.0`.

"Jev" is named after **William Stanley Jevons** (the Jevons paradox economist) — cheaper intelligence will increase total consumption, not reduce it.

### How It Differs from LLMs

| Aspect | Standard LLMs | Jev |
|--------|---------------|-----|
| Output | Free-form text (strings) | Typed structured values: choices, scores, probabilities |
| Sampling | Sequential autoregressive (one token at a time) | Parallel (all outputs in a single forward pass) |
| Speed | 3–329 seconds for frontier models | 70ms–500ms |
| Input cost | $0.20–$10 / MTok | $0.042 / MTok |
| Output cost | ~5× input cost | **$0 (free)** |
| Hallucination | Always possible | Structurally impossible (outputs constrained to predefined schema) |
| Training | RLHF / RLVR | RLCD (Reinforcement Learning for Calibrated Decisions) |
| Confidence | Overconfident, inconsistent | Calibrated probabilities with every output |

### What Jev Cannot Do

Generate text, write code, hold a conversation, produce explanations, or do anything involving open-ended string generation. It is not a chatbot or a general-purpose language model.

### What Jev Is Good At

Classification, routing, scoring, semantic branching ("smart if-statements"), guardrailing LLM outputs, judging/verifying, real-time decision-making, and map-reduce over large datasets.

## Three Primitives

| Type | Purpose | Returns |
|------|---------|---------|
| **Noul** | Yes/no probability | `noul` (0–1 float) |
| **Choice** | Pick one from a set (up to 255 options) | `choice`, `probabilities`, `confidence` |
| **Score** | Rate on ordered levels (2–10 levels) | `score`, `probabilities`, `legend`, `confidence` |

All three can be mixed in a single API call. Every question is evaluated in parallel and independently against the same state. Adding questions barely changes response time.

## Available Models

| Model | Description | Released |
|-------|-------------|----------|
| `jev-latest` | Stable (currently jev-1.13.0) | 2026-09-10 |
| `jev-preview` | Preview, should be better in most ways | 2026-09-10 |

## Key Links

- [TypeSafe docs](https://docs.typesafe.ai)
- [API reference](https://docs.typesafe.ai/api)
- [AI primer (RLCD explanation)](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [System One concept](https://docs.typesafe.ai/concepts/system-one)
- [Model jaggedness (known limitations)](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [Launch blog post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Manifesto](https://typesafe.ai/manifesto)
- [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [JS SDK](https://github.com/typesafe-ai/typesafe-sdk-js)
- [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)
- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe)

## Document Index

- [00-overview.md](00-overview.md) — This document
- [01-api-testing.md](01-api-testing.md) — API testing results with all three primitives
- [02-architecture-analysis.md](02-architecture-analysis.md) — Architecture deep dive and reproduction analysis
- [03-open-source-ecosystem.md](03-open-source-ecosystem.md) — Open-source projects and community ecosystem
