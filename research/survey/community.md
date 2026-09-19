# Community Resources

Trackers, awesome lists, benchmarks, and discussion channels.

## HuggingFace Reproductions Tracker

[multimodalart/jev-reproductions-tracker](https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker) — 38 artifacts cataloged by Apolinario. Searchable/filterable web app. Created 2026-09-17, actively updated.

## Awesome Lists

- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe) — official community list
- [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev) — 100+ projects
- [thevibeworks/awesome-typesafe-jev](https://github.com/thevibeworks/awesome-typesafe-jev)

## Benchmark Datasets

- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — 2,000 decisions across 4 domains (Agent Trace, Customer Service, Invoice Processing, Security). Teacher self-agreement ceiling: 0.735. Jev scored 0.727, Laya scored 0.766. Created by codelion.
- [C-Tianyu/NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) — procedurally generated maze/Snake game decisions

## Published Models

| Model | Backbone | Training | Calibration |
|---|---|---|---|
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | REINFORCE + proper scoring rules | Temperature-fitted ECE 0.081 |
| [C-Tianyu/NanoJev](https://huggingface.co/C-Tianyu/NanoJev) | Qwen3-0.6B | CE + RLCD experiment | Early-stage |
| [AlexWortega/openjev](https://huggingface.co/AlexWortega/openjev) | Qwen3.5-4B | NLI approach | — |
| [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | Qwen2.5-1.5B | No training (inference trick) | Uncalibrated |

## Independent Probing Studies

- [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked/) — 1,600+ probes, tokenizer fingerprinting, IIA analysis
- [SamuelSacco/jev-exploration](https://github.com/SamuelSacco/jev-exploration) — calibration audit
- [FirasSX914/calibre](https://github.com/FirasSX914/calibre) — confidence-based routing
- [anessbelbati/jev-rerank-bench](https://github.com/anessbelbati/jev-rerank-bench) — reranking benchmark

## ArXiv Note

"RLCD" was already used for a different technique (arXiv:2307.12950, "Reinforcement Learning from Contrastive Distillation"). TypeSafe's RLCD (Reinforcement Learning for Calibrated Decisions) is distinct. No independent academic paper reproducing TypeSafe's RLCD has been published as of 2026-09-19.

The paper most relevant to the reproduction landscape is arXiv:2407.02423 ("On the Anatomy of Attention"), cited by jevlike's author as the theoretical basis for reverse-engineering Jev's architecture from its type signature.
