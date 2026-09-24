# Krino

*κρίνω — to judge, to separate, to decide*

Probing, analysis, and open replication of [TypeSafe.ai](https://typesafe.ai)'s Jev — the first System One model for typed, probabilistic decisions.

## Highlights

- **Black-box probing** of the Jev API (21 probes, 5,620 API calls) identified the architecture: causal Qwen-family backbone + typed decision heads + RLCD calibration
- **Open replication** achieving **95.2% on Banking77** (vs Jev's 77.8%) with a 150M reranker-pretrained encoder + trained decision heads
- **Novel finding:** cross-encoder reranker pretraining transfers to decision scoring — Ettin-150m beats vanilla ModernBERT by +4.8pp at identical params
- **19-benchmark evaluation suite** across 3 question types (noul, choice, score) and 6 domains

## Project structure

```
probing/          # Black-box probes of the Jev API
  scripts/        #   Probe implementations
  results/        #   Raw JSONL results
research/         # Analysis and writeups
  00-overview.md  #   What is Jev / TypeSafe
  02-architecture-analysis.md  # Architecture fingerprinting conclusions
  04-model-replication.md      # Model replication findings (main results doc)
model/            # Open decision model
  src/            #   Backbone, heads, scorers
  data/           #   Pipeline: 19 benchmarks → unified TypedQuestion format
  training/       #   Supervised + calibration training
  evaluation/     #   Accuracy, ECE, Brier, bias metrics
  scripts/        #   Train, eval, serve entry points
```

## What is Jev?

Jev is a "System One" model that returns typed, probabilistic decisions instead of generated text. It evaluates a `state` against typed `questions` and returns structured `answers` with calibrated probabilities.

| Type | Purpose | Returns |
|------|---------|---------|
| **Noul** | Yes/no probability | `noul` (0–1 float) |
| **Choice** | Pick one from a set | `choice`, `probabilities`, `confidence` |
| **Score** | Rate on ordered levels | `score`, `probabilities`, `legend`, `confidence` |

## Results

### Trained heads (Banking77, 77-class intent classification)

| Backbone | Params | Accuracy |
|----------|--------|----------|
| **Ettin-150m r128** | 150M + ~800K | **95.2%** |
| Qwen3-0.6B r64 | 596M + 402K | 93.2% |
| ModernBERT-base r64 | 149M + 301K | 89.0% |
| Jev (reference) | unknown | 77.8% |

### Zero-training baselines (average across 19 benchmarks)

| Backbone | Params | Avg Accuracy |
|----------|--------|-------------|
| Qwen2.5-7B-Instruct | 7B | 68.5% |
| Ettin-400m | 400M | 59.4% |
| Qwen2.5-1.5B | 1.5B | 59.2% |

See [research/04-model-replication.md](research/04-model-replication.md) for full analysis.

## Setup

```bash
pip install -e '.[data]'    # core + data pipeline
pip install -e '.[train]'   # + training dependencies
```

Or from PyPI (placeholder for now):

```bash
pip install krino
```

## Quick start

```bash
# Download and convert benchmarks
python -m data.pipeline banking77 sst2

# Evaluate a backbone (zero-training logit readout)
python model/scripts/eval_logit.py --model Qwen/Qwen3-0.6B

# Train decision heads on a frozen backbone
python model/scripts/train.py --encoder --model answerdotai/ModernBERT-base \
    --train-data data/benchmarks/banking77.jsonl \
    --eval-data data/benchmarks/banking77.jsonl \
    --epochs 5
```

## Probing

The [controlled probe suite](probing/controlled/README.md) preregisters five stdlib-only experiments and records raw HTTP evidence without credentials.

```bash
python probing/scripts/controlled/probe_1_output_serialization.py --mode smoke --seed 20260918 --dry-run
```

## Links

- [TypeSafe docs](https://docs.typesafe.ai)
- [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)
- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe)
