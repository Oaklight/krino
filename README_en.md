<p align="center">
  <a href="README_en.md">English</a> | <a href="README_zh.md">中文</a>
</p>

# Krino

*κρίνω — to judge, to separate, to decide*

Krino is a research project studying **decision models** — a new class of AI systems that return calibrated probabilities over structured options instead of generating text. The project reverse-engineers, replicates, and extends [TypeSafe AI](https://typesafe.ai)'s Jev, the first commercially deployed decision model.

## What are decision models?

A decision model takes unstructured input (a support ticket, a contract clause, a code snippet) and a set of typed questions, and returns structured probabilistic answers — no text generation, no parsing, no retries.

| Question type | What it answers | Output |
|---|---|---|
| **Noul** | Is this true? | Probability 0–1 |
| **Choice** | Which of these options? | Per-option probability distribution |
| **Score** | Where on this scale? | Probability-weighted position |

The key differentiator: these probabilities are designed to be **calibrated** — when the model says 0.8, the answer should be correct ~80% of the time. This enables threshold-based automation (route, block, escalate) without per-task classifier training.

## Project components

| Directory | What it does | Highlights |
|---|---|---|
| [`probing/`](probing/) | Black-box API probing of Jev | 21 probes, 5,620 API calls; 5 preregistered controlled experiments |
| [`model/`](model/) | Open decision model replication | 95.2% on Banking77 (vs Jev's 75%); 10 backbones × 19 benchmarks |
| [`data/`](data/) | Data pipeline + synthetic generation | 26 benchmark loaders; synthetic data with counterfactuals and paraphrases |
| [`research/`](research/) | Analysis writeups + ecosystem survey | Architecture fingerprinting; reproduction landscape |

## Key findings

1. **Reranker pretraining transfers to decision scoring.** A 150M cross-encoder reranker (Ettin-150m) with trained decision heads achieves 95.2% on Banking77 — beating a vanilla encoder at identical params by +4.8pp, and outperforming Jev itself (75.0%).

2. **Zero-training logit readout works surprisingly well.** Reading option probabilities directly from frozen LLM logits (no training, no text generation) reaches 68.5% average accuracy across 19 benchmarks with Qwen2.5-7B — within 19pp of Jev's 87.2%.

3. **The decision model ecosystem is rapidly forming.** JevBench tracks 52 systems across proprietary, open-trained, and zero-training approaches. Multiple independent projects (SemIf, djev, Decider) demonstrate that decision-making capability is latent in pretrained LLMs and can be extracted through structured logit readout.

## Getting started

```bash
# Clone and install
git clone https://github.com/Oaklight/krino.git
cd krino
pip install -e '.[data]'

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

## Links

| | |
|---|---|
| Paper | [Oaklight/krino-paper](https://github.com/Oaklight/krino-paper) |
| PyPI | [krino](https://pypi.org/project/krino/) |
| Dataset | [oaklight/krino-synthetic](https://huggingface.co/datasets/oaklight/krino-synthetic) (private) |
| Docs | [oaklight.github.io/krino](https://oaklight.github.io/krino) |
| TypeSafe docs | [docs.typesafe.ai](https://docs.typesafe.ai) |

## Citation

```bibtex
@misc{krino-2026,
    title={Krino: Probing, Replication, and Analysis of JEV-Class Decision Models},
    author={Peng Ding},
    year={2026},
    url={https://github.com/Oaklight/krino}
}
```

## License

MIT
