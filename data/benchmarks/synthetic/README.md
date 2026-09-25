---
license: cc-by-nc-4.0
task_categories:
  - text-classification
  - question-answering
language:
  - en
tags:
  - synthetic
  - typed-decisions
  - calibration
  - distillation
  - noul
  - choice
  - score
  - multi-teacher
size_categories:
  - 100K<n<1M
pretty_name: Krino Synthetic
---

# Krino Synthetic Dataset

Synthetic training data for typed decision models — models that output structured probabilistic answers (yes/no probabilities, categorical distributions, ordinal ratings) instead of free text.

## Quick Start

```python
# Install
pip install krino

# Load all 154K synthetic items with teacher labels
from data.pipeline import load_all
items = load_all(["synthetic"])
print(f"{len(items)} items loaded")

# Each item has multi-teacher soft labels
item = items[0]
print(item.question)        # {"type": "noul", "instructions": "..."}
print(item.label)           # True/False (gold label)
print(item.teacher_probs)   # {"jev": {"true": 0.92, "false": 0.08}, "gpt_5_6_luna": {"true": 0.98, "false": 0.02}}
```

No intermediate files needed — items are assembled on-the-fly from the per-domain source files.

## Dataset Summary

- **154K items** across 23 domains, 12 cognitive types
- **Multi-teacher labels**: Jev (System One) + GPT-5.6 Luna (reasoning_effort=high)
- **86% teacher agreement** — 14% disagreement items are calibration-diagnostic gold
- **Hybrid generation**: 14 seeded domains (real benchmark states) + 9 pure-generation domains
- **Family-based**: each state produces ~35 related items (base + counterfactual + paraphrase + negation + shuffle variants)

## Domains

### Original Domains (10)

| Domain | Type | Seed Source | Items |
|---|---|---|---|
| Medical triage | seeded | MedNLI | 6,849 |
| Legal judgment | seeded | ContractNLI | 6,382 |
| Code review | seeded | CodeSearchNet | 6,980 |
| Financial analysis | seeded | TabFact | 6,860 |
| Scientific reasoning | seeded | ARC | 6,983 |
| Content analysis | seeded | FEVER | 6,969 |
| Spatial reasoning | pure-gen | — | 6,968 |
| Product categorization | pure-gen | — | 6,985 |
| Education assessment | pure-gen | — | 6,899 |
| Safety/moderation | pure-gen | — | 6,980 |

### Reasoning Domains (5)

| Domain | Type | Seed Source | Items |
|---|---|---|---|
| Academic reasoning | seeded | MMLU | 6,917 |
| Commonsense decision | seeded | CommonsenseQA | 6,984 |
| Logical inference | seeded | LogiQA | 6,916 |
| Adversarial inference | seeded | ANLI | 6,941 |
| Passage decision | seeded | BoolQ | 6,978 |

### Sequential Decision Domains (5)

| Domain | Type | Items |
|---|---|---|
| Game strategy | pure-gen | 6,722 |
| Navigation planning | pure-gen | 6,972 |
| Resource management | pure-gen | 6,988 |
| Sequential action | pure-gen | 6,978 |
| Multi-agent coordination | pure-gen | 6,957 |

### Long Context Domains (3)

| Domain | Type | Seed Source | Items |
|---|---|---|---|
| Long document | seeded | QuALITY | 2,374 |
| Multi-hop reasoning | seeded | HotpotQA | 6,704 |
| Numerical reasoning | seeded | DROP | 6,806 |

## Multi-Teacher Labels

Every item has probability distributions from two teachers:

| Teacher | Model | Intelligence | Calibration | Coverage |
|---|---|---|---|---|
| Jev | System One (jev-1.13.0) | baseline | baseline | 99.3% |
| Luna | GPT-5.6 Luna (reasoning_effort=high) | 96.8 | 89.8 | 91.7% |

**Teacher agreement: 86.1%** across all domains. Disagreements are concentrated in ambiguous domains (navigation 79%, spatial 81%, safety 82%) — exactly where calibration training is most valuable.

## Question Types

Three typed question primitives following the [TypeSafe System One](https://docs.typesafe.ai) API:

- **Noul** (Bernoulli): yes/no proposition → P(true) ∈ [0, 1]
- **Choice**: pick one from labeled options → categorical probability distribution
- **Score**: rate on ordered levels → ordinal probability distribution

Type ratio: noul 57% / choice 28% / score 14%

## Context Length Distribution

```
     0-100 chars     354 items (  8%)  short single-sentence states
   100-500          1354       ( 30%)  paragraph-length states
   500-1K           1020       ( 23%)  multi-paragraph states
  1K-2K             1021       ( 23%)  sequential decision states
  2K-5K              398       (  9%)  long-form states
  5K-10K             282       (  6%)  long context (documents, multi-hop)
  10K+                47       (  1%)  very long (legal contracts)
```

Median: 695 chars. Mean: 1,441 chars.

## File Structure

```
{domain}_families.jsonl              — raw family dicts (state + questions + gold labels)
{domain}_variants.jsonl              — variant dicts (counterfactual, paraphrase, negation)
{domain}_jev_labels.jsonl            — Jev soft-label annotations
{domain}_gpt_5_6_luna_labels.jsonl   — Luna soft-label annotations
```

Items are assembled on-the-fly by `load_synthetic()` — no monolithic output file. This avoids duplication (states are shared across ~35 variant items per family).

## CLI Usage

```bash
# Report generation status
python -m data.synthetic --stages report

# Generate new domain data
python -m data.synthetic --domains game_strategy --families-per-domain 200 \
    --stages base,counterfactual,paraphrase,negation,shuffle

# Add Jev labels
python -m data.synthetic --domains game_strategy --stages jev-label

# Add Luna labels
python -m data.synthetic --domains game_strategy --stages llm-label \
    --llm-teacher argo:gpt-5.6-luna

# Push to HuggingFace
python -m data.synthetic --push-to-hf oaklight/open-decisions-synthetic
```

## Item Format

```json
{
    "id": "synthetic-medical_triage-0042-noul-causal",
    "state": "Premise: 78F HTN presents with LLQ pain...",
    "question": {
        "type": "noul",
        "instructions": "Could the hypertension contribute to the abdominal presentation?"
    },
    "label": true,
    "source": "synthetic",
    "split": "train",
    "group": "synthetic-medical_triage-0042",
    "teacher_probs": {
        "jev": {"true": 0.82, "false": 0.18},
        "gpt_5_6_luna": {"true": 0.95, "false": 0.05}
    }
}
```

## Training Integration

```python
from data.pipeline import load_all

# Load synthetic + benchmark data
items = load_all(["synthetic", "banking77", "mnli", "arc"])

# Use teacher_probs for distillation
for item in items:
    if item.teacher_probs:
        # KL-divergence loss: student || teacher
        jev_probs = item.teacher_probs.get("jev")
        luna_probs = item.teacher_probs.get("gpt_5_6_luna")
        # Choose teacher or ensemble
```

## Citation

```bibtex
@misc{krino-synthetic-2026,
    title={Krino Synthetic Dataset},
    author={Peng Ding},
    year={2026},
    url={https://huggingface.co/datasets/oaklight/open-decisions-synthetic}
}
```

Source code: [Oaklight/krino](https://github.com/Oaklight/krino)

---

[中文版本](README_zh.md)
