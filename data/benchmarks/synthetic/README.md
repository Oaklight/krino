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
size_categories:
  - 10K<n<100K
pretty_name: Open Decisions Synthetic
---

# Open Decisions Synthetic Dataset

Synthetic training data for typed decision models — models that output structured probabilistic answers (yes/no probabilities, categorical distributions, ordinal ratings) instead of free text.

## Dataset Summary

- **68.8K items** across 10 domains, 12 cognitive types
- **Hybrid generation**: 6 seeded domains (real benchmark states) + 4 pure-generation domains
- **Family-based**: each state produces ~35 related items (base + counterfactual + paraphrase + negation + shuffle variants)
- **Cross-model generation**: Claude Sonnet (base), GPT-4.1 (variants), with label repair via GPT-4.1-nano

## Domains

| Domain | Type | Seed Source | Families | Items |
|---|---|---|---|---|
| Medical triage | seeded | MedNLI | 196 | 6,849 |
| Legal judgment | seeded | ContractNLI | 194 | 6,382 |
| Code review | seeded | CodeSearchNet | 200 | 6,980 |
| Financial analysis | seeded | TabFact | 197 | 6,860 |
| Scientific reasoning | seeded | ARC | 200 | 6,983 |
| Content analysis | seeded | FEVER | 200 | 6,969 |
| Spatial reasoning | pure-gen | — | 200 | 6,968 |
| Product categorization | pure-gen | — | 200 | 6,985 |
| Education assessment | pure-gen | — | 198 | 6,899 |
| Safety/moderation | pure-gen | — | 200 | 6,980 |

## Question Types

Three typed question primitives following the [TypeSafe System One](https://docs.typesafe.ai) API:

- **Noul** (Bernoulli): yes/no proposition → P(true) ∈ [0, 1]
- **Choice**: pick one from labeled options → categorical probability distribution
- **Score**: rate on ordered levels → ordinal probability distribution

Type ratio: noul 57% / choice 28% / score 14%

## Cognitive Types (12)

Each noul question targets a specific reasoning pattern: entailment, fact verification, comparison, temporal, causal, sufficiency, consistency, possibility, classification, safety, counterfactual, threshold.

## Variant Types

Each family generates ~35 items through:

| Variant | Description | Trains |
|---|---|---|
| Base | Original state + questions | Standard accuracy |
| Counterfactual | Modified state → flipped labels | Contrastive reasoning |
| Paraphrase (state) | Reworded state → same labels | Surface-form invariance |
| Paraphrase (question) | Reworded questions → same labels | Question-form invariance |
| Negation | Flipped question polarity → flipped labels | Bidirectional noul |
| Shuffle | Permuted choice option order → same labels | Ordering bias reduction |

## File Structure

```
{domain}_families.jsonl    — raw family dicts (state + questions + gold labels)
{domain}_variants.jsonl    — variant dicts (counterfactual, paraphrase, negation)
```

The final `TypedQuestion` items are assembled deterministically from these files using:

```bash
pip install krino
python -m data.synthetic --stages report   # verify data
python -m data.pipeline synthetic           # load as TypedQuestion objects
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
    "group": "synthetic-medical_triage-0042"
}
```

## Quality Controls

- **Label validation**: choice labels checked against criteria keys, score labels checked against range
- **Cross-model validation**: judge model (Gemini) verifies label correctness
- **LSH dedup**: MinHash on character n-grams detects near-duplicate states
- **Repair**: invalid labels fixed via cheap LLM (GPT-4.1-nano)
- **Alignment verification**: family-variant pairing checked before use

## Generation Pipeline

```
base → counterfactual,paraphrase,negation → shuffle → fill-variants → repair → dedup → validate → llm-label → jev-label
```

Source code: [Oaklight/krino](https://github.com/Oaklight/krino)

## Citation

```bibtex
@misc{krino-synthetic-2026,
    title={Open Decisions Synthetic Dataset},
    author={Peng Ding},
    year={2026},
    url={https://huggingface.co/datasets/oaklight/krino-synthetic}
}
```

---

[中文版本](README_zh.md)
