# Synthetic Data Generation Pipeline

[English](README_en.md) | [中文](README_zh.md)

Generates ~50K typed decision training items across 10 domains and 12 cognitive types, with contrastive pairs, surface-form variants, and Jev API soft-label annotations.

## Pipeline Design

### Hybrid Approach

| Method | Domains | Source |
|---|---|---|
| **Seeded** | Medical (MedNLI), Legal (ContractNLI), Financial (TabFact), Scientific (ARC), Code (CodeSearchNet), Content (FEVER) | Real benchmark states + LLM-generated questions |
| **Pure generation** | Spatial reasoning, Product categorization, Education assessment, Safety/moderation | LLM generates both states and questions |
| **Augmentation** | All domains | Counterfactual, paraphrase, negation, option-order shuffle variants |

### Cross-Model Generation

Different LLMs at each stage to avoid single-model bias:

- **Base families**: Claude Sonnet (via `LLM_GEN_MODEL`)
- **Variants**: GPT-4.1 (via `LLM_VARIANT_MODEL`)
- **Validation**: Gemini Flash (via `LLM_JUDGE_MODEL`)

### Family Structure

Each family produces ~20 training items from one base state:

```
Family {
    base:           4 noul + 3 choice + 2 score questions
    counterfactual: modified state → flipped labels
    paraphrase:     reworded state/questions → same labels
    negation:       flipped question polarity → flipped noul labels
    shuffle:        permuted choice option order → same labels
}
```

### Composable Stages

Each stage is independently runnable and resumes from cached intermediates:

| Stage | Description | Intermediate file |
|---|---|---|
| `base` | Generate families (state + questions + gold labels) | `{domain}_families.jsonl` |
| `counterfactual` | Modified state with flipped labels | `{domain}_variants.jsonl` |
| `paraphrase` | Reworded state/questions preserving labels | `{domain}_variants.jsonl` |
| `negation` | Flipped question polarity for noul | `{domain}_variants.jsonl` |
| `shuffle` | Option-order permutation for choice | (computed at conversion) |
| `dedup` | LSH near-duplicate removal | overwrites `_families.jsonl` |
| `validate` | Cross-model validation via judge LLM | filters families in-memory |
| `jev-label` | Jev API soft-label annotation | adds `teacher_probs` field |

### Quality Controls

- **Label validation**: choice labels checked against criteria keys, score labels checked against range
- **Anti-leak prompts**: states contain raw scenarios without explanations or judgments
- **Confidence bucketing**: Jev soft labels partitioned into high (>0.9), medium (0.6–0.9), uncertain (<0.6)
- **LSH dedup**: MinHash on character n-grams detects near-duplicate states

## CLI Usage

```bash
# Full run: all 10 domains, 200 families each, all stages
python -m data.synthetic

# Pilot: 10 families in one domain
python -m data.synthetic --domains spatial_reasoning --pilot

# Base families only (no variants)
python -m data.synthetic --stages base --domains education_assessment

# Add variants to existing families
python -m data.synthetic --stages counterfactual,paraphrase,negation

# Dedup only (no LLM calls)
python -m data.synthetic --stages dedup

# Jev soft labels only
python -m data.synthetic --stages jev-label

# Push to HuggingFace
python -m data.synthetic --push-to-hf Oaklight/jev-synthetic

# Custom concurrency and models
LLM_GEN_MODEL="argo:claude-sonnet-4.6" \
LLM_VARIANT_MODEL="argo:gpt-4.1" \
python -m data.synthetic --max-concurrent 10
```

## Configuration

Set in `.env` at repo root:

```
LLM_BASE_URL=http://your-llm-endpoint:port
LLM_API_KEY=your-key-if-needed
LLM_GEN_MODEL=claude-sonnet-4-20250514
LLM_VARIANT_MODEL=GPT-4.1-mini
LLM_JUDGE_MODEL=gemini-2.0-flash
```

## File Layout

```
data/benchmarks/synthetic/
├── README_en.md                         # This file
├── README_zh.md                         # Chinese version
├── README.md -> README_en.md            # Symlink
├── {domain}_families.jsonl              # Raw family dicts (resumable)
├── {domain}_variants.jsonl              # Raw variant dicts (resumable)
└── synthetic.jsonl                      # Final TypedQuestion items
```

## Output Format

Each item is a `TypedQuestion` dict:

```json
{
    "id": "synthetic-spatial_reasoning-0042-noul-causal",
    "state": "A rectangular museum gallery measures 20m east-to-west...",
    "question": {"type": "noul", "instructions": "Could rearranging the exhibits cause a bottleneck?"},
    "label": true,
    "source": "synthetic",
    "split": "train",
    "group": "synthetic-spatial_reasoning-0042",
    "teacher_probs": {"true": 0.82, "false": 0.18}
}
```

The `teacher_probs` field (added by `jev-label` stage) contains Jev's full probability distribution for distillation training.
