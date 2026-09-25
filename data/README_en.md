# Data Pipeline

[English](README_en.md) | [中文](README_zh.md)

Downloads, converts, and manages benchmark datasets for decision model evaluation. All benchmarks are converted to a unified `TypedQuestion` format.

## Quick start

```bash
pip install -e '.[data]'

# Download and convert specific benchmarks
python -m data.pipeline banking77 sst2 arc

# Download all benchmarks
python -m data.pipeline
```

Converted data is saved as JSONL files in `data/benchmarks/`.

## Benchmarks

26 loaders across 3 question types and multiple domains:

**Choice** (pick one option):
banking77, agnews, arc, race, hellaswag, fever, swag, codesearchnet, mmlu, winogrande, piqa, commonsenseqa, logiqa, typed_decisions

**Noul** (yes/no probability):
sst2, mnli, tabfact, multirc, mednli, contractnli, anli, boolq

**Score** (ordered levels):
stsb, sst5, yelp

**Synthetic**:
synthetic (generated via the synthetic pipeline)

## TypedQuestion format

All benchmarks convert to `TypedQuestion` (defined in `format.py`):

```python
TypedQuestion(
    id="banking77-test-00042",
    state="I was charged twice for the same transaction",
    question={"type": "choice", "instructions": "...", "criteria": {...}},
    label="transaction_charged_twice",
    source="banking77",
    split="test",
    group="banking77-42",
)
```

Three question types: `noul` (bool label), `choice` (str label), `score` (float label).

## Synthetic data pipeline

```bash
python -m data.synthetic --stages base,counterfactual,paraphrase,negation,shuffle,dedup,validate
python -m data.synthetic --stages report    # check status without running
```

Stages: `base` → `counterfactual,paraphrase,negation` → `shuffle` → `fill-variants` → `repair` → `dedup` → `validate` → `llm-label` → `jev-label` → `report`

Data is stored on HuggingFace: [`oaklight/krino-synthetic`](https://huggingface.co/datasets/oaklight/krino-synthetic) (private).

## Structure

```
data/
├── format.py           # TypedQuestion dataclass
├── pipeline.py         # 26 benchmark loaders
├── sampler.py          # Stratified sampling utilities
├── synthetic.py        # Synthetic data generation orchestrator
├── synthetic_*.py      # Stage implementations
├── balance.py          # Class balancing
├── lsh.py              # MinHash/LSH deduplication
└── benchmarks/         # Downloaded data (gitignored)
    ├── synthetic/      #   Synthetic data + HF dataset card
    └── openjev/        #   OpenJev benchmark data
```

## Key files

| File | Purpose |
|---|---|
| `format.py` | `TypedQuestion` dataclass — the universal data format |
| `pipeline.py` | All 26 benchmark loaders + CLI entry point |
| `synthetic.py` | Synthetic data generation pipeline orchestrator |
| `lsh.py` | Standalone MinHash/LSH for deduplication |
| `sampler.py` | Stratified sampling for balanced evaluation subsets |
