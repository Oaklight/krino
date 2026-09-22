# CLAUDE.md

Project-specific instructions for Claude Code sessions in jev-explore.

## Project Overview

jev-explore is a research project reverse-engineering and replicating TypeSafe AI's "Jev" System One model — a typed decision model that outputs structured probabilities (noul/choice/score) instead of free text.

## Repository Layout

```
model/
  data/           — data pipeline: format, loaders, synthetic generation, dedup, LSH
  src/            — model architecture: backbone, decision heads, scorers
  training/       — supervised training, calibration losses
  evaluation/     — accuracy, bias, calibration metrics
  scripts/        — train/eval CLI scripts
  configs/        — training config YAML files
probing/
  scripts/        — Jev API client, black-box probes, httpclient
  results/        — probe outputs (JSONL)
research/         — writeups and analysis
_vendor/          — vendored zerodep modules (httpclient, dotenv)
tests/            — pytest test suite
```

## Key Conventions

### Synthetic Data Pipeline

- Pipeline entry: `python -m data.synthetic`
- Composable stages: `--stages base,counterfactual,paraphrase,negation,shuffle,fill-variants,repair,dedup,validate,llm-label,jev-label,report`
- Data lives in `data/benchmarks/synthetic/` (gitignored)
- Data is stored on HuggingFace: `oaklight/open-decisions-synthetic` (private)
- Config via `.env` at repo root (LLM_BASE_URL, LLM_API_KEY, etc.)

### Vendored Modules

- `_vendor/` contains zerodep modules only (installed via `zerodep add <module> -d _vendor`)
- `probing/scripts/jev_client.py` stays in place (shared with probe scripts)
- Do NOT move jev_client to _vendor

### Data Format

- `TypedQuestion` dataclass in `data/format.py` — the universal format
- Three question types: noul (bool), choice (str), score (float)
- `teacher_probs` field: multi-teacher namespaced dict `{"jev": {...}, "gpt_5_6_luna": {...}}`

### Testing

- Run: `python -m pytest tests/`
- All tests must pass before committing
- Synthetic pipeline tests run offline (no API calls)

### Environment

- Python projects use conda environments (check `conda info -e`)
- LLM access via rosetta gateway: `LLM_BASE_URL` in `.env`
- Jev API access: `TYPESAFE_API_KEY` in `.env`
- Seeded domains require pyarrow (available on ts-lambda10, not laptop)

### HuggingFace Data Sync

```bash
# Push synthetic data to HF
python -m data.synthetic --push-to-hf oaklight/open-decisions-synthetic

# Or via CLI
hf upload oaklight/open-decisions-synthetic data/benchmarks/synthetic/ . --repo-type dataset
```
