# AGENTS.md

Instructions for AI coding agents working in Krino.

## Read These First

Read all `.md` files in `~/.config/agent-rules/` at session start for global conventions (git, testing, code style, submit workflow, etc.).

## Project Overview

Krino is a research project reverse-engineering and replicating TypeSafe AI's "Jev" System One model — a typed decision model that outputs structured probabilities (noul/choice/score) instead of free text.

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

## Key Files

| Path | Purpose |
|---|---|
| `data/format.py` | `TypedQuestion` dataclass — the universal data format |
| `data/synthetic.py` | Synthetic data generation pipeline orchestrator |
| `data/pipeline.py` | Benchmark data loaders (18 sources + synthetic) |
| `probing/scripts/jev_client.py` | Jev API client (stdlib only) |
| `_vendor/httpclient.py` | Zerodep async/sync HTTP client |
| `_vendor/dotenv.py` | Zerodep .env parser |
| `data/lsh.py` | Standalone MinHash/LSH module |

## Key Conventions

### Synthetic Data Pipeline

- Pipeline entry: `python -m data.synthetic`
- Composable stages: `--stages base,counterfactual,paraphrase,negation,shuffle,fill-variants,repair,dedup,validate,llm-label,jev-label,report`
- Data lives in `data/benchmarks/synthetic/` (gitignored)
- Data is stored on HuggingFace: `oaklight/krino-synthetic` (private)
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
python -m data.synthetic --push-to-hf oaklight/krino-synthetic

# Or via CLI
hf upload oaklight/krino-synthetic data/benchmarks/synthetic/ . --repo-type dataset
```

## Before Committing

1. Run `python -m pytest tests/` — all tests must pass
2. Do not commit data files (they are gitignored)
3. Do not add AI co-author lines to commit messages
4. Follow the submit workflow in `~/.config/agent-rules/submit-workflow.md`
