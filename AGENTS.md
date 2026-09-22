# AGENTS.md

Instructions for AI coding agents working in jev-explore.

## Read These First

Read all `.md` files in `~/.config/agent-rules/` at session start for global conventions (git, testing, code style, submit workflow, etc.).

Read `CLAUDE.md` in this repo root for project-specific context.

## Project Context

This is a research project replicating TypeSafe AI's Jev "System One" model. The model takes a state (context) and typed questions (noul/choice/score) and returns structured probabilistic answers.

## Working With Data

- Synthetic data pipeline: `python -m data.synthetic --help`
- Data is gitignored — stored on HuggingFace (`oaklight/open-decisions-synthetic`)
- After generating data, push to HF: `--push-to-hf oaklight/open-decisions-synthetic`
- Use `--stages report` to check generation status without running anything

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

## Before Committing

1. Run `python -m pytest tests/` — all tests must pass
2. Do not commit data files (they are gitignored)
3. Do not add AI co-author lines to commit messages
4. Follow the submit workflow in `~/.config/agent-rules/submit-workflow.md`
