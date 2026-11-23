# Historical harness scripts

One-off scripts kept for provenance. They produced or recorded results cited in
`research/`, but they are not part of CI and are not maintained.

Run the shell scripts with `bash` (or `./name.sh` after this change made them
executable) from the repository root, with the variables below set.

Two groups:

- `233-lora-kl/` — the nine-arm LoRA/KL scan harness behind PR #252 and the
  canonical precompute/gate runs. Original names began with a dot; the dot is
  dropped here so the files are visible to normal tooling.
- `scratch/` — corpus-band selection, teacher-label recording, sweep
  registration, and tokenizer/readout diagnostics from the same period.

## Paths

Operator-local absolute paths were replaced with variables. Set them before
running a script:

| Variable | Meaning |
| --- | --- |
| `KRINO_ROOT` | the Krino checkout |
| `KRINO_DATA` | the canonical corpus and precompute directory |
| `KRINO_WORKTREE` | the scratch worktree the scan ran in |
| `KRINO_V3` | the v3 latched worktree |
| `KRINO_PYTHON` | the Python interpreter with the training dependencies |
| `HOME` | user home, used for the remaining cache paths |

`redact_pass.py` and `scan_leaks.py` are redaction tools. Their identifier
lists are sanitized examples; replace them with your operator-local names
before use.

## Data

`data/benchmarks/abstention_train_20k.jsonl` is not carried in git. It is
pinned in `data/hf.lock` and restored with `python -m data.hf_sync`.
