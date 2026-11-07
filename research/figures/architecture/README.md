# Krino architecture figure

`architecture.pdf` is the **v4** model — the one that produced the canonical
arm and the noul arm. `architecture-v3-b2.tex` is the previous B2 figure,
kept because research/11 and research/12 still reference it.

- `architecture.tex` / `architecture.pdf` — v4, current
- `architecture-v3-b2.tex` — v3 B2 marker-residual arm (superseded)

## Build

```bash
make build        # pdflatex twice in oaklight/texlive:science-cn
make clean        # remove PDF and aux files
make clean-aux    # remove aux files, keep PDF
make fmt          # latexindent in the same image
```

The Makefile runs the container as the current uid/gid with `HOME=/tmp`, so
generated files stay user-owned. The image is `oaklight/texlive:science-cn`
(pulled locally, 4.34 GB).

## What the v4 figure shows

- **Backbone**: Qwen3.5-4B, hybrid GDN decoder. Base weights frozen; LoRA on
  the last 8 blocks, `r=16`, `alpha=32`, seven targets, about 3.27M trainable
  parameters.
- **Prefix**: `state` + `instructions` encoded once into a KV cache. The GDN
  recurrent state cannot be cropped, so a forked prefix is re-encoded; shape
  bucketing keeps the kernels warm.
- **Suffix**: listwise. Every option is concatenated into one suffix with a
  marker `M` after each, and the scores are read at the `M` positions. This is
  the difference from v3, which gave each option its own row and could not see
  the other options.
- **Heads**: a frozen base scorer that yields `p_base` (precomputed and
  cached), and one trainable `OptionMarkerHead` (about 251K parameters) shared
  by noul, choice and score.
- **AnswerableHead**: implemented and drawn dashed. It is wired off: on the
  rows it is meant to triage its AUROC is 0.458, below chance.
- **Fusion**: `z = log p_base + alpha_t * u`, one learned scalar per type.
  Measured values on the canonical arm are `alpha_noul = 0.00` (the type was
  not trained there), `alpha_choice = 0.09`, `alpha_score = 0.19`.
- **Loss**: `1.0 * KL(p_base || p) + 1.0 * CE(z, y)`. The KL anchor is on by
  default in v4; v3 used a weight of 0.2.
- **Inference**: LoRA folds into the base weights, so the shared-prefix cache
  stays valid and a decision costs 133 ms instead of 1.29 s.

## Status shown in the figure

Frozen held-out, 6,254 scored rows of 6,424, three seeds:

| type | delta vs frozen base | significant seeds |
| --- | ---: | ---: |
| choice | +1.83 to +2.61 pp | 3/3 |
| score | +2.89 to +3.05 pp | 2/3 |
| noul | +-0.00 pp | untrained in this arm |

JevBench public, 231 items: `cc` 0.5187, equal to the control. The canonical
arm did not pass the standard C gate; the admissibility clause was written
above the control value and is not attainable by construction (#263).

## Metric

In the figure and this file, `cc` is chance-corrected accuracy,
`(acc - 1/K) / (1 - 1/K)`, reported per type. It is not the official
Intelligence board metric.

## Sources

- `research/10-v3-redesign.md` (the plan the v3 figure came from)
- `research/29-lora-throughput-244.md` (LoRA targets, bucketing, folding)
- `research/27-matched-heldout-spec.md` (the frozen held-out and its power)
- `model/src/decision_model.py`, `model/src/listwise.py`,
  `model/scripts/train_listwise.py`, `model/scripts/eval_listwise.py`
