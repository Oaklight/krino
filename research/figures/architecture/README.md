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
- **AnswerableHead**: implemented and drawn dashed, and wired off. No
  committed measurement shows it clears the base-error floor on the rows it is
  meant to triage.
- **Fusion**: `z = log p_base + alpha_t * u`, one learned scalar per type.
  Measured on the canonical arm over three seeds: `alpha_noul = 0.00` (the
  type was not trained there), `alpha_choice` 0.0928 / 0.1335 / 0.0929 and
  `alpha_score` 0.1908 / 0.1408 / 0.1906 — that is 0.09–0.13 and 0.14–0.19.
- **Loss**: `1.0 * KL(p_base || p) + 0.2 * CE(z, y) + 1.0 * KL(p_base || q)`,
  where `q` is the readout distribution before fusion. The weights are the
  ones recorded in the canonical seed-0 checkpoint. In code `--kl-weight` is
  1.0 and `--ce-weight` is 0.2; `--lora-kl-weight`, the third term, defaults
  to 0 and was set to 1.0 for the canonical arm.
- **Inference**: LoRA folds into the base weights, so the shared-prefix cache
  stays valid and a decision costs 133 ms instead of 1.29 s.

## Status shown in the figure

Frozen held-out, 6,254 scored rows of 6,424, three seeds:

| type | delta vs frozen base |
| --- | ---: |
| choice | +1.83 to +2.61 pp |
| score | +2.89 to +3.05 pp |
| noul | +-0.00 pp (untrained in this arm) |

The deltas are ranges over the three seeds. The paired McNemar test is run on
the pooled held-out, not per type, so no per-type significance count is
reported.

JevBench public, 231 items, against a control `cc` of 0.5187:

| seed | `cc` | paired McNemar |
| --- | ---: | --- |
| 0 | 0.5187 | 1 win / 1 loss, p = 1.0 |
| 1 | 0.5129 | 1 win / 2 losses, p = 1.0 |
| 2 | 0.5187 | 0 wins / 0 losses, p = 1.0 |

The canonical arm did not pass the standard C gate: seed 1 misses the
admissibility floor by 0.58 pp, so JevBench is not unanimous. The exact
control value is 0.5186868686868686 and the admissibility clause is written
above it, so it is not attainable by construction (#263).

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
