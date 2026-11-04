# Krino v3 B2 architecture figure

Standalone TikZ figure of the B2 marker-residual arm. `architecture.pdf` is
20.9 cm x 17.1 cm, one page, fonts embedded.

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

## What the figure shows

- Frozen Qwen3-Reranker-4B. The shared prefix (`state` + `instructions`) is
  encoded once into a KV cache; each option is appended with an EOS marker
  token `M`, and the score is read at `M`. `chunk=1` is the exact setting;
  `option_batch=16` expands the cache along the batch dimension without
  changing the semantics.
- Three paths from the frozen model: the base reranker distribution
  (`p_base`, precomputed and cached), a trainable NoulHead over the pooled
  context, and a trainable OptionMarkerHead (about 251K parameters) shared by
  choice and score.
- Per-type zero gates: `z = log p_base + alpha_t * u` with
  `alpha_noul = alpha_choice = alpha_score = 0` at initialisation, so step 0
  reproduces the raw base arm B0.
- Loss: `1.0 * KL(p_base || p) + 0.2 * CE(z, y)`, with the teacher soft label
  when present and the hard label otherwise.
- Outputs: `noul` sigmoid, `choice` softmax over K, `score` softmax over
  levels.

The dashed strip lists proposed work that is not in B2: per-epoch band
re-scoring with the current head, a 4096 long-context arm (blocked on data),
and the full-corpus base-probability cache (96k/195k). The encoder ablation
is closed and is deliberately absent.

## Metric

In the figure and this file, `cc` is chance-corrected accuracy on the
231-item JevBench public proxy. It is not the official Intelligence board
metric.

## Status shown in the figure (2026-11-03)

| Arm | Status | Evidence |
| --- | --- | --- |
| B2 + confirm | validated baseline | chance-corrected accuracy 0.4613 on the 231-item JevBench public proxy, mean over 3 seeds |
| Sweep knobs | closed | D20K, LoRA, band and KL 0.5 all closed; KL 0.5 rejected on confirm (chance-corrected 0.4538 / 0.4545 on the 231-item JevBench public proxy, seeds 43/44) |
| Teacher-CE | running | data arm, not architecture |

## Sources

- `research/11-4b-primary-arm.md` (B0/B1/B2 and the residual formulation)
- `research/12-marker-readout-result.md` (marker vs pooled readout)
- `research/13-throughput-levers.md` (`option_batch`)
- `model/src/decision_model.py`, `model/src/heads.py`,
  `model/scripts/train_residual.py`, `model/scripts/eval_gate.py`

## Suggested caption

> The B2 marker-residual arm. A frozen Qwen3-Reranker-4B encodes a shared
> prefix and each marker-tagged option; a 251K-parameter OptionMarkerHead
> scores the marker positions. A zero-initialised per-type gate preserves the
> base reranker distribution, `z = log p_base + alpha_t * u`, and training adds
> a KL anchor to `p_base`. Dashed items are proposed, not in B2.
