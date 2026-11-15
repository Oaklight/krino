# Readout taxonomy figure

One standalone TikZ page: **"Where does the per-option score come from? Eight
readout designs"**, plus a "Where Krino sits" panel. It maps the readout
designs the project considered, not only the ones it measured.

- `readout-taxonomy.tex` / `readout-taxonomy.pdf` — the figure
- `preview-1.png` — 150 dpi preview of the PDF

## Build

```bash
make build        # pdflatex twice in oaklight/texlive:science-cn
make png          # 150 dpi preview as preview-1.png
make clean        # remove the PDF, preview and aux files
make clean-aux    # remove aux files, keep the PDF
```

The figure is pure TikZ. The palette matches
`research/figures/architecture/`.

## Panel claims and their sources

Each panel is a design sketch. The source column names where the claim is
measured, or marks the design as an external reference that this repo does not
re-measure.

| panel | claim as drawn | source |
|---|---|---|
| 1 Full-option likelihood | score one option as the continuation of the shared context; `K` forwards; no head | design of the early baselines named in the figure (Oumi, Dibia); measured here as the continuation rows of `research/35-jevbench-readout-matrix.md` |
| 2 Vocabulary single-token code | one forward; read the `K` code logits at the answer position; at most 255 codes | `research/43-code-tiers.md`, `research/46-position-probes.md`, `results/jevbench/panel2_k80_Qwen_Qwen3-Reranker-4B.json`; the H2O-Lightning reference is measured in `research/47-h2o-lightning.md` |
| 3a Last-position head | one decision token; one linear to `K` logits, initialised from LM-head rows | external design ("Releasing the Inner Jev" as named in the figure); not measured in this repo |
| 3b Pointer / cross-attention | one query per decision; `q^T h_k` is option `k`'s score; one forward; no vocabulary anchor | external design (Kev, noted in `research/13-throughput-levers.md`); not measured in this repo |
| 3c Per-option marker | read the hidden state at each marker; one shared linear; 2 forwards for any `K`; listwise | `research/figures/architecture/` (the v4 figure and its README), `research/29-corpus-data-card.md` |
| 4 Diffusion fill | denoise every code slot at once; `T` steps; label set = options | external design (djev / Maisa, `diffusion-gemma`); the name is recorded in `model/experiments/jevbench_official_public_hard.json` and its score in `research/13-jevbench-public.md` |
| D1 Code logits at one answer position | one read position; bias-free linear to 255 code logits; held because a 255-way linear on a frozen backbone is the 0.351-vs-0.441 configuration and an untrained mask collapses | #324, as recorded in the panel's own note |
| R Pointwise reranker | one forward per option; JevBench public 150/231 = 64.9%; candidates never interact | `research/35-jevbench-readout-matrix.md` (reranker × judgement), `research/47-h2o-lightning.md` |
| Where Krino sits | A 64.9%, J-R1 59.3% (exact `p = 0.14` against A, unresolved), J-R2 32.0% at chance; Krino v4 today = 3c | `research/38-judgement-architecture.md` |

External design names (Oumi, Dibia, Drex, H2O-Lightning, "Releasing the Inner
Jev", Kev, djev) identify the published design each panel sketches. They are
not bibliography entries, and the figure does not claim a reproduction of any
of them.

## Status

The taxonomy is a design map. The accuracies quoted in panels R and "Where
Krino sits" belong to the cells measured in `research/35` and `research/38`;
the other panels are drawn for shape, not for a score.
