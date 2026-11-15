# Long-list readout figure

One standalone TikZ page with three panels at `K = 150`, on the CLINC150
in-scope classes:

- **A — schematic.** The 52-letter working alphabet, blocking into 3/5/6
  blocks, the three final-decision variants and their forward costs.
- **B — frontier.** Accuracy against forwards per item for schemes A, B, C and
  full per-option judgement; the single-pass `K = 52` and `K = 80` cells are
  kept in a grey band marked as a different problem.
- **C — decomposition.** `P(gold wins its block) x P(final picks the gold |
  it won) = end-to-end` at `b = 52 / 32 / 26`.

## Build

```bash
make build        # pdflatex twice in oaklight/texlive:science-cn
make clean        # remove the PDF and aux files
make clean-aux    # remove aux files, keep the PDF
make fmt          # latexindent in the same image
```

The figure uses pure TikZ (no pgfplots). The palette and type match
`research/figures/architecture/`.

## Data sources

Every number on the page is read from the committed result files:

| Number | Source |
|---|---|
| Scheme A/B/C counts and forwards at `b = 52/32/26` | `results/jevbench/clinc_blocking_Qwen_Qwen3-Reranker-4B.json`, `by_block_size` |
| Won-block rates, within-block first/last 20% | same file, `by_block_size` |
| Shuffled-assignment counts 129/138/148 | `results/jevbench/clinc_blocking_shuffle_Qwen_Qwen3-Reranker-4B.json` |
| Full judgement 127/200 at 150 forwards | `results/jevbench/clinc_judgement_full_Qwen_Qwen3-Reranker-4B.json` |
| Single pass `K = 52`: 146/200 | `results/jevbench/clinc_k52_Qwen_Qwen3-Reranker-4B.json` |
| Single pass `K = 80`: 125/200 | `results/jevbench/clinc_k80_Qwen_Qwen3-Reranker-4B.json` |
| `p = 0.0525`, symbol-tier failures | `research/45-blocking-frontier.md`, `research/43-code-tiers.md` |

## Status

CLINC150 200-item diagnostic. Blocking has not been run on JevBench or on
held-out data. `b = 26` against `b = 32` is unresolved (+5.0 pp,
`p = 0.0525` under the shuffle); the figure shows both points and names
neither the winner.
