# 09 — Expanding the ordinal (`score`) data supply

**Date:** 2026-10-28
**Scope:** find, evaluate, and wire external sources of ordinal / graded /
severity / quality-rated data that convert to Krino's 3-5 level score question.

## TL;DR

Three external sources are wired in, each capped at 25,000 train items:

| Source | Licence | Native label | Train items added |
|---|---|---:|---:|
| HelpSteer | CC-BY-4.0 | human 0-4 ratings on five attributes | 25,000 |
| WMT direct assessment | Apache-2.0 | human 0-100 translation quality | 25,000 |
| UltraFeedback (binarized) | MIT | GPT-4 1-10 overall response score | 24,983 |

In the score-balanced 0.6B config the score pool goes from **39,357 to
114,340** items. The pool table after the change:

```
Type         Pool    Slots  Resampling
choice     77,594    5,000       0.06x
noul      100,311    5,000       0.05x
score     114,340    5,000       0.04x
```

Score is no longer the starved primitive. It now has the largest pool and the
lowest resampling multiplier; the sampler's 1:1:1 type ratio is no longer
being paid for by repeating score items three to four times more often.

The default 4B config was at 27,235 score items per epoch (issue #142). The
score-balanced config had already raised the yelp and synthetic caps before
this change, so its pre-change pool was 39,357.

## Selection criteria

1. **Licence clarity.** Prefer permissive licences that allow commercial use;
   record non-commercial and share-alike terms explicitly.
2. **Label integrity.** Human ratings first, independent-model judgements
   second, synthetic labels last.
3. **Size.** At least 10,000 usable items per source.
4. **Low conversion risk.** A released scalar or ordered scale, not a
   judgement we have to infer from free text.
5. **Label diversity.** At least three distinct levels, no single level above
   roughly 90% of items.
6. **No answer leakage.** The instruction must not contain the gold label.

## Ranked candidates (top 10)

`Rows` is the released row count, not the item count after expansion. `State`
is the estimated token length (characters / 4) of the converted state.

| # | Source | Family | Rows | Licence | Commercial | Label → levels | State | Decision |
|---:|---|---|---:|---|---|---|---|---|
| 1 | [WMT DA human evaluation](https://huggingface.co/datasets/RicardoRei/wmt-da-human-evaluation) | translation quality | 1,285,190 | Apache-2.0 | yes | human 0-100 DA → five 20-point bands | short, p50 ~59 | **wired** |
| 2 | [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | response quality | 35,331 train + 1,789 val | CC-BY-4.0 | yes | human 0-4 on helpfulness, correctness, coherence, complexity, verbosity | long, p50 ~775 | **wired** |
| 3 | [UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | response quality | 61,135 preference rows (122,270 ratings) | MIT | yes | GPT-4 1-10 overall score → five equal-width bands | long, p50 ~394 | **wired** |
| 4 | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | data quality | ~1.3T tokens | ODC-By 1.0 | yes, attribution | model 0-5 educational score | very long web docs | issue #151 |
| 5 | [DCLM-BASELINE](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | data quality | ~4T tokens | CC-BY-4.0 | yes | fastText educational score | long web docs | issue #151 |
| 6 | [Open-Jev v2](https://huggingface.co/datasets/ZefanCai/Open-Jev) | other decision model | 79,116 (11,923 score) | CC0-1.0 data | yes | 3-5 ordered options, one-hot/soft target | p50 ~97 | issue #152 |
| 7 | [AES2 essay scoring](https://huggingface.co/datasets/tasksource/AES2-essay-scoring) | human-graded writing | 17,307 | CC-BY-NC-4.0 | **no** | human holistic 1-6 | long essay | issue #153 |
| 8 | [DICES](https://github.com/google-research-datasets/dices-dataset) | safety severity | 115,153 rater rows (1,340 conversations) | CC-BY-4.0 | yes | five-point safety rating | long multi-turn | issue #154 |
| 9 | [TREC-COVID qrels](https://huggingface.co/datasets/BeIR/trec-covid-qrels) | graded relevance | 66,336 qrels (171,332 docs) | CC-BY-SA-4.0 | yes, share-alike | 0-2 relevance grades | abstract length | issue #155 |
| 10 | [WMT MQM human evaluation](https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation) | translation quality | 150,347 | Apache-2.0 | yes | negative error counts plus a 100 sentinel | short | rejected, semantics unclear |

## What was wired in

All three loaders live in `data/pipeline.py`, are registered in `LOADERS`, and
have tests in `tests/test_pipeline.py`.

### HelpSteer (`load_helpsteer`)

* **Licence:** CC-BY-4.0. Attribution required; commercial use allowed.
* **Conversion:** one item per (row, attribute). `state` is
  `Prompt: ...\n\nResponse: ...`; each attribute has its own 5-level rubric.
  `label` is the released integer 0-4.
* **Cap:** 5,000 train rows × 5 attributes = 25,000 items; 1,000 validation
  rows × 5 = 5,000 eval items (config caps eval to 500).
* **Observed labels:** 6.8% / 17.7% / 25.7% / 30.8% / 19.0%.
* **No leakage:** the instruction names the attribute, never the rating.
* **Red flags:** none material. Attribution must be preserved.

### WMT direct assessment (`load_wmt_da`)

* **Licence:** Apache-2.0 on the compiled dataset. The underlying WMT news,
  parliamentary, and web text retains its own source terms.
* **Conversion:** `state` is `Source: ...\n\nTranslation: ...`. The 0-100
  `raw` human score is banded into five 20-point levels: 0-19, 20-39, 40-59,
  60-79, 80-100. `label` is the band index.
* **Split:** year 2022 is held out; earlier years are train. This is a
  temporal split, so no segment is shared across the boundary.
* **Cap:** 25,000 train rows, 2,000 eval rows, fixed seed.
* **Observed labels:** 7.2% / 7.7% / 13.1% / 25.7% / 46.2%.
* **No leakage:** the reference translation is deliberately excluded from the
  state, and the instruction never contains the score.
* **Red flags:** states are much shorter than the corpus mean (p50 ~59 tokens
  vs ~238); this source adds many short ordinal examples. The upstream text
  licence should be reviewed before commercial distribution.

### UltraFeedback (`load_ultrafeedback`)

* **Licence:** MIT for the released annotations. The prompts come from
  ShareGPT, UltraChat, Evol-Instruct, TruthfulQA, FalseQA, and Flan; those
  upstream collections may carry additional terms.
* **Conversion:** each preference row contributes two items, one for the
  chosen and one for the rejected response. `state` is
  `Instruction: ...\n\nResponse: ...`. The 1-10 overall score maps to five
  equal-width bands over [1, 10]: [1, 2.8), [2.8, 4.6), [4.6, 6.4),
  [6.4, 8.2), [8.2, 10]. `label` is the band index.
* **Cap:** 12,500 train rows × 2 = 25,000 items; 1,000 test rows × 2 = 2,000
  eval items (config caps eval to 500).
* **Observed labels:** 4.0% / 9.9% / 13.5% / 48.5% / 24.0%.
* **No leakage:** the instruction is generic; the score is not in the state.
* **Red flags:** labels are GPT-4 judgements, not human ratings; the
  upstream prompt terms need review.

### Why these three

They cover three different ordinal judgements (translation quality, response
quality with five separate human attributes, and instruction-following
quality), all have clear released licences, and all convert from a released
scalar without a judgement call. Together they add 74,983 distinct score
items, enough to reach parity with noul and choice without touching the
sampler.

## Label distributions measured after materialisation

These are the actual distributions in the materialised JSONL, not the mock
distributions used by the unit tests.

| Source | n | Level 0 | 1 | 2 | 3 | 4 | Max share |
|---|---:|---:|---:|---:|---:|---:|---:|
| helpsteer | 30,000 | 6.8% | 17.7% | 25.7% | 30.8% | 19.0% | 30.8% |
| wmt_da | 27,000 | 7.2% | 7.7% | 13.1% | 25.7% | 46.2% | 46.2% |
| ultrafeedback | 26,981 | 4.0% | 9.9% | 13.5% | 48.5% | 24.0% | 48.5% |

All three pass the integrity test: at least three levels, no level above 90%,
and no gold label in the instruction.

## Score pool after the change

Measured on `r2_reranker0.6b_scorebalanced.yaml` with the new sources
materialised and the existing benchmarks symlinked from the main checkout.

| Source | Score items |
|---|---:|
| sst5 | 8,544 |
| stsb | 5,749 |
| yelp | 20,000 |
| helpsteer | 25,000 |
| wmt_da | 25,000 |
| ultrafeedback | 24,983 |
| synthetic | 4,269 |
| typed_decisions | 795 |
| **score total** | **114,340** |

```
Type         Pool    Slots  Resampling
choice     77,594    5,000       0.06x
noul      100,311    5,000       0.05x
score     114,340    5,000       0.04x
```

Before this change the score pool in the same config was 39,357. The default
config (without the score-balanced caps) was 27,235, the figure in #142.

## Rejected and deferred candidates

### Review star ratings

| Source | Licence | Why not |
|---|---|---|
| Amazon Reviews Multi | "other" (Amazon conditions of use) | No clear commercial grant; derived from Amazon content. |
| McAuley Amazon-Reviews-2023 | no licence field | Redistribution terms unclear. |
| Yelp review full | "other" (Yelp dataset licence) | Already in the corpus; research/non-commercial terms should be reviewed separately. |
| App reviews (`sealuzh/app_reviews`) | unknown | No licence; 288,065 rows but cannot be cleared. |
| Trustpilot (`crawlfeeds/...20K`) | CC-BY-SA-4.0 | Scraped; share-alike and site ToS concerns. |
| TripAdvisor / Glassdoor | mostly undisclosed | Scraped; no reliable licence. |
| IMDb | n/a | The public release is binary sentiment; per-title ratings are not per-item labels. |

### Human-graded writing

* **AES2 / ASAP 2.0** — 17,307 essays, CC-BY-NC-4.0. Clean ordinal target but
  non-commercial; issue #153.
* **Persuade 2.0** — derived from ASAP, same non-commercial chain.
* **TOEFL11, ICNALE, Write & Improve** — restricted or not redistributable.

### Translation and generation quality

* **WMT DA** — wired.
* **WMT MQM** — 150,347 rows, Apache-2.0, but the `score` field mixes negative
  error counts with a `100` sentinel (10,002 rows) and a long tail to -400.
  The intended scale is not documented well enough to band without guessing;
  rejected until the scoring convention is confirmed.
* **GEMBA-MQM / learned metrics (CometKiwi, MetricX)** — the public releases
  are model scores over third-party translation sets; reproducing them needs
  GPU inference and adds another model's bias. Deferred.

### Data-quality and educational-value regressions

* **FineWeb-Edu** — ODC-By 1.0, per-document 0-5 score. Blocked by shard
  size: a single parquet shard is ~1.9 GB and `_load_hf_parquet` downloads
  every shard. Issue #151.
* **DCLM-BASELINE** — CC-BY-4.0, fastText educational score. Same streaming
  problem. Issue #151.
* **QuRating** (`princeton-nlp/QuRating-GPT3.5-Judgments`) — 100k-1M rows,
  but the dataset card has no licence field and the texts come from
  books/web corpora. Rejected until the licence is stated.

### Graded relevance

* **TREC-COVID qrels** — 66,336 judgements, 0-2 grades, CC-BY-SA-4.0. Good
  three-level source; needs a qrel-to-corpus join and a query-level split.
  Issue #155.
* **TREC Deep Learning 2019/2020** — graded 0-3, but the passage corpus is
  MS MARCO (research terms) and the qrels are distributed through NIST rather
  than a single HF parquet. Deferred.
* **MS MARCO / BEIR** — mostly binary relevance; not enough levels without
  the graded TREC qrels.

### Safety and severity

* **DICES** — CC-BY-4.0, five-point safety ratings, 1,340 conversations.
  Small but clean; issue #154.
* **Civil Comments** — CC0, but the label is the fraction of annotators who
  flagged toxicity, and the distribution is a spike at zero. Binning it would
  put far more than 90% of items in the lowest level, so it fails the
  degeneracy guard.
* **Jigsaw toxicity** — binary and severe-toxicity flags, not an ordered
  severity scale.
* **Medical triage** — MIMIC-IV and similar are credential-gated; no public
  per-note severity release with a clear licence.
* **Credit and risk ratings** — tabular, not text-grounded; no usable state.

### Likert and preference intensities

* **HelpSteer** is the strongest Likert-style source and is wired.
* **MT-Bench human judgments** — CC-BY-4.0, 3,355 rows, but the human file is
  pairwise preference, not a per-item ordinal rating.
* **UltraFeedback fine-grained ratings** — the four 1-5 annotations are
  available, but the overall 1-10 score is the cleaner single target; the
  fine-grained scores can be a later soft-label variant.
* **DICES** is the safety Likert case.

## Other decision-model projects

| Project | Training data public? | Ordinal targets? | Licence | Note |
|---|---|---|---|---|
| JevBench | Public split yes; sealed split no | Yes, including score questions | Apache-2.0 (HF card) / MIT (GitHub) | Already loaded as `jevbench`; 2,240 question rows, public eval only. |
| Open-Jev (ZefanCai) | Yes, v2 redistributable | Yes, 11,923 score items | CC0-1.0 data for v2; v1.1 data is "other" | Issue #152. Upstream wording is not relicensed. |
| jev-9b (smha1012) | No, model only | Unknown | CC-BY-NC-4.0, gated | Cannot reuse training data. |
| Open-Jev-9B | Model only | n/a | Apache-2.0 | Adapter, no data release. |
| Nimble (Bespoke Labs) | No; decision-index repo is code only | n/a | Apache-2.0 model; index repo CC-BY-NC-4.0 | No training data release. |
| Laya / Lia (Convai) | No, model only | n/a | Apache-2.0 | No public training set. |
| CLM-8B (Contrastive-LM) | Pretrain embeddings public | No ordinal targets | Apache-2.0 model; embeddings MIT | Retrieval embeddings, not graded labels. |
| Clef (Cloudflare) | No, model only | n/a | Apache-2.0 | No data release. |

## Synthetic generation with Jev

`research/00-overview.md` records Jev input at **$0.042 per million tokens**
and output at **$0**. A score question needs the state plus instructions and
3-5 criteria; at ~350 input tokens per item:

* 10,000 generated score items ≈ 3.5M tokens ≈ **$0.15**.
* 100,000 items ≈ $1.47.
* Re-labelling all 154,092 existing synthetic states with one score question
  ≈ 54M tokens ≈ **$2.27**.

At 1,000-token states the same 10,000 items cost about $0.42. These are
estimates from the project's API notes, not a published price list; verify
against the current account before a large run.

This is much cheaper than sourcing and clearing new external data, and it can
target domains the external sources do not cover (medical triage, contract
review, code review). The caveat is that the labels are Jev's own judgements,
so synthetic score items should complement, not replace, the real ratings.
Issue #156 tracks the generator.

## Licensing risks to keep visible

1. **WMT DA** — Apache-2.0 covers the compiled annotations, not necessarily
   the underlying news and parliamentary text. Keep the dataset attribution
   and review before commercial distribution.
2. **UltraFeedback** — MIT covers the released annotations; the prompts come
   from six upstream collections with their own terms. Treat as research use
   until the prompt licences are reviewed.
3. **FineWeb-Edu / DCLM** — permissive dataset licences, but Common Crawl
   pages can carry third-party rights. Attribution alone may not clear every
   document.
4. **AES2** — CC-BY-NC-4.0 is non-commercial. Do not wire it in without an
   explicit decision.
5. **TREC-COVID** — CC-BY-SA-4.0 is share-alike; a derived benchmark file may
   need to carry the same licence.
6. **Open-Jev** — CC0 covers the generated records only; the data card says it
   does not relicense upstream wording, sources, assets, or weights.
7. **Yelp and Amazon** — already-used review corpora carry
   research/non-commercial terms that predate this change; this PR does not
   alter that exposure.

## Reproduction

```bash
# Materialise the three new benchmark files (needs pyarrow and network).
python data/scripts/materialize_benchmarks.py helpsteer wmt_da ultrafeedback

# Run the tests.
python -m pytest tests/
```

On the training host, the benchmark directory must be symlinked into the
worktree before the pool report (issue #141). The pool table above was
produced from a fresh worktree at the PR branch with every entry of the main
checkout's `data/benchmarks` linked in.

## Follow-up issues

* #151 — bounded FineWeb-Edu / DCLM educational-quality loader.
* #152 — Open-Jev's 11,923 CC0 score items.
* #153 — ASAP 2.0 essay scores (non-commercial).
* #154 — DICES five-point safety severity.
* #155 — TREC-COVID graded relevance (0-2).
* #156 — generate score questions over existing states with Jev.
