# H2O-Lightning-4B on our instruments

`h2oai/h2o-lightning-4b` is an Apache 2.0 fine-tune of Qwen/Qwen3.5-4B, the backbone of our v4 line. Its model card reports 206/231 on JevBench public with the tier split easy 48/48, standard 72/72, hard 86/111. That split matches ours item for item — easy 48, original 72, hard 111 — so for the first time a third-party JevBench number can be read as the same items we measure. This note records the reproduction, the two-way/one-way contrast, the model on our dev and held-out sets, and latency (#327).

The run began with a fidelity gate and nothing else: if the card's own command and serving configuration did not give 206/231, nothing downstream was to be reported. It gave 206/231 exactly, so the rest ran.

## What ran

| field | value |
|---|---|
| model | `h2oai/h2o-lightning-4b`, revision `659fc412de8486dbc2477b0d1cb4121e97e72471` |
| serving | vLLM `0.30.0` plus the 30-file per-request attention build from `h2oai/vllm` at `b05cc29ab813895947954c268c4ec41ca0da1306`, checked against `engine/vllm_per_request_attention.SHA256SUMS` |
| harness | `fstandhartinger/jevbench` at `609a664e81f46e03731600bd7bd9f1b7a61257c5`, the card's command verbatim |
| host | one NVIDIA H100 NVL (95 GB), CUDA 13 driver, one engine, the shipped `serve_config.json` |
| interpreter | python 3.10.12, torch 2.13.0+cu130, transformers 5.20.2 |

No setting in `serve_config.json`, the engine layout, the batch budget or any request option that reads a question more than once was changed. Each cell carries its `_provenance` block with the command, the build, the revision and the dates.

## The two-way effect

| mode | overall | easy | standard (original) | hard | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| two-way (`serve.sh` default) | **206/231 = 89.2%** | 48/48 | 72/72 | 86/111 | 126/139 | 64/74 | 16/18 |
| one-way (`TWO_WAY=0`) | **202/231 = 87.4%** | 48/48 | 70/72 | 84/111 | 124/139 | 62/74 | 16/18 |

Two-way reading is +4 items, +1.73 pp. This is the cleanest estimate we have of what bidirectional reading buys once the model is trained: a 1.7 pp effect, not 22 pp. Both hard and original lose two items; easy loses none, and the score family is unchanged.

## Our instruments

Both runs used the two-way endpoint, one request per decision.

| set | n | correct | accuracy | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|
| dev | 812 | 589 | **72.5%** | 239/322 | 287/391 | 63/99 |
| held-out | 6,254 | 4,549 | **72.7%** | 1,844/2,473 | 2,269/2,995 | 436/786 |

Every row was valid and answered; coverage is 1.0 on both sets. The dev set ranks; the held-out set is promotion evidence under `research/37`. The held-out number is internal and is not part of the model's card.

The dev file holds 828 records; 812 are scored. "Dev 812" has always meant the scored subset, and every dev accuracy we have quoted — before this run and in it — is over those 812, not over 828. The 16 omitted are withheld-gold abstention items, the same construction as the 170 below, excluded for the same reason. Nobody should later read 828 beside a quoted dev number and conclude the set changed; it did not, the denominator was always the scoreable rows.

### The 6,424 → 6,254 difference is a decision, not a count

The held-out selection names 6,424 records. 170 of them are withheld-gold abstention items (`label_kind: withheld_gold`): the gold label was deliberately removed from the criteria, so a typed readout over the supplied options cannot score them. We exclude them from the accuracy rather than count them wrong, and that is a decision about what the instrument can measure, not a neutral subtraction. It changes the denominator from 6,424 to 6,254, and it should be quoted that way — "4,549 of 6,254 scoreable held-out records" — never as an accuracy over 6,424. The same decision removes 16 of the 828 dev records.

### The corpus on disk is not the corpus the spec names

The selection was recomputed from the canonical corpus with `model/experiments/canonical/select_heldout.py`. It reproduces the committed ids exactly: `ids_sha256 492ee0af…`, 6,424 ids, 928 clusters, by-type 2,995 / 2,643 / 786, matching `model/experiments/canonical/heldout_6400.json` field for field.

The corpus file itself does not. It hashes to `8eb3b2a490a04c600bd652ebeed738c02976bb77e731b337efa513805c62e4f6`; `heldout_6400.json` records `corpus_sha256` as `07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53`. Same ids, different bytes — the record set is the intended one, and the artifact on disk is not the artifact the spec names. Until the corpus is reconciled, the held-out figure should be quoted with that caveat attached. This is written here, in the body, because a footnote would let the number travel without it.

## Latency

These are two different stacks, and the ratio between them is not a readout-level comparison.

**H2O-Lightning-4B** — end-to-end through vLLM 0.30.0 and the release shim, one request per decision, two-way, the 231 public items, concurrency 1: mean **38.8 ms**, median **24.3 ms**, p95 100.9 ms. Mean input 653.6 tokens (median 157, max 3,952), exactly one output token on every request. That is 22.9 decisions/s single-stream. Throughput at higher concurrency on the same server and the same 231 requests: 57.1/s at 8, 64.9/s at 32, 54.5/s at 64 — the server saturates between 8 and 32 and is over-subscribed at 64.

**Our pointwise judgement path** — batch-1 transformers on Qwen/Qwen3-Reranker-4B, per-option yes/no, one forward per option, the same host, the same 231 items: mean **0.493 s**, median **0.420 s**, and it scored 150/231 = 64.9%, reproducing the published cell. Timing is per item, around the whole read.

The 12.7× mean and 17.3× median between those two lines mix a vLLM server with a batch-1 PyTorch loop and a shim in front, and must not be quoted as the readout-level difference. The clean readout-level comparison against our own per-request stack is 0.32–0.40 s against 1.16–1.75 s (#329), measured with the same readout implementation on both arms; that comparison is not part of this note.

## Files

| file | contents |
|---|---|
| `results/jevbench/h2o-lightning-4b.json` | two-way, 206/231, per-tier and per-type, latency block |
| `results/jevbench/h2o-lightning-4b-oneway.json` | one-way, 202/231, per-tier and per-type, latency block |
| `results/external/h2o-lightning-4b-dev-812.json` | dev, 589/812, per-type, latency block |
| `results/external/h2o-lightning-4b-heldout-6254.json` | held-out, 4,549/6,254, per-type, latency block, the exclusion and hash notes above |

The dashboard reads the two JevBench cells as readout `native decision` (two-way) and `native decision, one-way`, method `third-party`; they are not zero-shot readout cells and do not join the panel-2 rows.
