# Decision Index reconnaissance: two benchmarks share the name, one can be run

**Decision: run the community Decision Index public suite when a GPU is available; do not
run the Artificial Analysis Decision Index.** The first is an MIT-licensed reproduction
kit whose public half we can rebuild, run and score ourselves. The second is a private
composite that Artificial Analysis runs on its own data; one quarter of the index is a
private component with no harness.

The community run is **not done**. Three things block it: the frozen suite is not
downloadable prebuilt, the local rebuild hit a truncated source download, and scoring the
v4 control needs a GPU. The exact blockers and the command sequence are below. No board
number is copied anywhere in this note, and no home-made composite is proposed.

Scope: issue #236, Workstream A. Reconnaissance run 2026-11-06 on `oaklight`'s laptop
(no GPU; `nvidia-smi` absent, `torch` absent from the base interpreter).

## The two things called "Decision Index"

| | Artificial Analysis Decision Index | Community Decision Index |
|---|---|---|
| Publisher | Artificial Analysis | Apolinário Passos (`apolinario/decision-index`) |
| Versions | v1.0, launched 2026-10-13; live page checked 2026-11-06 | editions 0.1, 0.2, 0.2.1, 0.3 in the kit; live board 0.4 with the public suite unchanged from 0.3 |
| What it scores | a fixed weighted composite of four evaluations | 37 public benchmarks in five areas, chance-corrected, plus private parts the maintainers run |
| Public code | none | MIT, `decision_index/` package, engines and scorers |
| Public data | HLE, When2Call, WildGuardTest; AA-Omniscience is private | the recipe; the suite is rebuilt from pinned upstream sources |
| Runnable locally | **no** | **yes**, after a local rebuild; the 2026-11-06 attempt was stopped by a truncated source download |

`research/08-clef-analysis.md` uses "Jev Decision Index" for the classification and
reasoning rankings (BANKING77, CLINC150, API-Bank, BBH, MMLU-Pro, GPQA Diamond). Those
benchmarks are in the community suite. The project's prior use of the name therefore
means the community benchmark, not the Artificial Analysis one.

## Artificial Analysis Decision Index: do not run

Source: <https://artificialanalysis.ai/articles/artificial-analysis-decision-index>,
<https://artificialanalysis.ai/methodology/decision-index>, and
<https://artificialanalysis.ai/evaluations/artificial-analysis-decision-index>, fetched
2026-11-06.

- **Components and weights.** `AA-Omniscience-Decision` (500 private questions adapted
  from AA-Omniscience; accuracy 12.5% + non-hallucination rate 12.5%), `HLE-MultiChoice`
  (513 text-only multiple-choice questions from `cais/hle`; 25%), `When2Call` (3,423 of
  NVIDIA's 3,652 test tasks after 229 ambiguous rows are dropped; 25%), and
  `WildGuardTest` (835 of 1,725 rows from `allenai/wildguardmix`, ODC-BY; average F1 of
  prompt harm, response harm and refusal; 25%).
- **Why it cannot be run.** The methodology says every evaluation is conducted
  independently by Artificial Analysis and there is no published harness. One quarter of
  the index is a private dataset. Running the other three would produce a number for
  three components, not the index, and it would not be comparable to the board. The
  licence of the composite itself is not published; the component datasets keep their own
  terms (HLE is gated, `wildguardmix` is ODC-BY).
- **Domains, separately.** Knowledge and hallucination, reasoning, tool calling, and
  safety classification. A single aggregate would hide which one moves, so report them
  separately if this benchmark is ever reproduced.
- **What we do instead.** Do not run it, do not approximate it, and do not quote a
  leaderboard value as our result. Revisit only if Artificial Analysis publishes the
  private component and a harness.

## Community Decision Index: run this one

Source: <https://github.com/apolinario/decision-index>, cloned 2026-11-06 at commit
`25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` (2026-10-20T15:51:50+02:00). The package
metadata says version 0.3; `decision_index/editions.py` defaults to `0.3`; the README
announces the live 0.4 board with the public suite unchanged from 0.3.

- **Licence.** Code is MIT (`LICENSE`, Copyright (c) 2026 Apolinário Passos), confirmed by
  the GitHub API. The suite is not redistributed in the repository. Each upstream source
  keeps its own terms: ANLI is CC BY-NC 4.0; RAGTruth's MS MARCO contexts are
  research-only and its Yelp contexts forbid redistribution; SGD and HoVer are
  share-alike; GPQA asks that its questions not be posted in plain text; BBH carries the
  BIG-bench canary; HLE is gated. Treat the suite as evaluation data: keep it private, do
  not train on it, do not republish it.
- **Public suite size.** Edition 0.3 has 110,201 base requests and 30,419 requests for the
  seven benchmarks added in 0.2: 140,620 requests, 140,178 scoreable after the 442
  exclusions. 37 benchmarks count; MMLU, ARC-Easy, ARC-Challenge, SimpleBench,
  RouterBench and SGD are shown but not counted.
- **Areas and weights.** Knowledge & Reasoning 25.8%, Language Understanding 25.8%,
  Retrieval & Classification 20.0%, Tools & Automation 18.3%, Arts & Human Taste 10.0%.
  Gold-starred benchmarks weigh 1.2 inside an area.
- **Scoring.** `raw = native score × answered / requests`; unanswered, unsupported,
  errored and abstained requests count as wrong. `skill = clip((raw − chance) / (1 −
  chance), 0, 1)`. The index is 100 × the weighted mean of the five areas. The kit forbids
  truncation, option filtering and per-benchmark prompt tuning; an engine that cannot fit
  a request must raise `Unsupported` instead.
- **Engines.** `http` (a `/v1/systemone` server), `transformers` (a stock causal LM,
  constrained decoding), `random`, and any `module:Class` passed with `--engine`.

### Availability evidence

The kit's `suite download` defaults to Hub repositories `multimodalart/decision-index-suite`
and `...-suite-0.2` / `...-suite-0.3`. They are not publicly readable:

```text
$ curl -s https://huggingface.co/api/datasets/multimodalart/decision-index-suite-0.3
{"error":"Invalid username or password."}                 # HTTP 401
$ hf download multimodalart/decision-index-suite-0.3 manifest.json --repo-type dataset
Error: Repository not found. If the repo is private, make sure you are authenticated ...
```

The second command ran as `oaklight`, not anonymously. So a run requires the local
rebuild: about **7 GB of downloads** (the HoVer Wikipedia database alone is 2.2 GB) and
**17 GB of working space**, with the HLE terms accepted on the Hub and a login. That is a
CPU job; it does not need a GPU.

### Rebuild attempt

On 2026-11-06 the kit was installed in a Python 3.12 virtual environment (`uv venv
--python 3.12`; `pip install -e ".[transformers,rebuild]"`). Dependency resolution
succeeded, and the HLE gate is open: `hf download cais/hle README.md` returned the file.

`suite rebuild --edition 0.3` then failed in source acquisition, not in the scoring code.
The Zenodo `BPoMP_datasets_p1_out_of_3.json` download was truncated at 11,846,880 bytes
against a `Content-Length` of 11,867,336; the kit's hash check rejected it (actual
`7e840b47...`, pinned `2500e828...`). `http_fetch` in `suite/build/acquire.py` does not
check `Content-Length`, so a dropped connection surfaces as a hash mismatch. A second
attempt was slower on the same source and was stopped. No suite was produced, so no
score was run.

This is an operational failure in the rebuild, not a licence or code failure. On the GPU
host, pre-fetch the flaky sources with a resumable downloader (`curl -C - --retry`) and
verify the pinned hashes before rerunning `suite rebuild`; the completed sources are
cached in the work directory.

### v4 control: identified, not scored

The control is manifest row 78 of
`model/experiments/residual/manifest.jsonl` in the `feat/v4-listwise-gate` worktree
(`/home/pding/.cache/krino-v4-gate`), arm `v4_control_qwen35_4b_raw`. Row numbers are
positional; use the arm name.

| field | value |
|---|---|
| backbone | `Qwen/Qwen3.5-4B` |
| path | `eval_gate.py --raw-reranker --shared-prefix`; `raw_scoring = {path: shared_prefix, readout: float32_two_token}` |
| data | JevBench public, 231 items, sha `dcb90f87` |
| acc / cc / hard / noul | 0.6883 / 0.5187 / 0.5045 / 0.6757 |
| choice / score / ECE | 0.6835 / 0.7778 / 0.1175 |
| evidence | `results/gate/qwen35_shared/s2_qwen35_4b_instruct.json` |

**Status: blocked (阻塞), not failed.** The full public suite is 140,178 scoreable
requests. The kit's reference run uses one RTX PRO 6000 (96 GB). This machine has no GPU
and no `torch` in the base interpreter; running Qwen3.5-4B over 140k requests on CPU is
not a measurement we can take. The adapter is written and unit-tested; the run waits for
a GPU allocation.

One more incompatibility: the kit's stock `transformers` engine loads through
`AutoModelForCausalLM`, and the Qwen3.5 4B config declares `model_type: qwen3_5` and
`architectures: [Qwen3_5ForConditionalGeneration]` with separate `text_config` and
`vision_config` blocks, so it has no causal-LM auto mapping (the same reason
`eval_gate.RawReranker` has a `_load_qwen35` fallback). The control must be driven through
our own loader, which the adapter below does.

## Adapter

`model/scripts/decision_index_engine.py` implements the kit's engine protocol with our
per-option scorer:

- `KrinoRawEngine` builds an `eval_gate.RawReranker` with the v4 control's path and calls
  `predict_row` once per question.
- Choice questions keep every option and its normalized probability; the chosen key is the
  argmax. Noul questions return `p_yes`. Score questions are refused with `Unsupported`
  because the Decision Index interface has no score primitive.
- `strict_length=true` reproduces the prompt-length accounting of `RawReranker` and raises
  `Unsupported` when a row would be truncated. This is required: a truncated run is not a
  valid submission.
- It is an in-process engine, not the HTTP server. `model/src/server.py` rounds every
  probability to two decimals in `_format_response`, which would corrupt calibration and
  can break the kit's sum-to-one check. The HTTP path is fine for a demo; it is not fine
  for scoring.

The adapter is tested with a fake scorer, so the tests run without `torch`:

```bash
python -m pytest tests/test_decision_index_engine.py
```

## Run plan, once a GPU is allocated

```bash
# 1. Rebuild the suite (CPU, ~7 GB down, 17 GB work, HLE terms accepted)
git clone --depth 1 https://github.com/apolinario/decision-index.git
cd decision-index
uv venv --python 3.12 .venv && . .venv/bin/activate
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1
hf auth login
python -m decision_index suite rebuild --work work
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz

# 2. Score the v4 control from the krino tree (PYTHONPATH must see both packages)
cd <krino-worktree>
python -m decision_index pipeline \
    --engine model.scripts.decision_index_engine:KrinoRawEngine \
    --model Qwen/Qwen3.5-4B \
    --option shared_prefix=true \
    --option strict_length=true \
    --option device=cuda \
    --out runs/v4-control
```

`pipeline` writes `results.jsonl`, `benchmark-summary.json`, `index.json` and
`scores.json`. Record the per-benchmark and per-area values; do not report only the
aggregate. A run with `unsupported` rows is valid only if the refusals are declared
capacity limits, and it loses the corresponding coverage before chance correction.

## What would change this decision

- A public prebuilt suite at the default Hub path, or a rebuild that clears the source
  licences for our use.
- A GPU allocation for the 140k requests.
- An Artificial Analysis harness and the private AA-Omniscience component. Until then,
  their index is a published number, not a measurement we can reproduce.
