# Decision Index reconnaissance: two benchmarks share the name, one can be run

**Decision: run the community Decision Index public suite when a GPU is available; do not
run the Artificial Analysis Decision Index.** The first is an MIT-licensed reproduction
kit whose public half we can rebuild, run and score ourselves. The second is a private
composite that Artificial Analysis runs on its own data; one quarter of the index is a
private component with no harness.

The community run is **partly done**. The suite is rebuilt as `di-0.3-krino-1` (140,620
requests, 140,178 scoreable) and is not byte-identical to the pinned edition: the kit fixed
a future-turn leak in the SGD rows after the pinned suite was frozen, so the pinned hash
cannot be reproduced without restoring the leak. HLE access is cleared, the BPoMP
truncation is resolved, and the remaining blocker is a GPU for scoring the v4 control. The
exact evidence and the command sequence are below. No board number is copied anywhere in
this note, and no home-made composite is proposed.

Scope: issue #236, Workstream A. Reconnaissance run 2026-11-06. No GPU was available for
the reconnaissance, and the base interpreter had no `torch`.

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

The second command ran with our Hugging Face credentials, not anonymously. So a run
requires the local rebuild: about **7 GB of downloads** (the HoVer Wikipedia database
alone is 2.2 GB) and **17 GB of working space**, with the HLE terms accepted on the Hub
and a login. That is a CPU job; it does not need a GPU.

### Rebuild attempt

On 2026-11-06 the kit was installed in a Python 3.12 virtual environment (`uv venv
--python 3.12`; `pip install -e ".[transformers,rebuild]"`). Dependency resolution
succeeded, and the HLE gate is open: `hf download cais/hle README.md` returned the file.

`suite rebuild --edition 0.3` then failed in source acquisition, not in the scoring code.
The Zenodo `BPoMP_datasets_p1_out_of_3.json` download was truncated at 11,846,880 bytes
against a `Content-Length` of 11,867,336; the kit's hash check rejected it (actual
`7e840b47...`, pinned `2500e828...`). `http_fetch` in `suite/build/acquire.py` does not
check `Content-Length`, so a dropped connection surfaces as a hash mismatch.

The retry on the compute host resolved it. `curl -L --retry 20 --retry-all-errors -C -`
downloaded all three BPoMP parts complete, and each SHA-256 matches the kit's pin:
p1 11,867,336 bytes `2500e828...`, p2 17,662,368 bytes `ee17bc0f...`, p3 17,655,733 bytes
`cc0b7564...`. The earlier truncation was local, not a short object; no mirror was needed.
The edition 0.3 rebuild was then started on the compute host. Source acquisition reached
HLE and stopped: `cais/hle` returned `GatedRepoError: 403 ... Access to dataset cais/hle is
restricted and you are not in the authorized list`. The gate is account-level, not
host-specific; the README is public, but the test parquet is not. HLE is catalog 45,
513 requests, in the Knowledge & Reasoning area. No suite hash is produced until the HLE
access request is approved. The other sources are cached in the work directory.

The kit declares `requires-python >= 3.10`, but `adapters_scored.chessbench` calls
`hashlib.file_digest`, which exists only from Python 3.11. On 3.10 the rebuild dies at
ChessBench with `AttributeError: module 'hashlib' has no attribute 'file_digest'`. Use
3.11 or newer; the edition 0.3 rerun uses 3.12.

### Edition 0.3 rebuild result

HLE access was cleared on the account, and the rebuild ran to completion on the compute
host under Python 3.12. The 0.1 base is **not byte-identical** to the pinned suite:

| file | rebuilt | pinned | result |
|---|---|---|---|
| release-v1 `selected-rows.jsonl` | 132,422 requests, 117,764 cases, 775,202 fields; sha256 `7068220a...` | sha256 `288d3720...` | **mismatch** |
| `added-rows.jsonl` | 30,419 requests; sha256 `7429f3c9...` | sha256 `7429f3c9...` | match |

The cause is a kit fix that postdates the freeze, not dependency or source drift. Commit
`2d7791c` (2026-10-13), "SGD builder: copy the dialogue history per row", records that
every SGD row stored a reference to the growing `history` list, so each row ended up with
the whole dialogue, future turns included. The fix copies the history per row. Rebuilding
SGD from the same pinned source with the pre-fix builder reproduces the pinned hash
exactly (`8496e556...`); the current builder produces the corrected file (`e57d11fd...`).

The per-file check confirms the isolation. `hub/manifest.json` pins the sha256 of all 37
normalized source files; 36 match the rebuilt files and only
`SGD-service-given-intent.jsonl` differs. GSM8K v3, the one component read through
pyarrow, reproduces its pinned hash `75933489...` under pyarrow 26.0.1. The dependency
versions resolved here are pandas 3.1.1, pyarrow 26.0.1, numpy 2.5.4 and scipy 1.18.1,
later than the 2026-09-22 editions, but they do not move the 36 matching files or GSM8K,
so pinning them cannot make the SGD file match.

The pinned edition 0.1, and editions 0.2 and 0.3, which read the 0.1 rows unchanged,
therefore carry the SGD future-turn leak in their 2,500 selected SGD requests. Our rebuilt
suite applies the fix. Do not revert the fix to match the pinned hash; reproducing it
would restore a benchmark-validity defect. The kit's `docs/suite.md` still claims all 37
normalizers reproduce byte-identically; that claim predates the fix and is stale.

The corrected suite is assembled and recorded as `di-0.3-krino-1`:

| identity | value |
|---|---|
| requests / scoreable | 140,620 / 140,178 |
| `rows_sha256` (uncompressed, v2 cut) | `8224546d...` |
| `added_sha256` | `7429f3c9...` (matches the pin) |
| `gsm8k_sha256` | `75933489...` (matches the pin) |
| `suite_sha256` | `f725070c14046bb86baf1e3911b1af209ae898e9a0ab29dee5a2548555f7c3bf` |

`suite_sha256` is the SHA-256 of the canonical JSON object
`{suite_version, rows_sha256, added_sha256, gsm8k_sha256}`. The counts above are the
edition-0.3 cut: 110,201 base requests plus 30,419 added requests. They are not
comparable to the release-v1 `selected-rows` cut in the table above (132,422 requests,
117,764 cases), which includes benchmarks this suite shows but does not count; the
difference between the two pairs is a difference of cuts, not of editions. Identity
against the pin is carried by `added_sha256` and `gsm8k_sha256`, both of which match;
the content differs only in the SGD rows.

### Suite identity and gate-row provenance

The rebuilt suite is ours, not a byte copy of the pinned one. Record its identity on
every gate row:

- `suite_version`: `di-0.3-krino-1`; bump for any content change, never for a rerun.
- `suite_sha256`: the composite above, or the three component hashes.
- `builder_git_sha`: kit commit `25c6ec7b...`, plus the SGD fix commit `2d7791c...`.
- `requests` / `scoreable`: 140,620 / 140,178.

Rules:

1. A gate row is not a gate row without `suite_version` and `suite_sha256`.
2. Never compare a result measured on one suite version with a result on another without
   rerunning the control on the new version. The arm-minus-control difference is valid
   only within one version.
3. When the #234-matched held-out replaces the interim split, existing gate rows keep
   their version and hash; the replacement is a new version, and the control is rerun on
   it before any arm is promoted.
4. The suite data stays private; only the identity tuple is committed.

Under the dual acceptance standard, JevBench public is a hard non-regression constraint
and the matched held-out is the promotion signal. The gate compares an arm with the
control on the same suite version, so board comparability is not required for the gate.
It is required only to quote a number next to someone else's.

### v4 control: identified, not scored

The control is manifest row 78 of
`model/experiments/residual/manifest.jsonl` in the `feat/v4-listwise-gate` worktree, arm
`v4_control_qwen35_4b_raw`. Row numbers are positional; use the arm name.

| field | value |
|---|---|
| backbone | `Qwen/Qwen3.5-4B` |
| path | `eval_gate.py --raw-reranker --shared-prefix`; `raw_scoring = {path: shared_prefix, readout: float32_two_token}` |
| data | JevBench public, 231 items, sha `dcb90f87` |
| acc / cc / hard / noul | 0.6883 / 0.5187 / 0.5045 / 0.6757 |
| choice / score / ECE | 0.6835 / 0.7778 / 0.1175 |
| evidence | `results/gate/qwen35_shared/s2_qwen35_4b_instruct.json` |

**Status: blocked (阻塞), not failed.** The full public suite is 140,178 scoreable
requests. The kit's reference run uses one GPU. No GPU was available for this work;
running Qwen3.5-4B over 140k requests on CPU is not a measurement we can take. The
adapter is written and unit-tested; the run waits for a GPU allocation.

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

The suite is already built at `work/suite-0.3` as `di-0.3-krino-1`; do not rebuild it.
The rebuild sequence, for provenance, is: `suite rebuild --edition 0.3 --work work` for
the v1 rows, the kit's `release_v2` cut with `added=[]` (the added rows were already
built and match the pin), `gsm8k_v3.main`, then `suite import` with `--no-verify`. The
`--no-verify` is deliberate: the rows hash cannot match the pin without the SGD leak.

```bash
# Score the v4 control from the krino tree (PYTHONPATH must see both packages)
cd <krino-worktree>
python -m decision_index pipeline \
    --suite-dir <build>/work/suite-0.3 \
    --no-verify \
    --engine model.scripts.decision_index_engine:KrinoRawEngine \
    --model Qwen/Qwen3.5-4B \
    --option shared_prefix=true \
    --option strict_length=true \
    --option device=cuda \
    --out runs/v4-control
```

Record `suite_version` and `suite_sha256` with the result.

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
