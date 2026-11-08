# Zero-shot logit baselines with embedded provenance

Nine result files from one run, covering #277 (19 benchmarks) and #278 (JevBench
public). Every file carries a `_provenance` block: model id, resolved revision,
dtype, device, code commit, readout, library versions and the command.

The September Phase A files carry none of that, which is why their numbers have
never been auditable. These do.

## One run, one commit

| field | value |
|---|---|
| `code_commit` | `33506b86990f0eaa3fdcdf5c0ef3075905edd80f` |
| tree | detached worktree at that commit, `git status` empty, `git diff --stat HEAD` empty |
| interpreter | python 3.12.16 |
| torch | 2.11.0+cu128 |
| transformers | 5.21.0 |
| dtype | bfloat16 |
| device | `NVIDIA H100 NVL` |
| readout | option-text continuation, mean-normalised log-prob (`--strategy description --norm mean`) |
| written at | 2026-11-08T04:33:49Z to 2026-11-08T04:42:07Z |

`code_commit` is read by `build_run_provenance` when the file is written, not
when the process starts (#282), so the field cannot prove its own claim. The
thing that can is the process record taken while the runs were live:

| model | pid | `/proc/<pid>/cwd` | `git rev-parse HEAD` in that directory |
|---|---:|---|---|
| Qwen3-4B-Base | 201993 | the detached worktree | `33506b86990f0eaa3fdcdf5c0ef3075905edd80f` |
| Qwen2.5-3B | 201994 | the detached worktree | `33506b86990f0eaa3fdcdf5c0ef3075905edd80f` |
| Phi-4-mini-instruct | 201995 | the detached worktree | `33506b86990f0eaa3fdcdf5c0ef3075905edd80f` |
| Qwen2.5-1.5B | 201996 | the detached worktree | `33506b86990f0eaa3fdcdf5c0ef3075905edd80f` |

The tree was not touched between launch and the last write.

## Architecture eligibility

The scan exists to replace a GDN backbone, not to rank scores. Qwen3.5-4B has 24
GDN and 8 full-attention layers, so only 8 of 32 can be made bidirectional and
the recurrent state cannot be cropped. A checkpoint that is not full attention at
every layer does not solve that problem, whatever it scores.

Fields below were read from the resolved `AutoConfig` on the machine that ran the
models, not inferred from the class name. `layer_types` is the resolved list:
`Qwen2Config`/`Qwen3Config` build one entry per layer, `Phi3Config` has no per-layer
typing at all.

| checkpoint | class | `layer_types` | `sliding_window` raw → resolved | `max_position_embeddings` | recurrent keys | verdict |
|---|---|---|---|---:|---|---|
| `Qwen/Qwen3-4B-Base` | `Qwen3Config` | 36 × `full_attention` | `None` → `None` | 32768 | none | **full-attention** |
| `Qwen/Qwen2.5-3B` | `Qwen2Config` | 36 × `full_attention` | `32768` → `None` (`use_sliding_window: false`) | 32768 | none | **full-attention** |
| `Qwen/Qwen2.5-1.5B` | `Qwen2Config` | 28 × `full_attention` | `131072` → `None` (`use_sliding_window: false`) | 131072 | none | **full-attention** |
| `microsoft/Phi-4-mini-instruct` | `Phi3Config` | absent (uniform) | `262144` → `262144` | 131072 | none | **full-attention** (sliding-window mask path; window 262144 = 2× max_pos 131072, never crops) |
| `Qwen/Qwen3-Reranker-4B` | `Qwen3Config` | 36 × `full_attention` | `None` → `None` | 40960 | none | **full-attention** |

No checkpoint carries a `linear_attn_config`, `linear_num_key_heads`,
`linear_key_head_dim`, `mamba_d_state`, `ssm_cfg`, `conv_kernel` or
`full_attention_interval` attribute, so none of them has a recurrent or linear
state to crop.

`Phi-4-mini-instruct` is the one row that needs the evidence rather than the
class name. Its config keeps `sliding_window: 262144`, and `modeling_phi3.py` in
transformers 5.21.0 selects `create_sliding_window_causal_mask` whenever that
value is not `None`. The window is twice the model's own
`max_position_embeddings`, so no input the model can accept is cropped: at any
length up to 131072 tokens every one of the 32 layers attends to the whole
prefix, and a bidirectional variant would be masked identically. It is the
sliding-window mask code path, not the causal one — that distinction is recorded
rather than smoothed over — but it is full attention in effect, and it is
eligible on that basis.

That verdict is a property of the library as much as of the checkpoint: the mask
path is chosen inside `modeling_phi3.py`. It holds for transformers **5.21.0**,
the version in every `_provenance` block here. If the version moves, this row has
to be re-read before the verdict is reused.

**Method note — read the resolved config, not the JSON.** `Qwen2.5-3B` declares
`sliding_window: 32768` and `Qwen2.5-1.5B` declares `131072`, but transformers
resolves both to `None` because each also sets `use_sliding_window: false`. A
config read that stops at the raw key would mark two eligible models ineligible
on a window they never use. Every field in the table above is the value after
`AutoConfig.from_pretrained` has applied its own resolution rules, and the raw
value is shown beside it so the two can be compared.

An ineligible row would keep its score and its reason; none of the five is
ineligible.

## #277 — 19 benchmarks

200 items per benchmark, except `authored144`, which has 144 in the file. Same
19 files and the same item order as Phase A.

The average uses the ruled denominator, `scripts/transform_results.py::
get_accuracy_for_aggregate`: `choice` and `noul` benchmarks contribute their
accuracy, `typed_decisions` is `mixed` and contributes the mean of its `noul`
and `choice` subtypes, the three `score` benchmarks contribute nothing, and the
mean is unweighted.

| model | variant | average | n | contributing | excluded |
|---|---|---:|---:|---|---|
| Qwen3-4B-Base | base | **65.5%** | 16 | all 16 choice+noul benchmarks below | sst5, stsb, yelp |
| Phi-4-mini-instruct | instruct | **64.7%** | 16 | same 16 | sst5, stsb, yelp |
| Qwen2.5-3B | base | **62.6%** | 16 | same 16 | sst5, stsb, yelp |
| Qwen2.5-1.5B | base | **58.3%** | 16 | same 16 | sst5, stsb, yelp |

These are averages over 16 benchmarks, not over 19. The three score benchmarks
are reported below as MAE and are not folded into the column.

| benchmark | type | Qwen3-4B-Base | Phi-4-mini | Qwen2.5-3B | Qwen2.5-1.5B |
|---|---|---:|---:|---:|---:|
| agnews | choice | 72.0 | 65.5 | 75.5 | 73.5 |
| arc | choice | 70.0 | 73.5 | 68.0 | 56.5 |
| authored144 | choice | 56.2 | 51.4 | 45.8 | 38.9 |
| banking77 | choice | 6.0 | 3.0 | 0.0 | 0.0 |
| codesearchnet | choice | 71.5 | 69.5 | 68.5 | 69.5 |
| contractnli | noul | 77.5 | 66.0 | 62.5 | 47.5 |
| fever | choice | 56.5 | 58.0 | 55.0 | 49.5 |
| hellaswag | choice | 64.0 | 63.5 | 60.5 | 54.5 |
| mednli | noul | 89.5 | 79.5 | 87.0 | 80.5 |
| mnli | noul | 87.5 | 80.5 | 80.0 | 83.0 |
| multirc | noul | 86.0 | 82.0 | 78.0 | 58.0 |
| race | choice | 65.5 | 65.5 | 62.5 | 60.5 |
| sst2 | noul | 80.0 | 83.5 | 80.0 | 88.0 |
| swag | choice | 65.0 | 58.5 | 62.5 | 56.5 |
| tabfact | noul | 55.5 | 77.5 | 75.0 | 71.5 |
| typed_decisions | mixed | noul 50.0 / choice 40.0 | noul 75.0 / choice 40.0 | noul 50.0 / choice 30.0 | noul 50.0 / choice 41.2 |
| sst5 | score | MAE 0.734 | MAE 0.755 | MAE 0.768 | MAE 0.962 |
| stsb | score | MAE 0.902 | MAE 0.875 | MAE 0.735 | MAE 1.169 |
| yelp | score | MAE 0.652 | MAE 0.610 | MAE 0.764 | MAE 0.938 |

`banking77` has 77 options and scores 0 to 6% for every causal model. Random is
1.3%. It is a diagnostic of the logit readout failing at high cardinality, not
a discriminator between these checkpoints.

`typed_decisions` also carries a `score` subtype (MAE 0.797 to 1.250 across the
four models). The ruled denominator does not use it: `mixed` contributes the mean
of `noul` and `choice` only, which is why the cell above shows two numbers and
not three.

## #278 — JevBench public, all 231 items

231 items, 0 failed, so nothing was skipped and every denominator is 231. Tiers
are easy 48, hard 111, original 72; the task-level chance baseline for this set
is 0.318.

**The 231 are predictions, not questions.** The `original` tier repeats each
source question as a paraphrase pair — `…-01-0` and `…-01-1`, same instructions,
same criteria, same gold answer, different state text. That is 72 predictions
over **36 distinct questions**, so the tier's effective n is 36, and the whole
set is 231 predictions over **195 distinct questions** (48 + 111 + 36). A reader
comparing the `original` 72 against another benchmark's 72 is comparing nothing,
and any statistic that treats all 231 rows as independent is slightly
anti-conservative because 36 of them form correlated paraphrase pairs. The
per-subtask counts are 12 predictions over 6 questions for each of the six
original subtasks.

All five rows below use the continuation readout, one named instrument each.

| model | variant | all 231 | easy 48 | hard 111 | original 72 | noul 74 | choice 139 | score 18 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phi-4-mini-instruct | instruct | **50.22%** | 79.17 | 36.94 | 51.39 | 51.35 | 47.48 | 66.67 |
| Qwen3-4B-Base | base | **48.48%** | 83.33 | 39.64 | 38.89 | 51.35 | 47.48 | 44.44 |
| Qwen3-Reranker-4B | (none) | **48.05%** | 87.50 | 32.43 | 45.83 | 54.05 | 43.17 | 61.11 |
| Qwen2.5-1.5B | base | **47.19%** | 77.08 | 36.94 | 43.06 | 56.76 | 42.45 | 44.44 |
| Qwen2.5-3B | base | **44.59%** | 72.92 | 33.33 | 43.06 | 50.00 | 40.29 | 55.56 |

**The Qwen3-Reranker-4B row is not the 0.628 anchor.** The anchor is
`score_options_native` — the reranker's own prompt, an independent P(yes) per
option, one option at a time — recorded in `research/13`. The row above is the
same checkpoint under option-text continuation, and it lands 14.8 pp lower.
That gap is the instrument, not the model: the pair is here so a reader who
sees only one of them cannot conclude the reranker lost 15 points, or that the
continuation rows are comparable to the anchor without saying so.

**The `original` column is not wholly a measurement.** Two of the subtasks in
that tier are answered with the same option index by every checkpoint anyone has
scored here: `original/intent` (12 predictions over 6 questions) at index 2 and
`original/routing` (12 over 6) at index 3, constant in 8 of 8 models across the
two sweeps (#288). In the five files committed here the same two cells are
constant at the same indices, 5 of 5, and `original/adequacy` (12 over 6) is
constant at index 0 in 5 of 5 as well (7 of 8 overall); `original/policy` is
constant in 4 of 8. No `easy` or `hard` subtask is constant in any model.

So 24 of the 231 predictions — **12 distinct questions** — carry no information
from this readout for every model tested, and a further 12 predictions (6
questions) are near-constant. The `original` percentage is partly a property of
where each model's constant answer falls against the gold distribution of those
subtasks, which differs by item. It does not reorder the rows — every model pays
the same cells — but the absolute value in that column should not be read as
accuracy on those subtasks. Scoring them through `score_options_native` is the
follow-up that would settle it, and it is recorded on #288 rather than guessed at
here.

## Determinism check

`Qwen/Qwen2.5-3B` was scored a second time from a separate checkout of the same
commit, with the same revision, the same 19 files and the same item order, on the
same machine. All 19 benchmarks returned identical `items` counts, `noul` and
`choice` accuracy, ECE, Brier and `score` MAE. The only values that differ are the
latency blocks, which are wall-clock and are not part of any row.

This is a determinism check, not a second measurement. The row of record is
`logit_Qwen_Qwen2.5-3B.json`; the rerun is not committed as a result, and no
number in the table above comes from it.

## What these numbers do not decide

The top two rows are 65.48 and 64.68 — 0.80 pp apart. The only two checkpoints
in this sweep that have been run twice on different stacks give the size of that
drift directly: Phi-4-mini 64.38 at Phase A → 64.68 here (+0.30 pp) and
Qwen2.5-1.5B 58.36 → 58.34 (−0.02 pp). Same data, same readout, different code
commit and a different GPU generation; 0.30 pp is not 0.80, so this is not an
argument that the gap is inside the noise band on the internal suite alone.

The second instrument does not break the tie either: on JevBench public
Phi-4-mini is 1.74 pp ahead (50.22 against 48.48), the opposite order to the
internal suite. Two instruments, opposite rankings, margins of 0.8 and 1.7 pp.
**These numbers do not order `Qwen3-4B-Base` against `Phi-4-mini-instruct`**, and
choosing between them has to rest on grounds this sweep does not supply.

The 1.74 pp is also smaller than it sounds. On those 231 items the two checkpoints
agree on the correctness verdict for 197 (85.3%) and predict the same option index
for 187 (81.0%). Of those 187, **36 are the constant `original` cells** — 18
distinct questions, each appearing twice, which both models answer identically
and which carry no information. The other **151 are genuine agreement** on items
where both models actually chose, across two model families and two tokenizers.
The margin is a net
of 4 items: 15 that only `Qwen3-4B-Base` gets right against 19 that only
`Phi-4-mini` does. A 4-item net on 231, next to 151 items of real agreement, is
not two models reading the set differently.

A base checkpoint sitting slightly behind an instruct checkpoint of the same size
is also the expected direction, not a failure. More to the point, neither column
says which is the better *starting point* for training, which is the only
question this screen exists to answer. A zero-shot readout measures what a
checkpoint can do under someone else's prompt format before a head has seen it;
it does not measure how much of that survives or improves under training.

## Data identity

All 19 benchmark files are byte-identical between two independent checkouts of
this repository (`mismatches=0`), so the rows above cannot differ through the
data.

| benchmark | sha256 |
|---|---|
| agnews | `7919b0a2c0bd3e0c67ed1f38002c878fc7a88fe8d21c45cb52ba5bb5bfa3882e` |
| arc | `f69ade2684e7602ed5ec9092dbd5df7ffe9258ceaad48575fd2c90ea993f1819` |
| authored144 | `6414a3a3a2434cf4cf3ffc35caf84aec7fe310bd84acd4b6189ccf389f323bbc` |
| banking77 | `04910f45c1ddaee278c04d7930f09d87e4e86ec67595b587df7afb006ee5b897` |
| codesearchnet | `f42471f458e7d876246e42025f24479ab95c1970d215785537fa383e6470cb0e` |
| contractnli | `1250b38330a27b063490d191520f7031a7ea0851253d7fac4573da552ed1e12e` |
| fever | `3420581c75e353066a4730850f049697c5127f495ae82d1b690e7a13dbebbb05` |
| hellaswag | `be97d17044b4d8c81b21c02acbc533c9cce2e718a7f0d1ccd437e6c1dca157ac` |
| mednli | `eff43aa673540371c144291b5e9cc4737efb3aeeb8b375b99d24d99b4783c121` |
| mnli | `fe19d1759c7a1493739235739ea0f92578a7067290b3e90e25aa7499cfe169e7` |
| multirc | `90a9ca562464db475be0b2d7b68ae2cc648447612cfb8d6f7aa2d60efa731b5f` |
| race | `673f95ba0db6ebc7f11e81f0335becaae6c6cda221fffe6969fb81395f777b26` |
| sst2 | `0532fe3a84fa63d4fb089eb9d943e240636725accc24e72b0c482812bdb1f575` |
| sst5 | `0a9dff821f86f8b2b34baf0083bb55d40c7b07c7dc96be82d326e682041ef6f5` |
| stsb | `9bc907d0edca1e82921d619f879eb126541289003a1e9c59750df32895585230` |
| swag | `596c077efba20efc9c6047142e606bf90175e1d0a1f76ed5e8e5ccaf3d087de8` |
| tabfact | `21061ae3ad4feb0c1366aa01e65d6bad351fa7f04a6809b1ba1043c1d62c1392` |
| typed_decisions | `3c1aacaddc1de5f7f313120c384eea9e255e5205cea281aa0d8a0a181b9c1077` |
| yelp | `aab33c61d929fe7626f08723d7c54b87b0e2978b84f42d3eb0da2c94bb1076f0` |

JevBench tiers: easy `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b`,
hard `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb`,
original `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`.

## Licence

Keyed by `model_id` and the `revision` in the file, so the join to any row is
unambiguous.

| model_id | revision | licence |
|---|---|---|
| `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | Apache-2.0 |
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | Apache-2.0 |
| `Qwen/Qwen3-Reranker-4B` | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Apache-2.0 |
| `microsoft/Phi-4-mini-instruct` | `cfbefacb99257ffa30c83adab238a50856ac3083` | MIT |
| `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | **Qwen Research, non-commercial** |

`Qwen2.5-3B` is usable for research only. If it becomes a production candidate,
that licence changes the decision, and it should not be discovered after the
fact. The licence was read from the Hugging Face API, not from a model card in
the cache, because the cached snapshots do not all carry a README.

## How the paths were removed

Two fields of `_provenance` record where the run lived: `snapshot_path` and
`command`. They were rewritten before commit by `scripts/redact_provenance.py`
(merged as `dc4d2b8`), which replaces the absolute root with `<host>` and keeps
the tail, so the cache layout, the script path and the benchmark arguments stay
readable. No other field changed. `--check` reported 9 files with local paths
before the rewrite and 9 already clean after it, and an independent comparison
against the raw copies confirmed every field outside those two is byte-identical
and every result value is unchanged.

Nothing else was hand-edited.

## Footnote: the documented averages do not reproduce

`research/04-model-replication.md` lists 19-benchmark averages that this rule
does not produce from the same committed files:

| model | documented | recomputed, ruled rule (n=16) |
|---|---:|---:|
| Qwen2.5-7B-Instruct | 68.5% | 67.1% |
| Phi-4-mini | 65.0% | 64.4% |
| Qwen3-4B | 64.3% | 66.3% |
| Qwen2.5-1.5B | 59.2% | 58.4% |
| Qwen3-1.7B | 54.2% | 53.6% |
| Qwen2.5-0.5B | 50.9% | 52.4% |
| Qwen3-0.6B | 47.8% | 47.7% |
| SmolLM2-1.7B | 51.1% | 50.7% |

The difference is not a constant offset and does not share a direction: some
rows are higher in the document, some lower. A second candidate rule — every
`noul` and `choice` subtype as its own entry, so `typed_decisions` counts twice —
gives 65.9 / 63.9 / 65.0 / 57.6 / 53.1 for the first five rows and does not
reproduce the column either. The original rule is not recoverable from the
files, and no attempt is made here to reverse-engineer it. The four #277 rows
above are computed under the ruled denominator and are not comparable to the
documented column without that difference in view.

The two checkpoints that appear in both places also differ slightly from the
Phase A files: Phi-4-mini 64.68 here against 64.38 recomputed there, Qwen2.5-1.5B
58.34 against 58.36. Same data and same readout, different code commit and a
different GPU generation, and bf16 reduction order is not guaranteed across
kernels. The gap is 0.02 to 0.30 pp; it is not evidence of a change in the model.

`Qwen3-4B-Base` is **not** a rerun of the September `logit_Qwen_Qwen3-4B.json`.
That file is the instruct checkpoint and scores 66.30 under the same rule; the
0.82 pp between 66.30 and 65.48 is a different checkpoint on different weights,
not drift between runs, and must not be used as an estimate of run-to-run noise.
