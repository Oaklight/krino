# Zero-shot logit baselines on the 19-benchmark set (#277) — 2026-11-07

Three checkpoints that have never seen a Krino head, scored on the same
benchmark set and with the same script as the existing zero-shot column. The
question is what the task itself looks like to a base model: if a raw LM reads
the options and still scores near chance, the head is the problem; if it scores
well, the head is not adding the capability.

## What was measured

| repo id | revision | variant | params | dtype |
|---|---|---|---|---|
| `openbmb/MiniCPM5-2B-Base` | `96a57cd572a02506b4500f54427dca24970c1bac` | base | 2B | bfloat16 |
| `mistralai/Ministral-3-3B-Base-2512` | `6f9c4b12a95b139af68670a6713616b757923735` | base | 3B | bfloat16 |
| `openbmb/MiniCPM3-4B` | `d6b14ddaefdb11c624dd75c3c779549bc90b08cb` | instruct | 4B | bfloat16 |

The readout is option-text continuation: the option text is scored as the
continuation of `State: ... Question: ... Answer:`, the option log-probs are
mean-normalised, and the argmax is the prediction. No prompt is tuned and no
example is given. Every run uses `model/scripts/eval_logit.py` at
`code_commit` 33506b8 with `--max-items 200`, `--strategy description`,
`--norm mean`, matching the cap and settings of the existing column.

Weights were downloaded on the compute host and stayed there. Nothing in this
repository is a checkpoint.

## The aggregate rule

The 19-benchmark average is the published rule, `get_accuracy_for_aggregate` in
`scripts/transform_results.py`, not a rule written for this note:

- noul and choice benchmarks contribute their accuracy;
- `typed_decisions` contributes the mean of its noul and choice subtypes;
- the score benchmarks (`sst5`, `stsb`, `yelp`) contribute nothing, because
  their metric is MAE, not accuracy;
- the average is the unweighted mean of the remaining 16 entries.

The 16 entries: `agnews`, `arc`, `authored144`, `banking77`, `codesearchnet`,
`contractnli`, `fever`, `hellaswag`, `mednli`, `mnli`, `multirc`, `race`,
`sst2`, `swag`, `tabfact`, `typed_decisions`.

## Results

200 items per benchmark, 144 for `authored144`. Accuracy on the model's argmax
option.

| benchmark | MiniCPM5-2B-Base | Ministral-3-3B-Base-2512 | MiniCPM3-4B (instruct) |
|---|---:|---:|---:|
| agnews | 0.735 | 0.795 | 0.805 |
| arc | 0.650 | 0.745 | 0.510 |
| authored144 | 0.514 | 0.465 | 0.493 |
| banking77 | 0.005 | 0.000 | 0.000 |
| codesearchnet | 0.700 | 0.705 | 0.695 |
| contractnli | 0.430 | 0.430 | 0.435 |
| fever | 0.495 | 0.495 | 0.530 |
| hellaswag | 0.540 | 0.610 | 0.575 |
| mednli | 0.420 | 0.340 | 0.430 |
| mnli | 0.645 | 0.380 | 0.480 |
| multirc | 0.610 | 0.465 | 0.470 |
| race | 0.600 | 0.620 | 0.640 |
| sst2 | 0.740 | 0.675 | 0.630 |
| swag | 0.570 | 0.610 | 0.595 |
| tabfact | 0.735 | 0.775 | 0.780 |
| typed_decisions | 0.456 | 0.450 | 0.425 |
| **16-entry average** | **55.3** | **53.5** | **53.1** |

The three score benchmarks (`sst5`, `stsb`, `yelp`) are reported in the files
and contribute nothing here, because their metric is MAE.

`banking77` is at chance for all three: 77 options, chance 1.3%, and the argmax
of the option-text continuation barely moves off a constant option. That is a
property of this readout on a wide label set, not of these checkpoints: the
existing column lands between 0.0 and 2.5 on the same benchmark with 200 items.

The two base checkpoints land 53.5 and 55.3. Against the recomputed existing
column that puts MiniCPM5-2B-Base between Qwen3-1.7B (53.6) and Qwen2.5-1.5B
(58.4), and Ministral-3-3B-Base slightly below Qwen3-1.7B; both are below
Qwen3-4B (66.3). The instruct checkpoint at 53.1 is the lowest of the three, but
it is a different model family from MiniCPM5, so the pair is not a controlled
base-versus-instruct comparison and this note does not read it as one.

## Tracks are not pooled

`openbmb/MiniCPM3-4B` is the instruction-tuned release; the other two are base
checkpoints. Their averages are reported in one table for reading convenience
and are never averaged together: an instruct number is a different instrument
from a base number, and the difference is not the thing this note measures.

## Comparison with the existing column

Recomputed from the committed files with the same rule: Phi-4-mini-instruct
64.4, Qwen2.5-1.5B 58.4, Qwen3-1.7B 53.6, Qwen2.5-7B-Instruct 67.1,
Qwen3.5-9B 67.3, Qwen3-4B 66.3.

`research/04-model-replication.md` lists 65.0, 59.2, 54.2, 68.5, 65.5 and 64.3
for the same six runs. The two columns do not reproduce each other: the
differences are not a constant offset and they change sign. This note uses the
recomputed values, because they come from the committed files through the
published function; where a comparison against the doc's column would change a
conclusion, the doc's column is not used.

## Provenance

Every file carries its own `_provenance` block: `model_id`, resolved `revision`
from the snapshot path, `dtype`, `device` and `device_name`, `code_commit`,
`date_utc`, `readout`, `strategy`, `norm`, `variant`, `command`, and the python,
torch and transformers versions the run imported.

Absolute paths in `snapshot_path` and `command` are redacted with
`scripts/redact_provenance.py` before the file is committed: the home root
becomes `<host>` and the tail is kept, so the revision and the benchmark
arguments stay readable. No committed file carries an absolute path.

## Limits

- 200 items per benchmark, not the full set. The numbers are comparable with the
  existing column, which uses the same cap, and not with a full-set run.
- Logit scoring is deterministic given the checkpoint, the dtype and the
  tokenizer, so there is no seed to vary; a rerun on different hardware can
  still differ in the last digits through kernel selection.
- JevBench is a separate issue (#278). Nothing here is comparable with the
  0.628 anchor, which is a reranker readout, not option-text continuation.
