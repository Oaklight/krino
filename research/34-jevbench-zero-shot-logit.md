# JevBench with a zero-shot logit readout (#278) — 2026-11-07

JevBench's 231 public items were scored by three untuned checkpoints, with no
Krino head and no training. The trained heads are gated on this set, so the
question is what a model that has never seen the task scores there under a
readout that needs no training at all.

## The instrument, named

**Option-text continuation.** Each option is scored as the continuation of
`State: ... Question: ... Answer:`; the option log-probs are mean-normalised and
the argmax is the prediction. This is the same instrument as
`research/33-zero-shot-logit-baselines.md`, and it is **not** the instrument
behind the 0.628 anchor in `research/13-jevbench-public.md`.

0.628 is `raw Qwen3-Reranker-4B` scored through `score_options_native`: the
reranker's own prompt, an independent P(yes) per option, one option at a time.
A reranker is trained for that prompt. A continuation row for the same
checkpoint would not reproduce 0.628, and no row in this note is placed beside
it as if it were the same measurement.

Every file records `readout` = `option-text continuation, mean-normalised
log-prob` in `_provenance`, so a reader of the JSON does not have to take this
paragraph's word for it.

## Results

231 items, chance as reported in `research/13` (0.284 easy, 0.336 hard, 0.311
original, 0.318 overall).

**The set is 231 predictions over 195 distinct questions.** The `original`
tier's 72 rows are 36 source questions, each asked twice with a different state
and the same gold (`…-01-0` and `…-01-1`), so its effective size is 36. Every
count in this note is a count of predictions unless it says questions; where a
statistic treats the 231 rows as independent it is slightly anti-conservative.

| run | variant | overall | easy | hard | original (72 rows / 36 questions) |
|---|---|---:|---:|---:|---:|
| MiniCPM5-2B-Base | base | 0.416 | 0.729 | 0.306 | 0.375 |
| Ministral-3-3B-Base-2512 | base | 0.455 | 0.729 | 0.360 | 0.417 |
| MiniCPM3-4B | instruct | 0.459 | 0.729 | 0.387 | 0.389 |
| _items_ | | _231_ | _48_ | _111_ | _72_ |
| _mean options_ | | _3.68_ | _4.06_ | _3.41_ | _3.83_ |
| _chance_ | | _0.318_ | _0.284_ | _0.336_ | _0.311_ |

Chance is computed per item — 0.5 for noul, `1/|options|` otherwise — and the
tiers do not share one floor: easy carries the most options on average (4.06),
so its chance is the lowest (0.284) and hard's, with 3.41, is the highest
(0.336). The aggregate 0.318 is not a floor a tier can be read against. With the
per-tier columns and their own floors above, a model that only clears easy
cannot hide behind the aggregate: MiniCPM5-2B-Base is the case, 44.5 points
above chance on easy and 3.0 points below it on hard.

By question type:

| run | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|
| MiniCPM5-2B-Base | 0.500 | 0.367 | 0.444 |
| Ministral-3-3B-Base-2512 | 0.527 | 0.439 | 0.278 |
| MiniCPM3-4B | 0.500 | 0.453 | 0.333 |

The score column is 18 items, one item is 5.6 points, and the 0.278-to-0.444
swing is a 3-item difference (5 of 18 against 8 of 18). It is not a ranking.

### Constant answers in the `original` tier

Eight checkpoints have now been scored on JevBench with this readout: the three
above plus `Qwen/Qwen3-4B-Base`, `microsoft/Phi-4-mini-instruct`,
`Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-3B` and `Qwen/Qwen3-Reranker-4B`. Their
overall scores span 0.416-0.502. On five of the six `original` subtasks at least
one model answers every row with the same option index, and in four of them most
models do:

| subtask | n | constant in | index |
|---|---:|---|---|
| `original/intent` | 12 | **8 of 8** | 2 |
| `original/routing` | 12 | **8 of 8** | 3 |
| `original/adequacy` | 12 | 7 of 8 | 0 or 1 |
| `original/policy` | 12 | 4 of 8 | 0 or 1 |
| `original/ordinal` | 12 | 1 of 8 | 2 |

**No `easy` or `hard` subtask is constant in any of the eight.** The split is by
subtask family, not by difficulty, and not by model: eight checkpoints across six
model families — a llama config, a minicpm3, a mistral3, two qwen2s, two qwen3s
and a phi3 — all answer `intent` with index 2 and
`routing` with index 3. Those two families are dead cells for every checkpoint —
24 predictions over **12 source questions** (six per family, each asked twice)
carry no information from this readout. The gold sits at the constant index twice
in each family, so all eight score 2 of 12 there, and the `original` column is
partly a property of where the constant lands against the gold distribution.

The other constants: MiniCPM3-4B answers `ordinal` with index 2 (12/12);
MiniCPM5-2B-Base answers `policy` with index 1 (12/12); Ministral-3-3B-Base-2512
answers `adequacy` with index 1 (12/12). The models scored by the other sweep
answer `adequacy` with index 0 (Qwen3-4B-Base, Qwen2.5-1.5B, Qwen2.5-3B,
Qwen3-Reranker-4B, Phi-4-mini), and `policy` with index 0 (Qwen2.5-3B,
Qwen3-4B-Base, Qwen3-Reranker-4B). `adequacy` carries six gold at each index, so
a constant there returns exactly 6 of 12 — chance, presented as a performance.

This is a property of the instrument. Continuation scores the options against
one shared context, and when the options are close to uniform the argmax is
decided by a token-level prior that barely changes across items, so it can name
the same index every time. An independent P(yes) per option scores each option
against the question rather than against its siblings, so it has no term that
would favour one position across items; a collapse there would need every
option to score exactly equal. That is still an argument here, not a measurement
— these files carry no native-readout per-item predictions — and #288 records
the experiment that would settle it: score these five subtasks through
`score_options_native` on at least two checkpoints.

MiniCPM5-2B-Base is **below chance on the hard tier** (0.306 against 0.336).
That is not an artefact of the tier cut: the same 111 items are above chance for
the other two checkpoints, and the model is above chance on the other two tiers.
On the hardest third of the set this readout is not reading the task for this
checkpoint.

The three easy totals are equal at 35 of 48, but the sets of correct items
differ, so it is a coincidence of counts rather than three models making the
same predictions. Across all 231 items the three agree on 147: 61 all correct,
86 all wrong.

## Against the rows in research/13

The rows other than the three continuation rows are reproduced from
`research/13`, not recomputed here. The `readout` column is the point of the
table: these numbers come from three different instruments and are not
interchangeable.

| model | readout | overall |
|---|---|---:|
| raw Qwen3-Reranker-4B | native P(yes) per option | **0.628** |
| Qwen3-Reranker-4B | option-text continuation | 0.481 |
| raw Qwen3-Reranker-0.6B | native P(yes) per option | 0.476 |
| Krino 4B (`arch_4b_loraAll_weighted`) | the heads' own readout | 0.463 |
| MiniCPM3-4B | option-text continuation | 0.459 |
| Ministral-3-3B-Base-2512 | option-text continuation | 0.455 |
| Krino 0.6B (`arch_06b_loraAll_last_s123`) | the heads' own readout | 0.433 |
| Krino 4B heads (HF release) | the heads' own readout | 0.429 |
| MiniCPM5-2B-Base | option-text continuation | 0.416 |
| Krino ettin-150m heads (HF) | the heads' own readout | 0.368 |
| Krino reranker-0.6B heads (HF) | the heads' own readout | 0.364 |
| Krino qwen3.5-4b heads (HF) | the heads' own readout | 0.346 |
| Krino qwen3-0.6b heads (HF) | the heads' own readout | 0.342 |
| Krino modernbert-base heads (HF) | the heads' own readout | 0.325 |
| _chance_ | | _0.318_ |

`research/13` also scores two fine-tuned backbones through the native readout,
which is the same instrument as the top two rows: 4B 0.623 and 0.6B 0.541.

What the table supports:

1. **The three untuned continuation rows land inside the trained-head band.**
   Their spread is 0.416-0.459; the eight trained-head rows span 0.325-0.463.
   Of the eight, one is above all three continuation rows (0.463), two are above
   the lowest of them (0.433, 0.429) and below the top two, and five are below
   all three (0.368, 0.364, 0.346, 0.342, 0.325). Model choice among untuned
   checkpoints moves this benchmark by about 4 points for these three; across
   all eight checkpoints scored with this readout it is 8.6 points (0.416 to
   0.502). The trained-head rows span about 14, but that span mixes backbones
   from 150M to 4B, so it is the width of a heterogeneous band, not a training
   effect: no checkpoint in this table appears both with and without a head.
2. **The same checkpoint under two instruments is the strongest comparison
   here.** Qwen3-Reranker-4B scores 0.628 through the native readout and 0.481
   through continuation — **14.8 points on identical weights and items**.
   Nothing about the model changes between those two rows, so the difference is
   the instrument. The native rows still occupy the top of the table, but with
   the reranker's own continuation row in it, "native beats continuation" is an
   observation about these rows rather than a rule: Phi-4-mini's continuation
   score (0.502) is above the 0.6B reranker's native 0.476.

What it does not support:

1. **No ordering between an individual continuation row and an individual
   trained-head row.** The instruments differ and adjacent rows are 1-4 points
   apart. The one comparison that can be read is the aggregate claim above, and
   it is about bands, not pairs.
2. **No statement that an untuned checkpoint is "better than Krino".** The Krino
   rows measure the head the project ships under the head's readout; these rows
   measure a different question asked of other backbones under a different
   readout. `research/13` already establishes the head's deficit against the
   same backbone's native readout; nothing here changes that measurement.
3. **Nothing about the 0.628 anchor.** No row in this note is that instrument.
4. **A small gap between two checkpoints of the same size.** Qwen3-4B-Base
   (0.485) and Phi-4-mini-instruct (0.502) are 1.7 points apart and agree on
   whether each of 197 of the 231 items is correct, naming the same option on
   187. One item on this set is 0.4 points, so a 1.7-point gap between two rows
   that agree on 85% of items is not a separation.

## What this says for the shortlist

All three checkpoints land in a narrow band above chance — 0.416 to 0.459
against 0.318 — and far below the reranker's native 0.628. #278 pre-registered
the reading for exactly this outcome: every base below 0.628 means the reranker
readout is the thing to port, not the base to change. That is what the numbers
say. At this size, no full-attention base on the shortlist beats the reranker
readout on JevBench, so the next arm should be about the readout rather than the
backbone.

The dead cells above are the mechanism behind that sentence, not just a
correlation with it: a readout that answers six different questions — each asked
twice — with the same option index is not reading the items, and scoring the
options against one shared context is what lets it do that. It is the strongest
piece of evidence in this note for porting the native per-option readout rather
than changing the backbone.

The bound belongs in the same breath: this is a finding about the instrument's
sensitivity, not about the candidates' quality. The same three checkpoints score
55.3, 53.5 and 53.1 on the 19-benchmark set, where the committed column spans
53.6 to 67.3. Whether a base clears the reranker on one instrument says nothing
about where it sits on the other.

One row deserves its own sentence, because it is the one that will be misread:
**MiniCPM3-4B at 0.459 is one item from the trained Krino 4B at 0.463**, and it
is an untrained instruction-tuned model. That is not evidence that training adds
nothing, and it is not evidence that an untrained model matches Krino. The two
numbers come from different instruments — continuation for one, the heads' own
readout for the other — on 231 items, and this note's own rule is that a pair
across instruments cannot be ordered. The comparison is stated only so that a
reader who makes it anyway sees the bound next to it.

## Architecture eligibility

The screen this benchmark is part of needs a backbone that is full attention at
every layer, so a hybrid or windowed checkpoint is out whatever it scores.
Recorded from each checkpoint's `config.json` in the Hugging Face cache on the
machine that ran it:

| run | model_type | architectures | layers | layer_types | sliding_window | linear / gdn / mamba keys | verdict |
|---|---|---|---:|---|---|---|---|
| MiniCPM5-2B-Base | `llama` | `LlamaForCausalLM` | 42 | absent | absent | none | `full-attention` |
| Ministral-3-3B-Base-2512 | `mistral3`, `text_config.model_type` `ministral3` | `Mistral3ForConditionalGeneration` | 26 | absent | key present, value `null` | none | `full-attention` |
| MiniCPM3-4B | `minicpm3` | `MiniCPM3ForCausalLM` | 62 | absent | absent | none; `q_lora_rank` / `kv_lora_rank` present, which is MLA, still full causal attention | `full-attention` |

All three are full attention at every layer. Ministral-3-3B-Base-2512 was the
one to check: it is a Mistral-family multimodal wrapper, and those configs
commonly carry a non-null `sliding_window`. This one carries the key with value
`null`, so its 0.455 stands as an eligible row rather than a capability-only
number. MiniCPM3-4B's MLA keys compress the KV cache; they do not window or
recur over the context.

## Provenance

Each file carries `_provenance` with `model_id`, resolved `revision`, `dtype`,
`device`, `device_name`, `code_commit` (33506b8), `date_utc`, `readout`,
`strategy`, `norm`, `variant`, `command`, and the python / torch / transformers
versions (3.12.16 / 2.11.0+cu128 / 5.21.0). Absolute paths were redacted with
`scripts/redact_provenance.py` before commit.

Files are in `results/jevbench/` for the three runs above and
`model/experiments/logit_prov/` for the five from the other sweep; all eight
carry `code_commit` 33506b8 and the same readout label, and
`scripts/transform_results.py` reads both shapes with one rule — `method` is
`logit` for these rows and `trained-heads` for the rest.

## Limits

- One readout, one prompt template. A different option construction would move
  these numbers; the note reports what this instrument says, not what the best
  prompt would say.
- The score column is 18 items; 0.278 against 0.444 is three items.
- 231 predictions over 195 questions: the `original` tier's 72 rows are 36
  paraphrase pairs with the same gold, so a statistic that treats every row as
  independent — a McNemar over paired outcomes, for instance — is slightly
  anti-conservative on this set.
- `0.628` remains the native reranker readout and is not reproduced by anything
  in this note.
