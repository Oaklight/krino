# Appending irrelevant text costs 24 points, so the long-state wall is mostly length

`research/13` measures a wall: JevBench's hard tier falls about 30 points from the
shortest to the longest quartile of state lengths, for the trained head and the
backbone's native readout alike. Two things follow from reading it as a length
effect, and both are cheapest to settle before building anything. The first is
that the comparison is across items, so length is confounded with whatever else
makes those items hard. The second is that the corpus audit in the same note
finds the conjunction — a long state asked as a choice — 25 times in 172,262
training items, so a length effect here lands on inputs the model has barely seen.

This note measures length on its own. The item is held fixed: same state, same
question, same options, same gold label. Neutral archive prose is appended until
the state reaches a target length, so the only thing that changes is how many
tokens the pooled position has to read over. Padding goes after the state, never
before it, because a causal backbone pools the last token — prefix padding would
change what that position sees, which is a different experiment.

## Setup

| | |
|---|---|
| Items | 300 choice items per source, seed 42, one shuffle, the same 300 at every length |
| Lengths | untouched, then 500 / 1,500 / 4,000 / 8,000 characters |
| Padding | `model/evaluation/state_padding.py`: repeated archive prose, then a closing sentence so the question does not read as continuing |
| Models | 4B `arch_4b_loraAll_32k_s123` epoch 10 (readout `last`); 0.6B `arch_06b_weighted_lr3e4_split138_s123` epoch 5 (readout `weighted`) |
| Code | `model/scripts/state_length_probe.py`; artifacts `state_length_agnews_4b_ep10.json`, `state_length_agnews_06bw_ep5.json`, `state_length_mmlu_4b_ep10.json` |

## Results

Accuracy over the same 300 agnews questions with only the amount of appended
prose changing:

| state | tokens | 4B | 0.6B |
|---|---:|---:|---:|
| untouched (235 chars) | 61 | **0.9067** | **0.9033** |
| 502 chars | 110 | 0.9067 | 0.8800 |
| 1,501 | 288 | 0.8267 | 0.8600 |
| 4,001 | 732 | 0.7100 | 0.7667 |
| 8,001 | 1,444 | **0.6633** | **0.7233** |
| change | | **−24.3 pp** | **−18.0 pp** |

Nothing about the question changed. The model reads the same four option texts and
answers the same instruction; the only difference is 1,383 additional tokens of
prose that carry no information about the task.

## Finding 1 — two-thirds of the quartile slope is length, not content

The quartile table in `research/13` drops 33 points from the shortest to the
longest quartile of hard-tier states. Padding costs 24 points over a comparable
range of lengths. So the majority of that slope is reproducible without any of the
content that makes hard items hard, and the residual — the part padding cannot
reach — is the difficulty of the items themselves.

This is the difference between two prescriptions that were previously
indistinguishable. If the wall were about *content* — long states carrying many
interacting facts that the model must combine — the fix would be data: more long
multi-fact states. If it is mostly *length*, the fix is anything that stops a
1,400-token sequence from diluting a pooled position, and more data of the same
kind would only partly help. The measurement says mostly length.

## Finding 2 — the smaller model is less sensitive, which is the opposite of the obvious guess

The 0.6B loses 18 points where the 4B loses 24, at the same lengths and on the
same items. A capacity story would predict the small model to be more fragile. A
dilution story predicts the opposite: what is being diluted is the pooled
position's ability to concentrate on the tokens that matter, and a model with a
sharper readout has more to lose when the signal is spread over four times as
many positions. The 0.6B's curve also starts falling earlier (−2.3 pp at 110
tokens, where the 4B is flat to four decimals) and stays flatter afterwards,
which is what a model with weaker attention to begin with looks like.

Two models is not enough to call that a mechanism, and this note does not.

## Finding 3 — the control fails for the right reason

The same probe on `mmlu`, where the 4B scores 0.3300 against a 0.25 chance floor:

| state | tokens | 4B mmlu |
|---|---:|---:|
| untouched | 70 | 0.3300 |
| 1,501 chars | 288 | 0.2733 |
| 8,001 | 1,441 | 0.2967 |

The curve moves within noise (the standard error on 300 items at p≈0.3 is 2.7
points) and does not order by length. That is the right control: there is almost
no signal to dilute. If padding hurt accuracy by some mechanism unrelated to the
model's use of the state — a tokenizer edge, a positional artefact, a corrupted
prompt — it would show here too. It does not.

## What this changes about the wall

The hard tier's ceiling is not only about what the states contain; it is partly
about their size. `research/13`'s data prescription — assemble long-state choice
items — is still the right next experiment, because a model trained at the length
it is evaluated at has less to dilute. But this measurement raises the value of
the alternatives it was ranked against: a readout that pools the state instead of
attending over all of it, truncation that keeps the state's most relevant span, or
retrieval over a long state. Before this, those were guesses about what the wall
was; now they are addressable against a number.

## Caveats

- **One source, one seed, 300 items.** agnews is a 4-option classification task
  with short states, and the hard tier is a multi-option decision over a long
  structured state. The manipulation is clean, the surface it is measured on is
  easier than the target.
- **The filler is neutral prose, not noise.** A model could in principle be
  distracted by any text rather than by length per se, and this probe cannot
  separate "irrelevant text" from "more tokens". Both are present in a real long
  state, so the distinction does not rescue the wall, but it does mean "length"
  here means "length of unrelated material", which is the honest reading.
- **The 0.6B checkpoint is the clause-5 run mid-training** (epoch 5 of 10), not a
  finished arm, and its readout is `weighted` where the 4B's is `last`. The
  cross-size comparison in Finding 2 is therefore not matched on readout.
- **An earlier 0.6B attempt used `arch_06b_loraAll_last_10ep_s123`, which scores
  0.248 on agnews — chance.** Its flat 0.2700 at every length is a broken arm, not
  a null result, and it is not reported above.
