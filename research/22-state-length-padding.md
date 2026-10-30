# Padding costs the tower 24 points and the cross-encoder 1, so dilution is a read-out defect

`research/13` measures a wall: JevBench's hard tier falls about 30 points from the
shortest to the longest quartile of state lengths. Reading that as a length effect
is the cheap explanation, and it has a consequence worth settling before anyone
builds against it — if the wall is about *how much* text there is, the fix is
anything that stops a long sequence diluting the pooled position; if it is about
*what* the text says, the fix is data.

This note measures length on its own. The item is held fixed: same state, same
question, same options, same gold label. Neutral archive prose is appended until
the state reaches a target length, so the only thing that changes is how many
tokens the head reads over. Padding goes after the state, never before it, because
a causal backbone pools the last token — prefix padding would change what that
position sees, which is a different experiment.

The answer is that padding costs a great deal to one read-out and almost nothing
to another, which says the effect belongs to the architecture rather than to the
task. That is a narrower and more useful result than "the wall is length", and it
is the one the measurements support.

## Setup

| | |
|---|---|
| Items | 300 choice items per cell, seed 42, one shuffle, the same 300 at every length |
| Lengths | untouched, then 500 / 1,500 / 4,000 / 8,000 characters |
| Padding | `model/evaluation/state_padding.py`: repeated archive prose, then a closing sentence so the question does not read as continuing |
| Models | 4B `arch_4b_loraAll_32k_s123` ep10 (`last`); 4B `arch_4b_xenc_split138_s123` ep3 (`weighted`, cross-encoder); 0.6B `arch_06b_weighted_lr3e4_split138_s123` ep5 (`weighted`) |
| Code | `model/scripts/state_length_probe.py`, `--choice-mode` selects the head shape; artifacts `state_length_agnews_{4b_ep10,4bxenc_ep3,06bw_ep5}.json`, `state_length_mmlu_4b_ep10.json` |

## Results

Accuracy over the same 300 agnews questions with only the amount of appended
prose changing:

| state | tokens | 4B tower | 4B cross-encoder | 0.6B tower |
|---|---:|---:|---:|---:|
| untouched (235 chars) | 61 | **0.9067** | **0.8833** | **0.9033** |
| 502 chars | 110 | 0.9067 | 0.8833 | 0.8800 |
| 1,501 | 288 | 0.8267 | 0.8733 | 0.8600 |
| 4,001 | 732 | 0.7100 | 0.8733 | 0.7667 |
| 8,001 | 1,444 | **0.6633** | **0.8733** | **0.7233** |
| change | | **−24.3 pp** | **−1.0 pp** | **−18.0 pp** |

Nothing about the question changed. The model reads the same four option texts and
answers the same instruction; the only difference is 1,383 additional tokens of
prose that carry no information about the task.

## Finding 1 — the padding cost is a property of the read-out, not of the state

The tower attends over the encoded context sequence, so 1,400 tokens of filler
spread the same attention over twenty times as many positions. The cross-encoder
puts the state into each option's prompt and reads a number out of the pooled
hidden state, so the filler is not competing for the same weights. On the same
items, at the same lengths, one loses 24 points and the other loses 1.

That is the whole of the mechanism this probe can establish, and it is enough to
re-rank the fixes. For the tower, dilution is a defect that can be removed by
changing the read-out — and has been, in `research/21`. For the cross-encoder, the
padding cost is already gone, so any remaining long-state deficit on real items is
not this.

## Finding 2 — and yet the cross-encoder still spreads 27 points across the hard tier's quartiles

The tempting conclusion is that the hard tier's long-state wall was dilution all
along. The quartiles say no: on one cut, the cross-encoder scores 0.704 / 0.393 /
0.250 / 0.429 across the hard tier's length quartiles while the tower scores
0.630 / 0.393 / 0.357 / 0.250 (`research/21`, `model/scripts/jevbench_quartiles.py`).
The cross-encoder is 18 points ahead on the longest quartile — the direction this
probe predicts — but it is not flat, and with 27–28 items per cell its
non-monotonicity is not resolvable either way.

So dilution explains a difference *between* architectures on long items and does
not explain the wall itself. The hard tier's long states are long because of what
they contain; padding shows that the tower is hurt twice, once by the content and
once by the length, and the cross-encoder only once.

## Finding 3 — the smaller tower is less sensitive, the opposite of a capacity guess

The 0.6B tower loses 18 points where the 4B tower loses 24, on the same items at
the same lengths. A capacity story would predict the small model to be more
fragile. A dilution story predicts the opposite: what is being diluted is the
read-out's ability to concentrate on the positions that matter, and a model with a
sharper read-out has more to lose when the signal is spread over four times as
many positions. The 0.6B also starts falling earlier (−2.3 pp at 110 tokens, where
the 4B is flat to four decimals) and stays flatter afterwards, which is what a
model with weaker attention to begin with looks like. Two models is not enough to
call that a mechanism and this note does not.

## Finding 4 — the control fails for the right reason

The same probe on `mmlu`, where the 4B tower scores 0.3300 against a 0.25 chance
floor:

| state | tokens | 4B mmlu |
|---|---:|---:|
| untouched | 70 | 0.3300 |
| 1,501 chars | 288 | 0.2733 |
| 8,001 | 1,441 | 0.2967 |

The curve moves within noise (the standard error on 300 items at p≈0.3 is 2.7
points) and does not order by length. That is the right control: there is almost
no signal to dilute. If padding hurt accuracy by some mechanism unrelated to how
the model uses the state — a tokeniser edge, a positional artefact, a corrupted
prompt — it would show here too. It does not.

## What this changes about the wall

`research/13` proposed assembling long-state choice items, on the grounds that the
corpus has 25 that reach the tier's median length. That remains the only fix
aimed at *content*, and Finding 2 says content is still there to fix.

What this note adds is a second, independent fix that is already implemented: a
read-out that scores each option in the state's full context rather than attending
over a dilated sequence. It buys 18 points on the longest quartile and 3.6 points
on the tier (`research/21`), and the padding probe explains why those two numbers
are the same phenomenon.

## Caveats

- **One source, one seed, 300 items.** agnews is a 4-option classification task
  with short states, and the hard tier is a multi-option decision over a long
  structured state. The manipulation is clean; the surface it is measured on is
  easier than the target.
- **The three models are not matched on epoch or readout.** The tower is 4B ep10
  with `last`; the cross-encoder is ep3 with `weighted`; the 0.6B is ep5 with
  `weighted`. The *slope* comparison is within-model, which is what the
  manipulation supports, but the baselines differ by up to 2.3 points and the
  tower-versus-cross-encoder contrast is confounded with readout as well as head
  shape.
- **The filler is neutral prose, not noise.** A model could in principle be
  distracted by any text rather than by length per se, and this probe cannot
  separate "irrelevant text" from "more tokens". Both are present in a real long
  state, so the distinction does not rescue the wall, but it does mean "length"
  here means "length of unrelated material", which is the honest reading.
- **The cross-encoder's top row is three items of movement.** 0.8833 / 0.8833 /
  0.8733 / 0.8733 / 0.8733 is 262–265 correct out of 300, so its −1.0 point change
  is a statement that the effect is absent rather than a precise estimate of a
  small one.
- **An earlier 0.6B attempt used `arch_06b_loraAll_last_10ep_s123`, which scores
  0.248 on agnews — chance.** Its flat 0.2700 at every length is a broken arm, not
  a null result, and it is not reported above.
