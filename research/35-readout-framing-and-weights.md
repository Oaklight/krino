# The readout, the weights, and what post-training enables — 2026-11-08

The zero-shot screen (#277, #278) left a gap it could not explain: Qwen3-Reranker-4B
scores 64 on JevBench public, Qwen3-4B-Base scores 48 under the same option-text
continuation readout, and the difference could be the reranker's *framing* — it
asks a yes/no judgement question per option — or its *weights*. This note crosses
four checkpoints with three readouts over the same 231 items and separates them.

Answer in one paragraph: **post-training, not family and not the reranker's prompt,
is what puts a 3.8–4B checkpoint into the 62–65 band.** Under a plain-text judgement
readout, three post-trained checkpoints from three families land within 3.04 pp of
each other with no pair distinguishable, and a same-size base sits 19 to 26 items
behind all three. The reranker's own chat-wrapped prompt is not the mechanism: it is
worth less than the plain-text framing for every post-trained checkpoint measured,
and worth nothing at all to the base. Post-training is not sufficient on its own —
a post-trained 1.7B scores 41.13 and misses the band — and the six-row base track
shows that the readout cannot choose among bases either, which leaves the licence as
the only measured difference between the tied leaders.

## The three readouts

| readout | prompt | score | file |
|---|---|---|---|
| continuation | `State/Question/Answer:` | option text scored as a completion | `eval_jevbench_logit.py` |
| native chat | `<Instruct>/<Query>/<Document>` in the conversation template | independent P(yes) per option | `eval_jevbench_reranker.py` |
| plain-text judgement | `State/Question/Option:/Does this option correctly answer the question? Answer:` | `log P(" yes")` against `log P(" no")` | `eval_jevbench_judgement.py` |

The continuation readout is the one used by every row in `research/34` and by the
19-benchmark suite. The judgement readout is new here; its prompt and option
construction are in `model/evaluation/judgement_prompt.py`, torch-free, and its
`_provenance.prompt_template` records the exact template used.

## The rectangle

JevBench public, 231 items, 0 failed on every cell.

| checkpoint | continuation | native chat prompt | plain-text judgement |
|---|---:|---:|---:|
| Qwen3-4B-Base | 48.48 | 47.62 | 53.68 |
| Qwen3-4B (post-trained) | 48.05 | 56.71 | **61.90** |
| Phi-4-mini-instruct | 50.22 | 59.74 | **64.94** |
| Qwen3-Reranker-4B | 48.05 | 64.07 | **64.94** |

## What each readout buys

Paired exact McNemar over the same 231 items, two-sided:

| comparison | discordant | net | pp | p |
|---|---:|---:|---:|---:|
| base: judgement vs its continuation | 78 | +12 | +5.19 | 0.213 |
| Qwen3-4B-post: judgement vs its continuation | 86 | +32 | +13.85 | **0.0007** |
| Phi-4-mini: judgement vs its continuation | 74 | +34 | +14.72 | **0.0001** |
| Phi-4-mini: native vs its continuation | 62 | +22 | +9.52 | **0.0071** |
| Phi-4-mini: judgement vs native | 44 | +12 | +5.19 | 0.096 |
| **Qwen3-4B-post vs base, native prompt** | 71 | +21 | +9.09 | **0.0170** |
| **Qwen3-4B-post vs base, judgement** | 81 | +19 | +8.23 | **0.0448** |
| **Qwen3-4B-post vs Phi-4-mini, judgement** | 59 | +7 | +3.03 | 0.435 |
| **Qwen3-4B-post vs reranker, judgement** | 53 | +7 | +3.03 | 0.410 |
| **Phi-4-mini vs reranker, judgement** | 46 | 0 | 0.00 | 1.000 |
| base vs Phi-4-mini, judgement | 66 | +26 | +11.26 | **0.0019** |
| base vs reranker, judgement | 60 | +26 | +11.26 | **0.0011** |
| reranker: judgement vs native | 54 | +2 | +0.87 | 0.892 |

Three things follow.

**Post-training is the property, and it is worth about nine points on a fixed
prompt.** The Qwen3-4B pair holds family, size, tokenizer and prompt constant and
differs only in whether the checkpoint was post-trained: +9.09 pp on the native
prompt and +8.23 pp on the judgement prompt, both significant. That pair has one
variable, which is the strongest design in this note.

**The judgement readout is what unlocks that capability.** The same post-trained
checkpoint gains +13.85 pp moving from continuation to judgement; the base gains
+5.19 and not significantly. Under judgement the three post-trained checkpoints —
Qwen3-4B, Phi-4-mini, Qwen3-Reranker-4B, three families — sit at 61.90, 64.94 and
64.94, and no pair is distinguishable (p = 0.435, 0.410, 1.000). The base is 19 to
26 items behind all three.

**The reranker's own prompt is not the mechanism.** It gives the reranker 64.07 and
the judgement prompt gives it 64.94 — a 2-item difference at p = 0.892. On
Qwen3-4B-Base the same native prompt scores 47.62, one item *below* continuation.
On Phi-4-mini it helps (+9.52 pp) but the plain-text judgement prompt helps more
(+5.19 pp over native, p = 0.096). Whatever the reranker learned, it is not a
dependence on `<Instruct>/<Query>/<Document>`.

## What this does not say

**It does not say the reranker is redundant.** The reranker holds 64.07 on the
prompt it was trained for and 64.94 on one it was not. That is robustness, and it
is the reason it can anchor a comparison at all. The finding is about the base:
the capability is latent in a post-trained model and the readout surfaces it.

**It does not say the three post-trained checkpoints are equal.** A 3.04 pp spread
that no pair can resolve is a statement about this design — one run per cell, 231
items — not about the checkpoints. Family, recipe, size (3.8B against 4B) and noise
are four candidate explanations for three data points.

**The 7-item family term is not a measurement of family.** Qwen3-4B-post is 7 items
behind Phi-4-mini on the same prompt. That is one checkpoint pair, one prompt, no
replicate, and the design cannot separate a family effect from this checkpoint
being better at this task. The matched pair gets a confidence claim because it
holds four things constant; this does not, and the note does not treat the small
effect as fragile and the large one as solid.

**The parity is on one instrument and one option profile.** 231 rows, mean 3.4
options, 74 of them binary. Nothing here says the post-trained checkpoints tie at
77 options or on an ordinal scale. `score` is 18 rows with chance 0.247: Phi-4-mini
61.11 against the reranker 72.22 is 2 items and is not read either way.

**Each readout has its own dead cells.** Continuation: `original/intent` at index 2
and `original/routing` at index 3, constant across eight checkpoints (#288, #290).
Native chat prompt: `original/routing` and `original/policy` at 11/12 on
Qwen3-4B-post, `original/adequacy` 12/12 on the reranker. Plain-text judgement:
`original/adequacy` 12/12 on Phi-4-mini, `original/policy` 11/12 and
`original/ordinal` 9/12 on the base. No family is near-constant under all three,
and `original/adequacy` is the only one near-constant under two.
"24 of 231 for every model" has to name the instrument every time it is used.

## The noul column

The judgement prompt deepens an existing yes-lean on the base and does not create
it: base continuation answers "yes" for 53 of 74 noul items (72%, 51.35% accurate),
judgement 64 of 74 (86%, 50.00%), and gold is 35 yes / 39 no, so an always-yes
responder scores 47.3%. The post-trained checkpoints do not show this: Phi-4-mini
answers 46/28 and scores 60.81%, the reranker 50/24 and 63.51%. The yes-saying is
that base under that prompt, not a property of the readout.

The baseline comparison is per-type and has to be quoted that way: an always-yes
responder scores 47.30% on the 74 noul items, while an always-index-0 responder on
all 231 rows scores 30.30%, which every row in every table above beats. The
Qwen3-1.7B-Base row is the one place the statement bites — 71 of 74 noul predictions
at index 0 and 43.24% on noul, below the always-yes baseline — and it is a
statement about noul, not about the model or the readout overall.

## The native anchor moved by three items

The native reproduction here is **64.07%** (148/231). The documented anchor is
**62.80%** (145/231). Easy matches exactly (48/48); the difference is hard +2 and
original +1.

The move is not explainable from the artifacts. The prompt construction is
byte-identical to the pre-#180 version, `reranker_prompt.py` has not changed since
2026-10-29, and the 0.628 run recorded no model revision, no code commit and no
per-item predictions, so there is nothing to diff against. The candidates are a
different checkpoint revision behind the same repository id, or bf16 ties resolving
differently on a different GPU generation. Neither can be confirmed or excluded.

64.07 is the number of record: it carries provenance, a pinned revision, a clean
tree and predictions. 62.80 is history, and anyone quoting it after this note is
quoting a number that cannot be audited.

## The base track (#292)

The judgement readout was then run on six base checkpoints, to ask whether it can
choose a base where the continuation column could not. The rule was fixed before
the cells landed: a row is above another only at paired exact McNemar p < 0.05, and
if no pair separates, the readout cannot choose a base whatever the point estimates.

| row | variant | overall | easy 48 | hard 111 | original 72 |
|---|---|---:|---:|---:|---:|
| Qwen2.5-3B | base | **57.14** | 97.92 | 39.64 | 56.94 |
| Ministral-3-3B-Base-2512 | base | **55.84** | 93.75 | 38.74 | 56.94 |
| Qwen3-4B-Base | base | **53.68** | 89.58 | 36.94 | 55.56 |
| Qwen2.5-1.5B | base | 47.62 | 91.67 | 34.23 | 38.89 |
| MiniCPM5-2B-Base | base | 47.19 | 70.83 | 37.84 | 45.83 |
| Qwen3-1.7B-Base | base | 46.75 | 79.17 | 35.14 | 43.06 |

Seven of the fifteen pairs separate, and they split the track in two with no leader
inside either group. The top three — Qwen2.5-3B, Ministral, Qwen3-4B-Base — are all
pairwise null (p = 0.788, 0.322, 0.603). The bottom three — Qwen2.5-1.5B, MiniCPM5,
Qwen3-1.7B-Base — are all pairwise null (p = 1.000, 0.902, 1.000). Seven of the nine
cross-group pairs separate; Qwen3-4B-Base against Qwen2.5-1.5B (p = 0.125) and
against MiniCPM5 (p = 0.101) do not.

**So the judgement column ranks groups and cannot rank within them.** That is a
stronger statement than "uninformative": it resolves about 4–6 pp and stops. And the
continuation column does not order the top either — ten pairwise tests over the same
shortlist gave one separation. Neither 231-item instrument chooses a base, and the
choice has to come from something else.

**The licence is then the only decision-relevant difference measured among the
tied top three.** `Qwen/Qwen2.5-3B` is under the Qwen Research licence
(non-commercial); `mistralai/Ministral-3-3B-Base-2512` and `Qwen/Qwen3-4B-Base` are
both Apache-2.0. A null result that leaves the top three indistinguishable is still
usable if it says the remaining difference is one that constrains deployment rather
than one that shows up in accuracy.

**Post-training is not sufficient for the 62–65 band.** `Qwen/Qwen3-1.7B` is a
post-trained checkpoint (`base_model: Qwen/Qwen3-1.7B-Base` on its card) and scores
41.13 under this readout — below every 3.8–4B post-trained row and below its own
base twin, 46.75, which is the sixth row above. The base-versus-post-trained
comparison at 1.7B is +13 items for the base at p = 0.213, so it is null and the
direction is a hypothesis, not a result: two sizes, one pair each. What survives is
narrower and solid — neither 1.7B checkpoint reaches the band the 3.8–4B
post-trained checkpoints occupy.

Whether post-training is *necessary* for that band is not tested. One base row
misses it; the six-row track is the test, and it fails if any base reaches 61.90.

## The 19-benchmark comparison

JevBench is 231 items with mean 3.4 options. Every shortlist row in `research/34`
has a 19-benchmark cell; the reranker had none, so JevBench parity alone could not
say whether the reranker is level on a second instrument. The row below was added
for that reason, at the same settings as the other eight.

| model | 19-benchmark average, ruled denominator (n=16) | JevBench judgement |
|---|---:|---:|
| Qwen3-4B-Base | 65.48 | 53.68 |
| Phi-4-mini-instruct | 64.68 | 64.94 |
| **Qwen3-Reranker-4B** | **64.19** | 64.94 |
| Qwen2.5-3B | 62.55 | — |
| Qwen2.5-1.5B | 58.34 | — |

**Parity holds on the second instrument.** The reranker is 64.19 against Phi-4-mini's
64.68 on 19 benchmarks, and both trail Qwen3-4B-Base's 65.48. So the JevBench tie is
not an artefact of 231 rows with mean 3.4 options: on the larger suite, with 3,744
items and the same continuation readout, the reranker is level with the best 3.8B
instruct checkpoint and below a base.

That is the same picture as the judgement column, with the ordering inverted at the
top and every difference small: these four checkpoints sit within 1.3 pp of each
other on the 19-benchmark suite, whereas on JevBench under the judgement readout
they span 11.3 pp. The suite's composition — 16 choice+noul entries, no score
benchmarks in the aggregate — is a different instrument again, and the row is
reported in its own table for that reason.

The reranker's per-benchmark behaviour is not uniform: `banking77` at 3.5% (77
options, the high-cardinality failure every causal model shows), `arc` 64.0, `race`
58.0, `typed_decisions` noul 50.0 / choice 33.8, and score MAE 0.927 / 0.823 / 0.903
on sst5 / stsb / yelp.

## Reproduce

```
python model/scripts/eval_jevbench_judgement.py \
    --model Qwen/Qwen3-4B-Base --variant base \
    --output model/experiments/logit_prov/jevbench_judgement_Qwen_Qwen3-4B-Base.json

python model/scripts/eval_jevbench_reranker.py \
    --model Qwen/Qwen3-Reranker-4B \
    --output model/experiments/logit_prov/jevbench_reranker_native_Qwen_Qwen3-Reranker-4B.json
```

Every cell above ran from a tree at `57c33df` (reranker cells) or `2819b6f`
(judgement cells, which differ only in writing `predicted_idx` on every record);
the continuation rows are from `33506b8` and are committed with #287. Result files
are in `model/experiments/logit_prov/`. Absolute paths in `_provenance` were removed
with `scripts/redact_provenance.py`; every other field is as the run wrote it.
