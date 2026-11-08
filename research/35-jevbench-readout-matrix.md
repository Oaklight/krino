# JevBench readout matrix: framing, checkpoints, and the anchor — 2026-11-08

Four checkpoints on up to three readouts each, over JevBench-public's 231 rows,
plus a six-row judgement track for choosing a base. The question behind the
matrix is whether the distance to the 0.628 anchor is the model, the wording, or
the readout. The measurement says: the readout decides both whether a difference
exists and how large it is; the wording helps only the checkpoint that can parse
it; and the anchor itself reproduces at 0.641 rather than 0.628.

## The cells, with the readout named

| cell | model | readout | overall |
|---|---|---|---:|
| base × continuation | `Qwen/Qwen3-4B-Base` | option-text continuation | 112/231 = **0.4848** |
| base × reranker prompt | `Qwen/Qwen3-4B-Base` | `score_options_native`, reranker template | 110/231 = **0.4762** |
| base × judgement | `Qwen/Qwen3-4B-Base` | per-option yes/no, judgement wording | 124/231 = **0.5368** |
| Phi-4-mini × continuation | `microsoft/Phi-4-mini-instruct` | option-text continuation | 116/231 = **0.5022** |
| Phi-4-mini × reranker prompt | `microsoft/Phi-4-mini-instruct` | `score_options_native`, reranker template | 138/231 = **0.5974** |
| Phi-4-mini × judgement | `microsoft/Phi-4-mini-instruct` | per-option yes/no, judgement wording | 150/231 = **0.6494** |
| reranker × continuation | `Qwen/Qwen3-Reranker-4B` | option-text continuation | 111/231 = **0.4805** |
| reranker × judgement | `Qwen/Qwen3-Reranker-4B` | per-option yes/no, judgement wording | 150/231 = **0.6494** |
| reranker × native | `Qwen/Qwen3-Reranker-4B` | `score_options_native`, reranker template | 148/231 = **0.6407** |
| Qwen3-4B (post) × continuation | `Qwen/Qwen3-4B` | option-text continuation | 111/231 = **0.4805** |
| Qwen3-4B (post) × native | `Qwen/Qwen3-4B` | `score_options_native`, reranker template | 131/231 = **0.5671** |
| Qwen3-4B (post) × judgement | `Qwen/Qwen3-4B` | per-option yes/no, judgement wording | 143/231 = **0.6190** |

Chance is 0.318 overall (0.284 easy, 0.336 hard, 0.311 original). All twelve
cells score the same 231 rows with the same denominators.

Continuation scores the option text as the continuation of one shared context.
The reranker template scores each option independently against an
`<Instruct>/<Query>/<Document>` prompt. The judgement readout scores
`log P(" yes") − log P(" no")` per option under a plain question wording. The
first normalises options against each other; the other two never do.

`predicted_idx` is present on every row of the judgement files and of the two
`p_yes` files (base × reranker prompt, reranker × native). The continuation files
record `logits` and no `predicted_idx` on any row, including Qwen3-4B (post);
index counts for them come from the argmax of `logits`.

By tier and by question type:

| cell | overall | easy | hard | original | noul | choice | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| base × continuation | 112/231 | 40/48 | 44/111 | 28/72 | 38/74 | 66/139 | 8/18 |
| base × reranker prompt | 110/231 | 40/48 | 38/111 | 32/72 | 35/74 | 66/139 | 9/18 |
| base × judgement | 124/231 | 43/48 | 41/111 | 40/72 | 37/74 | 79/139 | 8/18 |
| Phi-4-mini × continuation | 116/231 | 38/48 | 41/111 | 37/72 | 38/74 | 66/139 | 12/18 |
| Phi-4-mini × reranker prompt | 138/231 | 48/48 | 48/111 | 42/72 | 42/74 | 84/139 | 12/18 |
| Phi-4-mini × judgement | 150/231 | 47/48 | 49/111 | 54/72 | 45/74 | 94/139 | 11/18 |
| reranker × continuation | 111/231 | 42/48 | 36/111 | 33/72 | 40/74 | 60/139 | 11/18 |
| reranker × judgement | 150/231 | 48/48 | 50/111 | 52/72 | 47/74 | 90/139 | 13/18 |
| reranker × native | 148/231 | 48/48 | 51/111 | 49/72 | 47/74 | 86/139 | 15/18 |
| Qwen3-4B (post) × continuation | 111/231 | 42/48 | 38/111 | 31/72 | 43/74 | 60/139 | 8/18 |
| Qwen3-4B (post) × native | 131/231 | 48/48 | 42/111 | 41/72 | 39/74 | 81/139 | 11/18 |
| Qwen3-4B (post) × judgement | 143/231 | 48/48 | 49/111 | 46/72 | 44/74 | 89/139 | 10/18 |

## Verification

Every cell was recomputed from its own `predictions` array by a script that does
not read the summary: the overall count, every tier count, every type count, and
`correct` against the argmax of the prediction's own scores (or the recorded
`predicted_idx`). All twelve are consistent; no cell's summary disagrees with its
predictions.

The six-track files were checked the same way: the two run here
(MiniCPM5-2B-Base, Ministral-3-3B-Base-2512), the three run by krino-ab
(Qwen2.5-3B, Qwen2.5-1.5B, Qwen3-1.7B-Base) and Qwen3-4B-Base from the earlier
judgement run, all at the same commit, have `items = 231`, `failed = 0`,
`predicted_idx` on 231/231, and summaries that recompute from `predictions`. The
fifteen pairwise tests were then run independently by both sessions from the same
files; the two grids agree line by line.

The judgement files used here are the v2 rerun at `2819b6f`, which reproduced
every shared field of the pre-fix A/A2/C rows exactly and added `predicted_idx` on
all 231 records. The earlier copies had no `predicted_idx` anywhere; that key is
part of the fix, not an optional column.

JevBench is 231 predictions over **195 distinct questions** — the `original`
tier's 72 rows are 36 questions asked twice, with different states and the same
gold — so concentration is reported at question level: a question is "at an
index" only when both paraphrases agree, and a group is degenerate at 5 of 6
questions. `original/adequacy` is degenerate in the cells named below.

## The anchor reproduces at 0.641, not 0.628

`research/13` records the reranker under its native readout at 0.628 (145/231).
Rerun from current code: **148/231 = 0.6407**, three items higher — `hard`
49 → 51, `original` 48 → 49, `easy` 48/48 in both.

The 0.628 run recorded no revision, no code commit and no per-item predictions,
so it cannot be item-diffed. The prompt is not the explanation: the wrapper,
`build_query_doc` and `get_options` are byte-identical to the pre-change
versions, and `reranker_prompt.py` has not changed since. What remains is a
different checkpoint revision or bf16 ties on near-ties, and neither can be
settled from the old file. Both numbers are reported; the older one is not
silently replaced, because the reason it cannot be reproduced matters more than
the three items.

## The readout decides the size of the difference

The same three weights under three readouts:

| model | continuation | reranker template | judgement |
|---|---:|---:|---:|
| `Qwen3-4B-Base` | 0.4848 | 0.4762 (wrapped) | 0.5368 |
| `Qwen/Qwen3-4B` (post) | 0.4805 | 0.5671 (native) | 0.6190 |
| `Qwen3-Reranker-4B` | 0.4805 | 0.6407 (native) | 0.6494 |

Continuation puts all three within 0.4 points of one another (111, 112, 111 items;
every pair null at p = 1.0). Judgement spans 11.3 points (124 to 150), and native
spans 16.5 (110 to 148). The readout does not merely rank the checkpoints; it sets
how far apart they can get.

## The framing helps the checkpoint that can parse it

Base model, per subtask, continuation against the reranker's template: the
templates differ on net −2 (46–48 discordant, p = 0.92), redistributing rather
than moving. The largest moves are `original/intent` +6 (2/12 → 8/12), the
`hard/sol` family −7 (sol-a −5, sol-c −2), `easy/extraction` −3, against gains
on `hard/opus-b` +3, `easy/fact` +2 and `original/ordinal` +2.

The same prompt on a post-trained model is a different readout. `Phi-4-mini` gains **+22 items** over its own continuation row (138 against 116, 42–20 discordant, p = 0.0071; Holm 0.057, BH 0.017 — uncorrected-only), with tier gains of +10 easy, +7 hard, +5 original. `Qwen/Qwen3-4B` gains **+20** (131 against 111, 48–28, p = 0.0286; Holm 0.200, BH 0.057) — **this one does not survive either correction and is a point estimate, not an established template lift.** Under the same native template the post-trained model is 21 items above the untrained base (131 against 110, +9.09 pp, 46–25, p = 0.017; Holm 0.068, BH 0.034 — uncorrected-only); that is a cross-checkpoint contrast, not the template lift.

So the sentence "the reranker's format buys nothing on a base model" is true of
`Qwen3-4B-Base` (−0.9) and false for both post-trained checkpoints (+9.5, +8.7).
The surviving explanation is that a prompt must be in distribution. The rectangle
is now closed: post-trained `Qwen/Qwen3-4B` moves with the prompt, so the gain
belongs to post-training, not to Phi-4-mini's recipe.

## The 62–65 band: post-training at 4B, clean against the weak bases only

Under the judgement readout the three post-trained rows are:

| model | overall | hard | original |
|---|---:|---:|---:|
| `microsoft/Phi-4-mini-instruct` | 150/231 = 0.6494 | 49/111 | 54/72 |
| `Qwen/Qwen3-Reranker-4B` | 150/231 = 0.6494 | 50/111 | 52/72 |
| `Qwen/Qwen3-4B` (post) | 143/231 = 0.6190 | 49/111 | 46/72 |

No pair separates: Phi-4 vs reranker 23–23 (p = 1.000, agreeing on 185 of 231);
Qwen3-4B post vs Phi-4 26–33 (p = 0.435); post vs reranker 23–30 (p = 0.410). All three are ahead of the untrained same-family base — Phi-4 and the reranker by 26 items (p = 0.0019 and 0.0011, both surviving Holm at 0.015 and 0.0096), Qwen3-4B post by 19 (p = 0.0448, **Holm 0.179, BH 0.053 — not established**). Against the two stronger bases the picture is thinner than an uncorrected read suggests. Qwen2.5-3B (132) and Ministral (129) are not separable from Qwen3-4B post (45–34, p = 0.260; 40–26, p = 0.109). Phi-4 against Qwen2.5-3B is 46–28 at p = 0.047, which survives neither correction (Holm 0.179, BH 0.053). The reranker against Qwen2.5-3B is 41–23 at p = 0.033, which survives BH (0.042) but not Holm (0.164). Phi-4 against Ministral is 41–20 at p = 0.0099, BH 0.014 and Holm 0.059. So the band is cleanly above the weak bases and **not established above the two strongest**; the in-family contrast, which was the only controlled comparison, is itself uncorrected-only.

Under the native template the reranker is still ahead of the general model: 148
against 131, 36–19 discordant, p = 0.0300. So "the reranker is not special" is a
judgement-readout statement, not a general one. Under its own template it retains
a 7.4-point edge over an instruct model of the same size.

Post-training is not sufficient for the band. `Qwen/Qwen3-1.7B` is post-trained —
its card declares `base_model: Qwen/Qwen3-1.7B-Base` — and scores 95/231 = 0.4113
under the same readout, 13 items below its own base (40–53, p = 0.213, so a point
estimate, not a separation) and below all six base rows. The band is a property
of post-training at roughly 4B, not of post-training as such.

## A second instrument: the 19-benchmark suite

Under the published aggregate rule for the 19-benchmark suite (choice and noul
only; the three score benchmarks contribute nothing; unweighted mean of 16
entries; n = 16, 3,744 items), the same three checkpoints land within 1.3 pp:

| model | 19-benchmark average |
|---|---:|
| `Qwen/Qwen3-4B-Base` | 65.48 |
| `microsoft/Phi-4-mini-instruct` | 64.68 |
| `Qwen/Qwen3-Reranker-4B` | 64.19 |

On JevBench under the judgement readout the same three span 11.3 pp (53.68,
64.94, 64.94), and the reranker is tied for first. The JevBench tie is therefore
a property of the 231 short-option rows, not of the checkpoints: a different
instrument, in composition as well as size, puts 1.3 pp between them and the
reranker last.

The second instrument does not order the base shortlist either: 65.48 against
64.68 is 0.8 pp. So neither readout, on either suite, picks a single base.

The caveat belongs to the section: the 19-benchmark aggregate excludes the three
score benchmarks, so this compares two instruments, not two samples of one.

## Dead cells: three classes, and one that is not about the readout

1. **Readout-specific.** `original/intent` and `original/routing` are constant
   under continuation for every base checkpoint tested (2/12 for all eight in
   `research/34`), and both independent-per-option readouts break the constant
   for intent:

   The cells are **accuracies, `correct/12`** — 6 questions times 2 paraphrases —
   for that subtask in that cell. The index notes are the separate concentration
   fact and are stated where they matter, not encoded in the fractions.

   | subtask | continuation accuracy (base / reranker / post) | native accuracy (reranker / post) | judgement accuracy (reranker / base / post) |
   |---|---:|---:|---:|
   | original/intent | 2/12 / 2/12 / 2/12, all constant at index 2 | 10/12 / 6/12 | 9/12 / 9/12 / 10/12 |
   | original/routing | 2/12 / 2/12 / 3/12, index 3 for the first two | 5/12 / 2/12 | 6/12 / 3/12 / 3/12 |

   Routing has exceptions inside the independent readouts: the wrapped reranker
   prompt moves the base to index 5 for 11 of 12 predictions (3/12 correct), and
   Qwen3-4B post under its native template puts 11 of 12 at index 1 (2/12
   correct). The collapse does not always break; what changes is where the
   constant sits. These are the cells that argue for keeping a second readout in
   the eval, not for changing the benchmark.

   `easy/fact` shows the same shift without being exactly constant: 10/12 and
   11/12 at index 1 under continuation (base, Phi-4), 8/12 and 10/12 for the
   reranker and Qwen3-4B post. It then splits 6/6 between the two indices in
   every post-trained independent cell except Qwen3-4B post under native, and
   moves to 10/12 at index 0 for the base under judgement.

2. **Readout-independent in the weak sense.** `original/adequacy` sits at one
   option under every readout tested, but **not the same option**: 6 of 6
   questions at index 0 under continuation for all four checkpoints here (in
   `research/34`'s eight-checkpoint sweep it was constant in seven, at index 0 or
   1); 6 of 6 at index 0 in both 64.94 cells, for Qwen3-4B post, and 5 of
   6 for the base under judgement; 5 of 6 at **index 1** for the native reranker;
   4 of 6 under the wrapped prompt, at index 1 for the base and index 0 for
   Phi-4-mini. The exceptions are informative: the native template splits
   Qwen3-4B post's adequacy 6/6 between the two indices, so the cell can carry
   information for a checkpoint that the other cells collapse. A stable position,
   not a stable answer — and because the gold is split 6 at each option, every
   cell that is 6/6 at one index scores exactly 6/12.

   That makes it a **data question, not an instrument question**: eight
   differently-trained checkpoints across six families are constant at one of the
   two options under continuation, and the six-row judgement track does not break
   the pattern — every base row leans to index 0 (at least four of six questions
   for four rows) and the one instruct row collapses to index 1. The two adequacy
   options are probably not discriminable as written. It is neutral for
   comparison — everyone pays the same chance-level 12 rows — and costly for
   power: **the informative denominator is 219, not 231.**

3. **Model-specific.** `policy` and `ordinal` vary by cell with no pattern.
   `policy`: 11 of 12 at index 0 for the base under judgement, 11 of 12 at index
   1 for Qwen3-4B post and for the reranker under its native template, and an
   even 6/6 split between indices 0 and 1 for the reranker under judgement.
   `ordinal`: 10 of 12 at index 2 for the base under continuation, 9 of 12 for
   the base and for Qwen3-4B post under judgement, while the reranker spreads
   over four indices and scores 12/12. These should be checked per cell, not
   assumed.

## The judgement prompt turns this base into a yes-sayer, not the readout

noul answers by predicted index, recomputed from the score arrays:

| cell | index 0 | index 1 | correct |
|---|---:|---:|---:|
| base × judgement | **64** | 10 | 37/74 = 50.0% |
| base × reranker prompt | 50 | 24 | 35/74 |
| Phi-4-mini × judgement | 46 | 28 | 45/74 = 60.8% |
| reranker × judgement | 50 | 24 | 47/74 = 63.5% |
| reranker × native | 36 | 38 | 47/74 |
| Qwen3-4B (post) × judgement | 31 | **43** | 44/74 = 59.5% |
| Qwen3-4B (post) × native | 40 | 34 | 39/74 |
| Qwen3-1.7B-Base × judgement | **71** | 3 | 32/74 = 43.2% |

The base models can collapse to "yes" under this prompt: Qwen3-4B-Base answers
86.5% at index 0 and exactly chance; Qwen3-1.7B-Base answers 95.9% at index 0 and
scores 32/74, three below the always-yes baseline (35/74). The post-trained
checkpoints answer it more evenly and score 59–64%, so the yes-sayer is a
model-and-prompt pair, not a property of independent per-option scoring. Base ×
judgement's 53.68 is therefore a choice-and-original result.

## Test families and correction

Every p-value in this note is a two-sided exact McNemar test on 231 paired items, and none of them was corrected for multiplicity when the note was first written. The tests are also not independent: the same cells appear in many pairs. This revision reports two corrections for every declared family — **Holm**, which controls the family-wise error rate and holds under any dependence, and **Benjamini–Hochberg** at 5%, which controls the false discovery rate but assumes independence or positive dependence. Where the two disagree, both values are printed.

The families, declared after the fact and therefore themselves a judgement:

| family | tests | what it supports |
|---|---:|---|
| within-checkpoint readout pairs | 12 | "the readout decides how far apart checkpoints can get" |
| band vs base rows, judgement | 18 | "the 62–65 band is above the bases" |
| base track, judgement | 15 | "the judgement readout orders the base shortlist" |
| continuation ordering | 10 | "neither readout orders the top" |
| native-template cross-checkpoint | 6 | "the reranker keeps an edge under its own template" |

A claim is called **established** here only if it survives both corrections. Anything that survives one and not the other is printed with both values and called **uncorrected-only**.

## Paired tests

Exact McNemar over the 231 shared rows:

| comparison | discordant | p |
|---|---|---:|
| base × judgement vs base × continuation | 45–33 | 0.213 |
| base × judgement vs base × reranker prompt | 40–26 | 0.109 |
| Phi-4-mini × judgement vs its own reranker prompt | 28–16 | 0.096 |
| reranker × judgement vs reranker × native | 28–26 | 0.892 |
| reranker × judgement vs base × judgement | 43–17 | 0.0011 |
| Phi-4-mini × judgement vs base × judgement | 46–20 | 0.0019 |
| Phi-4-mini × judgement vs reranker × judgement | 23–23 | 1.000 |
| Qwen3-4B (post) × judgement vs base × judgement | 50–31 | 0.0448 |
| Qwen3-4B (post) × judgement vs Phi-4-mini × judgement | 26–33 | 0.435 |
| Qwen3-4B (post) × judgement vs reranker × judgement | 23–30 | 0.410 |
| Qwen3-4B (post) × native vs reranker × native | 19–36 | 0.030 |
| Qwen3-4B (post) × native vs base × reranker prompt | 46–25 | 0.017 |
| Qwen3-4B (post) × native vs its continuation | 48–28 | 0.029 |
| Qwen3-4B (post) × judgement vs its native | 34–22 | 0.141 |
| reranker × continuation vs base × continuation | 12–13 | 1.000 |
| Qwen3-4B (post) × continuation vs base × continuation | 13–14 | 1.000 |
| Qwen3-4B (post) × continuation vs reranker × continuation | 14–14 | 1.000 |
| base × reranker prompt vs base × continuation | 46–48 | 0.918 |
| Phi-4-mini × reranker prompt vs its continuation | 42–20 | 0.0071 |

Three lines carry the matrix. Under continuation the three checkpoints are
pairwise indistinguishable (all p = 1.0). Under judgement the post-trained rows
are pairwise indistinguishable and the base is behind. Under the native template
the reranker is ahead of the general instruct model.

## The six-row judgement track (#292)

Pre-registered before the cells landed: across the six base rows, one row is
above another only at paired exact McNemar p < 0.05. If no pair separates, the
judgement readout cannot choose a base. The six rows are `Qwen3-4B-Base`,
`Qwen2.5-3B`, `Qwen2.5-1.5B`, `Qwen3-1.7B-Base`, `MiniCPM5-2B-Base` and
`Ministral-3-3B-Base-2512`. `MiniCPM3-4B` is `variant: instruct` and is reported
as a seventh row outside the rule; so are Phi-4-mini and Qwen3-4B (post).

All six rows are at `2819b6f` with `items = 231`, `failed = 0` and
`predicted_idx` on 231/231:

| model | overall | easy | hard | original |
|---|---:|---:|---:|---:|
| `Qwen/Qwen2.5-3B` | 132/231 = 0.5714 | 47/48 | 44/111 | 41/72 |
| `mistralai/Ministral-3-3B-Base-2512` | 129/231 = 0.5584 | 45/48 | 43/111 | 41/72 |
| `Qwen/Qwen3-4B-Base` | 124/231 = 0.5368 | 43/48 | 41/111 | 40/72 |
| `Qwen/Qwen2.5-1.5B` | 110/231 = 0.4762 | 44/48 | 38/111 | 28/72 |
| `openbmb/MiniCPM5-2B-Base` | 109/231 = 0.4719 | 34/48 | 42/111 | 33/72 |
| `Qwen/Qwen3-1.7B-Base` | 108/231 = 0.4675 | 38/48 | 39/111 | 31/72 |

Seven of the fifteen pairs separate at the uncorrected threshold of p < 0.05, all between the top three and the bottom three: Qwen2.5-3B from Qwen2.5-1.5B (49–27, p = 0.015), MiniCPM5 (52–29, p = 0.014) and Qwen3-1.7B-Base (41–17, p = 0.002); Ministral from Qwen2.5-1.5B (40–21, p = 0.020), MiniCPM5 (52–32, p = 0.038) and Qwen3-1.7B-Base (46–25, p = 0.017); Qwen3-4B-Base from Qwen3-1.7B-Base (35–19, p = 0.040).

**Under correction the structure does not survive.** Holm leaves one test: Qwen2.5-3B over Qwen3-1.7B-Base (41–17, p = 0.0022, Holm 0.0335). Benjamini–Hochberg at 5% leaves the same one. The remaining six fall to Holm 0.20–0.38 and to BH 0.20–0.38. **The two-group reading in the earlier revision of this note is withdrawn**: what the column establishes is that one base is above one other base, not that the track splits in three and three.

The top three are mutually inseparable: Qwen2.5-3B vs Ministral 29–26 (p = 0.79),
Qwen2.5-3B vs Qwen3-4B-Base 29–21 (p = 0.32), Ministral vs Qwen3-4B-Base 32–27
(p = 0.60). The bottom three are also mutually inseparable: Qwen2.5-1.5B vs
MiniCPM5 43–42 (p = 1.000), Qwen2.5-1.5B vs Qwen3-1.7B-Base 34–32 (p = 0.902),
MiniCPM5 vs Qwen3-1.7B-Base 37–36 (p = 1.000).

### The tied top: a post-hoc ruling

The pre-registered rule specified only the all-null case. The measured case — a
tied top group with separations underneath — it did not cover, so the ruling
below is post-hoc.

Ruling: a tied top group is the same decision state as all-null for choosing a
base. The column eliminated the bottom half and did not order the top half.
Therefore the judgement column cannot choose a base; the choice among the tied
leaders must come from something else.

Continuation fails the same way. Paired exact McNemar on the continuation files:

| pair | net (discordant) | p |
|---|---:|---:|
| Qwen3-4B-Base vs Qwen2.5-3B | +9 (18–9) | 0.1221 |
| Qwen3-4B-Base vs Phi-4-mini | −4 (15–19) | 0.6076 |
| Qwen3-4B-Base vs Qwen3-Reranker-4B | +1 (13–12) | 1.0000 |
| Qwen2.5-3B vs Phi-4-mini | −13 (9–22) | **0.0294** |
| Qwen2.5-1.5B vs Phi-4-mini | −7 (17–24) | 0.3489 |
| Phi-4-mini vs Qwen3-Reranker-4B | +5 (19–14) | 0.4869 |

Ten pairs, one separates uncorrected (Qwen2.5-3B vs Phi-4-mini, p = 0.0294); under correction that one is gone as well (Holm 0.295, BH 0.295), and the closest remaining pair is Qwen3-4B-Base against Qwen3-Reranker-4B at one item, 13–12, p = 1.0000. So **neither readout orders the top of this shortlist on 231 items, and correction makes that conclusion stronger rather than weaker.**

`MiniCPM3-4B` (instruct) is 78/231 = 0.3377 — **the plain-text judgement template
does not work for this chat checkpoint**; the number is a format failure, not a
capability reading. Its easy tier is 17/48 (35.4%), against 66.7% for the next
lowest checkpoint measured, and the response is degenerate: adequacy, ordinal and
policy 6 of 6 questions at index 1, intent 5 of 6 at index 2. It is also 28 items
below its own continuation row (106/231). Seventh row, outside the six-row rule.

Against the six base rows, Phi-4-mini and the reranker separate from every row at the uncorrected threshold. Under correction the picture splits: both clear Qwen3-4B-Base, Qwen2.5-1.5B, Qwen3-1.7B-Base and MiniCPM5 under Holm; Phi-4 against Qwen2.5-3B survives neither (Holm 0.179, BH 0.053) and Phi-4 against Ministral survives only BH (Holm 0.059, BH 0.014); the reranker against Qwen2.5-3B survives only BH (Holm 0.164, BH 0.042). Qwen3-4B (post) against Qwen3-4B-Base is 50–31 at p = 0.045, which survives neither correction (Holm 0.179, BH 0.053); it does clear Qwen2.5-1.5B, MiniCPM5 and Qwen3-1.7B-Base under both, and is null against Qwen2.5-3B (45–34, p = 0.260) and Ministral (40–26, p = 0.109). **The in-family contrast is the only controlled comparison, and after correction it is a point estimate.**

## What this means for the build plan

- **The readout decides whether a difference is visible, and how big it is.**
  Continuation makes three different checkpoints identical; judgement opens an
  11-point span over the same rows and weights.
- **The 62–65 band is a point-estimate cluster, not a separable tier, and correction makes that weaker still.** The three post-trained rows tie each other; Qwen3-4B (post) does not separate from Qwen2.5-3B or Ministral; Phi-4 and the reranker clear those two only at the margin; and the in-family post-trained-vs-base contrast, which used to be the one comparison that cleared p < 0.05 uncorrected, does not survive either correction (Holm 0.179, BH 0.053). An instruct model ties the reranker under judgement; under the reranker's native template the gap is uncorrected-only (p = 0.030, Holm 0.090, BH 0.045).
- **The chat template helps only a model that was post-trained on one, and only one of the two post-trained gains is established.** Base `Qwen3-4B` moves by −0.9 pp from the reranker's prompt. `Phi-4-mini` moves by +9.5 (p = 0.0071, Holm 0.057, BH 0.017 — uncorrected-only) and `Qwen/Qwen3-4B` by +8.7 (p = 0.029, Holm 0.200, BH 0.057 — not established). The direction is the same for both; the size of the claim is not.
- **Porting the readout means training the head with the independent-per-option
  objective, not re-prompting a model differently.** The one thing the prompt
  does by itself is redistribute accuracy across subtasks.

## Limits

- 231 predictions over 195 questions; 219 informative ones once the adequacy
  family's 12 chance-level rows are excluded.
- The 0.628 file cannot be item-diffed; the 0.641 explanation is a candidate
  list, not a cause.
- One run per cell. The totals are 231 rows each, and a 3-item difference is
  inside the resolution of the set.
- The fifteen pairwise tests over the six-row track are exact McNemar. A null
  means "not separable at this n", not "equal": the largest null inside the top
  group is 32–27 (p = 0.60), and the smallest separation is 52–29 (p = 0.014).
- The judgement prompt is krino-ab's design (`model/evaluation/judgement_prompt.py`);
  these cells are its measurement, not its motivation.
