# The routed system's ECE passes only when each path gets its own temperature

Goal item 1 is met by a rule rather than a model: short lists go to the
backbone's native reranker readout and long ones to the trained head, which is
worth +3.77 pp overall (`research/17`) and +1.74 pp more with the blend
(`research/19`). Item 2 asks for `ECE <= 0.05` after per-type temperature scaling,
and three different quantities have been read as answering it. The first is the
head's **CV-selected monotone map**, 0.0264 / 0.0137 / 0.0289 over noul / choice /
score, which is what `research/17` quotes as "met on every type". The second is
the head's **per-type temperature** from the same table: 0.0273 / 0.0180 /
**0.1244**, where the score primitive stays 2.5× the bar — a temperature does not
fix score, a map does. The third is the slot the clause actually has to be read
in: the deployed system, which has two scorers with different calibration and is
not the head at all. This note measures that system. Its records are choice-only,
so the per-question-type reading of the routed system is *not measured* here; what
is measured is one temperature for the whole choice set, which misses 0.05 on
every checkpoint, and one temperature per **path**, which clears it on every one.

## Setup

| | |
|---|---|
| Input | the `--dump-records` files from `model/scripts/blend_scorers.py`: 7,026 choice items per checkpoint, each with its head logit vector and its native log-odds |
| Rule | `k < 20` options → native readout, `k >= 20` → trained head (6,526 / 500 items; the 16 six-to-nineteen items follow the `rule_sweep` boundary) |
| Temperatures | fitted by minimising ECE on the other folds' items, applied to the held-out fold; 4 folds, folds alternate by list position |
| Checkpoints | 4B `arch_4b_loraAll_32k_s123` epoch 10, and the epoch-5 pair (seeds 123 and 42) on the cluster-disjoint split |
| Code | `model/evaluation/routed_calibration.py`, `model/scripts/eval_routed_calibration.py`; artifact `model/experiments/routed_calibration.json` |

Temperature scaling is monotone, so every accuracy below is the argmax's and does
not move with the temperature; only ECE and Brier do.

## Results

ECE, Brier and accuracy on the held-out folds, averaged over the four folds.

| | s123 ep10 | s123 ep5 | s42 ep5 |
|---|---:|---:|---:|
| head only, T=1 | **0.0509** | 0.0352 | 0.0284 |
| head only, fitted T (0.99–1.11) | 0.0227 | 0.0338 | 0.0258 |
| routed, T=1 | **0.1716** | 0.1733 | 0.1721 |
| routed, one temperature for everything (fitted) | **0.0607** | **0.0526** | **0.0774** |
| routed, one temperature per path (fitted) | **0.0355** | **0.0371** | **0.0382** |
| accuracy (all rows) | 0.7048 | 0.6641 | 0.6890 |
| Brier, head only (fitted T) | 0.4695 | 0.5648 | 0.5331 |
| Brier, routed (per-path T) | 0.4046 | 0.4494 | 0.4269 |

## Finding 1 — the native readout is the miscalibrated one, by a factor of three

At `T = 1` the routed system's ECE is 0.172 while the head alone is 0.051 / 0.035 /
0.028. The fitted temperatures say where it comes from: the native path wants
**2.7–3.6**, the head wants **0.94–1.37**. The head arrives close to calibrated and
the reranker does not, which is unsurprising — the reranker was trained on a
pointwise relevance objective, not on the option set of a choice question.

That also explains why one number for the type cannot work: a single temperature
that divides the native logits by 3 would soften the head into uselessness, and
one that leaves the head alone leaves the native path at 0.17.

## Finding 2 — on the head the literal per-type reading fails on score; on the system it is not measured

`research/17`'s per-type table is where the clause's literal reading lives. After
a temperature of its own per type the trained head is at noul 0.0273, choice
0.0180 and **score 0.1244** — one primitive fails, by two and a half times the
bar — and it is the CV-selected map, not the temperature, that takes score to
0.0289. So on the head the clause passes only if "per-type temperature scaling"
is read as "whatever per-type post-hoc map"; read literally, it fails on score.

**Which primitive carries the failure, and two numbers that are not the per-type
column.** Split by type, the clause as worded passes on noul and choice and fails
only on score. A second checkpoint measured the same way
(`arch_4b_loraAll_32k/best_heads.pt`, `model/experiments/calibration_ecefit_4b_32k.json`,
temperature fitted on half the type's items and applied to the other half) gives
noul 0.0127, choice 0.0127 and **score 0.0631** — the same shape, a narrower
score miss. That artifact also carries the unscaled column for the same
checkpoint, **0.0274 / 0.0407 / 0.1568**, and both of those triples circulate as
"the per-type ECE": 0.0407 is choice **at T=1** on that checkpoint, not under a
fitted temperature, and it would read as the clause passing on choice for the
wrong reason. What survives every reading is the claim about *which type*:
choice and noul pass under a per-type temperature on both checkpoints, score
fails on both (0.1244 and 0.0631), and score is the primitive whose gold labels
`#177` shows were half fabricated — so the clause cannot be read as failing on
that number until the repaired corpus is re-measured.

Beside the routed system's 0.0607 the per-type head numbers are not comparable
and the note should not put them in one sentence without saying so: 0.0607 is two
scorers sharing one divisor over choice items, and the head's numbers are one
scorer with a temperature per type. Neither is a per-question-type ECE of the
deployed system.

For the deployed system the per-type question cannot be answered from these
records at all: they are choice-only, so there is no noul or score item to fit a
temperature on. What can be answered is how many temperatures the choice set
gets. **One temperature for the whole choice set** lands at **0.0607, 0.0526 and
0.0774** against the 0.05 bar — a miss on all three checkpoints, and not a
marginal one on two of them. That number is *not* the head's 0.0180: 0.0180 is
choice items under a temperature of their own, with no native path in the mix,
and 0.0607 is two scorers forced to share one divisor. The prose in earlier
drafts of this note called 0.0607 "per type"; it never was one.

## Finding 3 — per-path temperatures clear it on two seeds and two training lengths

With one temperature per path — still two numbers, still fitted on items the
evaluation does not see, still no per-item adaptation — the same three checkpoints
give **0.0355, 0.0371 and 0.0382**. All three are under 0.05, and the two epoch-5
numbers are two seeds of the same training length, so this is not a one-seed
artefact.

## Finding 4 — routing does not cost calibration, and improves Brier

The routed system is more accurate than the head alone (0.7048 against 0.6285 at
epoch 10) and also better on multiclass Brier (0.4046 against 0.4695). So the
routing gain in `research/17` and `research/19` is not being paid for in
calibration quality — once each path is calibrated, the system is better on both
axes at once.

## What this means for the goal

Item 2's ECE half is **met on the routed system with per-path temperatures, and
failed with a single one**; on the head it is met with the CV-selected map and
failed under the temperature the clause literally names, because score needs the
map. All of these are global, held-out calibrations; the difference is how many
of them there are. That makes it an author decision, and the note should not
pretend otherwise:

- **If per-path scaling counts**, item 2's ECE half is met on the system that
  meets item 1, on two seeds, at 0.0355–0.0382.
- **If the clause means one temperature per question type** (noul / choice / score), this note
  does not measure it on the routed system — the records are choice-only, so that reading needs a
  calibration dump over all three primitives; where it is measured, on the head, it fails on
  score (0.1244) and passes only with the map. What this note can say is that within choice, a
  single temperature fails and per-path succeeds. If the choice reading is the one that counts,
  the repair is not a better temperature but a better-calibrated native path: fit its
  temperature at serving time from training items, or replace the reranker score with a
  calibrated head. Software alone will not close 0.0607 → 0.05 without touching
  how the native scores are produced.

Either way, the previously quoted 0.0264 / 0.0137 / 0.0289 is the CV-selected
**map** column on the head — not a temperature, and not the system — and should
be labelled as such wherever it appears. The head's temperature column is
0.0273 / 0.0180 / 0.1244, and that is the one that answers the clause as worded.

## Caveats

- **The 16 six-to-nineteen items follow the deployed rule** (`k < 20` → native).
  They are 0.2% of the set and cannot move any number here by more than 0.001.
- **The fold assignment alternates by list position**, which is the order the
  script wrote the items in. It is not a random split, so the numbers were
  re-measured with `--folds 2`, which changes how much of the set each fit sees:

  | | folds 4 | folds 2 |
  |---|---|---:|
  | routed, per path: s123 ep10 / s123 ep5 / s42 ep5 | 0.0355 / 0.0371 / 0.0382 | 0.0278 / 0.0336 / 0.0324 |
  | routed, one temperature for everything: s123 ep10 / s123 ep5 / s42 ep5 | 0.0607 / 0.0526 / 0.0774 | 0.0589 / 0.0517 / 0.0632 |

  The largest move is 0.014 (s42, one temperature for everything). **No verdict
  flips**: the single-temperature arm stays above 0.05 on all three checkpoints
  under both schedules and per-path stays below on all three. The two-seed spread on the per-path number is 0.0011 at
  four folds, so the pass is not resting on the fold size.

  The verdicts are mean-level statements and the individual folds are not all on
  the same side of the bar. On the passing side, s42 ep5 has a per-path fold at
  **0.0501**, over 0.05, on a mean of 0.0382; its per-fold spread (sd 0.0141) is
  three times the other two checkpoints', so its 2-SE upper bound is 0.0523. On
  the failing side, s123 ep5 has a single-temperature fold at **0.0429**, under
  0.05, on a mean of 0.0526. Neither flips a verdict, but "no verdict flips" is a statement
  about the means, and the choice this note asks the author to make is exactly
  the place where a fold-level margin matters.
- **The head's per-type numbers predate the STS-B repair.** 0.0273 / 0.0180 /
  0.1244 were measured while the score scale carried GLUE's withheld-label
  sentinel. On the repaired corpus (`#177`) the same harness gives noul 0.0465,
  choice 0.0446 and score **0.1264** on the s123 epoch-9 checkpoint
  (`cal_4b_split138_s123_ep9_repaired.json`). The score primitive is the one that
  fails on both corpora, at about the same level, so the claim above does not
  depend on the repair; the map's 0.0289 does, and has not been re-measured.
- **The criterion is ECE, not NLL**, because the clause is written in ECE. An NLL
  fit would pick different temperatures (the two criteria disagree by a factor of
  ~2 on the score primitive, `research/17`) and this note does not report that
  variant.
- **The head-only T=1 number at epoch 10 is 0.0509** — the head alone is right at
  the bar before scaling, which is worth knowing when reading any claim that the
  head "is calibrated" without a temperature.
- **The records are committed, and so is the two-fold pass.** Both were inputs
  the note quoted but could not be re-run from a checkout: the per-item dumps
  now live in `model/experiments/blend_records/` (3.4 MB each; the `/tmp` path
  each run records as its `source` in `routed_calibration.json` is where they
  were first written, not where they are now) and
  `model/experiments/routed_calibration_folds2.json` is the output of the
  fold-count check, so the 0.0278 / 0.0336 / 0.0324 and 0.0589 / 0.0517 / 0.0632
  rows reproduce with
  `eval_routed_calibration.py --records NAME=PATH … --folds 2` rather than from
  the note alone.
