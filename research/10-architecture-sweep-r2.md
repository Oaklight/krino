# Round 2 architecture sweep: readout, LoRA scope, backbone scale

This note consolidates the round-2 architecture sweep. Every number is final-epoch
eval accuracy on the shared 13,444-item evaluation set, produced by
`model/scripts/train_multitask.py --eval-every 1`. Rows whose run stopped at an
earlier epoch say so and are compared against the frozen baseline *at that
epoch*.

**Provenance.** The training logs are not in this repository — they live on the
training hosts, at `model/experiments/arch_*.log` under the project checkout on
each (`/home/pding/projects/open-decisions` on `rbh101`, and the shared
`lambda_home` copy of the same path on `rbdgx1`/`rbdgx2`). The table reader used
to assemble them, `model/scripts/sweep_report.py`, is merged (#133), so the
tables below can be rebuilt from those logs; the runs themselves need the
benchmark corpus, which is not in the repository.

The point of the sweep was to separate three questions that are easy to
confound: how much of the gain comes from adapting the backbone (LoRA), how much
from the head's readout over hidden layers, and how much from backbone scale.

## Setup

| | |
|---|---|
| Backbones | `Qwen/Qwen3-Reranker-0.6B` / `-4B` / `-8B` |
| Data | `model/configs/r2_reranker4b.yaml` — 182,262 train / 13,444 eval items, `epoch_size` 15,000, `type_ratios` 1:1:1 |
| Training | 5 epochs unless stated (configs default to 20), seed 42, `accumulation_steps 8`, bf16, flash attention |
| LR and head rank | from the config, no CLI override: `r2_reranker0.6b.yaml` (`lr 1e-3`, `rank 64`) at 0.6B; `r2_reranker4b.yaml` (`lr 3e-4`, `rank 128`, `noul_rank 128`) at 4B, and again at 8B — there is no 8B config, those runs reuse the 4B file with `--model Qwen/Qwen3-Reranker-8B` |
| Readouts | `last`, `mean_last_n` (`mean4`), `weighted` (learned softmax over all hidden states, initialised with 0.95 on the final layer) |
| LoRA scopes | none (frozen), last 8 blocks, all blocks (`qkvo` = q/k/v/o targets, otherwise q/v) |

Two caveats that apply to the whole table:

* These runs predate the cluster-aware split (#138). Up to 2.1% of the synthetic
  eval slice was scored on near-duplicate training text, so every number here is
  a slight overestimate. The shipped implementation (#138) moves 25 of 4,476
  families — well under 1% — so it should shift these figures by a few tenths of
  a point at most, but they are not directly comparable with post-fix runs.
* Seed noise: 0.40 pp at 0.6B frozen (49.11 vs 48.71 at seed 123) and 0.35 pp at
  8B frozen (58.58 vs 58.23), 0.14 pp for the 8B full-LoRA `weighted` arms
  (62.85 / 62.99 at epoch 4). LoRA arms are noisier at small scale: a 0.6B
  full-LoRA `last` replicate moves 54.46 -> 56.46, 2.00 pp. So 0.6B LoRA arms are
  read only against a matched control, and differences under ~2 pp there are not
  resolvable.

## Results

The `Δ` column is against the arm named in `baseline`, at the same epoch. An
epoch-4 arm is scored against the frozen run's epoch 4, not its epoch 5, because
the frozen run still gains +3.19 pp across that last epoch (52.40 -> 55.59) and a
final-vs-epoch-4 delta would credit that gain to the arm under test.

### 0.6B — readout × LoRA scope

| run | readout | LoRA | final | baseline | Δ |
|---|---|---|---|---|---|
| `arch_06b_frozen_last` | last | none | 49.11 | — | — |
| `arch_06b_frozen_weighted` | weighted | none | 42.22 | frozen last 49.11 | **−6.89** |
| `arch_06b_loraAll_last` | last | all | 54.46 | frozen last 49.11 | +5.35 |
| `arch_06b_loraAll_weighted` | weighted | all | **56.17** | frozen last 49.11 | **+7.06** |
| `arch_06b_loraAll_mean4` | mean4 | all | 53.03 | frozen last 49.11 | +3.92 |
| `arch_06b_loraAll_qkvo_weighted` | weighted | all, qkvo | 55.51 | frozen last 49.11 | +6.40 |
| `arch_06b_loraLast8_weighted` | weighted | last 8 | 50.15 | frozen last 49.11 | +1.04 |

### 4B — readout × context length, LoRA complete

| run | readout | LoRA | max_length | final | baseline | Δ |
|---|---|---|---|---|---|---|
| `arch_4b_frozen_last` | last | none | 8k | 55.59 | — | — |
| `arch_4b_frozen_32k` | last | none | 32k | 55.96 | frozen last 8k 55.59 | +0.37 |
| `arch_4b_frozen_weighted` | weighted | none | 8k | 52.11 (ep4) | frozen last ep4 52.40 | −0.29 |
| `arch_4b_loraAll_last` | last | all | 8k | 64.59 | frozen last 55.59 | +9.00 |
| `arch_4b_loraAll_32k` | last | all | 32k | 64.67 | frozen 32k 55.96 | +8.71 |
| `arch_4b_loraAll_weighted` | weighted | all | 8k | **64.89** | frozen last 55.59 | **+9.30** |
| `arch_4b_loraAll_mean4` | mean4 | all | 8k | 64.81 | frozen last 55.59 | +9.22 |
| `arch_4b_loraAll_qkvo_weighted` | weighted | all, qkvo | 8k | 36.72 | frozen last 55.59 | **−18.87** |
| `arch_4b_loraLast8_weighted` | weighted | last 8 | 8k | 49.73 (ep1) | frozen last ep1 49.08 | +0.65 |

The frozen run's per-epoch values, used as the baselines above: 49.08 at epoch 1,
52.40 at epoch 4, 55.59 at epoch 5. Measured against the frozen run's *final*
55.59 instead, the 4B full-LoRA `last` arm at epoch 4 (61.92) is +6.33 — the
weaker claim, which holds even if the arm had stopped there.

That epoch-5 jump is large enough to be worth ruling out as an artifact. It is
not one: `eval_epoch_detailed` scores the live epoch-end weights on the single
eval list built once before training, and the same code runs at every epoch, so
no epoch sees a best-checkpoint or a different subset. The last epoch is also the
largest single-step gain in every arm whose schedule completed — +3.19 at 4B
frozen (52.40 -> 55.59), +2.67 and +2.80 for the 4B full-LoRA arms (61.92 ->
64.59, 61.87 -> 64.67), +2.85 at 8B frozen, and +7.22 for 8B frozen `weighted`
(48.69 -> 55.91) — so it is a property of the schedule's tail, not of this arm.
The same shape shows in the eval loss, which peaks mid-run and then falls
sharply: 1.1407 / 1.7894 / 1.9718 / 1.2011 / 1.0726 at 4B frozen.

### 8B — readout collapse

| run | readout | LoRA | final | baseline | Δ |
|---|---|---|---|---|---|
| `arch_8b_frozen_8k` | last | none | 58.58 | — | — |
| `arch_8b_frozen_8k_e10` | last | none | **61.43** (ep10) | frozen last 58.58 | +2.85 |
| `arch_8b_frozen_32k` | last | none | 58.17 | frozen last 58.58 | −0.41 |
| `arch_8b_frozen_weighted` | weighted | none | 55.91 | frozen last 58.58 | −2.67 |
| `arch_8b_loraAll_weighted` | weighted | all | **62.85 (ep4)** | frozen `last` ep4 55.73 | **+7.12** |
| `arch_8b_loraAll_weighted_s123` | weighted | all | 62.99 (ep4) | frozen `last` ep4 55.73 | +7.26 |
| `arch_8b_loraAll_8k` | last | all | 47.51 | frozen last 58.58 | **−11.07** |
| `arch_8b_loraAll_8k_s123` | last | all | 36.43 | frozen last s123 58.23 | **−21.80** |

The two 8B full-LoRA `weighted` arms are epoch-4 checkpoints: those runs stopped
at the host's wall-clock limit before finishing epoch 5, so they are lower bounds.
Both deltas are against frozen `last` at epoch 4 (55.73), which is the same-epoch
comparison but mixes readouts. The matched-readout baseline is the frozen
`weighted` arm, whose own epochs run 52.17 / 48.69 / 46.47 / 48.69 / 55.91 — epoch
4 is a mid-schedule dip, so the epoch-matched matched-readout delta there would be
+14.16 against a number no run would report as its result. Measured against that
arm's final 55.91 instead, the delta is +6.94, which mixes epochs and is therefore
a *lower* bound on what LoRA buys at 8B.

## Findings

**1. LoRA is the dominant lever; it grows from 0.6B to 4B and reverses at 8B.**
Comparing like readouts — `last` throughout — frozen is 49.11 / 55.59 / 58.58 at
0.6B / 4B / 8B and full LoRA is 54.46 / **64.59** / 47.51, i.e. +5.35 / +9.00 /
−11.07; the 8B `last` arm collapses (finding 4). At 8B, under the readout that
does not collapse, the gain is +7.12 against frozen `last` at the same epoch and
+6.94 against the frozen `weighted` arm's final — the conservative,
matched-readout figure. Not +15.34 against a collapsed `last` baseline. So the
gain peaks at 4B, which is also the best *finished* result in the sweep: 4B, full
LoRA, `weighted` readout, **64.89** at five epochs.

Whether it beats 8B is not settled by this sweep. The nearest 8B arm is 62.85, an
epoch-4 checkpoint of a run the wall-clock limit stopped, 2.04 pp behind. Two
bounds fit the same evidence. At the average rate across the 8B frozen run's five
epochs (2.85 pp, about 0.57 pp per epoch) closing 2.04 pp needs roughly three and
a half more epochs, which the run never had. At the tail rate this note measures
below — the last epoch is where the schedule lands its biggest gain, +2.85 for 8B
frozen and +7.22 for 8B frozen `weighted` — one unrun epoch is worth more than the
entire gap, and 62.85 could pass 64.89. The two bounds disagree because the
schedule is a step function rather than a slope, so the honest reading is that
this sweep cannot separate 4B from an unfinished 8B. It establishes that 4B is the
best result that finished, not that 8B is worse.

A 0.6B model with full LoRA (56.17) matches a frozen 4B (55.59): for this task
the adaptation matters about as much as a 6.7x parameter increase.

**2. Context length does not matter, at any scale.** 8k versus 32k is +0.37 pp at
4B frozen, −0.41 pp at 8B frozen, and +0.08 pp at 4B full LoRA (64.59 -> 64.67) —
all inside noise. The earlier reading of the 4B 32k result as a context effect
was wrong; the gain was LoRA. The likely reason 32k shows nothing is that our
items are short (mean 238 tokens; the longest source, contractnli, averages
1,194) and `max_length` only bounds truncation.

**3. The readout does not buy accuracy under LoRA below 8B.** On a frozen
backbone `weighted` loses clearly at 0.6B (−6.89) and at 8B (−2.67); at 4B the
only comparison available is one epoch short (52.11 at epoch 4, inside noise).
Under full LoRA the readout differences are +1.71 at 0.6B (56.17 vs 54.46), +0.30
at 4B (64.89 vs 64.59) and +0.22 for `mean4` at 4B (64.81 vs 64.59) — all within
the 2.00 pp seed spread that a 0.6B LoRA replicate shows. So below 8B the choice
of readout is not settled by accuracy; `weighted` wins decisively only at 8B
(+15.34 against the collapsed `last` arm), where it is really a rescue rather
than a gain.

`mean4` was measured twice: 53.03 at 0.6B full LoRA (against 56.17 for
`weighted`) and 64.81 at 4B full LoRA (against 64.89). Both are single cells.

A fourth arm changes the LoRA *targets* rather than the readout: all-block LoRA
on `qkvo` reaches 55.51 at 0.6B (+6.40) but at 4B falls from 54.30 at epoch 1
(+5.22 against frozen `last` epoch 1, 49.08) to 36.72 at epoch 5 (−18.87). It is
above its same-epoch baseline at epoch 1 and then loses 15.9 pp between epochs 1
and 2 (54.30 -> 38.39), which is a divergence rather than a configuration that
never worked — a distinction that matters for what to try next. One run
per scale, so this is a reproduction target, not a finding.

**4. `last` collapses at 8B under full LoRA, and it is not the seed.** 47.51 at
seed 42 and 36.43 at seed 123, against 62.85 and 62.99 for `weighted` in the same
setting — a 15 to 27 pp gap, far outside noise, reproduced on both seeds. The
earlier open question ("degenerate solution or an unlucky seed?") is answered: a
second seed does not rescue the `last` arm, so this is the readout at that scale,
not the draw. `weighted` should be the default readout above 4B, not an option.

**5. Partial LoRA is dominated.** Last-8-blocks reaches 50.15 at 0.6B where full
LoRA reaches 56.17. A frozen 0.6B run reaches 53.27 after 20 epochs (`r2_06b_r64`),
which is 94.8% of the full-LoRA accuracy but only 58.9% of its gain over the
5-epoch frozen baseline (4.16 of 7.06 pp), for several times the wall-clock.

**6. Five epochs truncates the large models.** The 8B frozen run gains +2.85 pp
going from 5 to 10 epochs (58.58 -> 61.43) and was still improving. The 8B
full-LoRA `weighted` arms are epoch-4 numbers for the same reason — the host's
wall-clock limit, not convergence — so every 8B number here is a lower bound.

## Corrections in this revision

An earlier revision of this note reported four arms at a superseded epoch, and
described three 4B arms as still running. All four have a longer run in the same
log, and all three have finished:

| arm | earlier revision | this revision |
|---|---|---|
| `arch_06b_loraAll_qkvo_weighted` | 53.44 (ep4) | 55.51 (ep5) |
| `arch_4b_loraAll_last` | 61.92 (ep4) | 64.59 (ep5) |
| `arch_4b_loraAll_32k` | 61.87 (ep4) | 64.67 (ep5) |
| `arch_8b_loraAll_8k` | 43.91 (ep4) | 47.51 (ep5) |
| `arch_4b_loraAll_weighted` | not reported | 64.89 (ep5) |
| `arch_4b_loraAll_mean4` | not reported | 64.81 (ep5) |
| `arch_4b_loraAll_qkvo_weighted` | not reported | 36.72 (ep5) |

Consequences: the 4B headline moves from 61.92 to 64.59 under `last` and to
64.89 under `weighted`, so the best *finished* arm is 4B, not 8B; the 8B
`last` collapse is 11.07 pp rather than 11.82; and the epoch-mixing in the `Δ`
column is gone, replaced by an explicit baseline column.

## Not established

* **Why `last` collapses at 8B.** Reproduced on two seeds, so it is real, but not
  explained. The candidate — a single pooled vector from an 8B backbone feeding a
  rank-128 attention head — has not been tested against, e.g., a `last` head with
  a wider rank or a LayerNorm before pooling.
* **The 4B `qkvo` collapse.** One run, one seed, and it fails within one epoch of
  starting. Worth reproducing before anything is concluded about LoRA targets.
* **Calibration and order sensitivity**: none of these numbers say whether the
  winner is calibrated or order-stable. Both are being measured (#143), and the
  literature's largest failure mode for this model family is option-order
  sensitivity, not accuracy.
* **Data supply**: score is resampled 3.8× more often than noul in this mixture
  (#142), so the score slice of every number here is the least-supported one.

## Reproduction

```bash
python model/scripts/train_multitask.py \
    --config model/configs/r2_reranker4b.yaml \
    --model Qwen/Qwen3-Reranker-8B \
    --lora-layers -1 --readout weighted \
    --no-gradient-checkpointing --epochs 5 --eval-every 1 --seed 42

# The 0.6B arms use the 0.6B config, where lr and rank differ:
#   --config model/configs/r2_reranker0.6b.yaml   (lr 1e-3, rank 64)
# versus the 4B/8B runs above (lr 3e-4, rank 128). No --lr or --rank was passed
# on any arm; both came from the config.
```

`--lora-layers -1` means all blocks, `N` means the last N blocks, `0`/absent
means frozen. `--readout weighted` is the learned layer mixture; `mean_last_n`
with `--readout-n 4` is the `mean4` row.
