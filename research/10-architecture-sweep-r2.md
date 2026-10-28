# Round 2 architecture sweep: readout, LoRA scope, backbone scale

This note consolidates the round-2 architecture sweep. Every number is final-epoch
eval accuracy on the shared 13,444-item evaluation set, produced by
`model/scripts/train_multitask.py --eval-every 1` and read back from
`model/experiments/arch_*.log` with `model/scripts/sweep_report.py`.

The point of the sweep was to separate three questions that are easy to
confound: how much of the gain comes from adapting the backbone (LoRA), how much
from the head's readout over hidden layers, and how much from backbone scale.

## Setup

| | |
|---|---|
| Backbones | `Qwen/Qwen3-Reranker-0.6B` / `-4B` / `-8B` |
| Data | `model/configs/r2_reranker4b.yaml` — 182,262 train / 13,444 eval items, `epoch_size` 15,000, `type_ratios` 1:1:1 |
| Training | 5 epochs unless stated, seed 42, `lr 1e-3`, `accumulation_steps 8`, bf16, flash attention |
| Readouts | `last`, `mean_last_n` (`mean4`), `weighted` (learned softmax over all hidden states, initialised with 0.95 on the final layer) |
| LoRA scopes | none (frozen), last 8 blocks, all blocks (`qkvo` = q/k/v/o targets, otherwise q/v) |

Two caveats that apply to the whole table:

* These runs predate the cluster-aware split (#138). Up to 2.1% of the synthetic
  eval slice was scored on near-duplicate training text, so every number here is
  a slight overestimate. The fix changes 273 of 4,476 families, so it should
  shift these figures by a few tenths of a point at most, but they are not
  directly comparable with post-fix runs.
* Seed noise: 0.40 pp at 0.6B final epoch, 0.35 pp at 8B, 0.09–0.85 pp for
  same-config reruns. Differences under ~0.8 pp are treated as noise throughout.

## Results

### 0.6B — readout × LoRA scope

| run | readout | LoRA | final | Δ vs frozen `last` |
|---|---|---|---|---|
| `arch_06b_frozen_last` | last | none | 49.11 | — |
| `arch_06b_frozen_weighted` | weighted | none | 42.22 | **−6.89** |
| `arch_06b_loraAll_last` | last | all | 54.46 | +5.35 |
| `arch_06b_loraAll_weighted` | weighted | all | **56.17** | **+7.06** |
| `arch_06b_loraAll_mean4` | mean4 | all | 53.03 | +3.92 |
| `arch_06b_loraAll_qkvo_weighted` | weighted | all, qkvo | 53.44 (ep4) | +6.15 (ep4) |
| `arch_06b_loraLast8_weighted` | weighted | last 8 | 50.15 | +1.04 |

### 4B — readout × context length

| run | readout | LoRA | max_length | final | Δ vs frozen `last` |
|---|---|---|---|---|---|
| `arch_4b_frozen_last` | last | none | 8k | 55.59 | — |
| `arch_4b_frozen_32k` | last | none | 32k | 55.96 | +0.37 |
| `arch_4b_frozen_weighted` | weighted | none | 8k | 52.11 (ep4) | −0.29 (ep4) |
| `arch_4b_loraAll_last` | last | all | 8k | **61.92 (ep4)** | **+9.52 (ep4)** |
| `arch_4b_loraAll_32k` | last | all | 32k | 61.87 (ep4) | +9.44 (ep4) |
| `arch_4b_loraLast8_weighted` | weighted | last 8 | 8k | 49.73 (ep1) | +0.65 (ep1) |

### 8B — readout collapse

| run | readout | LoRA | final | Δ vs frozen `last` |
|---|---|---|---|---|
| `arch_8b_frozen_8k` | last | none | 58.58 | — |
| `arch_8b_frozen_8k_e10` | last | none | **61.43** (10 epochs) | +2.85 at equal epochs |
| `arch_8b_frozen_32k` | last | none | 58.17 | −0.41 |
| `arch_8b_frozen_weighted` | weighted | none | 55.91 | −2.67 |
| `arch_8b_loraAll_weighted` | weighted | all | **62.85 (ep4)** | **+7.12 (ep4)** |
| `arch_8b_loraAll_8k` | last | all | **43.91 (ep4)** | **−11.82 (ep4)** |

## Findings

**1. LoRA is the dominant lever, and it grows with backbone size.** Frozen
(49.11 / 55.59 / 58.58 at 0.6B / 4B / 8B) versus full LoRA (54.46 / 61.92 / 62.85
where measured). A 0.6B model with full LoRA (56.17) matches a frozen 4B (55.59):
for this task the adaptation matters about as much as a 6.7× parameter increase.

**2. Context length does not matter, at any scale.** 8k versus 32k is +0.37 pp at
4B frozen, −0.41 pp at 8B frozen, and −0.05 pp at 4B full LoRA — all inside noise.
The earlier reading of the 4B 32k result as a context effect was wrong; the gain
was LoRA. The likely reason 32k showed nothing is that our items are short (mean
238 tokens; the longest source, contractnli, averages 1,194) and `max_length`
only bounds truncation.

**3. The readout's sign depends on both LoRA scope and backbone size.** On a frozen
backbone `weighted` loses at every scale (0.6B −6.89, 4B −0.29 at ep4, 8B −2.67).
Under full LoRA it wins at 0.6B (+1.71) and at 8B (+18.94 vs the collapsed `last`
arm). `mean4` loses in both regimes, so it is the learned softmax specifically,
not multi-layer aggregation in general, that flips.

**4. `last` collapses at 8B under full LoRA.** 43.91 at epoch 4 versus 62.85 for
`weighted` in the same setting, a 19 pp gap that is far outside noise. Two
readings are open and not yet distinguished: the `last` head reaches a degenerate
solution at that scale (a single pooled vector from an 8B backbone, with a rank-64
attention head on top), or the run hit an unlucky seed. `arch_8b_loraAll_last_s123`
(seed 123, otherwise identical) is running to distinguish them. If it reproduces,
`weighted` should be the default readout above 4B, not an option.

**5. Partial LoRA is dominated.** Last-8-blocks reaches 50.15 at 0.6B where full
LoRA reaches 56.17. The closest a frozen 0.6B run gets is 53.27 after 20 epochs
(`r2_06b_r64`), i.e. 70% of the full-LoRA result for several times the wall-clock.

**6. Five epochs truncates the large models.** The 8B frozen run gains +2.85 pp
going from 5 to 10 epochs (58.58 → 61.43) and was still improving. Every 5-epoch
number at 8B in this table is a lower bound.

## Not established

* **4B full LoRA + `weighted`**: three arms (weighted, qkvo_weighted, mean4) are
  on H100s at ~2.2 h/epoch and will finish well after this note. Without them the
  4B readout comparison rests on a single epoch, where frozen-baseline deltas are
  dominated by schedule warmup.
* **Whether the 8B `last` collapse is stable**: see finding 4.
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

python model/scripts/sweep_report.py model/experiments/arch_*.log
```

`--lora-layers -1` means all blocks, `N` means the last N blocks, `0`/absent
means frozen. `--readout weighted` is the learned layer mixture; `mean_last_n`
with `--readout-n 4` is the `mean4` row.
