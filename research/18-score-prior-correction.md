# Is clause 2's QWK reachable by a class-prior correction?

Clause 2 asks for `QWK >= 0.85` on the score primitive. On the best checkpoint the
mixed-scale QWK is **0.6562**, and the two candidate repairs discussed so far —
temperature scaling and a monotone confidence map — cannot touch it: both leave
every argmax where it was, so neither changes a single ordinal prediction. That
made the score head's scale mismatch the last untested lever on a failing clause.
This note measures it and reports a negative result.

Everything below is one checkpoint (`arch_4b_loraAll_32k_s123`, epoch 10), one
seed, the 1,832 score items inside the 13,444-item cluster-split eval, at
`model/scripts/score_vote_adjust.py`; the numbers are in
`model/experiments/score_vote_adjust_s123_ep10.json`.

## The correction

Logit adjustment shifts every level by a constant and re-takes the argmax:

```
adjusted_i = logit_i + alpha * (log p_ref_i - log p_hat_i)
```

`p_hat` is the marginal the head produces on the items being corrected, `alpha`
scales the shift (0 is the identity, 1 is a full correction), and `p_ref` is the
marginal being aimed at. Two references are measured, and the difference between
them is the point of the experiment:

- **train** — the training split's label marginal. Deployable: it is known before
  any eval item is seen.
- **gold** — the eval items' own label marginal. An oracle, so it bounds what any
  prior correction could reach on those items. It is not a method.

The `p_hat` term is transductive — it comes from the model's outputs on the batch
being scored. No eval label enters either reference's use in the sweep.

## What the marginals say

Per level scale, as probabilities over levels:

| scale | n | train | model argmax | gold |
|---|---|---|---|---|
| 4 | 218 | 0.101 / 0.291 / 0.362 / 0.247 | **0.060** / 0.362 / 0.317 / 0.261 | **0.133** / 0.257 / 0.372 / 0.239 |
| 5 | 1113 | 0.153 / 0.246 / 0.200 / 0.239 / 0.162 | 0.164 / 0.232 / 0.184 / 0.251 / 0.169 | 0.152 / 0.243 / 0.208 / 0.221 / 0.176 |
| 6 | 500 | 0.107 / 0.141 / 0.163 / 0.230 / 0.254 / 0.105 | 0.140 / 0.152 / 0.144 / 0.228 / 0.256 / 0.080 | **0.534** / 0.108 / 0.082 / 0.118 / 0.124 / 0.034 |

(A one-item three-level scale is excluded everywhere; its QWK is noise.)

Three things are visible before any correction is applied.

**The five-level scale is nearly matched** and needs no repair: the model lands
within 2.4 pp of the gold marginal at every level, and its QWK there is 0.827.

**The six-level scale is the STS-B split, and the head is faithful to training.**
The model predicts 0.140 on level 0 where its training split is 0.107 — closer to
training than to the 0.534 its eval items actually contain. This is the
train/test marginal mismatch flagged in `research/17`, and the head's behaviour
is what a model trained on that split should do. Nothing is broken; the split
is.

**The four-level scale does show the bottom-level aversion, and it is smaller
than the eval wants it to be.** The model emits 0.060 on level 0 against 0.133 of
gold, and the surplus sits on level 1 (0.362 against 0.257). But the training
marginal is also only 0.101, so most of the gap to gold is distribution, not
aversion: a correction to *train* has little to do, and a correction to *gold* is
an oracle.

## What the correction does

Mixed-scale QWK, item-weighted over the three readable scales:

| alpha | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| reference = train | 0.6562 | 0.6537 | 0.6522 | 0.6513 | 0.6533 | 0.6548 | **0.6570** |
| reference = gold (oracle) | 0.6562 | 0.6569 | 0.6560 | **0.6576** | 0.6572 | 0.6568 | 0.6530 |

Per scale, at the unmodified argmax, at the deployable optimum, and at the
oracle optimum:

| scale | baseline | train, alpha 1.5 | gold, alpha 0.75 |
|---|---|---|---|
| 4-level QWK | **0.631** | 0.619 | 0.612 |
| 5-level QWK | 0.827 | 0.832 | 0.831 |
| 6-level QWK | 0.287 | 0.284 | 0.292 |

**The correction does not fix QWK.** The best deployable point is +0.0008, the
best oracle point +0.0014, against a clause that needs +0.19. Accuracy does not
move either: 0.5478 at the baseline, 0.5440 at the deployable optimum, 0.5483 at
the oracle optimum.

The per-scale rows say more than the mean.

**The four-level aversion is real and correcting it costs QWK.** Every non-zero
alpha lowers the four-level QWK — from 0.631 to between 0.580 and 0.619 — under
*both* references, including the one that knows the true eval marginal. Moving
level 0 up takes correct level-1 predictions with it. The aversion is real at the
marginal, but the marginal is not what caps this scale: the head's within-item
discrimination is, and no shift of the levels changes it.

**STS-B does not reach 0.85 even with an oracle prior.** The six-level QWK goes
0.287 → 0.292 against a gold marginal the model would never be given in
production. If the marginal were the whole story, an oracle correction would
recover most of it. It recovers 0.006. So re-splitting STS-B — the open decision
in `research/17` — removes an unfair comparison but does not by itself produce a
passing QWK on that scale.

## Excluding STS-B does not rescue the clause either

That raises the other open decision from `research/17` directly: if STS-B is a
split artefact rather than a model defect, should it be in the aggregate at all?
The engine that produced the table above answers it with no new measurement.

Dropping the six-level scale entirely leaves the four- and five-level scales,
where the model is not fighting its own split:

| aggregate | n | QWK |
|---|---|---|
| all three readable scales | 1,831 | 0.6562 |
| without STS-B | 1,331 | **0.7947** |
| five-level scale alone | 1,113 | 0.8268 |
| four-level scale alone | 218 | 0.6309 |

Excluding STS-B moves the number by 0.139 and **still leaves it 0.055 short of
0.85**. Nor does any other subset of the scales: every weighted mean of 0.631,
0.827 and 0.287 sits between the smallest and largest of them, and the best
single scale — the five-level one — is 0.827. **No choice of scales reaches the
clause.** And excluding STS-B concentrates the shortfall on one scale: holding
the five-level scale at its measured 0.827 and dropping STS-B, the four-level
scale would have to reach **0.968** for the aggregate to hit 0.85, where it sits
at 0.631 — the same 0.631 it had before any correction. With STS-B kept, the
four-level scale would have to reach 2.26, outside the range of the metric.

So the STS-B question changes the reported number but not the verdict, and it
names the binding constraint: **the four-level scale is what stands between this
checkpoint and clause 2**, and the five-level scale is already close enough that
a realistic gain there would not matter on its own. The four-level scale is 218
items, `typed_decisions` (161) plus synthetic (57). Any training-side work aimed
at clause 2 should be aimed at that data, not at the calibration layer and not at
the aggregate.

## What this closes

Clause 2's QWK half is **not post-hoc fixable**. The three cheap post-hoc levers
are now all measured and all null: temperature leaves the argmax alone, a
monotone map leaves the argmax alone, and a class prior moves the argmax by an
amount that does not help. A repair has to come from training — the loss, the
level balancing, or more data on the low end of the four-level scales. The section
above rules out the one remaining non-training alternative.

The remaining two decisions from `research/17` are unchanged in substance.
Whether STS-B belongs in the QWK aggregate is now answered as a matter of
arithmetic rather than judgement: excluding it moves the number to 0.7947 and
still fails, so the decision affects reporting but not the verdict. What clause
2's Brier criterion was meant to be still needs its author.

## Caveats

- One checkpoint, one seed. There is no second-seed QWK for this checkpoint, so
  +0.0014 cannot be separated from checkpoint-to-checkpoint variation by this
  experiment alone; the direction of the conclusion does not depend on it,
  because a repair would have to clear +0.19.
- 1,832 score items, of which 500 are STS-B and 218 are four-level. The
  four-level result is the smallest sample in the table and carries the most
  weight in the argument against the aversion fix; its direction is consistent at
  every alpha and under both references.
- The `alpha` grid is coarse (seven points to 1.5). The curves are not peaky and
  the peak is not near a boundary, except the deployable one, which is still
  rising at 1.5 — so the deployable optimum is reported as a lower bound on the
  best a train-prior correction could do. It would have to rise by 0.19 from
  there.
- `p_hat` is estimated on the same items the correction is applied to, so the
  sweep is a batch correction, not a per-item one. On a different batch it would
  differ by sampling noise; on 1,832 items that noise is small, and it is the same
  magnitude for every alpha in the sweep.
