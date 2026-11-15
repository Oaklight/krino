# The marker token does not matter at JevBench resolution

Status: measured, one seed, six trained heads plus a swap pass on the public
231-item set. All five non-default markers are unresolved against `<|im_end|>`,
so the default stays.

## Question

The option-marker readout scores each option at a marker token appended to the
option's ids. The shipped token is the `eos` token, 151645 `<|im_end|>`. Does
the choice of marker id matter? Evaluating one head under a different marker
measures head mismatch, so the primary experiment trains six heads from scratch
on the same corpus, identical except for `marker_token`.

## Setup

Backbone `Qwen/Qwen3-Reranker-4B`, tokenizer revision
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`, code commit `f7ec100`. Data:
`residual_train_20000.jsonl` (sha256 `f2fe77ec...9f32`) with
`base_probs_20000.jsonl` (`6b020e1d...5f`). Recipe, unchanged across arms:
`--option-marker --option-batch 16 --rank 32 --max-length 2048 --kl-weight 1.0
--ce-weight 0.2 --seed 42 --epochs 4 --patience 1 --val-fraction 0.2`.

| marker | token | config hash |
|---|---|---|
| 151645 | `<|im_end|>` (default) | `b48acf3068600fde...bee57ae` |
| 151647 | `<|object_ref_end|>` | `4cf5e422ec3c36c1...a8d972` |
| 151649 | `<|box_end|>` | `8cbe8d85a20f99a9...14e5e1` |
| 151651 | `<|quad_end|>` | `9c1040c5224cae7b...c14890` |
| 151662 | `<|fim_pad|>` | `536be7ef736c628c...1c02cd` |
| 151664 | `<|file_sep|>` | `1f657466e3e7ba2e...60e03b` |

The six printed hashes match the ones recorded in #362. Diffing the printed
configs leaves only `marker_token` and its hash different. The predictor mask
is identical across arms: the context is left-padded with 151643
(`<|endoftext|>`) under `attention_mask=0` and `position_ids=0`; the batched
suffix buffer's unused tail is zero-filled and fully masked, and the marker
hidden state is read at index `n-1` of each option block. `split_items` takes
no marker input: with items, seed 42 and fraction 0.2 it yields 16,001 train /
3,999 validation rows, and the validation digest over sorted
`(id, sha256(question))` pairs is `edca4993020997798948d425e67bc08083ab1619ad5ec92b856ddb5467580abf`.
All six candidates pass `check_marker_tokens.py` on this tokenizer.

One gap surfaced: `residual.pt` does not store the marker id. The evaluation
rebuilds each arm from its checkpoint and then overrides `marker_token_id` to
the arm's token; the driver is `eval_marker_arm.py`, sha256 `09493cde...d618`.
Future arms should store the marker id in the checkpoint config.

## Instrument

JevBench public 231 items (easy 48, hard 111, original 72), paired per item.
The pre-registered rule from #362: a non-default marker is adoptable only if it
exceeds 151645 by at least 9.3 pp raw accuracy with a paired 95% CI excluding
0; below that it is unresolved. The 9.3 pp is the instrument's detectable raw
difference at n=231 (`research/37-gate-policy.md`). Exact two-sided McNemar is
reported beside each comparison, with both discordant counts. The paired CI is
a 10,000-resample item-level bootstrap over the raw accuracy difference.

## Primary result

All six arms ran all four epochs; none early-stopped, and every arm selected
epoch 4 on the synthetic validation split.

| marker | raw | cc | hard | easy | original | epoch | +/− | Δ raw pp | McNemar p | state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 151645 | 0.6407 | 0.4574 | 0.4595 | 1.0000 | 0.6806 | 4 | — | — | — | baseline |
| 151647 | 0.6364 | 0.4446 | 0.4595 | 1.0000 | 0.6667 | 4 | 4/5 | −0.43 | 1.0 | unresolved |
| 151649 | 0.6364 | 0.4484 | 0.4414 | 1.0000 | 0.6944 | 4 | 4/5 | −0.43 | 1.0 | unresolved |
| 151651 | 0.6537 | 0.4745 | 0.4775 | 1.0000 | 0.6944 | 4 | 5/2 | +1.30 | 0.45 | unresolved |
| 151662 | 0.6450 | 0.4518 | 0.4595 | 1.0000 | 0.6944 | 4 | 7/6 | +0.43 | 1.0 | unresolved |
| 151664 | 0.6450 | 0.4540 | 0.4505 | 1.0000 | 0.7083 | 4 | 5/4 | +0.43 | 1.0 | unresolved |

`+/−` is items correct only under the arm / only under 151645. Raw bootstrap
95% CIs on the delta (pp): 151647 [−3.03, +2.16], 151649 [−3.03, +2.16],
151651 [−0.87, +3.46], 151662 [−2.60, +3.46], 151664 [−2.16, +3.03]. The
chance-corrected gate screen passes the coarse safety screen for all five;
151651 alone additionally shows "improvement suggested but unresolved" (cc
+1.71 pp, CI [−1.72, +5.14]). No marker comes within 7 pp of the floor.

## Swapping the marker under the default head

The head trained at 151645, re-evaluated at each other marker. The baseline
column is the identical head at 151645, so this is one run, not two.

| marker | raw | cc | +/− | Δ raw pp | McNemar p | state |
|---|---:|---:|---:|---:|---:|---|
| 151647 | 0.6494 | 0.4693 | 4/2 | +0.87 | 0.69 | unresolved |
| 151649 | 0.6580 | 0.4812 | 5/1 | +1.73 | 0.22 | unresolved |
| 151651 | 0.6494 | 0.4693 | 4/2 | +0.87 | 0.69 | unresolved |
| 151662 | 0.6407 | 0.4571 | 3/3 | +0.00 | 1.0 | unresolved |
| 151664 | 0.6450 | 0.4628 | 4/3 | +0.43 | 1.0 | unresolved |

The head does not collapse under the other markers: every swap is within
+1.73 pp of the baseline marker, and 151649 is nominally the best cell in the
whole experiment (152/231). That is the behaviour to expect if the head reads
the option boundary position rather than the token's identity.

## Verdict

By the pre-registered rule, all five non-default markers are unresolved, so
`<|im_end|>` 151645 stays. The largest raw difference is +1.30 pp (three
items), and the five-arm spread is 1.73 pp, both far inside the instrument's
9.3 pp resolution. The swap experiment points the same way: the 151645 head
scores within noise at five other markers, so the marker id is not doing
measurable work at this resolution.

This is a null at 9.3 pp, not a proof of identity. The swap CIs reach about
+3.9 pp, so a token effect smaller than roughly 4 pp is not excluded. The
correct sentence is "no measurable marker effect at JevBench resolution", not
"the marker is irrelevant".

## Caveats

- The runs used `f7ec100`. `main` has since adopted the group-aware
  `split_items` from #364: on the same data, seed and fraction it yields
  15,931 train / 4,069 validation rows with digest `43498054...`, so a rerun
  from `main` will not reproduce the recorded split (`edca4993...`, 16,001 /
  3,999) or necessarily the exact per-arm validation curves. The six config
  hashes are unchanged because the split is not part of the recipe hash. Every
  number in this note is the `f7ec100` run.
- One seed per arm. The training seed is fixed at 42 for every arm, so seed
  variance is not measured; the arm spread (1.7 pp) is the observable noise
  here.
- The easy tier saturates at 48/48 for every arm, so all discrimination is in
  hard (111) and original (72). The hard tier alone has a 14.9 pp detectable
  difference and is diagnostic only.
- Epoch selection uses the 3,999-row synthetic validation split, not JevBench;
  JevBench is never used for selection. All six arms chose epoch 4, so the
  comparison is epoch-matched.
- Evaluation recomputes the base distribution through the raw reranker instead
  of reading the precomputed cache; the established bf16 difference is
  `max |delta p| < 0.03` (`eval_gate.py`). This applies equally to every arm.
- The family of five comparisons was fixed in #362 before any outcome was
  seen. The p-values are unadjusted, as pre-registered; no result approaches a
  threshold where multiplicity could change the decision.

Per-arm result JSONs with config hash, data sha256, code commit, tokenizer
revision and the paired comparison are in `results/marker_ablation/`.
