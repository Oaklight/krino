# LoRA v4 throughput: corrected single-configuration numbers

Date: 2026-11-06. Scope: #244. This note supersedes the throughput and per-epoch figures
in the earlier #248 description. No code changes.

## Configuration

All training numbers below use one configuration, the ladder's:

| field | value |
|---|---|
| model | Qwen3.5-4B |
| LoRA | last 8 blocks, rank 16, alpha 32 |
| targets | `q_proj`, `v_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` |
| batch | 8 listwise items per step |
| device | one GPU |
| data | 20K listwise subset (`residual_train_20000`) |
| path | the #248 fixes (skip the discarded prefix pass, fixed right-padded shapes) |

The launch line is the #233 ladder script:

```bash
--lora-layers 8 --lora-rank 16 --lora-alpha 32 \
--lora-targets q_proj,v_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj
```

## Throughput

| | trainable items/s |
|---|---:|
| before (LoRA, #244 baseline) | 0.72 |
| after, six ladder arms (12.4 to 12.9) | **12.5 nominal** |
| recovery | **about 17x** |

Peak memory: 10.0 GiB.

The after rate is the steady-state rate of the six LoRA arms running the ladder
configuration. The before rate is the #244 measured LoRA-before throughput.

## Withdrawn numbers

- **16 items/s** was the per-200-item tail increment at the end of a 5,000-item run whose
  target list was the trainer's default `q_proj,v_proj`, not the ladder's seven-target
  list. It is not the ladder configuration and is withdrawn as the headline.
- **22x** was 16/0.72. With 16 withdrawn, 22x is withdrawn. The headline is
  12.5/0.72, about 17x.
- The 500/2,000/5,000-item cumulative figures (3.50, 7.55, 11.17 items/s) are from the
  same narrower-default run and are not used for the headline or the epoch estimate.

The like-for-like per-item component change is separate: the profile total moves
2111 to 994 ms per item (2.1x), on the same items and split. The recovery ratio above
compares the before path's end-to-end throughput with the after path's steady state; the
profile table is a component decomposition, not the same measurement.

## Per-epoch time

The 80/20 split is over all 243,366 items, one denominator:

- train: 194,693 items, of which 101,824 are trainable choice/score (52.3%);
- validation: 48,673 items, all types.

At 12.5 trainable items/s:

- training: 101,824 / 12.5 = 8,146 s = 2.26 h;
- validation at 2.65-2.66 model-scored rows/s (measured; see the canonical section):
  about 2.67 h;
- one device: **4.93 h**;
- eight GPUs, training and validation sharded: 0.28 + 0.33 = 0.61 h (**37 min**);
- eight GPUs, one gate GPU on full validation: 0.28 + 2.67 = 2.95 h (**3.0 h**).

At the low end of the arm range (12.4 items/s), training is 2.28 h and one device is
about 5.0 h; the eight-GPU range is 37 min to 3.0 h. The working-day target is met.

The 64-item listwise eval (67.3 s to 36.4 s) is 1.76 items/s. That is a cold-start mean
over 64 items and pays the bucket loads; it is not the steady validation rate used above.
Both are eval-only (no gradient, prefix cache retained); the difference is sample size,
not the path.

## Canonical confirmation arm (measured)

The confirmation arm runs on the canonical corpus with the frozen held-out
excluded, so its counts differ from the estimate above:

- corpus 243,366; held-out 6,424 (928 whole `(source, group)` clusters) excluded;
- 236,942 items remain; split seed 42, val 0.2;
- train 99,595 trained-type (42.0% of the non-held-out corpus, 40.9% of the full corpus);
- validation 47,389 items, all types;
- 1,479 train rows excluded as abstention rows (`answerable = False`, gold removed
  from the option set by design); 1,521 train score rows carry a fractional label
  that `_label_index` rounds to the nearest level.

The abstention exclusion is keyed on the flag, not on `_label_index < 0`, so a
genuine out-of-options label raises instead of disappearing.  It is an interim
measure: dropping and masking are equivalent only in this per-item loop, and the
answerable head (#254) should replace it.

Measured rate on this mix, at the end of each run: the three seeds finish the
99,595 trained-type rows at 12.71, 13.15 and 13.02 trainable items/s. The trainer
prints a cumulative `index/elapsed`, so a mid-run reading sits below the final
rate; an earlier version of this note quoted 10.9 to 11.2 as steady state, which
was an early cumulative reading and is corrected here.

State the two phases separately: training is 99,595 / 12.71-13.15 = 7,574-7,836 s
= 2.10-2.18 h. The full internal validation is 47,389 items; at the measured
2.65-2.66 model-scored rows/s and the 46.3% noul share that is about 25,445
scored rows, 9,602 s = 2.67 h, so the checkpoint lands about 4.8-4.9 h after the
15:53 start. The 6.1 items/s validation figure quoted here previously had no
measurement behind it and is replaced.

After #264, `TRAINED_TYPES` includes noul, and the same recipe trains 185,321
rows. The recorded all-type arms finish at 13.45-13.65 trainable items/s:
training is 3.77-3.83 h, and with the measured validation the single-device epoch
is about 6.4-6.5 h. Sharded across eight devices the training term is 0.47-0.48 h,
so the eight-GPU epoch is 0.80-0.81 h with sharded validation or 3.14-3.15 h with
one gate GPU on full validation. The working-day target still holds.

The internal validation is a diagnostic, not the standard C gate, and running it
in full on all three seeds costs about 6.6 GPU-hours for a number that cannot
promote or reject. The running arm keeps it for seed comparability; the next
canonical run should subsample it (10,000 rows is enough for early stopping) or
evaluate it once per configuration rather than once per seed. A validation pass
is not free and must not be quoted inside the training rate.

## What is not claimed

- No per-arm validation number is used here; the ladder's gate decides adoption.
- The throughput is measured on the 20K subset, not the full canonical corpus. The epoch
  estimate scales the measured per-item rate to the canonical count and assumes the
  bucket distribution is comparable. The 20K subset is 19% smaller than the canonical
  count, so the estimate carries that assumption explicitly.
