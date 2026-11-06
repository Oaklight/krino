# Adapter folding on the v4 shared-prefix path

Status: measured. The adapter is synthetic until #233 ships a trained one; the
folding code path is the same for a trained adapter.

## Answer

Every module the adapter targets can be folded into the base weights.

- `inject_lora` wraps only `nn.Linear`. Every projection in the Qwen3.5-4B text
  tower that the targets name is `nn.Linear`, including the four GDN input
  projections and the GDN output projection.
- `merge_lora` removes every wrapper: 16/16 for q/v, 136/136 for q/v plus all
  GDN projections. The merge is 3 ms and 20 ms respectively, once at load.
- Merged latency equals the frozen path: 89.1 ms mean against 89.5 ms (q/v) and
  88.8 ms against 89.0 ms (all GDN projections). The wrapper costs +3.2% and
  +27.3%. The 100 ms gate holds after merging.
- Folding is not bit-identical. Merged against wrapper changes 1/231 (q/v) and
  4/231 (all GDN) JevBench predictions. Every changed item is a top-2 tie: the
  largest log-odds margin among them is 0.0175.

No module blocks folding. The only `Conv1d` in the stack is
`linear_attn.conv1d`, which `inject_lora` never wraps.

## Setup

- Backbone: `Qwen/Qwen3.5-4B`, raw reranker, shared-prefix path, JevBench 231
  items. The frozen condition reproduces the 89 ms baseline from #218 on the
  same accelerator class.
- Adapter: rank 16, alpha 32, seeded. `lora_b` is filled with a seeded
  Gaussian of standard deviation 0.01. The relative Frobenius norm of the
  update against the base weight is 6.5-6.7% mean and 9.7% maximum.
- Target sets: `q_proj,v_proj` (16 modules) and
  `q_proj,v_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj` (136
  modules).
- Conditions: frozen, adapter wrapper, adapter merged. Three repeats over the
  same 231 items, per-item timing.
- The seed is part of the method. `inject_lora` draws `lora_a` from the global
  RNG; without seeding, two runs get different adapters and the parity flips
  measure the draw, not folding. `bench_adapter_fold.py` seeds before
  injection, and a repeated q/v run gives identical flips and differences.
  Every fold benchmark run made before this fix measured a different adapter,
  so no earlier fold number is comparable to these.

## Module survey

The text tower has 248 `nn.Linear` layers:

| module | count | parent |
| --- | ---: | --- |
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | 8 each | `Qwen3_5Attention` |
| `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` | 24 each | `Qwen3_5GatedDeltaNet` |
| `gate_proj`, `up_proj`, `down_proj` | 32 each | `Qwen3_5MLP` |

Each GDN layer also has one `linear_attn.conv1d` (`nn.Conv1d`, 24 in total).
`inject_lora` matches only `nn.Linear`, so the convolution is never adapted and
never needs folding. `inject_lora` now raises when any named target matches
zero layers, including after `last_n_layers` filtering. A future `conv1d`
target therefore fails loudly instead of training a smaller adapter silently;
close that gap before targeting it.

## Folding

| targets | modules | merged | remaining | merge time |
| --- | ---: | ---: | ---: | ---: |
| q/v | 16 | 16 | 0 | 3 ms |
| q/v + GDN + out_proj | 136 | 136 | 0 | 20 ms |

Every adapted module is a `nn.Linear` with no bias interaction: the wrapper
computes `base(x) + scaling * B A x`, so the folded weight is
`W + scaling * B A` and the bias is untouched.

## Latency

Per item, milliseconds, mean / median over 231 items and three repeats.

| targets | frozen | wrapper | merged | wrapper | merged vs frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| q/v | 89.5 / 78.3 | 92.3 / 80.9 | 89.1 / 78.0 | +3.2% | -0.4% |
| q/v + GDN + out_proj | 89.0 / 77.9 | 113.3 / 100.7 | 88.8 / 77.7 | +27.3% | -0.2% |

The wrapper overhead grows with the number of adapted modules in the prefix and
recurrent path: 16 modules cost 2.8 ms, 136 cost 24.3 ms. Folding removes it.
The merged and frozen paths are within run-to-run noise, so the merged adapter
keeps the 89 ms path and the 100 ms gate.

Fallback, measured only to price a module that could not fold: the per-option
path that re-encodes the full prompt for every option costs 1.25-1.41 s median
per item, 16-18x the shared path. No module requires it today.

## Numerics

Adapter wrapper against merged, same items and options:

| targets | label flips | max abs logit-gap diff | max abs probability diff | max top-2 margin among flips |
| --- | ---: | ---: | ---: | ---: |
| q/v | 1/231 (0.43%) | 0.234 | 0.058 | 0.0004 |
| q/v + GDN + out_proj | 4/231 (1.73%) | 0.313 | 0.056 | 0.0175 |

The logit gap is `log(p / (1 - p))` for each option's yes probability. The
top-2 margin is the smaller of the two top-1/top-2 margins on the two sides, so
a tiny value means the flip is a tie.

Frozen against wrapper changes 9/231 (q/v) and 17/231 (all GDN) predictions,
with margins up to 0.22 and 0.61. The adapter is doing real work; the
merged-vs-wrapper flips are a rounding effect on ties, not a lost update.

The cause is precision. The merged path rounds `W + scaling * B A` once to
bf16; the wrapper rounds the low-rank term separately before adding it. The
error scales with the adapter's relative norm and with the number of adapted
modules. If bit-exact parity with the wrapper is required, do not fold: keep
the wrapper and pay +3.2% (q/v) or +27.3% (all GDN).

## Recommendation

1. Merge the adapter at load. It is a one-time 3-20 ms cost, restores the 89 ms
   path, and keeps any prefix cache valid because the adapter becomes part of
   the weights that produced it.
2. Re-run this check on the trained adapter after #233. The flip count scales
   with the adapter norm; the synthetic 6.5% norm is a stand-in, not a
   prediction of the trained one. This is a standing gate requirement recorded
   on #230, not an optional follow-up: bf16 merge rounding changes with the
   weights.
3. Treat a future `conv1d` target separately: the current wrapper cannot adapt
   it, and the per-option fallback is 1.25 s per item. The new loud failure in
   `inject_lora` makes the attempt visible instead of silently adapting less.
4. Re-run the latency gate on the merged path after the ladder changes the
   model, so the 100 ms claim is measured on the model that ships, and record
   the result on #230.

## Reproduce

```bash
# q/v adapter
python model/scripts/bench_adapter_fold.py \
    --model Qwen/Qwen3.5-4B --targets q_proj,v_proj \
    --json results/gate/adapter_fold_qv.json

# every GDN projection too
python model/scripts/bench_adapter_fold.py \
    --model Qwen/Qwen3.5-4B \
    --targets q_proj,v_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj \
    --json results/gate/adapter_fold_all.json
```

The first command also writes the parity and folding fields. `--repeats`,
`--warmup`, `--adapter-scale`, and `--adapter-seed` are exposed for reruns.

## Acceptance mapping

- Latency for the adapter path, measured: section "Latency", mean and median.
- Numerical-parity report with flipped predictions: section "Numerics", counts
  and fractions, not only the maximum.
- Modules that cannot be folded: none among the modules `inject_lora` can
  adapt. The non-adapted `conv1d` is named.
