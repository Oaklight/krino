# Experiment Log: MLP + Per-Component LR Targeted Runs

## Issue #93

## Failed Runs (rbdgx1/rbdgx2, 2026-09-26)

### Attempt 1: Data corruption
- **Machines**: rbdgx1 (A100 40GB), rbdgx2 (A100 40GB)
- **Failure**: `json.decoder.JSONDecodeError` — benchmark JSONL files corrupted during tar pipe transfer from rbdgx3
- **Root cause**: SSH pipe-based tar transfer truncated files mid-transfer
- **Fix**: Re-synced data via `rsync -avzc` with checksum verification

### Attempt 2: OOM after max_length fix
- **Machines**: rbdgx1, rbdgx2
- **Failure**: `torch.OutOfMemoryError`
  - rbdgx1 (Reranker-4B): "Tried to allocate 6.36 GiB" on A100 40GB
  - rbdgx2 (Reranker-0.6B): "Tried to allocate 33.01 GiB" at F.linear
- **Root cause**: max_length fix (commit 4e9621d) removed 512 cap, defaulted to backbone's `max_position_embeddings` (32K for Qwen3-Reranker). Attention over 32K tokens exceeds A100 40GB memory.
- **Fix**: PR #94 (GPU memory optimization with auto-detect). Temporary workaround: `--max-length 4096`

### Attempt 3: Wrong heads LR for Reranker-4B
- **Machine**: rbdgx3 (H200 144GB) — no OOM
- **Issue**: Config file `multitask_qwen_reranker4b.yaml` has `lr: 0.001` but R1 training used `--lr 3e-4` CLI override. Targeted runs inherited the config's 0.001 without override.
- **Result**: Reranker-4B train loss exploded to 5.07 (mlp_lr=1e-4) and 9.41 (mlp_lr=3e-5) by epoch 2
- **Fix**: Killed runs, relaunched with `--lr 3e-4`

### Attempt 4: UnboundLocalError in per-component LR
- **Machines**: rbdgx1, rbdgx2
- **Failure**: `UnboundLocalError: cannot access local variable 'trainable'` in `supervised.py:608`
- **Root cause**: Per-component LR code path split `trainable` into `head_params` + `projector_params`, but print statement still referenced `trainable`
- **Fix**: Commit 4b81ceb — moved `trainable = [...]` before the if/else block

## Successful Runs (rbdgx3, 2026-09-26)

### Current targeted runs (in progress)

| Run | Machine | GPU | Model | Config | Heads LR | MLP LR | Status |
|-----|---------|-----|-------|--------|----------|--------|--------|
| t1 | rbdgx3 | 0 | Reranker-4B | r128 mlp=1 | 3e-4 | 1e-4 | Running |
| t2 | rbdgx3 | 5 | Reranker-4B | r128 mlp=1 | 3e-4 | 3e-5 | Running |
| t3 | rbdgx3 | 7 | Reranker-0.6B | r64 mlp=1 | 1e-3 | 3e-4 | Running |
| t4 | rbdgx3 | 7 | Reranker-0.6B | r64 mlp=1 | 1e-3 | 1e-4 | Running |

### Baselines (from head-size sweep, uniform LR)

| Model | Config | Uniform LR | Acc@5ep |
|-------|--------|-----------|---------|
| Reranker-4B | r128 mlp=0 | 3e-4 | 52.1% |
| Reranker-4B | r128 mlp=1 | 3e-4 | 51.2% |
| Reranker-0.6B | r64 mlp=0 | 1e-3 | 50.3% |
| Reranker-0.6B | r64 mlp=1 | 1e-3 | 46.7% |

### Final Results

**Reranker-0.6B (r64 mlp=1, heads_lr=1e-3) — MLP does NOT help**

| mlp_lr | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | vs mlp=0 (50.3%) |
|--------|-----|-----|-----|-----|-----|-------------------|
| **1e-4** | 44.9% | 42.2% | 46.2% | 47.4% | **48.0%** | **-2.3pp** |
| 3e-4 | 44.6% | 38.6% | 40.8% | 42.8% | 42.8% | **-7.5pp** |

**Reranker-4B (r128 mlp=1, heads_lr=3e-4) — MLP DOES help**

| mlp_lr | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | vs mlp=0 (52.1%) |
|--------|-----|-----|-----|-----|-----|-------------------|
| 1e-4 | 50.1% | 47.2% | 52.8% | 53.3% | **54.8%** | **+2.7pp** |
| **3e-5** | 50.1% | 48.1% | 52.8% | 54.2% | **55.4%** | **+3.3pp** ✅ |

### Key Finding

**MLP projector benefit is backbone-size-dependent:**
- 4B backbone (hidden=2560): MLP adds valuable nonlinear transformation, +3.3pp with mlp_lr=3e-5
- 0.6B backbone (hidden=1024): MLP adds noise, -2.3pp even with optimal mlp_lr=1e-4
- Optimal mlp_lr scales inversely with backbone size and hidden dim

### R2 Configuration Decision

| Model | Rank | MLP | Heads LR | MLP LR |
|-------|------|-----|----------|--------|
| **Reranker-4B** | r128 | **mlp=1** | 3e-4 | **3e-5** |
| **Reranker-0.6B** | r64 | mlp=0 | 1e-3 | — |

MLP will also be revisited during LoRA stage (Stage 5 of #73) for the 0.6B model,
where backbone adaptation may make the projector useful.

## Lessons Learned

1. **Always verify data integrity after cross-machine transfer** — use `rsync -c` or checksum validation
2. **Config files can have stale LR values** — R1 used CLI overrides that aren't reflected in YAML. Need to either update configs or always pass `--lr` explicitly
3. **max_length fix requires GPU-aware defaults** — PR #94 addresses this with auto-detection
4. **SSH nohup is unreliable for multi-process launches** — use `screen` instead
5. **Per-component LR code must define `trainable` before conditional branches** — the print statement needs it regardless of which branch is taken
