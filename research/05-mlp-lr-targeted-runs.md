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

### Epoch-by-epoch results (updating)

**Reranker-0.6B mlp_lr=1e-4:**
- Epoch 1: 44.9% (train_loss=1.23)
- Epoch 2: 42.2% (train_loss=1.71, spike)
- Epoch 3: 46.2% (train_loss=1.19, recovery)

**Reranker-0.6B mlp_lr=3e-4:**
- Epoch 1: 44.6% (train_loss=1.23)
- Epoch 2: 38.6% (train_loss=1.55, spike)
- Epoch 3: 40.8% (train_loss=1.23, recovering)

**Reranker-4B mlp_lr=1e-4:** (relaunched with correct LR)
- Epoch 1: 49.1% (train_loss=1.39)

**Reranker-4B mlp_lr=3e-5:** (relaunched with correct LR)
- Epoch 1: 49.1% (train_loss=1.39)

## Lessons Learned

1. **Always verify data integrity after cross-machine transfer** — use `rsync -c` or checksum validation
2. **Config files can have stale LR values** — R1 used CLI overrides that aren't reflected in YAML. Need to either update configs or always pass `--lr` explicitly
3. **max_length fix requires GPU-aware defaults** — PR #94 addresses this with auto-detection
4. **SSH nohup is unreliable for multi-process launches** — use `screen` instead
5. **Per-component LR code must define `trainable` before conditional branches** — the print statement needs it regardless of which branch is taken
