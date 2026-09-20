# GPU Fleet Inventory

Probed 2026-09-19.

## Machine Inventory

### Tier 1 — rbdgx3 (Flagship)

| Spec | Value |
|------|-------|
| GPUs | 8× NVIDIA H200 NVL |
| VRAM | 141 GB each, **1.1 TB total** |
| CPU | 384 cores |
| RAM | 2.2 TB |
| Driver | 580.126.20 |
| Local disk | 876 GB root (163 GB free), 56 TB `/raid` (3.6 TB free) |
| NFS | `/rbstor` (12 TB free), `/rbscratch` (33 TB free) |
| SSH | `ts-rbdgx3` |
| **Occupancy** | GPUs 1-4 loaded (~126 GB each), GPU 7 active (80% util). GPUs 0, 5, 6 free |

### Tier 2 — rbdgx1, rbdgx2 (High-end + NFS Bridge)

| Spec | rbdgx1 | rbdgx2 |
|------|--------|--------|
| GPUs | 8× A100-SXM4-40GB | 8× A100-SXM4-40GB |
| VRAM | 40 GB each, **320 GB total** | 40 GB each, **320 GB total** |
| CPU | 256 cores | 256 cores |
| RAM | 1.0 TB | 1.0 TB |
| Driver | 570.211.01 | 555.42.06 |
| Local disk | 1.8 TB root (321 GB free), 14 TB `/raid` (9 TB free) | 1.8 TB root (174 GB free), 14 TB `/raid` (3.7 TB free) |
| NFS | `/rbstor`, `/rbscratch`, **`/nfs/lambda_stor_01`** | `/rbstor`, `/rbscratch`, **`/nfs/lambda_stor_01`** |
| SSH | `ts-rbdgx1` | `ts-rbdgx2` |
| **Occupancy** | All GPUs ~37 GB alloc, 0% util (stale?) | All GPUs 19-24 GB alloc, 0% util (stale?) |

### Tier 3 — lambda0, lambda1, lambda2, lambda4 (Workhorse)

| Spec | Value (all identical) |
|------|----------------------|
| GPUs | 8× V100-SXM2-32GB |
| VRAM | 32 GB each, **256 GB total** |
| CPU | 80 cores |
| RAM | 503 GB |
| Local disk | 1.8 TB root, 3.5 TB `/scratch` |
| NFS | `/nfs/lambda_stor_01`, `/nfs/ml_lab` |
| SSH | `ts-lambda{0,1,2,4}` (double-hop via `ts-lambda5`) |
| Home | `/nfs/lambda_stor_01/homes/pding` (shared across all lambdas + rbdgx1/2) |

| Machine | Occupancy |
|---------|-----------|
| lambda0 | Mostly idle (small allocs on a few GPUs) |
| **lambda1** | **Active** — all GPUs 9-14 GB, 57-69% util (someone else's job) |
| lambda2 | Idle |
| lambda4 | Idle (GPU 0 ~720 MB) |

### Tier 4 — lambda5 (Massively Parallel)

| Spec | Value |
|------|-------|
| GPUs | 20× Tesla T4 |
| VRAM | 15 GB each, **300 GB total** |
| CPU | 32 cores |
| RAM | 754 GB |
| Driver | 550.163.01 |
| Local disk | 12 TB (95% full!) |
| NFS | `/nfs/lambda_stor_01`, `/nfs/ml_lab` |
| SSH | `ts-lambda5` |
| Home | `/home/pding` (local, NOT shared) |
| **Occupancy** | Idle |
| **Note** | T4 has fp16 but NO bf16. Slower than V100 for training. Best for inference parallelism |

### Tier 5 — lambda11, lambda12 (Light Duty)

| Spec | Value (both identical) |
|------|----------------------|
| GPUs | 2× V100-SXM2-32GB |
| VRAM | 32 GB each, **64 GB total** |
| CPU | 48 cores |
| RAM | 125 GB |
| Driver | 550.163.01 |
| NFS | `/nfs/lambda_stor_01`, `/nfs/ml_lab` |
| SSH | `ts-lambda{11,12}` |
| **Occupancy** | Idle |

### CPU-Only — lambda10

| Spec | Value |
|------|-------|
| GPUs | None |
| CPU | 48 cores |
| RAM | 251 GB |
| NFS | `/nfs/lambda_stor_01`, `/nfs/ml_lab` |
| SSH | `ts-lambda10` |
| **Use** | Data pipeline, preprocessing, CPU-bound tasks |

## Unreachable / Inaccessible

| Machine | Status |
|---------|--------|
| rbdgx0 | DNS unresolvable — likely decommissioned |
| lambda3 | No route to host — powered off |
| lambda6, 7, 9 | Auth denied — no account |
| lambda8 | Connection closed immediately |
| rbh101 | Auth denied — account not provisioned |

## Availability Summary

| Tier | Machines | Idle GPUs | Model | VRAM/GPU | Total Idle VRAM |
|------|----------|-----------|-------|----------|-----------------|
| 1 | rbdgx3 | 3 of 8 | H200 NVL | 141 GB | ~420 GB |
| 2 | rbdgx1 + rbdgx2 | 16* | A100-40GB | 40 GB | 640 GB* |
| 3 | lambda0/2/4 | 24 | V100-32GB | 32 GB | 768 GB |
| 4 | lambda5 | 20 | T4 | 15 GB | 300 GB |
| 5 | lambda11/12 | 4 | V100-32GB | 32 GB | 128 GB |
| — | lambda10 | 0 | (CPU) | — | — |
| | **Total** | **67*** | | | **~2.3 TB*** |

*rbdgx1/2 show 0% GPU util but memory allocated — may be reclaimable.

## Filesystem Topology

### Three NFS Domains (No Cross-Visibility)

```
┌─────────────────────────────────────────────────────────┐
│                    NFS Domain Map                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ lambda_stor  │    │   rbstor /   │    │  ml_lab   │  │
│  │   400 TB     │    │  rbscratch   │    │  149 TB   │  │
│  │ (4.8 TB free)│    │ (12T + 33T)  │    │ (86T free)│  │
│  └──────┬───────┘    └──────┬───────┘    └─────┬─────┘  │
│         │                   │                  │        │
│    ┌────┴────┐         ┌────┴────┐        ┌────┴────┐   │
│    │lambda0-4│         │rbdgx1/2 │        │lambda0-4│   │
│    │lambda5  │◄───────►│(BRIDGE) │        │lambda5  │   │
│    │lambda10 │         │         │        │lambda10 │   │
│    │lambda11 │         │         │        │lambda11 │   │
│    │lambda12 │         │         │        │lambda12 │   │
│    └─────────┘         │         │        └─────────┘   │
│                        │         │                      │
│                        │    ┌────┴────┐                 │
│                        │    │ rbdgx3  │                 │
│                        └───►│         │                 │
│                             └─────────┘                 │
└─────────────────────────────────────────────────────────┘
```

| NFS Volume | Mount Path | Size | Free | Accessible From |
|-----------|-----------|------|------|-----------------|
| `lambda_stor_01` | `/nfs/lambda_stor_01` | 400 TB | **4.8 TB** (99% full!) | lambda0-5, 10-12, **rbdgx1/2** |
| `radbiostor` | `/rbstor` | 43 TB | 12 TB | **rbdgx1/2/3** only |
| `radbioscratch` | `/rbscratch` | 43 TB | 33 TB | **rbdgx1/2/3** only (no pding subdir, root mkdir denied) |
| `ml_lab` | `/nfs/ml_lab` | 149 TB | 86 TB (projects/ml_lab) | lambda0-5, 10-12 only |

### Key Directories

| Path | Host(s) | Writable | Notes |
|------|---------|----------|-------|
| `/nfs/lambda_stor_01/homes/pding` | All lambdas + rbdgx1/2 | Yes | Main lambda home, shared, has projects/conda |
| `/nfs/ml_lab/homes/pding` | All lambdas | Yes | 50 GB quota, nearly empty |
| `/rbstor/pding` | rbdgx1/2/3 | Yes | Has conda, projects, models |
| `/home/pding` on rbdgx3 | rbdgx3 only | Yes | Local, 163 GB free |
| `/scratch` on lambda0 | lambda0 only | Yes | 3.5 TB, 1.8 TB free |
| `/raid` on rbdgx1 | rbdgx1 only | Needs subdir | 14 TB, 9 TB free |

### Data Transfer Paths

**rbdgx1/2 are the bridge** — the only machines that mount BOTH `lambda_stor` and `rbstor`.

| From → To | Method | Speed | Example |
|-----------|--------|-------|---------|
| lambda ↔ lambda | `/nfs/lambda_stor_01` (shared) | NFS native | No copy needed |
| lambda → rbdgx1/2 | `/nfs/lambda_stor_01` (shared) | NFS native | No copy needed |
| lambda → rbdgx1/2 | Direct SCP | Network | `scp file rbdgx1.cels.anl.gov:/path` |
| rbdgx1/2 ↔ rbdgx3 | `/rbstor` (shared) | NFS native | No copy needed |
| **lambda → rbdgx3** | **Bridge via rbdgx1/2** | NFS→NFS | `ssh rbdgx1 'cp /nfs/lambda_stor_01/.../file /rbstor/pding/file'` |
| lambda → rbdgx3 | Direct SCP | Network | `scp file rbdgx3.cels.anl.gov:/path` |
| rbdgx3 → lambda | Direct SCP | Network | `ssh rbdgx3 'scp /path lambda0.cels.anl.gov:/path'` |

### Recommended Working Directories

| Fleet | Working Dir | Reason |
|-------|------------|--------|
| lambda0-4, 10-12 | `/nfs/lambda_stor_01/homes/pding/projects/` | Shared NFS, visible from all lambdas + rbdgx1/2 |
| lambda5 | `/home/pding/projects/` or `/nfs/lambda_stor_01/homes/pding/projects/` | Local home is separate; use NFS for shared access |
| rbdgx1/2 | `/rbstor/pding/projects/` | Shared with rbdgx3, also can see lambda_stor |
| rbdgx3 | `/rbstor/pding/projects/` | Shared with rbdgx1/2 |
| Model weights cache | `/rbstor/pding/models/` (rbdgx) or `/nfs/lambda_stor_01/homes/pding/.cache/` (lambda) | Avoid duplicate downloads |

## SSH Connectivity

All `ts-*` aliases configured in `~/.ssh/config.d/anl-lockout-bypass`.

### Hop Chains (from local machine)

| Alias | Chain |
|-------|-------|
| `ts-rbdgx{1,2,3}` | local → `ts-docker-build` → target FQDN |
| `ts-lambda{5,10,11,12}` | local → `ts-docker-build` → target FQDN |
| `ts-lambda{0,1,2,3,4}` | local → `ts-docker-build` → `lambda5` → target (short name) |

### Inter-Machine SSH (via `lambda_keys`)

Full mesh operational — any machine can SSH to any other using `lambda_keys`:

```
rbdgx3 ↔ rbdgx1 ↔ lambda0-4
rbdgx3 ↔ rbdgx2 ↔ lambda5
rbdgx3 ↔ lambda*   (direct)
lambda* ↔ rbdgx*   (direct)
```

Key: `lambda_keys` (RSA 3072-bit, fingerprint `SHA256:DFsFvuB7Bt2j6/e0g/4+1CH86xau3mMs5PwuL4vXanY`), deployed to `~/.ssh/` on all machines, authorized in `~/.ssh/authorized_keys` on all machines.

## Setup Needed Per Machine

For machines that don't yet have jev-explore:
```bash
# Clone repo
cd /nfs/lambda_stor_01/homes/pding/projects/  # or /rbstor/pding/projects/
git clone <repo-url> jev-explore

# Set up conda env
conda create -n jev-env python=3.11
conda activate jev-env
pip install -e ./jev-explore
```
