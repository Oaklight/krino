"""Compare realised adapter deltas across the scan arms."""

import os
import torch
from pathlib import Path

ROOT = Path(os.path.expandvars("${KRINO_WORKTREE}/scan"))
TAGS = [
    "lr1e-3_kl0", "lr3e-4_kl0", "lr1e-4_kl0",
    "lr1e-3_kl0.5", "lr3e-4_kl0.5", "lr1e-4_kl0.5",
    "lr1e-3_kl1.0", "lr3e-4_kl1.0", "lr1e-4_kl1.0",
]


def norm(tensors):
    return sum(float(v.pow(2).sum()) for v in tensors.values()) ** 0.5


for tag in TAGS:
    path = ROOT / tag / "listwise.pt"
    if not path.exists():
        print(f"{tag:16s} no checkpoint")
        continue
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["heads"]
    cfg = ck.get("config", {})
    bs = {k: v.float() for k, v in sd.items() if "lora_b" in k}
    nnz = sum(1 for v in bs.values() if float(v.abs().max()) > 1e-12)
    print(
        f"{tag:16s} lr={cfg.get('lr')} kl={cfg.get('lora_kl_weight')} "
        f"layers={len(bs)} nonzero={nnz} ||B||={norm(bs):.6f} "
        f"max|B|={max(float(v.abs().max()) for v in bs.values()):.6f}"
    )
