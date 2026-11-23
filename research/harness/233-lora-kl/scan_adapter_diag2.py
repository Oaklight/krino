"""Compare the true LoRA update norm (B @ A) across the scan arms."""

import os
import torch
from pathlib import Path

ROOT = Path(os.path.expandvars("${KRINO_WORKTREE}/scan"))
TAGS = [
    "lr1e-3_kl0", "lr3e-4_kl0", "lr1e-4_kl0",
    "lr1e-3_kl0.5", "lr3e-4_kl0.5", "lr1e-4_kl0.5",
    "lr1e-3_kl1.0", "lr3e-4_kl1.0", "lr1e-4_kl1.0",
]


def factors(state):
    a = {
        key[: -len(".lora_a.weight")]: value.float()
        for key, value in state.items()
        if key.endswith(".lora_a.weight")
    }
    b = {
        key[: -len(".lora_b.weight")]: value.float()
        for key, value in state.items()
        if key.endswith(".lora_b.weight")
    }
    return a, b


reference = None
for tag in TAGS:
    path = ROOT / tag / "listwise.pt"
    if not path.exists():
        print(f"{tag:16s} no checkpoint")
        continue
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["heads"]
    config = checkpoint.get("config", {})
    a, b = factors(state)
    deltas = {prefix: b[prefix] @ a[prefix] for prefix in a}
    total = sum(float(delta.pow(2).sum()) for delta in deltas.values()) ** 0.5
    per_layer = {p: float(d.pow(2).sum()) ** 0.5 for p, d in deltas.items()}
    largest = max(per_layer.items(), key=lambda item: item[1])
    same_a = None
    if reference is not None:
        same_a = all(torch.equal(reference[p], a[p]) for p in reference if p in a)
    else:
        reference = a
    print(
        f"{tag:16s} lr={config.get('lr')} kl={config.get('lora_kl_weight')} "
        f"||BA||={total:.6f} max_layer={largest[1]:.6f} "
        f"A_identical={same_a}"
    )
