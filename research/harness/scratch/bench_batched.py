import time
from pathlib import Path

import torch

from data.pipeline import load_jevbench
from model.scripts.eval_gate import ResidualScorer
from model.scripts.train_residual import head_logits

ckpt = Path("model/experiments/residual/D20K/residual.pt/residual.pt")
scorer = ResidualScorer("Qwen/Qwen3-Reranker-4B", "cuda", ckpt)
model = scorer.model
model.eval()
items = [i for i in load_jevbench() if i.question.get("type") in ("choice", "score")][:40]
print(f"items {len(items)} options {[len(i.question['criteria']) if i.question.get('type')=='choice' else len(i.question['levels']) for i in items][:10]}", flush=True)

out = {}
for bs in (1, 16, 32):
    model.option_batch = bs
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        logits = [head_logits(model, item).float().cpu() for item in items]
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    out[bs] = logits
    print(f"batch {bs:3d}: {len(items) / elapsed:6.2f} items/s  ({elapsed:6.1f}s)", flush=True)

for bs in (16, 32):
    diff = max((a - b).abs().max().item() for a, b in zip(out[1], out[bs]))
    print(f"max |logit diff| batch 1 vs {bs}: {diff:.6f}", flush=True)
