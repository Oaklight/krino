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
items = [i for i in load_jevbench() if i.question.get("type") in ("choice", "score")][:60]
warm = items[:8]
print(f"items {len(items)}", flush=True)


def run(batch: int, subset):
    model.option_batch = batch
    torch.cuda.synchronize()
    start = time.monotonic()
    with torch.no_grad():
        out = [head_logits(model, item).float().cpu() for item in subset]
    torch.cuda.synchronize()
    return time.monotonic() - start, out


for batch in (1, 16, 32):
    run(batch, warm)  # warm the kernels and the clock

results = {}
for batch in (1, 16, 32):
    best = min(run(batch, items)[0] for _ in range(2))
    results[batch] = best
    print(f"batch {batch:3d}: {len(items) / best:6.2f} items/s  best {best:6.1f}s", flush=True)

_, a = run(1, items)
_, b = run(32, items)
print(f"max |logit diff|: {max((x - y).abs().max().item() for x, y in zip(a, b)):.6f}")
print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
