import time
from pathlib import Path

from data.pipeline import load_jsonl
from model.scripts.eval_gate import RawReranker, _raw_options

scorer = RawReranker("Qwen/Qwen3-Reranker-4B", "cuda")
items = [i for i in load_jsonl(Path("data/benchmarks/residual_train_5000.jsonl"))][:12]
diffs = []
t0 = time.monotonic()
for item in items:
    q = item.question
    opts = _raw_options(q)
    single = [scorer._score_option(item.state, q.get("instructions", ""), q["type"], l, d) for l, d in opts]
    batched = scorer.score_many(item.state, q.get("instructions", ""), q["type"], opts, group_size=16)
    diffs.append(max(abs(a - b) for a, b in zip(single, batched)))
    assert len(single) == len(batched)
print("n items", len(items), "max diff", max(diffs), "mean", sum(diffs) / len(diffs))
print("elapsed", round(time.monotonic() - t0, 1))
