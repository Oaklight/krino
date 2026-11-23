from pathlib import Path
from data.pipeline import load_jsonl
from model.scripts.eval_gate import RawReranker, _raw_options

scorer = RawReranker("Qwen/Qwen3-Reranker-4B", "cuda")
items = load_jsonl(Path("data/benchmarks/residual_train_5000.jsonl"))[:6]
for gs in (1, 4, 16):
    diffs = []
    for item in items:
        q = item.question
        opts = _raw_options(q)
        single = [scorer._score_option(item.state, q.get("instructions",""), q["type"], l, d) for l, d in opts]
        batch = scorer.score_many(item.state, q.get("instructions",""), q["type"], opts, group_size=gs)
        diffs.append(max(abs(a-b) for a,b in zip(single,batch)))
    print("group", gs, "max", max(diffs), "mean", sum(diffs)/len(diffs))
