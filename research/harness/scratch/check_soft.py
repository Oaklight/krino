from pathlib import Path
from data.pipeline import load_jsonl
from model.scripts.train_residual import soft_target, _options

items = load_jsonl(Path("data/benchmarks/residual_train_20000_jev.jsonl"))[:400]
ok = sum(soft_target(it, len(_options(it.question))) is not None for it in items)
print("soft targets", ok, "of", len(items))
