import torch
from pathlib import Path
from data.pipeline import load_jsonl
from model.scripts.eval_gate import RawReranker, _raw_options, RAW_RERANKER_INSTRUCTION

scorer = RawReranker("Qwen/Qwen3-Reranker-4B", "cuda")
item = load_jsonl(Path("data/benchmarks/residual_train_5000.jsonl"))[0]
q = item.question; label, desc = _raw_options(q)[0]
query = f"{item.state}\n\nQuestion: {q.get('instructions','')}"
document = f"Answer option {label}: {desc}"
text = f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
full = scorer.prefix_tokens + scorer.tokenizer.encode(text, add_special_tokens=False)[: scorer.max_length-len(scorer.prefix_tokens)-len(scorer.suffix_tokens)] + scorer.suffix_tokens
pre = scorer.tokenizer.encode(f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>:", add_special_tokens=False)
doc = scorer.tokenizer.encode(" " + document, add_special_tokens=False)
split = scorer.prefix_tokens + pre + doc + scorer.suffix_tokens
print("ids equal", full == split, len(full), len(split))
# fp32 comparison
scorer.model = scorer.model.float()
p1 = scorer._score_option(item.state, q.get("instructions",""), q["type"], label, desc)
p2 = scorer.score_many(item.state, q.get("instructions",""), q["type"], [(label, desc)])[0]
print("fp32 single", p1, "batched", p2, "diff", abs(p1-p2))
