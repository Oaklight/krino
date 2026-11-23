from pathlib import Path
from data.pipeline import load_jsonl
from model.scripts.eval_gate import RawReranker, _raw_options, RAW_RERANKER_INSTRUCTION

scorer = RawReranker("Qwen/Qwen3-Reranker-4B", "cuda")
item = load_jsonl(Path("data/benchmarks/residual_train_5000.jsonl"))[0]
q = item.question
label, desc = _raw_options(q)[0]
query = f"{item.state}\n\nQuestion: {q.get('instructions','')}"
kind = "score"
document = f"Rubric level {label}: {desc}" if q["type"] == "score" else f"Answer option {label}: {desc}"
full = f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
pre = f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: "
a = scorer.tokenizer.encode(full, add_special_tokens=False)
b = scorer.tokenizer.encode(pre, add_special_tokens=False) + scorer.tokenizer.encode(document, add_special_tokens=False)
print("qtype", q["type"], "len", len(a), len(b), "equal", a == b)
if a != b:
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:
            print("first diff at", i, a[max(0,i-2):i+3], b[max(0,i-2):i+3]); break
