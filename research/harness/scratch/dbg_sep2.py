from pathlib import Path
from data.pipeline import load_jsonl
from model.scripts.eval_gate import RawReranker, _raw_options, RAW_RERANKER_INSTRUCTION

tok = RawReranker("Qwen/Qwen3-Reranker-4B", "cuda").tokenizer
items = load_jsonl(Path("data/benchmarks/residual_train_5000.jsonl"))[:40]
variants = {
    "E": ("<Document>:", " "),
    "F": ("<Document>:", ""),
}
for name, (tail, lead) in variants.items():
    bad = 0
    for item in items:
        q = item.question
        query = f"{item.state}\n\nQuestion: {q.get('instructions','')}"
        for label, desc in _raw_options(q)[:4]:
            document = f"{lead}Answer option {label}: {desc}"
            pre = f"<Instruct>: {RAW_RERANKER_INSTRUCTION}\n<Query>: {query}\n{tail}"
            a = tok.encode(pre + document, add_special_tokens=False)
            b = tok.encode(pre, add_special_tokens=False) + tok.encode(document, add_special_tokens=False)
            if a != b:
                bad += 1
    print(name, "mismatches", bad, "/", 40 * 4)
