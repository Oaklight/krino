import json
from math import comb
D = "/home/pding/tmp/qwen35-raw"
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
models = {"qwen3_reranker_4b": "paired/p_qwen3_reranker_4b",
          "qwen35_4b_instruct": "paired/p_qwen35_4b_instruct_fla",
          "qwen35_4b_base": "paired/p_qwen35_4b_base"}
for m, pref in models.items():
    runs = {"po_bf16": load(f"{D}/{pref}.pred.jsonl"),
            "po_f32": load(f"{D}/shared2/o32_{m}.pred.jsonl"),
            "shared_f32": load(f"{D}/shared2/s2_{m}.pred.jsonl")}
    acc = {k: sum(v[i]["correct"] for i in v) / len(v) for k, v in runs.items()}
    ties = {k: sum(abs(v[i]["max_prob"] - 0.5) < 1e-9 for i in v) for k, v in runs.items()}
    print(m, {k: round(a, 4) for k, a in acc.items()}, "exact-0.5", ties)
    for a, b in (("po_bf16", "po_f32"), ("po_f32", "shared_f32"), ("po_bf16", "shared_f32")):
        A, B = runs[a], runs[b]
        flips = sum(A[i]["pred"] != B[i]["pred"] for i in A)
        g = sum(B[i]["correct"] and not A[i]["correct"] for i in A)
        l = sum(A[i]["correct"] and not B[i]["correct"] for i in A)
        dp = sum(abs(A[i]["max_prob"] - B[i]["max_prob"]) for i in A) / len(A)
        print(f"  {a:>8} -> {b:<10} flips {flips:2d} (+{g}/-{l}) mcnemar p={mcnemar(g, l):.3f} mean|dp| {dp:.4f}")
