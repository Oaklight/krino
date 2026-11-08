# JevBench base probabilities

`base_probs_qwen35_4b.jsonl` is the frozen Qwen3.5-4B option distribution for the
231 public JevBench items (`easy.jsonl` 48, `hard.jsonl` 111, `original.jsonl` 72),
in the format `model/scripts/precompute_base_probs.py` writes: one record per
item, `{"id", "q_type", "keys", "probs"}`, keys in the item's own criteria order.

- sha256: `fafa8e2662b967602b42779bb0dccc4c3692f62804465ae34452672de5bf69e9`
- model: `Qwen/Qwen3.5-4B`, the same checkpoint as
  `base_probs_canonical_243366_qwen35_4b.jsonl`, so the two files are the same
  distribution and not two different ones.
- command, run from a checkout of the commit that added this file:

```
python model/scripts/precompute_base_probs.py \
    --items <the 231 JevBench items as TypedQuestion JSONL, from data.pipeline.load_jevbench> \
    --model Qwen/Qwen3.5-4B \
    --output base_probs_qwen35_4b.jsonl \
    --device cuda --option-batch 16
```

## Coverage

Checked when the file was written: 231 of 231 ids present, no extra ids, no
duplicate ids, no null probabilities, every row's option count equal to its
item's option count, and every row summing to 1 within 1e-3. The item mix is 139
choice, 74 noul and 18 score.

## What this does and does not change

It is what a **future** arm in the base-conditioned family would need to be gated
on JevBench at all. It does not reopen anything: the base-conditioned arm is
killed under its own pre-registration (`results/gate/base_conditioned_preregistration.json`),
whose failure clause forbids re-running it to look for a positive. Any arm
pre-registered before this file existed cannot have been gated on JevBench, and
that is a fact about the record, not a defect in the arm.
