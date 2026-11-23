#!/bin/bash
# Gate the canonical arm under standard C, dual: JevBench hard constraint and
# frozen held-out signal.  Held-out evaluation is sharded across three GPUs.
set -u
ROOT=${KRINO_WORKTREE}
PY=${KRINO_PYTHON}
BASE=${KRINO_DATA}/precompute-243366/base_probs_canonical_243366_qwen35_4b.jsonl
JEVBASE=${KRINO_V3}/data/benchmarks/base_probs_jevbench_qwen35_4b.jsonl
CONTROL=${KRINO_V3}/results/gate/v4_control_shared_f32_qwen35_4b_instruct.pred.jsonl
HELD=${KRINO_DATA}/heldout_6400_items.jsonl
HOLDIDS=model/experiments/canonical/heldout_6400.json
cd "$ROOT"

gate_seed() {
  local k=$1
  local d="fullarm/seed$k"
  if [ ! -f "$d/listwise.pt" ]; then
    echo "seed$k: no checkpoint"
    return
  fi
  echo "seed$k: JevBench $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 "$PY" model/scripts/eval_listwise.py \
    --checkpoint "$d/listwise.pt" --jevbench --base-probs "$JEVBASE" \
    --device cuda --output "$d/jevbench.pred.jsonl" > "$d/jevbench.log" 2>&1

  echo "seed$k: held-out, three shards $(date -Is)"
  for p in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$p "$PY" model/scripts/eval_listwise.py \
      --checkpoint "$d/listwise.pt" \
      --items "${KRINO_DATA}/heldout_6400_part$p.jsonl" \
      --base-probs "$BASE" --device cuda \
      --output "$d/heldout_part$p.pred.jsonl" > "$d/heldout_part$p.log" 2>&1 &
  done
  wait
  cat "$d/heldout_part0.pred.jsonl" "$d/heldout_part1.pred.jsonl" "$d/heldout_part2.pred.jsonl" \
    > "$d/heldout.pred.jsonl"

  echo "seed$k: gate $(date -Is)"
  "$PY" .eval_c_gate.py \
    --seed-dir "$d" --heldout-items "$HELD" --base-probs "$BASE" \
    --heldout-ids "$HOLDIDS" --control-preds "$CONTROL" \
    --output "$d/gate_c.json"
}

for k in "$@"; do
  gate_seed "$k"
done

# The pre-registered rule combines the three seeds; no pooled test.  Run it
# only when all three gates exist, so a partial run cannot be mistaken for a
# combined verdict.
if [ -f fullarm/seed0/gate_c.json ] && [ -f fullarm/seed1/gate_c.json ] \
   && [ -f fullarm/seed2/gate_c.json ]; then
  echo "combine $(date -Is)"
  "$PY" .eval_c_combine.py \
    --gates fullarm/seed0/gate_c.json fullarm/seed1/gate_c.json fullarm/seed2/gate_c.json \
    --output fullarm/gate_c_combined.json
fi
