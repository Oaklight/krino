#!/bin/bash
# Launch the remaining precompute shards as scan arms release their GPUs.
set -u
ROOT=${KRINO_WORKTREE}
PY=${KRINO_PYTHON}
CORPUS=${KRINO_DATA}/krino-canonical-243366.jsonl
OUT=${KRINO_DATA}/precompute-243366
cd "$ROOT"

wait_ckpt() {
  local tag=$1 waited=0
  while [ ! -f "scan/$tag/listwise.pt" ]; do
    sleep 10; waited=$((waited+10))
    if [ "$waited" -ge 3600 ]; then echo "timeout waiting for $tag"; break; fi
  done
  sleep 10
}

launch() {
  local shard=$1 gpu=$2
  local out="$OUT/shard_${shard}of7.jsonl"
  local log="$OUT/shard_${shard}of7.log"
  if [ -e "$out" ]; then echo "exists, skip: $out"; return; fi
  echo "launch shard $shard gpu $gpu $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" model/scripts/precompute_base_probs.py \
    --items "$CORPUS" --model Qwen/Qwen3.5-4B --output "$out" \
    --shard "$shard" --shards 7 --device cuda --option-batch 16 \
    > "$log" 2>&1 < /dev/null &
}

wait_ckpt lr3e-4_kl0;      launch 0 4
wait_ckpt lr3e-4_kl0.5;    launch 3 5
wait_ckpt lr3e-4_kl1.0;    launch 4 6
wait_ckpt lr1e-4_kl0;      launch 5 1
wait_ckpt lr1e-4_kl0.5;    launch 6 2
echo "remaining shards launched $(date -Is)"
