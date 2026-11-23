#!/bin/bash
set -u
cd "$HOME/projects/krino-v3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate jev
launch () {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup env PYTHONPATH="$HOME/projects/krino-v3" \
    python -m model.scripts.precompute_base_probs \
      --items data/benchmarks/residual_train_15000_extra.jsonl \
      --model Qwen/Qwen3-Reranker-4B \
      --output "data/benchmarks/base_probs_extra_shard${shard}.jsonl" \
      --shard "$shard" --shards 4 \
      > "logs/precompute_shard${shard}.log" 2>&1 < /dev/null &
  echo "shard=$shard gpu=$gpu pid=$!"
}
launch 0 0
launch 1 1
launch 2 2
launch 4 3
sleep 10
ps -eo pid,cmd | grep precompute_base_probs | grep -v grep | cut -c1-80
