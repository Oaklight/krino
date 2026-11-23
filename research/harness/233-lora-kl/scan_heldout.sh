#!/bin/bash
# Evaluate the nine 20K scan checkpoints on the frozen canonical held-out.
# GPUs 3-7 only; GPUs 0-2 carry the canonical seeds and must not be touched.
# Each checkpoint is scored in three shards; the pool spreads 27 jobs over 5
# cards.  The shard split does not affect the result.
set -u
ROOT=${KRINO_WORKTREE}
PY=${KRINO_PYTHON}
BASE=${KRINO_DATA}/precompute-243366/base_probs_canonical_243366_qwen35_4b.jsonl
PARTS=(${KRINO_DATA}/heldout_6400_part0.jsonl
       ${KRINO_DATA}/heldout_6400_part1.jsonl
       ${KRINO_DATA}/heldout_6400_part2.jsonl)
TAGS=(lr1e-3_kl0 lr1e-3_kl0.5 lr1e-3_kl1.0
      lr3e-4_kl0 lr3e-4_kl0.5 lr3e-4_kl1.0
      lr1e-4_kl0 lr1e-4_kl0.5 lr1e-4_kl1.0)
cd "$ROOT"
mkdir -p scan/heldout

# Build the job list, skipping shards already scored.
JOBS=()
for tag in "${TAGS[@]}"; do
  for p in 0 1 2; do
    out="scan/heldout/${tag}_part${p}.pred.jsonl"
    if [ -f "$out" ]; then
      continue
    fi
    JOBS+=("$tag $p")
  done
done
echo "jobs ${#JOBS[@]} $(date -Is)"
if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "nothing to do"
else
  # Deal round-robin into five lists, one per card.
  rm -f scan/heldout/list_*.txt
  for i in "${!JOBS[@]}"; do
    echo "${JOBS[$i]}" >> "scan/heldout/list_$((i % 5)).txt"
  done

  run_worker() {
    local gpu=$1 list=$2
    while read -r tag p; do
      out="scan/heldout/${tag}_part${p}.pred.jsonl"
      echo "gpu$gpu $tag part$p $(date -Is)"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" model/scripts/eval_listwise.py \
        --checkpoint "scan/$tag/listwise.pt" \
        --items "${PARTS[$p]}" --base-probs "$BASE" --device cuda \
        --output "$out" > "scan/heldout/${tag}_part${p}.log" 2>&1
      echo "gpu$gpu $tag part$p exit $? $(date -Is)"
    done < "$list"
  }

  GPUS=(3 4 5 6 7)
  pids=()
  for i in 0 1 2 3 4; do
    list="scan/heldout/list_$i.txt"
    if [ -f "$list" ]; then
      run_worker "${GPUS[$i]}" "$list" &
      pids+=($!)
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

# Concatenate the shards of each checkpoint, then report.
for tag in "${TAGS[@]}"; do
  if [ -f "scan/heldout/${tag}_part0.pred.jsonl" ] \
     && [ -f "scan/heldout/${tag}_part1.pred.jsonl" ] \
     && [ -f "scan/heldout/${tag}_part2.pred.jsonl" ]; then
    cat "scan/heldout/${tag}_part0.pred.jsonl" \
        "scan/heldout/${tag}_part1.pred.jsonl" \
        "scan/heldout/${tag}_part2.pred.jsonl" > "scan/heldout/${tag}.pred.jsonl"
  fi
done
"$PY" .scan_heldout_report.py | tee scan/heldout/report.txt
touch scan/heldout/DONE
