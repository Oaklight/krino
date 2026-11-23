#!/bin/bash
# Nine-arm learning-rate x LoRA-KL scan on the 20K subset.
# GPUs 1-6 for training (GPU 0 is in use), GPU 7 for gates.
set -u
ROOT=${KRINO_WORKTREE}
cd "$ROOT"
PY=${KRINO_PYTHON}
DATA=${KRINO_V3}/data/benchmarks
ITEMS="$DATA/residual_train_20000.jsonl"
BASE="$DATA/base_probs_20000_qwen35_4b.jsonl"
JEVBASE="$DATA/base_probs_jevbench_qwen35_4b.jsonl"
TARGETS="q_proj,v_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj"
mkdir -p scan

TAGS=(
  lr1e-3_kl0 lr1e-3_kl0.5 lr1e-3_kl1.0
  lr3e-4_kl0 lr3e-4_kl0.5 lr3e-4_kl1.0
  lr1e-4_kl0 lr1e-4_kl0.5 lr1e-4_kl1.0
)

run_arm() {
  local tag=$1 gpu=$2 lr=$3 kl=$4
  mkdir -p "scan/$tag"
  echo "start $tag gpu=$gpu lr=$lr kl=$kl $(date -Is)" > "scan/$tag/train.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" model/scripts/train_listwise.py \
    --items "$ITEMS" --base-probs "$BASE" --model Qwen/Qwen3.5-4B \
    --output "scan/$tag" --epochs 1 --patience 1 \
    --lora-layers 8 --lora-rank 16 --lora-alpha 32 --lora-targets "$TARGETS" \
    --lr "$lr" --lora-kl-weight "$kl" \
    --torch-seed 0 --seed 0 --split-seed 42 --device cuda \
    >> "scan/$tag/train.log" 2>&1
  echo "exit $? $(date -Is)" >> "scan/$tag/train.log"
}

gate_one() {
  local tag=$1
  echo "gate $tag $(date -Is)" > "scan/$tag/gate.log"
  CUDA_VISIBLE_DEVICES=7 "$PY" model/scripts/eval_listwise.py \
    --checkpoint "scan/$tag/listwise.pt" --jevbench \
    --base-probs "$JEVBASE" --device cuda \
    --output "scan/$tag/jevbench.pred.jsonl" \
    >> "scan/$tag/gate.log" 2>&1
  echo "gate exit $? $(date -Is)" >> "scan/$tag/gate.log"
}

watch_gates() {
  for tag in "${TAGS[@]}"; do
    local waited=0
    while [ ! -f "scan/$tag/listwise.pt" ]; do
      if grep -q "Traceback" "scan/$tag/train.log" 2>/dev/null; then
        echo "train failed" > "scan/$tag/GATE_SKIPPED"
        break
      fi
      sleep 20
      waited=$((waited + 20))
      if [ "$waited" -ge 5400 ]; then
        echo "timed out" > "scan/$tag/GATE_SKIPPED"
        break
      fi
    done
    if [ -f "scan/$tag/GATE_SKIPPED" ]; then
      continue
    fi
    gate_one "$tag"
  done
  touch scan/GATES_DONE
}

watch_gates &
GATE_PID=$!

# Wave 1: GPUs 1-6
run_arm lr1e-3_kl0   1 1e-3 0 & p1=$!
run_arm lr1e-3_kl0.5 2 1e-3 0.5 & p2=$!
run_arm lr1e-3_kl1.0 3 1e-3 1.0 & p3=$!
run_arm lr3e-4_kl0   4 3e-4 0 & p4=$!
run_arm lr3e-4_kl0.5 5 3e-4 0.5 & p5=$!
run_arm lr3e-4_kl1.0 6 3e-4 1.0 & p6=$!
wait $p1 $p2 $p3 $p4 $p5 $p6

# Wave 2: GPUs 1-3
run_arm lr1e-4_kl0   1 1e-4 0 & p1=$!
run_arm lr1e-4_kl0.5 2 1e-4 0.5 & p2=$!
run_arm lr1e-4_kl1.0 3 1e-4 1.0 & p3=$!
wait $p1 $p2 $p3

wait $GATE_PID
touch scan/SCAN_DONE
