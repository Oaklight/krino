#!/bin/bash
# Gate the nine scan arms on GPUs 4-6 (freed training cards), not GPU 7.
# Each worker waits for its arm checkpoint and for its GPU to be free.
set -u
ROOT=${KRINO_WORKTREE}
PY=${KRINO_PYTHON}
JEVBASE=${KRINO_V3}/data/benchmarks/base_probs_jevbench_qwen35_4b.jsonl
cd "$ROOT"

gate_one() {
  local tag=$1 gpu=$2
  local waited=0
  while [ ! -f "scan/$tag/listwise.pt" ]; do
    if grep -q "Traceback" "scan/$tag/train.log" 2>/dev/null; then
      echo "train failed" > "scan/$tag/GATE_SKIPPED"; return
    fi
    sleep 10; waited=$((waited + 10))
    if [ "$waited" -ge 5400 ]; then
      echo "timed out waiting for checkpoint" > "scan/$tag/GATE_SKIPPED"; return
    fi
  done
  # Wait for this arm's trainer to exit.  Do not use a free-memory test:
  # a precompute shard may already be on the same card, which would make a
  # free-memory gate wait forever.
  while pgrep -f -- "--output scan/$tag" >/dev/null; do
    sleep 5
  done
  echo "gate $tag gpu=$gpu $(date -Is)" > "scan/$tag/gate.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" model/scripts/eval_listwise.py \
    --checkpoint "scan/$tag/listwise.pt" --jevbench \
    --base-probs "$JEVBASE" --device cuda \
    --output "scan/$tag/jevbench.pred.jsonl" \
    >> "scan/$tag/gate.log" 2>&1
  echo "gate exit $? $(date -Is)" >> "scan/$tag/gate.log"
}

worker() {
  local gpu=$1; shift
  for tag in "$@"; do gate_one "$tag" "$gpu"; done
}

worker 4 lr1e-3_kl0 lr3e-4_kl0 lr1e-4_kl0 &
worker 5 lr1e-3_kl0.5 lr3e-4_kl0.5 lr1e-4_kl0.5 &
worker 6 lr1e-3_kl1.0 lr3e-4_kl1.0 lr1e-4_kl1.0 &
wait
touch scan/GATES_DONE
echo "all gates finished $(date -Is)"
