#!/bin/bash
# Launch the canonical confirmation arm: lr 1e-3, LoRA-KL 1.0, wide targets,
# three seeds, frozen held-out excluded from training.
set -u
ROOT=${KRINO_WORKTREE}
PY=${KRINO_PYTHON}
CORPUS=${KRINO_DATA}/krino-canonical-243366.jsonl
BASE=${KRINO_DATA}/precompute-243366/base_probs_canonical_243366_qwen35_4b.jsonl
HOLD=model/experiments/canonical/heldout_6400.json
TARGETS="q_proj,v_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj"
cd "$ROOT"

if [ ! -f "$BASE" ]; then
  echo "missing base-probs file: $BASE"
  exit 2
fi
if [ ! -f "$HOLD" ]; then
  echo "missing held-out list: $HOLD"
  exit 2
fi

mkdir -p fullarm
for k in 0 1 2; do
  if [ -e "fullarm/seed$k/listwise.pt" ]; then
    echo "seed$k already has a checkpoint; skipping"
    continue
  fi
  echo "launch seed$k on GPU $k $(date -Is)"
  CUDA_VISIBLE_DEVICES=$k nohup "$PY" model/scripts/train_listwise.py \
    --items "$CORPUS" --base-probs "$BASE" --holdout-ids "$HOLD" \
    --model Qwen/Qwen3.5-4B --output "fullarm/seed$k" \
    --epochs 1 --patience 1 \
    --lora-layers 8 --lora-rank 16 --lora-alpha 32 \
    --lora-targets "$TARGETS" \
    --lr 1e-3 --lora-kl-weight 1.0 \
    --torch-seed "$k" --seed "$k" --split-seed 42 --device cuda \
    > "fullarm/seed$k.log" 2>&1 < /dev/null &
done
echo "launched; check fullarm/seed*/train logs"
