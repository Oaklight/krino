#!/usr/bin/env bash
# Launch the pooled column of the v3 screen matrix.
#
# Six arms: 3 backbones x K-augmentation off/on, one <accelerator> each. Frozen
# backbone, 5 epochs, 2K tokens, seed 42. Config hashes are written to
# model/experiments/screen/manifest.jsonl before any arm starts, so a result
# can always be tied to the config that produced it.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/miniforge3/envs/jev/bin/python}"
OUT="model/experiments/screen"
mkdir -p "$OUT"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
manifest="$OUT/manifest.jsonl"
: > "$manifest"

arms=(
  "S1_ettin150m_pooled_kaugoff screen_ettin150m.yaml 0 0"
  "S3_ettin150m_pooled_kaugon  screen_ettin150m.yaml 1 1"
  "S5_ettin400m_pooled_kaugoff screen_ettin400m.yaml 2 0"
  "S7_ettin400m_pooled_kaugon  screen_ettin400m.yaml 3 1"
  "S9_reranker06_pooled_kaugoff screen_reranker06.yaml 4 0"
  "S11_reranker06_pooled_kaugon screen_reranker06.yaml 5 1"
)

for spec in "${arms[@]}"; do
  read -r arm cfg gpu kaug <<< "$spec"
  dir="$OUT/$arm"
  mkdir -p "$dir/checkpoints"
  extra=()
  if [ "$kaug" = "1" ]; then
    extra=(--k-augment-fraction 0.2 --k-augment-min-k 2 --k-augment-max-k 256)
  fi
  sha="$(sha256sum "model/configs/$cfg" | cut -d' ' -f1)"
  printf '{"arm":"%s","config":"model/configs/%s","config_sha256":"%s","k_augment":%s,"gpu":%s,"started_at":"%s"}\n' \
    "$arm" "$cfg" "$sha" "$kaug" "$gpu" "$started_at" >> "$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$PY" model/scripts/train_multitask.py \
    --config "model/configs/$cfg" \
    --epochs 5 --seed 42 --max-length 2048 --device cuda:0 \
    --checkpoint-dir "$dir/checkpoints" \
    --output "$dir/results.json" \
    "${extra[@]}" > "$dir/train.log" 2>&1 < /dev/null &
  echo "$arm pid=$! gpu=$gpu kaug=$kaug"
done
