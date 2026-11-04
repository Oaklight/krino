#!/usr/bin/env bash
# Launch the corrected Ettin-150M column: pooled control reruns plus the
# option-marker arms, one H100 each.
#
# The earlier pooled arms were trained before _is_encoder_model was fixed.
# ModernBERT configs leave is_decoder unset, so those runs used the causal
# last-token pooling path.  All four arms here share the fixed detection, so
# the marker-vs-pooled and K-aug-vs-not contrasts are clean.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-$HOME/miniforge3/envs/jev/bin/python}"
OUT="model/experiments/screen"
mkdir -p "$OUT"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
manifest="$OUT/manifest.jsonl"

arms=(
  "S1b_ettin150m_pooled_kaugoff screen_ettin150m.yaml 0 0"
  "S3b_ettin150m_pooled_kaugon  screen_ettin150m.yaml 1 1"
  "S2_ettin150m_marker_kaugoff  screen_ettin150m_marker.yaml 2 0"
  "S4_ettin150m_marker_kaugon   screen_ettin150m_marker.yaml 3 1"
)

for spec in "${arms[@]}"; do
  read -r arm cfg gpu kaug <<< "$spec"
  dir="$OUT/$arm"
  mkdir -p "$dir/checkpoints"
  extra=()
  if [ "$kaug" = "1" ]; then
    extra=(--k-augment-fraction 0.2 --k-augment-min-k 2 --k-augment-max-k 80)
  fi
  sha="$(sha256sum "model/configs/$cfg" | cut -d' ' -f1)"
  printf '{"arm":"%s","config":"model/configs/%s","config_sha256":"%s","k_augment":%s,"k_min":%s,"k_max":%s,"gpu":%s,"attempt":3,"encoder_detection":"class_and_architectures","started_at":"%s"}\n' \
    "$arm" "$cfg" "$sha" "$kaug" "$([ "$kaug" = 1 ] && echo 2 || echo null)" "$([ "$kaug" = 1 ] && echo 80 || echo null)" "$gpu" "$started_at" >> "$manifest"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$PY" model/scripts/train_multitask.py \
    --config "model/configs/$cfg" \
    --epochs 5 --seed 42 --max-length 2048 --device cuda:0 \
    --checkpoint-dir "$dir/checkpoints" \
    --output "$dir/results.json" \
    "${extra[@]}" > "$dir/train.log" 2>&1 < /dev/null &
  echo "$arm pid=$! gpu=$gpu kaug=$kaug"
done
