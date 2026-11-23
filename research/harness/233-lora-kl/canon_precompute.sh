#!/bin/bash
# Shard the canonical base-probability precompute, verify every shard, and
# concatenate.  Refuses to concatenate if any shard is short or failed.
#
# Usage: .canon_precompute.sh CORPUS OUTDIR [GPU ...]
# Example: .canon_precompute.sh ${KRINO_DATA}/krino-canonical-243366.jsonl \
#            ${KRINO_DATA}/precompute-243366 1 2 3 4 5 6 7
set -uo pipefail
ROOT=${KRINO_WORKTREE}
PY=${KRINO_PYTHON}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
CORPUS=${1:?usage: $0 CORPUS OUTDIR [GPU ...]}
OUTDIR=${2:?usage: $0 CORPUS OUTDIR [GPU ...]}
shift 2
GPUS=("$@")
if [ ${#GPUS[@]} -eq 0 ]; then GPUS=(1 2 3 4 5 6 7); fi
S=${#GPUS[@]}
mkdir -p "$OUTDIR"
cd "$ROOT"

N=$(wc -l < "$CORPUS")
CORPUS_SHA=$(sha256sum "$CORPUS" | awk '{print $1}')
echo "corpus  $CORPUS"
echo "lines   $N"
echo "sha256  $CORPUS_SHA"
echo "shards  $S on GPUs ${GPUS[*]}"
echo "model   $MODEL"

for i in $(seq 0 $((S - 1))); do
  out="$OUTDIR/shard_${i}of${S}.jsonl"
  if [ -e "$out" ]; then
    echo "REFUSING: $out already exists; use a fresh OUTDIR" >&2
    exit 2
  fi
done

pids=()
for i in $(seq 0 $((S - 1))); do
  g=${GPUS[$i]}
  out="$OUTDIR/shard_${i}of${S}.jsonl"
  log="$OUTDIR/shard_${i}of${S}.log"
  echo "launch shard $i/$S gpu $g -> $out $(date -Is)"
  CUDA_VISIBLE_DEVICES="$g" "$PY" model/scripts/precompute_base_probs.py \
    --items "$CORPUS" --model "$MODEL" --output "$out" \
    --shard "$i" --shards "$S" --device cuda --option-batch 16 \
    > "$log" 2>&1 &
  pids[$i]=$!
done

fail=0
for i in $(seq 0 $((S - 1))); do
  if wait "${pids[$i]}"; then
    echo "shard $i exited 0 $(date -Is)"
  else
    echo "SHARD $i EXITED NONZERO" >&2
    fail=1
  fi
done

manifest="$OUTDIR/manifest.jsonl"
: > "$manifest"
for i in $(seq 0 $((S - 1))); do
  out="$OUTDIR/shard_${i}of${S}.jsonl"
  actual=$(wc -l < "$out" 2>/dev/null || echo 0)
  expected=$(( (N - 1 - i) / S + 1 ))
  sha=$(sha256sum "$out" 2>/dev/null | awk '{print $1}')
  echo "shard $i: $actual records, expected $expected, sha $sha"
  if [ "$actual" -ne "$expected" ]; then
    echo "SHARD $i SHORT: $actual != $expected" >&2
    fail=1
  fi
  printf '{"shard":%d,"of":%d,"records":%d,"expected":%d,"sha256":"%s"}\n' \
    "$i" "$S" "$actual" "$expected" "$sha" >> "$manifest"
done

if [ "$fail" -ne 0 ]; then
  echo "REFUSING TO CONCATENATE: a shard failed or is short. See $manifest" >&2
  exit 1
fi

concat="$OUTDIR/base_probs_canonical_${N}_qwen35_4b.jsonl"
: > "$concat"
for i in $(seq 0 $((S - 1))); do
  cat "$OUTDIR/shard_${i}of${S}.jsonl" >> "$concat"
done
total=$(wc -l < "$concat")
total_sha=$(sha256sum "$concat" | awk '{print $1}')
unique=$("$PY" - "$concat" <<'PYEOF'
import json
import sys
ids = set()
with open(sys.argv[1]) as handle:
    for line in handle:
        ids.add(json.loads(line)["id"])
print(len(ids))
PYEOF
)
printf '{"total":%d,"unique_ids":%d,"sha256":"%s","corpus_lines":%d,"corpus_sha256":"%s","shards":%d,"model":"%s"}\n' \
  "$total" "$unique" "$total_sha" "$N" "$CORPUS_SHA" "$S" "$MODEL" >> "$manifest"

echo "concat  $concat"
echo "records $total"
echo "unique  $unique"
echo "sha256  $total_sha"
if [ "$total" -ne "$N" ] || [ "$unique" -ne "$N" ]; then
  echo "TOTAL MISMATCH: records $total, unique ids $unique, corpus lines $N" >&2
  exit 1
fi
echo "OK: $total records, $unique unique ids"
