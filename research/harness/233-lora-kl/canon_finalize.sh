#!/bin/bash
# Verify the seven completed precompute shards, then concatenate.
# Refuses to concatenate on any short shard, count mismatch, or duplicate id.
set -u
OUT=${KRINO_DATA}/precompute-243366
PY=${KRINO_PYTHON}
N=243366
S=7
CORPUS_SHA=07600293087d207e1f60537f721a83d157a4f8091d300948aa3e1709bc6f1e53
CONCAT=$OUT/base_probs_canonical_${N}_qwen35_4b.jsonl
MANIFEST=$OUT/manifest.jsonl
FAIL=0

: > "$MANIFEST"
for i in $(seq 0 $((S - 1))); do
  out="$OUT/shard_${i}of${S}.jsonl"
  expected=$(( (N - 1 - i) / S + 1 ))
  if [ ! -f "$out" ]; then
    echo "MISSING shard $i: $out"
    FAIL=1
    actual=0
    sha="missing"
  else
    actual=$(wc -l < "$out")
    sha=$(sha256sum "$out" | awk '{print $1}')
  fi
  echo "shard $i: $actual records, expected $expected, sha $sha"
  if [ "$actual" -ne "$expected" ]; then
    echo "SHARD $i SHORT: $actual != $expected"
    FAIL=1
  fi
  printf '{"shard":%d,"of":%d,"records":%d,"expected":%d,"sha256":"%s"}\n' \
    "$i" "$S" "$actual" "$expected" "$sha" >> "$MANIFEST"
done

if [ "$FAIL" -ne 0 ]; then
  echo "REFUSING TO CONCATENATE: a shard is missing or short"
  exit 1
fi

if [ -e "$CONCAT" ]; then
  echo "REFUSING: $CONCAT already exists"
  exit 2
fi

: > "$CONCAT"
for i in $(seq 0 $((S - 1))); do
  cat "$OUT/shard_${i}of${S}.jsonl" >> "$CONCAT"
done

total=$(wc -l < "$CONCAT")
total_sha=$(sha256sum "$CONCAT" | awk '{print $1}')
unique=$("$PY" - "$CONCAT" <<'PYEOF'
import json
import sys
ids = set()
with open(sys.argv[1]) as handle:
    for line in handle:
        ids.add(json.loads(line)["id"])
print(len(ids))
PYEOF
)
printf '{"total":%d,"unique_ids":%d,"sha256":"%s","corpus_sha256":"%s","shards":%d,"model":"Qwen/Qwen3.5-4B"}\n' \
  "$total" "$unique" "$total_sha" "$CORPUS_SHA" "$S" >> "$MANIFEST"

echo "concat  $CONCAT"
echo "records $total"
echo "unique  $unique"
echo "sha256  $total_sha"
if [ "$total" -ne "$N" ] || [ "$unique" -ne "$N" ]; then
  echo "TOTAL MISMATCH: records $total, unique ids $unique, expected $N"
  exit 1
fi
echo "OK: $total records, $unique unique ids, zero id collisions"
