# Shared-prefix parity on JevBench public (231 items, sha dcb90f87)

Runs on rbh101 H100, env `jev-fla` (fla GDN kernels), `--no-order-sensitivity`.

| file | path |
|---|---|
| `s2_*.json` | `eval_gate.py --raw-reranker --shared-prefix` (one prefix pass, 8-row suffix batch, float32 two-token readout) |
| `o32_*.json` | `po32.py`: per-option full forward with the same float32 readout (control) |
| `parity.txt` | `cmp2.py` output: flips between per-option bf16 (`results/gate/qwen35_raw/paired`), per-option f32, and shared f32 |

Reading `parity.txt`: `po_bf16 -> po_f32` changes only the readout precision, so its
flip count is the bf16 noise floor. `po_f32 -> shared_f32` changes only the cache
path. The cache path flips 1-4 items, at or below the floor (2-10), and no
McNemar test is below p=0.5.

Prediction JSONL files stay on rbh101 under `~/tmp/qwen35-raw/{paired,shared2}`.
