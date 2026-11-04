# The 150M encoder's failure was mostly the pooled readout

Status: measured on JevBench public, 231 items, one seed. The arm is still
killed by the encoder Intelligence line; this note records what the screen
taught us about readouts.

## Question

S1/S3/S5/S9/S11 all landed at 33-36% raw on JevBench with a pooled readout,
and every one of them collapsed as K grew (K=4 0.5 -> K=256 0.0). Two
explanations fit: the backbones are too small, or the pooled readout cannot
represent a candidate set. S2 separates them: same Ettin-150M backbone, same
182K mixed corpus, same 5 epochs, same seed, same K-aug off, marker readout
instead of pooled, `option_chunk_size=1`.

## Result

`results/gate/{S1b_ettin150m_pooled_kaugoff,S2_ettin150m_marker_kaugoff}_jevbench.json`

| Arm | Raw acc | cc | ECE | easy | hard | original | choice | noul | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1b pooled | 0.3117 | -0.026 | 0.165 | 0.333 | 0.333 | 0.264 | 0.273 | 0.419 | 0.167 |
| S2 marker | 0.4848 | 0.195 | 0.119 | 0.854 | 0.360 | 0.431 | 0.532 | 0.419 | 0.389 |

K-sweep (distractors drawn from other items, 50 items per K):

| Arm | K=4 | K=16 | K=64 | K=256 |
|---|---:|---:|---:|---:|
| S1b pooled | 0.50 | 0.10 | 0.00 | 0.00 |
| S2 marker | 0.92 | 0.88 | 0.86 | 0.66 |

The readout is worth +17.3 raw points and +22.1 chance-corrected points on the
same weights, and it removes the K collapse: 56 pp of drop becomes 26 pp. The
gain is concentrated where pooled was worst (easy 0.333 -> 0.854, choice
0.273 -> 0.532), which is what a readout bottleneck looks like. Noul is
unchanged (0.419 both) because noul does not use the option readout.

Real JevBench items with K=6-20 still score 0.125 under marker versus 0.188
under pooled. Those are the long-context hard items, not a K effect: the
synthetic sweep at K=256 is 0.66. Long context, not option count, is the
remaining 150M limit.

## Verdict

S2 is killed on the same line as the rest of the encoder lane: cc 0.195 is
below the Intelligence 40 kill criterion, and the K=256 gap rule does not
rescue it because the goal's kill line is independent of the screen's
sibling rule.

The readout result outlives the arm. Any future arm that scores a candidate
set should use the marker readout; the pooled readout was the dominant error
source at 150M and there is no evidence it is not one at 4B.

## Caveats

- One seed per cell. The pooled cells agree with each other within 4 points,
  and the marker gain is 17, so the ordering is not seed noise.
- The K-sweep distractors come from other items and are easy; it measures
  degradation with K, not absolute difficulty.
- The first S2 gate run silently dropped 8 `hard-sol-a-multi_hop-*` items:
  their `state` is a dict, and the marker path's truncation fallback passed it
  to the tokenizer unstringified. Fixed in `_tokenize_context`
  (`str(state)`), covered by `tests/test_option_marker.py::TestStructuredState`.
  The numbers above are the 231-item rerun with zero failures.
