# Answerable head: joint training and reporting

Design note for #237 Part B. Training is blocked on #233; this note is the
contract another session can implement without asking questions. It fixes how
the head joins the main loss, what the ablation is, and what gets reported.

The head and the abstention path already exist from #232: `DecisionModel`
carries `answerable_head` and `forward_answerable`
(`model/src/decision_model.py:1299`), and `train_residual.py` has
`--answerable-head`, `--answerable-items`, and `--answerable-weight`. What is
missing is the joint-training contract: masked task loss, a share-or-own
forward decision, warm-up, and an ablation that separates the head from the
extra rows.

## 1. Data contract

The paired corpus is produced by `model/scripts/build_unanswerable_pairs.py`.
It writes three files on the compute host; none of the corpus is committed.

- `pairs.jsonl` — each answerable original followed by its variants.
- `negatives.jsonl` — variants only. **This is the training input.** Passing
  `pairs.jsonl` to `--answerable-items` duplicates the originals, which already
  live in the base corpus.
- `unanswerable_30k.stats.json` — committed at
  `results/abstention/unanswerable_30k.stats.json`.

Variant contract:

| field | value |
| --- | --- |
| `id` | `<base-id>-unans-<reason>-<k>` |
| `question` | byte-identical to the original |
| `state` | original with only the evidence span edited |
| `label` | the original label, kept for provenance |
| `answerable` | `false` |
| `reason` | `support_removed`, `out_of_scope`, or `entity_corrupted` |
| `teacher_probs`, `teacher_name` | stripped |

The original pair row sets `answerable: true` explicitly.

The builder verifies every pair before writing it: question, source, and group
are identical; the text before and after the evidence span is identical; the
evidence equals the constructed evidence; the id is paired to the original.
The run fails if any check fails.

Because the label is provenance, **a reason-bearing row must not contribute to
the task CE or the KL term**. Without masking, the task head is trained to say
"false" on removed support and "supports" on swapped evidence. Section 2.2
makes the mask mandatory.

Held-out split: `model/scripts/split_abstention.py` recognizes the `-unans-`
suffix (this PR). Run it with `negatives.jsonl` as `--abstention` and the
concatenated base corpus as `--corpus`. It writes the held-out mix and
`eval_base_ids.txt`; pass that file to training as `--exclude-ids` so no
held-out base item appears in training in answerable form. The corpus is two
source files, so concatenate them (dedup by id) into one training file first.

## 2. Loss

### 2.1 Formulation

```
L = w_kl * KL(base || fused) + w_ce * CE + lambda(t) * w_ans * BCEWithLogits(a, y, pos_weight)
```

- `y = answerable_target(item)`: the explicit `answerable` field wins; a
  gold-carrying item defaults to `true`.
- `pos_weight = n_neg / max(n_pos, 1)`, computed over `train_items` after the
  negatives are appended. At the shipped mix this is about 34k negatives over
  about 50k positives, so `pos_weight` is about 0.68. Do not resample; the
  weight handles the class imbalance.
- `w_ans = 1.0` to start. `w_kl` and `w_ce` come from the run config; the
  answerable term does not change them.

### 2.2 Mask reason-bearing rows from the task loss

In `item_loss` (`model/scripts/train_residual.py:167`), after the CE and KL are
computed:

```python
masked = item.answerable is False and bool(item.reason)
if masked:
    ce = torch.zeros((), device=ce.device)
    kl = torch.zeros((), device=kl.device)
```

Keep the logged `kl_mean` and `ce_mean` over unmasked rows only, and log
`n_masked` beside them, so the loss curves stay comparable with earlier runs.
The answerable BCE is still computed for the masked row.

### 2.3 Same forward pass or its own

`forward_answerable` currently re-encodes `state + instructions`, so it is a
second backbone pass. Decide per readout:

- **Plain noul (`noul_marker=False`)**: share the pass. Refactor `forward_noul`
  to return the pooled context as well (for example
  `forward_noul_with_pooled` returning `(logit, pooled)`), and feed
  `answerable_head(pooled)` from that call. The input is already
  `f"{state} {instructions}"`, so the head sees exactly what it sees today.
- **Choice, score, and marker-based noul**: keep the own pass. These readouts
  are option-conditioned; pooling the context out of the marker forward needs
  `output_hidden_states=True` and adds memory for no accuracy gain. The
  answerable pass is one context encode, not K option scorings, so its cost is
  a fraction of the option forward.

Do not change the head's input to include options or the option count. The head
must stay option-agnostic or it learns K-dependent shortcuts.

### 2.4 Warm-up

```
lambda(t) = min(1.0, t / W),  W = 10% of the first epoch's steps
```

Warm-up applies on the first LoRA rung only; later rungs use `lambda = 1`. The
head is randomly initialized, and with LoRA its gradient reaches the adapter
immediately, so the ramp keeps the main output from moving before the task and
KL terms settle. The first `W` steps are also the "head attached but inert"
control.

### 2.5 LoRA and KL

Run the joint arm on the first LoRA rung from #233 with `KL > 0` (use 0.1
unless the rung config fixes another value). The default `KL = 0` is the
byte-identical reproduction path, not a joint-training arm. The answerable BCE
updates the same adapter as the task loss; a nonzero KL is what bounds the
drift of the main distribution.

## 3. Ablation

Same base corpus, same LoRA scope/rank/alpha, same epochs, same shuffle seed,
same held-out split.

| arm | head | `--answerable-items` | task loss on reason rows |
| --- | --- | --- | --- |
| A0 | off | none | n/a |
| A1 | on | `negatives.jsonl` | masked (section 2.2) |
| A2 (diagnostic) | on | `negatives.jsonl` | not masked |

A2 exists only to show what the mask buys; it is not a candidate. A0 vs A1 is
the reported ablation.

Seeds: 42, 43, 44. Report every seed, then mean ± sd. Never report one seed as
the result.

Fixed evaluation sets:

1. Held-out abstention mix from `split_abstention.py` (15% of base ids).
2. The internal base eval set used by the rung.
3. JevBench public (231 items), as the acceptance gate.

### Metrics

Risk-coverage on the held-out mix, model `p_answerable` against the `max_prob`
baseline:

- AURC (lower is better), per seed.
- Risk at 90% coverage: answered accuracy, abstain-zero accuracy, and the
  fraction of abstained rows that carry a reason.
- The full curve at 50/70/90/100% coverage.
- Abstention recall by reason at 90% coverage: separate numbers for
  `support_removed`, `out_of_scope`, `entity_corrupted`.
- Answerable-head-only accuracy at threshold 0.5, with per-class recall. This
  is a classification number, not a risk-coverage number.

Main output with the head on:

- chance-corrected accuracy (`cc`), raw accuracy, per type (noul/choice/score)
  and per option-count bucket.
- ECE, Brier, and BSS against the label marginal, on the raw option
  distribution and on the `p_answerable` mixture separately.
- The same numbers for A0, so the delta is readable.

Cost: step time, items/s, peak memory. Report them; do not gate on them.

### Pass criteria

1. `AURC(A1) < AURC(max_prob)` on all three seeds, and the mean gap is larger
   than the seed sd.
2. `|cc(A1) - cc(A0)| <= 1 pp` on the held-out base eval, and the difference is
   within one seed sd. `ECE(A1) <= 0.05` and `|ECE(A1) - ECE(A0)| <= 0.01`.
3. No per-type `cc` regression larger than 2 pp.
4. At 90% coverage, A1's answered accuracy is not below A0's accuracy at the
   same coverage, and the abstained rows are enriched for reason-bearing
   items above the base rate. Report the enrichment, not only the accuracy.
5. A step-time increase above 2x is a blocker that needs a decision, even if
   the accuracy criteria pass.

If criterion 2 fails, halve `w_ans` to 0.5, rerun A1 on the three seeds, and
report both. Do not tune further against the eval set.

## 4. Why the mixture cannot change cc by itself

At inference:

```
p(abstain) = 1 - sigmoid(a)
p(option_i) = sigmoid(a) * p_i
```

Argmax over options is unchanged by a positive scale, so top-1 accuracy and
`cc` are invariant to the mixture. Any `cc` change between A0 and A1 is caused
by adapter drift, which is exactly what the ablation measures. ECE is not
invariant: scaling every probability down changes calibration by shrinkage.
That is why ECE is reported raw and mixture separately, and why the "no cost"
claim uses `cc` and raw ECE. A Brier/BSS improvement on the mixture alone is
not evidence of a better main output.

## 5. Confounds and limits to state in the write-up

- The three transforms are operational definitions. `out_of_scope` is
  out-of-distribution detection, not logical unanswerability; the head may
  learn "evidence does not belong to this question" rather than "no answer
  exists". Report it as such.
- Entity donors are same-source and same token shape, but not type-matched.
  `minecraft` can be replaced by a person name. If `entity_corrupted` recall is
  much higher than the other two, the head may be detecting the type clash.
  Build a type-matched donor pass before claiming the capability.
- Support removal on a long contract clause leaves a long, still-readable
  clause; the removed sentence may not be the only support. Per-reason recall
  is the diagnostic.
- The corpus is deduplicated by base id for the split, but near-duplicate base
  items can still cross the split boundary. The existing abstention split does
  not cluster; state this, and do not claim a clean held-out estimate beyond
  the base-id guarantee.
- The corpus lives on the compute host. Only the builder, its tests, the stats
  JSON, and this note are committed.

## 6. Work order after #233

1. Concatenate the two source corpora into one training file, dedup by id.
2. Run `split_abstention.py` with `negatives.jsonl` and the concatenated
   corpus; keep `eval_base_ids.txt` for `--exclude-ids`.
3. Implement sections 2.2–2.4 in the trainer that carries the first LoRA rung.
   If the rung is `train_residual.py`, the head and LoRA are already there; if
   it is a v4 trainer, port `answerable_head`, `forward_answerable`, the
   answerable term, and the mask. The data contract is identical.
4. Run A0 and A1 on seeds 42/43/44 with the same config otherwise.
5. Report the table from section 3 with the pass/fail per criterion.
6. Only then run the JevBench gate with the arm name and seed in every row.

## 7. Acceptance mapping

- Head joins the main loss: section 2, with the mask, the share-or-own rule,
  and the warm-up.
- Ablation head off vs head on, same corpus and seeds: section 3.
- AURC, risk at 90% coverage, cc and calibration with the head on: section 3.
- "Does not cost accuracy on the main output": pass criteria 2 and 3, measured
  as a delta against A0, not as an absolute number.
