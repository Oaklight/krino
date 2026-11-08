# Reranker size ladder on JevBench (#307) — 2026-11-08

JevBench's judgement band is 7 items wide and every checkpoint inside it sits
inside the instrument's resolution, so the band cannot say whether capacity or
the readout sets the level. This note scores one family, `Qwen3-Reranker` at
0.6B, 4B and 8B, with one recipe, one template and the same 231 items. The
question is what the ladder does, not which checkpoint is best.

## Cells

All six cells use Krino's templates, not each checkpoint's native prompt: the
judgement readout ranks each option on `log P(" yes")` after
`Does this option correctly answer the question? Answer:`, and the
continuation readout scores the option text at `Answer:` (`research/35`). All
six are at commit `880a9f1`; the two 4B rows were re-run there and replace the
older files, because every prediction reproduced (below).

| model | judgement | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|---:|
| Reranker-0.6B | 96/231 = 0.4156 | 39 | 54 | 3 |
| Reranker-4B | 150/231 = 0.6494 | 47 | 90 | 13 |
| Reranker-8B | 149/231 = 0.6450 | 44 | 93 | 12 |

| model | continuation | noul (74) | choice (139) | score (18) |
|---|---:|---:|---:|---:|
| Reranker-0.6B | 91/231 = 0.3939 | 43 | 42 | 6 |
| Reranker-4B | 111/231 = 0.4805 | 40 | 60 | 11 |
| Reranker-8B | 111/231 = 0.4805 | 42 | 55 | 14 |

Counts are predictions over 231 rows. The `original` tier's 72 rows are 36
source questions asked twice with different states (`research/34`), so a paired
test that treats the 231 rows as independent is slightly anti-conservative.

## The ladder

Exact paired McNemar, two-sided, on the 231 common items.

| pair | judgement | continuation |
|---|---|---|
| 0.6B vs 4B | +14 −68, p = 1.1e-9 | +21 −41, p = 0.015 |
| 0.6B vs 8B | +23 −76, p = 8.5e-8 | +21 −41, p = 0.015 |
| 4B vs 8B | +20 −19, p = 1.0 | +16 −16, p = 1.0 |

Judgement against continuation within a size: 0.6B +44 −39 (p = 0.66), 4B
+55 −16 (p = 3.8e-6), 8B +58 −20 (p = 2.0e-5). At 0.6B the two readouts are
indistinguishable; at 4B and 8B the judgement template is ahead of
continuation by 39 and 38 items.

## The pre-registered branches

- **Capacity** — monotone with at least 4 items per step and 8B at least 8
  items above 0.6B. Does not fire: the ladder is not monotone, because 4B to 8B
  is −1 item.
- **Readout** — 0.6B within 8 items of 8B. Does not fire: the gap is 53 items.
- **Intermediate** — 0.6B clearly below both, 4B and 8B indistinguishable.
  This is the branch that fires. 0.6B is 54 items below 4B and 53 below 8B
  (p = 1.1e-9 and 8.5e-8); 4B and 8B are one item apart (+20 −19, p = 1.0).
- **Other** — not needed. The ladder is non-monotone by one item (150 → 149),
  which is why Capacity did not fire; that single item is indistinguishable
  from zero (+20 −19, p = 1.0) and is read above, not as a separate outcome.

## What a null means at 231 items

The 4B/8B pair has 39 discordant items on judgement, 16.9% of 231. At alpha
0.05 two-sided and 80% power, the normal approximation to the paired test
reaches `2.80 × sqrt(0.169/231)` = **7.6 pp**, about 17.6 items. On
continuation the pair has 32 discordant items (13.9%) and reaches **6.9 pp**,
about 16 items.

So "4B and 8B are indistinguishable on JevBench" is a statement about a
231-item instrument with a 17-item floor, not a statement that the two
checkpoints are equivalent. It is the same arithmetic that made the current
band unreadable.

## The 4B rows

The committed 4B cells were produced at `2819b6f6` (judgement) and `33506b86`
(continuation). Both were re-run at `880a9f1` and compared per item, not on the
aggregate: all 231 judgement `logp_yes` values and chosen indices match to six
decimals, and all 231 continuation `logits` vectors match, so the totals stay
150/231 and 111/231. The re-runs replace the committed files, which gives the
ladder one `code_commit`. The replacement changes the judgement row shape only
by adding `logp_no` and `mode` (#310); no stored score moves.

## Conclusion

**Branch triggered: Intermediate. Inference withdrawn as instrument-limited.**

The trigger is exact: 0.6B is clearly below both sizes, and 4B and 8B are
indistinguishable on JevBench. The branch's stated inference — capacity
saturates near 4B — is contradicted by the instrument that has the power to
test it, so this note records the trigger and withdraws the inference, the way
`#298` recorded a mis-specified node check rather than substituting a branch.

On JevBench the 4B/8B judgement pair is 150 against 149, net −1, p = 1.0, with
39 discordant items and 7.6 pp detectable at 80% power. On the development set
the same pair is 478 against 527, net +49 = 6.0 pp, p = 5.2e-4. A 6.0 pp
difference sits below what JevBench resolves and above what dev resolves.

So the plateau is a property of the 231-item instrument, not of the models:
capacity continues to help from 4B to 8B, and the acceptance set cannot
arbitrate between the two checkpoints.

Continuation repeats the pattern: JevBench 111 against 111 (16–16, p = 1.0,
6.9 pp detectable at 80% power) against dev 450 against 483 (net 33,
p = 0.0120).

Two limits are stated rather than buried. The nine-test family behind the dev
p-values is post-hoc: it was not fixed before the second batch landed, and the
correction is named as post-hoc rather than presented as pre-registered. And
the dev set is validated as a checkpoint-ordering instrument (`#298`), not as a
capability measurement, so 527/812 orders the sizes; it does not license an
absolute claim.

The 4B dev cells behind the middle rung were re-run at `880a9f1` for this
comparison and reproduced `#298`'s 478 and 450 exactly. They are committed in
the companion dev-set PR; this note quotes them as the instrument that resolves
the pair, not as cells it ships.
