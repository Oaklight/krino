# Panel-2 code assignment and order sensitivity (#339) — 2026-11-09

Panel 2's codes are positional: the first option is always `A`. Reversing the
option list therefore changes which code names which option, and the shipped
cell pays for it — 2.60 pp on Qwen3-Reranker-4B and 0.87 pp on Qwen3-32B in
this run, with 23% and 18% of choices changing. This note asks whether
content-addressed codes remove that cost and whether averaging over three
random assignments cancels it instead. Three arms, declared before the run and
no others; nothing was fitted.

## Design

`positional` is the shipped scheme: `CODES[i]` sits at display position `i`.
`hashed` assigns codes by `sha256(text.strip().encode("utf-8")).digest()`,
sorts the options by the 32-byte digest ascending, and gives the i-th option in
that order `CODES[i]`. `mean3` draws
`numpy.random.default_rng(seed).permutation(K)` for seeds 0, 1, 2, re-seeded per
item, and averages the three per-option distributions as probabilities before
argmaxing. In the reversed display each option keeps its code under `hashed`
and `mean3` — asserted over all 231 items in the run — while under `positional`
the code follows the display.

No item has two identical stripped option texts, so no digest tie exists and
the sorted order is fully content-determined. The validity gate passes on both
checkpoints: every code is one token, zero items unscorable, and the structural
reversal invariance holds for both content arms.

## Results on JevBench

Natural and reversed display order, 231 items. W/L is the paired McNemar on
the natural-versus-reversed comparison, and `changed` is the fraction of items
whose chosen option differs between the two orders.

| checkpoint | arm | natural | reversed | Δ pp | changed | W/L | exact p |
|---|---|---:|---:|---:|---:|---:|---:|
| Reranker-4B | positional | 162/231 = 70.13% | 67.53% | −2.60 | 22.9% | 18/24 | 0.441 |
| Reranker-4B | hashed | 158/231 = 68.40% | 67.10% | −1.30 | 26.0% | 22/25 | 0.771 |
| Reranker-4B | mean3 | 162/231 = 70.13% | 62.77% | −7.36 | 20.3% | 8/25 | 0.0046 |
| 32B | positional | 178/231 = 77.06% | 76.19% | −0.87 | 18.2% | 16/18 | 0.864 |
| 32B | hashed | 180/231 = 77.92% | 74.89% | −3.03 | 13.0% | 9/16 | 0.230 |
| 32B | mean3 | 177/231 = 76.62% | 77.92% | +1.30 | 14.7% | 15/12 | 0.701 |

**One item separates this run's shipped cell from #336's.** `hard-opus-a-probability-03` is an exact tie in the reversed prompt: its two code logits are both 17.5 in bf16. The two harnesses break the tie differently. #326's harness takes the first maximum in display order, which lands on natural option 1 and marks the item correct; this harness takes the first maximum in natural option order, which lands on option 0 and marks it wrong. That one item is the whole of 156 against 157, and the difference is the tie-break rule, not numerics and not a different reversed-list construction — the natural-order prompt for the same item is not a tie (17.25 against 18.625). The rule used here is the one `record_correct` states, that ties are broken by the option's original index so reordering cannot change whether an item counts as correct; under it this item is wrong in both orders, and under #326's rule it flips.

Paired against the shipped arm on the same items, natural order: `hashed` is
−1.73 pp on the 4B (12/16, p = 0.572) and +0.87 pp on the 32B (12/10,
p = 0.832); `mean3` is 0.00 pp on the 4B (6/6, p = 1.0) and −0.43 pp on the
32B (5/6, p = 1.0). Paired on the churn indicator: `hashed` against
`positional` is 60 against 53 changed items on the 4B (27/20, p = 0.382) and
30 against 42 on the 32B (10/22, p = 0.050).

## Reading

**Accuracy is unchanged by content addressing on both checkpoints.** Every
cross-arm difference is far inside the 9.3 pp floor and every paired test is a
null. The arms move items around without moving the total: on the 4B `hashed`
loses four noul items (48 to 44) and nothing else; on the 32B it gains six
choice items (105 to 111) and loses four noul (60 to 56), net +2.

**The reversal cost is not removed, and the two checkpoints disagree about
what changes.** On the 4B the accuracy delta halves, 2.60 to 1.30 pp, but the
changed fraction rises, 22.9% to 26.0%. On the 32B the delta triples, 0.87 to
3.03 pp, while the changed fraction falls, 18.2% to 13.0% (p = 0.050). No
direction is consistent across models, every accuracy delta is a null, and the
difference between the two arms' deltas is 1.3 pp on the 4B and 2.2 pp on the
32B — both far inside the floor, so neither the shrink nor the growth is
resolved.

The mechanism is worth naming precisely. Under `hashed` the codes do not move
when the list is reversed, so the residual delta is the model's own
display-order sensitivity, not a code-remapping artefact; under `positional`
the measured delta contains both. The experiment was designed to separate the
two, and at 231 items it cannot: the difference between the arms' deltas is
itself unresolvable, and on one checkpoint it has the wrong sign.

**Pre-registered branch: the "other" clause, not branch 1.** Branch 1's
accuracy clause holds — `hashed` holds accuracy — but its "removes the
reversal cost" clause does not: the cost does not fall consistently, and 13-26%
of choices still change under reversal. Branch 2 does not fire; no arm loses
more than 9.3 pp. Branch 3 does not fire: `mean3` does not remove the reversal
cost. On the 4B it makes it markedly worse, −7.36 pp with 8/25 discordant
items and p = 0.0046; on the 32B it is +1.30 pp, a null. So the outcome is
unclassified: content addressing is accuracy-neutral, it does not demonstrably
stabilise the readout, and it should not be adopted on this evidence. The
shipped positional scheme stays.

**`mean3` is near-identical to `positional` in natural order.** Its natural
predictions agree with the shipped arm on 213 of 231 items, and for 198 items
all three seeds pick the same option, because the model's preference is strong
enough that most permutations do not move the argmax. That is why averaging
three assignments mostly costs three forwards and changes little.

**Hypothesis, not a result:** averaging three distributions flattens the
mixture relative to any one member, so its argmax is decided by smaller
margins and should be more sensitive to whatever the reversal changes in the
context. That would explain why the 4B reverses worse under `mean3` (−7.36 pp)
than under either single-pass arm (−1.3 to −2.6 pp) while its natural accuracy
is unchanged. It is not tested here — the per-forward logits were not stored,
so the mixture margins cannot be checked without a re-run — and it should not
be quoted as the reason until they are.

## Cost

The arms are 1, 1 and 3 forwards per display order, so `mean3` is three times
the forward cost of a single-pass arm and the run does five forwards per item
against one for the shipped cell. Whole-run wall clock including model load on
one GPU each, both runs started together on separate GPUs: 2.8 minutes for the
4B and 6.5 minutes for the 32B, for 2,310 forwards per checkpoint.

The per-call `latency_s` values recorded in the result files are **not usable**:
the forward loop has no device synchronisation point, so the time charged to a
call depends on what the previous call queued. The files show 320 ms mean for
`positional` and 25 ms for `hashed` against an identical single forward, and
76 ms for `mean3`'s three forwards — a 13x spread that is queueing, not cost.
The structural arithmetic above and the wall clock are the cost numbers; a
device-synchronised timing is needed for a per-arm wall-clock curve and was not
taken here.

## What this does not say

It does not say the model has no positional bias. It says the bias that the
code remapping contributes is too small to separate from the model's own
display-order sensitivity at 231 items, and that removing the remapping by
construction does not produce a stable readout: the choices churn as much as
before. A larger instrument, or an intervention that scores the same option
content under different letters with the letters held out of the answer
position, would be needed to attribute the residual.

## Provenance

Pre-registration: #339 comment, written before the run. Config hash
`a5c840066b7755c5afd4855aab49b1acc3ed9a7ad9ae32a6d2b3bf286b743e56`, data hash
`e85e155c8c14d7ebdb073c04efb049ded3e0186b0297ebcd8388aa259a365370`, code
commit `52c6456c08f80bd318240a91878de98aa47f8176`, branch
`feature/339-panel2-order-arms` from `b0cdb06`. Models at
`22e683669bc0f0bd69640a1354a6d0aebcfeede5` (Reranker-4B) and
`9216db5781bf21249d130ec9da846c4624c16137` (32B), bf16, `--code-form space`.
The same config hash covers both checkpoints because the config excludes the
model id; the model is in each file's provenance.

Files: `results/jevbench/panel2_order_arms_Qwen_Qwen3-Reranker-4B.json` and
`results/jevbench/panel2_order_arms_Qwen_Qwen3-32B.json`, with per-item
predictions, per-forward argmaxes, the per-arm metrics, the position
diagnostic and the validity gate.
