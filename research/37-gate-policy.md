# Gate policy: what each instrument is allowed to conclude

**Date:** 2026-11-08
**Adopted by:** the project owner
**Supersedes:** the point non-regression clause in `#263` and the blanket rule in `#308`

## The rule

**dev ranks. The held-out set establishes promotion. JevBench detects large regressions on the curated set, and nothing else. When the instruments disagree, the gate produces an investigation request — never a composite score, never an automatic verdict.**

A gate is a suggestion with evidence attached. A pass/fail verdict that people then rubber-stamp is the failure mode this policy exists to prevent, and so is a gate that says nothing actionable.

## Why this shape

Paired power, exact two-sided McNemar at alpha 0.05, 80% power, computed with `model/evaluation/paired_power.py`. **The discordance is measured per row and stated with it**, because a single header cannot reproduce four rows: the hard tier's rate is 0.333 and the public set's is 0.229 from the same pair.

| instrument | n | effective n | discordance | detectable difference |
|---|---:|---:|---:|---:|
| JevBench public | 231 | 231 | 0.229 measured (53/231, `Qwen3-4B` vs `Qwen3-Reranker-4B`) | 9.3 pp |
| JevBench hard tier | 111 | 111 | 0.333 measured (37/111, same pair) | 14.9 pp |
| dev | 812 | 562–709 | 0.238 measured (193/812, `Reranker-4B` vs `Reranker-8B`) | 4.9 pp at the raw n; 5.2–5.8 pp at the effective n |
| frozen held-out | 5,426 | about 3,600 | **0.20 assumed, not measured** | 1.7 pp at the raw n; 2.1 pp at the effective n |

The held-out row is the only one that does not rest on a measurement: nothing has been scored against it under this readout, so 0.20 is an assumption and the row is a planning figure. Treat it as "a couple of points", not as 2 pp.

The project's interesting effects are 1–3 pp.

The concrete case that forced this policy: an 8B reranker scored 149 against a 4B reranker's 150 on JevBench (net −1, p = 1.0, 39 discordant, 7.6 pp detectable) and 527 against 478 on the 812-row dev set (net +49, p = 5.2e-4). The acceptance instrument called a difference the dev set resolved a tie. That is not a model finding, it is a statement about 231 items.

## Instrument roles

| instrument | role | may promote | may block | what a null means |
|---|---|---|---|---|
| dev | ordering and diagnosis | no | no | the difference is under about 5 pp, or the set is not looking at it |
| held-out | promotion evidence | yes, subject to human review | yes | the effect is under about 2 pp |
| JevBench public | coarse safety screen on the curated set | no | only on a supported material regression | nothing about differences under about 9 pp |

No instrument may be averaged with another. They answer different questions at different resolutions, and a weighted sum of them would have no interpretation.

## What the gate outputs

For each arm-against-control comparison:

- the **paired effect estimate** and a 95% confidence interval;
- **discordant counts** in both directions and the exact two-sided p-value;
- the **detectable difference** for that n and that measured discordance, beside the p-value;
- a **state**, assigned by the interval and not by the point estimate;
- a **recommended action** and a one-sentence reason.

On JevBench the states are:

| state | interval |
|---|---|
| material regression supported | 95% upper bound below −8 pp |
| coarse safety screen passed | 95% lower bound above −8 pp |
| unresolved | the interval crosses −8 pp |

and on any instrument:

| state | interval |
|---|---|
| improvement supported | 95% lower bound above 0 |
| improvement suggested but unresolved | point estimate above 0, interval crosses 0 |

These states are not mutually exclusive. An arm can pass the coarse safety screen while showing no evidence of improvement, and for most comparisons that is the correct answer.

## The margin, and what it is not

The JevBench coarse margin is **delta = 8 pp of chance-corrected accuracy**. Derivation: the 9.3 pp floor implies a standard error of about 9.3/2.80 = 3.3 pp; a one-sided 80% test at a true difference of zero needs about (1.645 + 0.84) x 3.3 = 8.2 pp.

**Eight points is the instrument's resolution, not a scientific tolerance.** A genuine 6 pp regression passes this screen: 8 pp is the pass-side margin, and a true 8 pp regression lands in "material regression supported" only about 2.5% of the time under the two-sided 95% bound the states use. It is not the detection threshold, and it should not be read as one. The correct sentence is "passes the coarse JevBench safety screen", never "non-inferior".

The hard tier is worse — 14.9 pp — so it is reported diagnostically and does not carry its own veto.

## Multiplicity

**A family chosen after seeing the results is not a confirmatory family.** Correcting a nine-test family that was selected after the first batch was analysed does not repair the selection; the selection used the data.

When the family was not fixed in advance, the gate reports raw estimates, unadjusted p-values and intervals, labels the family **post hoc**, and states `confirmatory inference: unavailable`. Confirmation requires a fresh prespecified comparison or data not yet used for selection.

For work from here on, one of three designs must be chosen before the outcomes are seen: fix the family and the primary contrast in advance; use alpha-spending or a gatekeeping plan if the family grows through a known sequence; or use an online procedure (alpha-investing, online FDR, e-values) if tests genuinely arrive over time. The requirement in every case is that the procedure predates the outcomes that drive selection.

## Disagreement

When the instruments disagree, the gate produces an investigation request naming which instrument said what, what each can and cannot resolve, and what evidence would settle it. It does not average the instruments, does not pick the more favourable one, and does not convert a disagreement into a no.

Worked example from `#307`, whose dev comparison is post hoc:

> dev strongly favours the 8B reranker (+49 of 812, p = 5.2e-4). JevBench is uninformative about a difference of that size and does not support a material regression. Promotion remains contingent on the held-out result.

## What this replaces

- `#263`: the point non-regression clause is replaced by the three-state interval screen above, with delta = 8 pp and the derivation recorded.
- `#308`: the requirement to quote the detectable delta stays and becomes mandatory output; its "no claim below 5 pp on dev" is subsumed by the dev-is-ordering-only role.
- `research/10` section 7's "freeze the eval suite and every gate on day 0" stands, and this note is what the frozen gate is now defined by.
