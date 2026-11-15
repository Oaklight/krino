# Research notes — index

`research/` is the working record: one note per question, written before or
immediately after the run, with the result, the scope and what it does not
show. It is not the published site; that is `docs_en/`. It is not the report;
that lives outside this repository.

Read it in one of two ways:

- **By question.** The catalogue below groups every note by the phase that
  produced it and states what the note settles.
- **By work item.** `00-work-history.md` lists every issue and pull request in
  creation order, with its parent, close and stacking links.

## Phases

Phases are dated bursts of activity, not milestones. The repository has no
milestones.

| phase | dates | what changed | notes |
|---|---|---|---|
| P0 | 2026-09-18 .. 09-21 | black-box probing, then the first replication | 00–03, then 04 |
| P1 | 2026-09-22 .. 09-30 | multi-task training, Qwen3.5, web, JevBench | 05–07 |
| P2 | 2026-10-01 .. 10-26 | pause: Clef, data fixes, labels | — |
| P3 | 2026-10-27 .. 10-31 | round 2: LoRA, score, evaluation, calibration, leakage | 08–19 |
| P4 | 2026-11-01 .. 11-05 | v3 lands, v4 starts | 09, 10, 11, 12, 13, 14, 23, 24 |
| P5 | 2026-11-06 .. 11-08 | canonical corpus, v4 infrastructure, zero-shot, readout matrix | 25–37, 235, 247, 314 |
| P6 | 2026-11-09 .. 11-10 | judgement, panel 2, code space, blocking, position | 38–46 |
| P7 | 2026-11-11 .. 11-15 | consolidation, corpus v2, readout training | 47, 48, 49 |

## Catalogue

Status is `current` unless marked. `superseded` means a later note replaces
its conclusion; the older note stays for the record.

### P0 — probing and the first replication

| note | date | what it settles | status |
|---|---|---|---|
| `00-overview.md` | 2026-09-18 | what Jev is, its three primitives, and the project's starting context | historical |
| `01-api-testing.md` | 2026-09-18 | the three primitives behave as documented against the live API | historical |
| `02-architecture-analysis.md` | 2026-09-18 | what the black-box probes do **not** identify: backbone, mask and readout stay open (correction of 2026-09-18) | historical |
| `03-open-source-ecosystem.md` | 2026-09-18 | the open-source projects around TypeSafe | historical |
| `04-model-replication.md` | 2026-09-20 | the first replication results and the model comparison table | superseded by the v3/v4 arms |

### P1 — multi-task training and the web surface

| note | date | what it settles | status |
|---|---|---|---|
| `05-mlp-lr-targeted-runs.md` | 2026-09-26 | MLP plus per-component LR on a frozen backbone (#93) | historical |
| `06-head-size-sweep.md` | 2026-09-26 | head width and rank across seeds (#76, #93) | historical |
| `07-mtl-optimization.md` | 2026-09-26 | per-type LR, gradient balancing and adaptive weighting (#95) | historical |

### P2 — pause

No research notes were opened in this window; the work is data fixes and the
Clef analysis, later written up as `08-clef-analysis.md`.

### P3 — round 2

| note | date | what it settles | status |
|---|---|---|---|
| `08-clef-analysis.md` | 2026-10-28 | Cloudflare's Clef as a comparison recipe | historical |
| `08-rl-training-design.md` | 2026-10-27 | the RL training design for the decision model | design |
| `09-score-data-sources.md` | 2026-10-28 | where ordinal data can come from, and what converts to a 3–5 level question | current |
| `10-architecture-sweep-r2.md` | 2026-10-28 | round-2 sweep: readout, LoRA scope, backbone scale | superseded by `10-v3-redesign` |
| `11-ordinal-head-loss-arms.md` | 2026-10-28 | ordinal head x loss: the plain softmax baseline wins on accuracy | current |
| `12-score-calibration-ceiling.md` | 2026-10-28 | score calibration is a property of the arm, not of the primitive | current |
| `13-jevbench-public.md` | 2026-10-29 | the trained heads lose 14–21 points to our own readout on JevBench | superseded by `35` and `37` for the interpretation |
| `14-efficiency.md` | 2026-10-29 | the 0.6B meets the latency bar only when compiled; no configuration meets both halves | current |
| `15-option-order-and-count.md` | 2026-10-28 | option-order flip rate and option-count robustness (goal clause 3) | current |
| `16-corpus-leakage-audit.md` | 2026-10-29 | 100 of 253,701 eval items duplicate a training item; the STS-B sentinel | current, extended by `235` |
| `17-schedule-and-readout.md` | 2026-10-29 | more epochs buy in-distribution accuracy; the readout buys JevBench | current |
| `19-native-head-blend.md` | 2026-10-29 | native score plus trained head beats either on short lists | current |

### P4 — v3 lands, v4 starts

| note | date | what it settles | status |
|---|---|---|---|
| `09-architecture-exploration.md` | 2026-11-02 | readout layer, layer fusion, multi-backbone and head width | historical input to v3 |
| `10-v3-redesign.md` | 2026-11-02 | the v3 plan; supersedes the 19/73/106 roadmaps | superseded by v4 |
| `11-4b-primary-arm.md` | 2026-11-03 | the 4B primary arm: B1 is killed; no arm meets the gate | current |
| `12-marker-readout-result.md` | 2026-11-03 | the 150M encoder's failure was mostly the pooled readout | current |
| `13-throughput-levers.md` | 2026-11-03 | throughput levers surveyed across comparable repositories | reference |
| `14-jev-highk-probe.md` | 2026-11-04 | Jev's latency is flat in K and 80-way accuracy is far above chance | current |
| `23-public-tier-power.md` | 2026-11-03 | the public tier can see a deficit but not a gain | current |
| `24-v3-closeout.md` | 2026-11-04 | v3 misses the acceptance proxy; what the residual line taught | current |

### P5 — canonical corpus, v4 infrastructure, measurement instruments

| note | date | what it settles | status |
|---|---|---|---|
| `09-answerable-head-integration.md` | 2026-11-06 | how the answerable head joins the loss and what the ablation reports (#237) | design |
| `25-corpus-inventory.md` | 2026-11-06 | every candidate training file, counted on both machines (#234) | current |
| `235-corpus-audit.md` | 2026-11-06 | exact-duplicate and JevBench-contamination audit of the canonical corpus (#235) | current |
| `247-ability-coverage.md` | 2026-11-06 | ability labels and the coverage map (#247) | current |
| `26-adapter-folding.md` | 2026-11-06 | folding the adapter keeps the shared-prefix inference path (#245) | current |
| `26-decision-index-recon.md` | 2026-11-06 | which Decision Index can be run, and which cannot (#236) | current |
| `27-matched-heldout-spec.md` | 2026-11-06 | size, quotas, group key and freeze procedure for the matched held-out set | spec |
| `28-decision-index-suite-licences.md` | 2026-11-06 | licence position per upstream source of the Decision Index kit | current |
| `29-corpus-data-card.md` | 2026-11-06 | the one-page data card for the training arm | current |
| `29-lora-throughput-244.md` | 2026-11-06 | corrected LoRA throughput, superseding the earlier figures (#244) | current |
| `30-abstention-eval-instrument.md` | 2026-11-06 | the abstention evaluation instrument and its committed selection (#237) | instrument |
| `30-canonical-source-licences.md` | 2026-11-06 | licence verdict per source group; 35 groups, policy recorded | current |
| `31-score-question-pilot.md` | 2026-11-06 | the first score-question pilot: yield and label quality (#156) | pilot |
| `31-suite-freeze-reproducibility.md` | 2026-11-06 | why a pinned suite stopped reproducing, and what a freeze record needs | current |
| `32-score-pilot-round2.md` | 2026-11-06 | round two of score generation with state-carried rubrics | pilot |
| `33-zero-shot-logit-baselines.md` | 2026-11-07 | three checkpoints on the 19-benchmark set, zero-shot (#277) | current |
| `34-jevbench-zero-shot-logit.md` | 2026-11-07 | the same checkpoints on JevBench-public, zero-shot (#278) | current |
| `314-head-generations.md` | 2026-11-08 | what produced each trained head, by generation (#314) | current |
| `35-jevbench-readout-matrix.md` | 2026-11-08 | four checkpoints x three readouts; the 0.628 anchor reproduces at 0.641 | current |
| `36-reranker-size-ladder.md` | 2026-11-08 | 0.6B / 4B / 8B on the judgement readout (#307) | current |
| `37-gate-policy.md` | 2026-11-08 | which instrument may conclude what; owner-adopted | current |

### P6 — readout science

| note | date | what it settles | status |
|---|---|---|---|
| `38-judgement-architecture.md` | 2026-11-09 | readout shape, reading direction and a trained yes/no head (#324) | current |
| `39-panel2-calibration.md` | 2026-11-09 | per-type temperature and bias for the vocabulary-code readout (#338) | current |
| `40-panel2-order-arms.md` | 2026-11-09 | positional codes and the order arms, including content addressing (#339) | current |
| `41-set-head-option-states.md` | 2026-11-09 | a set-function head over frozen option states (#340) | current |
| `42-cascade-close-calls.md` | 2026-11-09 | panel 2, then a judgement re-read of the top two (#342) | current |
| `43-code-tiers.md` | 2026-11-09 | no tested tier extends the 52-letter alphabet (#354) | current |
| `44-code-lexical.md` | 2026-11-09 | code lexical content is not the discriminator; the alphabet stays at 52 | current |
| `45-blocking-frontier.md` | 2026-11-09 | blocking at K=150: B is best at every block size | current |
| `46-position-probes.md` | 2026-11-10 | the long-list tail: class rank on Banking77, display position on CLINC150 | current |

### P7 — consolidation

| note | date | what it settles | status |
|---|---|---|---|
| `47-h2o-lightning.md` | 2026-11-15 | H2O-Lightning-4B scored on our instruments (#327) | current |
| `48-near-duplicate-remediation.md` | 2026-11-15 | the per-row near-duplicate policy and canonical corpus v2 (#235 follow-up) | current |
| `49-judgement-readout-arm.md` | 2026-11-15 | the judgement-objective LoRA arm: a significant JevBench regression, calibration excluded | current |

## Numbering collisions

Several number prefixes are reused because notes were written in parallel
sessions. Renaming would break references in issues, PRs and later notes, so
the files stay. This table is the authority for what each number means.

| prefix | notes | how to read it |
|---|---|---|
| 08 | `08-clef-analysis`, `08-rl-training-design` | 08-clef is the Clef comparison; 08-rl is the RL design |
| 09 | `09-score-data-sources`, `09-architecture-exploration`, `09-answerable-head-integration` | score supply; architecture exploration; answerable head contract |
| 10 | `10-architecture-sweep-r2`, `10-v3-redesign` | the round-2 sweep; the v3 plan |
| 11 | `11-ordinal-head-loss-arms`, `11-4b-primary-arm` | ordinal head x loss; the 4B primary arm |
| 12 | `12-score-calibration-ceiling`, `12-marker-readout-result` | score calibration; the marker readout on the 150M encoder |
| 13 | `13-jevbench-public`, `13-throughput-levers` | JevBench results; the throughput survey |
| 14 | `14-efficiency`, `14-jev-highk-probe` | our latency; Jev's high-K behavior |
| 29 | `29-corpus-data-card`, `29-lora-throughput-244` | the data card; LoRA throughput |
| 30 | `30-canonical-source-licences`, `30-abstention-eval-instrument` | licences; abstention evaluation |
| 31 | `31-score-question-pilot`, `31-suite-freeze-reproducibility` | the pilot; the freeze retrospective |

Issue-numbered notes (`235-`, `247-`, `314-`) are unambiguous: the number is
the issue they answer.

## Conventions for new notes

1. File name `NN-topic.md`. If the note answers an issue, use the issue number
   as `NN` (for example `235-corpus-audit.md`); otherwise take the next free
   number in the current phase and record it here.
2. Open with `# Title — YYYY-MM-DD`, then `**Status:**`, `**Issues:**`,
   `**PRs:**`.
3. State in one paragraph what the note settles. Put the result after it, then
   scope and limits. Do not leave the conclusion to the end.
4. Link every reported number to the committed JSON when one exists. Never
   retype a hash; paste it.
5. Add the note to this catalogue in the same PR. A note that is not in the
   index is not part of the record.

## Related

- `00-work-history.md` — every issue and PR in creation order, with dependencies.
- `figures/` — the standalone figures and their build files.
- `docs_en/` — the published MkDocs site, a curated subset.
- The arXiv-style report is maintained outside this repository.
