# Work history: issues and pull requests, 2026-09-18 to 2026-11-15

This is the raw chronological record behind the research notes. Every issue and
pull request created in the repository appears once, under the phase in which it
was opened, with the dependency links found in its title or body. It is
deliberately exhaustive: later write-ups select from it, they do not repeat it.

Source: the GitHub API, all states: 246 pull requests (232 merged, 14 closed)
and 128 issues (92 closed, 36 open), 2026-09-18 to 2026-11-15. Relations are the
explicit ones: `Parent`, `Part of`, `Closes`, `Fixes`, `Depends on`,
`Stacked on`, `Follow-up to`, plus issue references in a PR title. A PR whose
base is not `main` is marked with that base.

Phases are dated bursts of activity, not milestones: the repository has no
milestones. The headings are descriptive.

Companion: `research/README.md` indexes the notes themselves.

## P0 2026-09-18 to 2026-09-21 — Probing and the first replication

27 pull requests and 28 issues opened in this window.

The project opens by fingerprinting the Jev API. Issue #1 is the probing
epic; nine probes are filed the same day (#2-#13). PRs #7, #14, #15, #17 and #18
fold the findings into `research/02`; #16 is closed as superseded by the
controlled probe suite. The replication line starts on 2026-09-19: epic #19 with
steps #20-#24, then shared infrastructure (#25), zero-training baselines (#26),
trained heads (#28), the calibration skeleton (#31), the backbone sweep (#33)
and the Ettin reranker (#36). The window closes with the 19-benchmark sweep
(#43), the Jev API evaluation (#45), the GitHub Pages site (#48), distillation
support (#51) and the synthetic pipeline (#52). The architecture note is
corrected on 2026-09-18: the black-box probes do not identify the backbone, the
attention mask or the readout, and the analysis says so.

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-09-18 | #1 | closed | Epic: Architecture probing: systematic tests to fingerprint Jev's model internals |  |
| 2026-09-18 | #2 | closed | Probe 1: Token accounting — verify tool-call template hypothesis | #1 (parent) |
| 2026-09-18 | #3 | closed | Probe 2: Noul precision quantization — output head fingerprint | #1 (parent) |
| 2026-09-18 | #4 | closed | Probe 3: Diffusion vs autoregressive signature — masking & scrambling robustness | #1 (parent) |
| 2026-09-18 | #5 | closed | Probe 4: Option ordering bias on ambiguous cases | #1 (parent) |
| 2026-09-18 | #6 | closed | Probe 5: Cross-model error fingerprinting | #1 (parent) |
| 2026-09-18 | #8 | closed | Probe 3b: Corrected diffusion test — word-level masking and bidirectional context | #1 (parent) |
| 2026-09-18 | #9 | closed | Probe 6: MoE vs dense — domain capability variance | #1 (parent) |
| 2026-09-18 | #10 | closed | Probe 7: Pretraining cutoff — temporal knowledge boundary | #1 (parent) |
| 2026-09-18 | #11 | closed | Probe 8: Readout head design — per-type behavior differences | #1 (parent) |
| 2026-09-18 | #12 | closed | Probe 9: Multilingual parity — language capability fingerprint | #1 (parent) |
| 2026-09-18 | #13 | closed | Investigate Jev API output pattern: single-payload vs streaming |  |
| 2026-09-19 | #19 | open | Epic: Decision model replication |  |
| 2026-09-19 | #20 | closed | Step 0: Shared infrastructure — data pipeline, eval suite, API server | #19 (parent) |
| 2026-09-19 | #21 | closed | Step 1: Zero-training baselines — causal logit readout + encoder scoring | #19 (parent), #20 (depends on) |
| 2026-09-19 | #22 | closed | Step 2: Trained scoring heads — frozen backbone + lightweight heads | #19 (parent), #20 (depends on) |
| 2026-09-19 | #23 | closed | Step 3: Calibration training — losses + RLCD-style RL | #19 (parent), #22 (depends on) |
| 2026-09-19 | #24 | closed | Step 4: Fresh architecture — bidirectional encoder vs causal comparison | #19 (parent) |
| 2026-09-19 | #27 | closed | Base model candidate sweep: causal + encoder + reranker backbones | #19 (parent) |
| 2026-09-19 | #29 | closed | Data pipeline: contrastive + synthetic data with diversity and balance fixes | #19 (parent) |
| 2026-09-20 | #44 | closed | Benchmark Jev API on our 19-benchmark eval suite | #19 (parent) |
| 2026-09-20 | #46 | closed | GitHub Pages site for research findings |  |
| 2026-09-20 | #47 | closed | Sweep: Qwen3/3.5 MoE + dense models (4B+) | #19 (parent) |
| 2026-09-20 | #49 | closed | Publish trained decision models to HuggingFace | #19 (parent) |
| 2026-09-21 | #50 | closed | Synthetic data generation pipeline: 50K items with Jev soft labels | #29 (parent) |
| 2026-09-21 | #53 | closed | Tracking: Pipeline infrastructure improvements (post-PR-52) |  |
| 2026-09-21 | #54 | closed | Tracking: New pipeline stages — dedup, repair, fill-variants, report (post-PR-52) |  |
| 2026-09-21 | #55 | closed | Tracking: Multi-teacher LLM soft-label annotation (post-PR-52) |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-09-18 | #7 | merged | Update architecture analysis with probe findings |  |
| 2026-09-18 | #14 | merged | Update architecture analysis with round 3 rigorous findings |  |
| 2026-09-18 | #15 | merged | Final architecture analysis: causal Qwen confirmed via cross-model validation |  |
| 2026-09-19 | #16 | closed | Superseded by controlled probe suite |  |
| 2026-09-19 | #17 | merged | Add controlled black-box probe suite |  |
| 2026-09-19 | #18 | merged | Update architecture analysis with controlled probe findings |  |
| 2026-09-19 | #25 | merged | Add shared infrastructure: data pipeline, eval suite, API server | #20 (closes) |
| 2026-09-19 | #26 | merged | Step 1: Zero-training baselines — causal logit readout + encoder scoring | #21 (closes) |
| 2026-09-19 | #28 | merged | Step 2: Trained decision heads on frozen backbone | #22 (part of) |
| 2026-09-19 | #30 | merged | Fix data pipeline infra: multi-shard parquet, recursive glob, NFS I/O |  |
| 2026-09-19 | #31 | merged | Add calibration losses and training skeleton (Step 3 Phase A) | #23 (part of), #19 (parent) |
| 2026-09-19 | #32 | merged | Step 2 follow-up: cross-benchmark generalization + surface-form sensitivity |  |
| 2026-09-19 | #33 | merged | Add backbone sweep Step 1: 4 causal + 3 Ettin reranker baselines |  |
| 2026-09-19 | #34 | merged | Add Phase A benchmark loaders and balance script (#29) | #29 (title-ref) |
| 2026-09-19 | #35 | merged | Add virtual batching for MMCE and gradient accumulation | #23 (part of), #19 (parent) |
| 2026-09-19 | #36 | merged | Step 2: Ettin reranker heads — 95.2% Banking77 with 150M backbone |  |
| 2026-09-19 | #37 | merged | Add calibration training script and sweep config (Phase A completion) | #23 (part of), #19 (parent) |
| 2026-09-19 | #38 | merged | Fix YAML lr parsing and verify training on GPU | #23 (part of) |
| 2026-09-19 | #39 | merged | Update Step 1 sweep results: noul fix + AG News + corrected Banking77 |  |
| 2026-09-19 | #40 | merged | Add remaining Phase A loaders: Yelp, SWAG, MultiRC, MedNLI, ContractNLI, CodeSearchNet |  |
| 2026-09-19 | #41 | merged | Replace KV cache deepcopy with DynamicCache.crop() in LogitScorer |  |
| 2026-09-20 | #42 | merged | Add batched inference to LogitScorer: 39x speedup on Banking77 |  |
| 2026-09-20 | #43 | merged | Phase A sweep: 19 benchmarks × 10 models with score-type support |  |
| 2026-09-20 | #45 | merged | Add Jev API evaluation: 19-benchmark sweep with latency | #44 (closes) |
| 2026-09-20 | #48 | merged | Add GitHub Pages site for research findings | #46 (closes) |
| 2026-09-21 | #51 | merged | Support float score labels and Jev teacher distillation in loss functions | #23 (part of) |
| 2026-09-21 | #52 | merged | Add synthetic data generation pipeline (#50) | #50 (closes), #50 (title-ref) |

## P1 2026-09-22 to 2026-09-30 — Multi-task training, Qwen3.5, the web surface, JevBench

39 pull requests and 19 issues opened in this window.

Multi-task training becomes the main line (#56, #73): type-balanced joint
training, per-type learning rates and adaptive loss weighting (#95, #96), a
unified decision head (#98, #99, #105) and noul-to-choice augmentation (#104,
#107-#113). Qwen3.5 GDN support arrives (#62, #64). The web dashboard and the
playground take shape (#63, #65, #77-#92). JevBench is integrated (#66, #85,
#91, #92). Reasoning, sequence and long-context loaders expand the corpus
(#67-#70, #75). The data epic (#106) is opened, with mixing (#97) and compact
architectures (#58-#60) recorded as future work.

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-09-22 | #56 | closed | Multi-task training pipeline: type-balanced joint training across all benchmarks | #73 (parent), #19 (parent) |
| 2026-09-22 | #58 | closed | Future: compact and hybrid decision model architectures | #19 (parent) |
| 2026-09-22 | #59 | open | Future: compact decision model (<10M params) | #58 (parent) |
| 2026-09-22 | #60 | open | Future: hybrid encoder-decoder decision model | #58 (parent) |
| 2026-09-23 | #61 | closed | Experiment: Qwen3-Reranker as causal reranker backbone for decision heads | #19 (parent) |
| 2026-09-23 | #62 | closed | Sweep: Qwen3.5 Base models (0.8B/2B/4B) with GDN support | #19 (parent) |
| 2026-09-23 | #63 | closed | Web: interactive benchmark dashboard + live inference playground (GitHub Pages) |  |
| 2026-09-24 | #66 | closed | Integrate JevBench: community standard decision model benchmark |  |
| 2026-09-24 | #68 | closed | Synthetic data: seeded domains from reasoning benchmarks (MMLU, CSQA, LogiQA, ANLI) | #50 (parent) |
| 2026-09-24 | #70 | closed | Synthetic data: long context + sequential decision domains | #50 (parent) |
| 2026-09-24 | #73 | open | Epic: Training pipeline — supervised → multi-task → calibration → PEFT | #19 (parent) |
| 2026-09-24 | #74 | open | Literature review: LLM routing models and transferable calibration strategies |  |
| 2026-09-25 | #76 | open | CLM-8B: open-source System One model with contrastive training (Stanford/NVIDIA) |  |
| 2026-09-26 | #93 | closed | Experiment: MLP projector + per-component LR (frozen backbone done, LoRA pending) | #73 (parent) |
| 2026-09-26 | #95 | closed | Multi-task optimization: per-type LR, gradient balancing, adaptive loss weighting | #73 (parent) |
| 2026-09-26 | #97 | open | Data mixing: task/source ratio optimization for multi-task training | #73 (parent) |
| 2026-09-26 | #98 | closed | Architecture: unified decision head with shared attention layers | #73 (parent) |
| 2026-09-27 | #104 | closed | Noul-to-Choice augmentation: convert binary questions to multi-option with LLM/Jev-generated alternatives | #73 (parent) |
| 2026-09-27 | #106 | open | Epic: Data pipeline — synthetic generation, augmentation, and mixing | #19 (parent) |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-09-22 | #57 | merged | Multi-task training pipeline: type-balanced joint training (#56) | #56 (part of), #56 (title-ref) |
| 2026-09-23 | #64 | merged | Add Qwen3.5 Base model support (GDN hybrid architecture) (#62) | #62 (part of), #62 (title-ref) |
| 2026-09-23 | #65 | merged | Web dashboard Phase 1: results dashboard + landing page |  |
| 2026-09-24 | #67 | merged | Add 7 reasoning benchmarks: MMLU, WinoGrande, PIQA, CSQA, LogiQA, ANLI, BoolQ |  |
| 2026-09-24 | #69 | merged | Add reasoning benchmark loaders and synthetic domains (#68) | #68 (closes), #68 (title-ref) |
| 2026-09-24 | #71 | merged | Update project metadata and package config |  |
| 2026-09-24 | #72 | merged | Add bilingual READMEs for root, probing, model, and data |  |
| 2026-09-24 | #75 | merged | Add sequential decision + long context domains (#70) | #70 (closes), #70 (title-ref) |
| 2026-09-25 | #77 | merged | Web Phase 2: playground, deploy migration, rename to Krino |  |
| 2026-09-25 | #78 | merged | Add configurable head architecture for head-size sweep |  |
| 2026-09-25 | #79 | merged | Fix deploy workflow: add contents:write permission |  |
| 2026-09-25 | #80 | merged | Add Colab/Kaggle inference notebook and configurable playground |  |
| 2026-09-25 | #81 | merged | Fix playground Gradio API for v6.x two-step protocol |  |
| 2026-09-25 | #82 | merged | Fix notebook Gradio compat + add qwen3-reranker-0.6b model |  |
| 2026-09-25 | #83 | merged | Fix notebook: total_mem → total_memory |  |
| 2026-09-25 | #84 | merged | Fix notebook/spaces: validate options, handle empty criteria |  |
| 2026-09-25 | #85 | merged | Integrate JevBench: loader, eval script, and baseline results | #66 (closes) |
| 2026-09-25 | #86 | merged | Lazy model loading + multi-question presets |  |
| 2026-09-25 | #87 | merged | Fix playground SSE parsing for heartbeat events |  |
| 2026-09-25 | #88 | merged | Add Jev API comparison to playground + SVG theme icons |  |
| 2026-09-25 | #89 | merged | Add Kaggle notebook link to playground |  |
| 2026-09-25 | #90 | merged | Add remote server stop button to playground |  |
| 2026-09-25 | #91 | merged | Add raw reranker baseline eval for JevBench |  |
| 2026-09-25 | #92 | merged | Add JevBench results for Krino trained models to dashboard |  |
| 2026-09-26 | #94 | merged | Add GPU memory optimization with auto-detect and CLI overrides |  |
| 2026-09-26 | #96 | merged | Add per-type LR and uncertainty-based loss weighting for multi-task optimization |  |
| 2026-09-26 | #99 | merged | Add shared attention for choice+score heads (issue #98) | #98 (closes), #98 (title-ref) |
| 2026-09-26 | #100 | merged | Add temperature-based source sampling (mT5/PaLM style) | #97 (part of) |
| 2026-09-26 | #101 | merged | Add difficulty-proportional type ratio override | #97 (part of) |
| 2026-09-26 | #102 | closed | Add synthetic data replay ratio support | #97 (part of) |
| 2026-09-26 | #103 | merged | Add synthetic data replay ratio support | #97 (part of) |
| 2026-09-27 | #105 | merged | Add unified decision head: all types as score problems (issue #98 Option A) | #98 (closes), #98 (title-ref) |
| 2026-09-27 | #107 | merged | Add noul-augment pipeline stage (#104) | #104 (closes), #104 (title-ref) |
| 2026-09-27 | #108 | merged | Fix teacher_probs quality: normalization + neutral templates + Luna gap filler |  |
| 2026-09-27 | #109 | merged | Improve LLM teacher labeling prompts for calibration |  |
| 2026-09-27 | #110 | merged | Add LLM-generated question-specific confounders for noul augmentation |  |
| 2026-09-29 | #111 | merged | Add LLM confounder soft labels for augmented noul questions |  |
| 2026-09-29 | #112 | merged | Add LLM augmented noul labeling script + include labels in HF sync |  |
| 2026-09-30 | #113 | merged | Route augmented noul through choice head for training (Phase C) |  |

## P2 2026-10-01 to 2026-10-26 — Pause: Clef, data fixes, label taxonomy

10 pull requests and 3 issues opened in this window.

A quieter window. Clef is analysed for its training recipe (#114). The
synthetic split moves to the family level (#117) after #120 finds near-duplicate
states leaking across train and test; the synthetic source is loaded on the fly
(#115, #116), MMLU training rows are included (#119, #122), and the mT5
temperature reciprocal is corrected (#121, #125, #128). The Reranker-4B head
sweep is corrected for the learning-rate confound (#124). The shared label
taxonomy lands (#126).

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-10-13 | #120 | closed | Synthetic corpus: near-duplicate states leak across the train/test split | #117 (follow-up to), #106 (parent) |
| 2026-10-13 | #121 | closed | sampling_temperature is the reciprocal of the mT5 alpha it cites, so the temperature sweep tested the wrong regime | #97 (parent) |
| 2026-10-14 | #125 | closed | Log the effective LR (and MLP settings) at training startup |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-10-12 | #114 | merged | Add Clef analysis: architecture and training recipe comparison |  |
| 2026-10-13 | #115 | merged | Load synthetic source on the fly when no path is configured |  |
| 2026-10-13 | #116 | merged | Address review comments on synthetic source loading | #115 (follow-up to) |
| 2026-10-13 | #117 | merged | Split synthetic data into train/test at the family level |  |
| 2026-10-13 | #118 | merged | Add script to materialize benchmark jsonl from HF parquet |  |
| 2026-10-13 | #119 | merged | Load MMLU training split so the source contributes training items |  |
| 2026-10-13 | #122 | merged | Cap mmlu at 5000 train items in the R2 configs | #119 (follow-up to) |
| 2026-10-13 | #123 | merged | Bump the npm_and_yarn group across 1 directory with 3 updates |  |
| 2026-10-14 | #124 | merged | Correct the Reranker-4B head sweep for the LR confound |  |
| 2026-10-18 | #126 | merged | chore: add .github/labels.yml (shared label taxonomy) |  |

## P3 2026-10-27 to 2026-10-31 — Round 2: LoRA, score, evaluation, calibration, leakage

60 pull requests and 19 issues opened in this window.

Round 2. LoRA and a multi-layer readout enter the model (#127); score-first
conversion (#129, #130) and the RPS loss (#132) make the score primitive
trainable, with RPS and MAE reported per epoch (#134). The score supply widens
(#157, #158) after #142 finds it is the least-supplied type. Choice accuracy
collapses with option count (#166), so option-count augmentation (#168) and
choice-balanced caps (#164) follow. Evaluation gains calibration and
option-order robustness (#167), the source/type breakdown (#165), the JevBench
OOD gap (#159), the latency benchmark (#160) and the public-tier power note
(#202). The leakage audit (#183) finds the STS-B withheld-label sentinel (#176,
#177). Two readout hybrids are tested: the native-score plus head blend (#186)
and the cross-encoder head (#192, #196, #198, #199). Notes 08-19 record these
results.

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-10-28 | #140 | closed | Synthetic corpus: duplicated _family_idx collides item ids across different families |  |
| 2026-10-28 | #141 | closed | Configured sources that resolve to zero items only warn; a worktree silently trains on 10k of 182k items |  |
| 2026-10-28 | #142 | closed | Score is the least-supplied primitive: 27k of 167k available, and each score item is resampled ~4x more often |  |
| 2026-10-28 | #143 | closed | Measure calibration and option-order robustness; nothing currently reports ECE, Brier, QWK or order sensitivity |  |
| 2026-10-28 | #144 | closed | Option-marker readout: score each candidate at its own marker, built for up to 256 options |  |
| 2026-10-28 | #145 | closed | Cumulative-link score head vs softmax: does rank-consistency help the ordinal primitive? |  |
| 2026-10-28 | #146 | closed | Do multiple representations help? Layer-index sweep, two-layer fusion, and 0.6B+0.6B / 4B+0.6B readouts |  |
| 2026-10-28 | #147 | closed | load_drop emits constant labels and prints the gold answer in the instruction |  |
| 2026-10-28 | #151 | open | Add a bounded educational-quality score loader (FineWeb-Edu / DCLM) |  |
| 2026-10-28 | #152 | open | Wire Open-Jev's CC0 score items into the score pool |  |
| 2026-10-28 | #153 | open | Add ASAP 2.0 essay scores as a non-commercial ordinal source |  |
| 2026-10-28 | #154 | open | Add DICES five-point safety severity as a score source |  |
| 2026-10-28 | #155 | open | Add TREC-COVID graded relevance (0-2) as a score source |  |
| 2026-10-28 | #156 | open | Generate score questions over existing states and label them with Jev |  |
| 2026-10-28 | #161 | open | 8B LoRA run grows to 89 GiB of device memory and OOMs on its fifth epoch |  |
| 2026-10-28 | #166 | closed | Choice accuracy collapses with option count: -46 pp from 4 to 256 options |  |
| 2026-10-28 | #172 | closed | Score calibration fails at 4B, and the error is concentrated in the 6-level source |  |
| 2026-10-28 | #176 | closed | stsb's withheld test labels enter the eval set as gold level 0 |  |
| 2026-10-28 | #178 | closed | The goal's 'Brier >= 0.85 on the score primitive' has no agreed definition, and fails under every reading |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-10-27 | #127 | merged | Add LoRA adapters and multi-layer readout to the decision model |  |
| 2026-10-27 | #128 | merged | Log the effective settings and their source at startup | #125 (closes) |
| 2026-10-27 | #129 | merged | Add score-first conversion for typed questions |  |
| 2026-10-27 | #130 | merged | Add a score-form training mode |  |
| 2026-10-27 | #131 | merged | Add the RL training design document |  |
| 2026-10-27 | #132 | merged | Add an RPS loss for the score primitive |  |
| 2026-10-27 | #133 | merged | Add a sweep report that renders training logs as a table |  |
| 2026-10-27 | #134 | merged | Report RPS and MAE for score items during evaluation |  |
| 2026-10-27 | #135 | merged | Fix the two NameErrors that made train() unusable |  |
| 2026-10-27 | #136 | merged | Honour an explicit --no-gradient-checkpointing when LoRA is on |  |
| 2026-10-27 | #137 | merged | Report the type distribution the sampler actually uses |  |
| 2026-10-28 | #138 | merged | Split synthetic data by near-duplicate cluster, not by family | #120 (closes) |
| 2026-10-28 | #139 | merged | Train across GPUs by averaging gradients explicitly |  |
| 2026-10-28 | #148 | closed | Report per-type pool resampling and widen the score supply | base `fix/type-distribution-reporting`, #137 (stacked on) |
| 2026-10-28 | #149 | merged | Fail the run when a configured source loads nothing | #141 (closes) |
| 2026-10-28 | #150 | merged | Consolidate the round-2 architecture sweep |  |
| 2026-10-28 | #157 | closed | Expand the score data supply with three ordinal sources |  |
| 2026-10-28 | #158 | merged | Expand the score data supply with three ordinal sources | #142 (closes), #148 (stacked on), #137 (stacked on) |
| 2026-10-28 | #159 | merged | Add an eval-only JevBench config and measure the OOD gap |  |
| 2026-10-28 | #160 | merged | Add a latency benchmark and measure the many-option regime |  |
| 2026-10-28 | #162 | merged | Generate deterministic eval variants for robustness probing |  |
| 2026-10-28 | #163 | merged | Log device memory at every epoch boundary | #161 (part of) |
| 2026-10-28 | #164 | merged | Add a choice-balanced config that widens the choice caps |  |
| 2026-10-28 | #165 | merged | Split the eval breakdown by source and question type |  |
| 2026-10-28 | #167 | merged | Add calibration and option-order robustness evaluation |  |
| 2026-10-28 | #168 | merged | Add option-count augmentation for choice training |  |
| 2026-10-28 | #169 | merged | Keep only the last LM-head logits in the causal backbone forward |  |
| 2026-10-28 | #170 | merged | Let an empty epoch shard to nothing, and state the shard duplication budget |  |
| 2026-10-28 | #171 | merged | Record the ordinal head x loss 2x2: the plain baseline wins on accuracy |  |
| 2026-10-28 | #173 | merged | Bump next from 16.3.8 to 16.4.1 in /web in the npm_and_yarn group across 1 directory |  |
| 2026-10-28 | #174 | closed | Report a per-level-count temperature beside the per-type one |  |
| 2026-10-28 | #175 | merged | Stop tracking the cluster cache, committed one version out of date |  |
| 2026-10-28 | #177 | merged | Drop STS-B rows whose label is the withheld-label sentinel | #176 (closes) |
| 2026-10-28 | #179 | merged | Measure score calibration on both 4B arms: a temperature clears the bar on one and misses it on the other |  |
| 2026-10-28 | #180 | merged | Report JevBench-public: the trained heads lose 14-21 points to our own readout |  |
| 2026-10-28 | #181 | merged | Measure goal item 5: the 0.6B meets the latency bar only compiled, and no model meets both halves |  |
| 2026-10-29 | #182 | merged | Report goal clause 3: option-order flip rate and option-count robustness |  |
| 2026-10-29 | #183 | merged | Audit the benchmark corpus for train/eval leakage, and find the STS-B sentinel on the eval side |  |
| 2026-10-29 | #184 | merged | Report the schedule and readout levers: 69.24% for the heads, 73.01% routed, and a matched option-count curve |  |
| 2026-10-29 | #185 | closed | Clause 2's QWK is not post-hoc fixable, and its worst scale was a label artefact |  |
| 2026-10-29 | #186 | merged | Blend the native reranker score with the trained head: +1.4 and +1.2 pp on two seeds |  |
| 2026-10-29 | #187 | merged | Warn when a long schedule is combined with the readout that has collapsed twice |  |
| 2026-10-29 | #188 | merged | Decide the readout from the checkpoint, not the config |  |
| 2026-10-29 | #189 | closed | The routed system's ECE passes only when each path gets its own temperature |  |
| 2026-10-29 | #190 | merged | Give each dumped choice record its source, so a dump can be decomposed |  |
| 2026-10-29 | #191 | merged | Add the evaluation bridge a score-first checkpoint needs |  |
| 2026-10-29 | #192 | merged | Score choice options with a cross-encoder head |  |
| 2026-10-29 | #193 | merged | Coerce structured states instead of crashing the cross-encoder prompt fit |  |
| 2026-10-29 | #194 | closed | Time the choice head the checkpoint actually has |  |
| 2026-10-29 | #195 | closed | Measure every option's suffix in one tokenizer call | base `fix/choice-prompts-nonstr-state`, #193 (stacked on) |
| 2026-10-30 | #196 | closed | research/21: what the cross-encoder head buys, and where the hard tier cannot tell |  |
| 2026-10-30 | #197 | merged | research/13: the long-state wall, measured instead of hypothesised |  |
| 2026-10-30 | #198 | merged | Measure every option's suffix in one tokenizer call |  |
| 2026-10-30 | #199 | closed | Padding costs the tower 24 points and the cross-encoder 1: dilution is a read-out defect |  |
| 2026-10-30 | #200 | merged | Let the option-count sweep run on either choice-head shape |  |
| 2026-10-30 | #201 | merged | Refuse to diff two calibration artifacts measured on different slices |  |
| 2026-10-30 | #202 | merged | research/23: the public tier can see a deficit but not a gain |  |
| 2026-10-30 | #203 | merged | data: a deterministic authored held-out set of 255 paired items |  |
| 2026-10-30 | #204 | closed | k_augment: do not pad around a document-length option | base `data/authored-holdout` |
| 2026-10-30 | #205 | merged | load_heads: refuse a checkpoint from another head architecture |  |

## P4 2026-11-01 to 2026-11-05 — v3 lands, v4 starts

21 pull requests and 2 issues opened in this window.

v3 lands in one PR (#211) after Qwen3.5 text and multimodal loaders (#209,
#210). The corpus moves to HuggingFace behind a pinned lock (#213, #217);
long-context, tool-call and noul label-repair builders (#212, #214) and
MMLU-Pro/RACE/LongBench/2Wiki adapters (#215) widen it. v4 starts: shared-prefix
scoring through hybrid caches (#218), the Qwen3.5-4B control (#220), marker
scoring through forked caches (#221), the listwise set head (#222), shape
bucketing (#224) and the two-stage pairwise head (#228). The split seed is
separated from the training seed (#227). Two v4 gates are recorded as
inconclusive (#219, #226).

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-04 | #207 | open | Epic: Training tasks and dataset crafting |  |
| 2026-11-04 | #208 | closed | Track Qwen3.5 text and multimodal loader support |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-04 | #206 | merged | Vendor zerodep yaml and drop the PyYAML import |  |
| 2026-11-04 | #209 | merged | Add Qwen3.5 text and multimodal loader support | base `feature/marker-encoder-readout` |
| 2026-11-04 | #210 | merged | eval_gate: build the raw reranker for --raw-reranker | base `feature/marker-encoder-readout` |
| 2026-11-04 | #211 | merged | Land Krino v3: marker readout, residual line, 133 ms path, Qwen3.5 loaders, closeout |  |
| 2026-11-04 | #212 | merged | Add long-context, tool-call and noul label-repair data builders |  |
| 2026-11-04 | #213 | merged | Move data to HuggingFace with a pinned lock and sync helper |  |
| 2026-11-04 | #214 | merged | Add tool-call argument and after-result decisions |  |
| 2026-11-04 | #215 | merged | Add MMLU-Pro, RACE, LongBench and 2Wiki choice adapters |  |
| 2026-11-04 | #216 | merged | Pin the six new pilot datasets in the HF lock | #213 (follow-up to) |
| 2026-11-04 | #217 | merged | Untrack the synthetic corpus and pin it in the HF lock |  |
| 2026-11-04 | #218 | merged | Port shared-prefix scoring to Qwen3.5 hybrid caches (v4 step 1) |  |
| 2026-11-04 | #219 | merged | Record the union-arm gate with in-tree evidence |  |
| 2026-11-05 | #220 | merged | Register the Qwen3.5-4B v4 control and record the top-m recall diagnostic |  |
| 2026-11-05 | #221 | merged | Score the option marker through forked caches on Qwen3.5 | #218 (follow-up to) |
| 2026-11-05 | #222 | merged | Add the listwise set head and its training and eval paths |  |
| 2026-11-05 | #223 | merged | Register the three v4 head arms |  |
| 2026-11-05 | #224 | merged | Bucket the listwise shapes so the kernels stay warm | #222 (follow-up to) |
| 2026-11-05 | #225 | merged | Load JevBench rows through load_jevbench in eval_listwise | #222 (follow-up to) |
| 2026-11-05 | #226 | merged | Record the v4 listwise s42 gate (inconclusive) |  |
| 2026-11-05 | #227 | merged | Separate the split seed from the training seed | #222 (follow-up to) |
| 2026-11-05 | #228 | merged | Add the two-stage pairwise head and its training and eval paths |  |

## P5 2026-11-06 to 2026-11-08 — Canonical corpus, v4 infrastructure, zero-shot baselines, readout matrix

56 pull requests and 37 issues opened in this window.

The corpus becomes the centre. Epics #229-#232 open. #234 inventories the
corpus and defines the canonical set (#241); #235 audits duplicates and JevBench
contamination (#246); #247 labels abilities and reports coverage (#255); #236
reconnoitres the Decision Index and fixes the matched held-out spec (#239,
#256); #237 builds unanswerable synthesis and the abstention instrument (#243,
#262). The canonical artifact carries unique ids, provenance and a hash (#249),
and the label contract is enforced (#260). The v4 trainers gain LoRA and a
default-off base KL (#238); the adapter folds into the shared-prefix path (#250);
throughput is recovered (#248); noul trains through the listwise path (#264,
#271); the answerable mask is enforced (#253) with mutation checks (#261).
Zero-shot logit baselines (#277, #278, #284, #285, #287), the judgement readout
(#292, #293), the reranker ladder (#307, #316, #317) and the 828-row development
set (#298, #300-#303) are measured. The gate policy is adopted (#320), the
base-conditioned arm is recorded as failed (#272), and the architecture figure
is redrawn for v4 (#267).

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-06 | #229 | open | Epic: Canonical corpus and label verification |  |
| 2026-11-06 | #230 | open | Epic: Backbone adaptation ladder — LoRA, then RL, then full SFT |  |
| 2026-11-06 | #231 | open | Epic: Benchmark expansion — Decision Index, held-out tracking, disclosure |  |
| 2026-11-06 | #232 | open | Epic: Answerable head and unanswerable data |  |
| 2026-11-06 | #233 | closed | LoRA and KL in the v4 trainers |  |
| 2026-11-06 | #234 | closed | Corpus inventory and canonical corpus definition |  |
| 2026-11-06 | #235 | closed | Dedup and JevBench contamination audit |  |
| 2026-11-06 | #236 | open | Decision Index reconnaissance and matched held-out spec |  |
| 2026-11-06 | #237 | open | Unanswerable synthesis and answerable head integration |  |
| 2026-11-06 | #244 | closed | GDN training throughput: recover the 10x lost to gradient re-encoding |  |
| 2026-11-06 | #245 | open | Adapter inference: keep the 89 ms path with a LoRA merged into the weights |  |
| 2026-11-06 | #247 | closed | Ability labels for the canonical corpus, and the coverage map they produce | #207 (part of) |
| 2026-11-06 | #251 | open | The listwise trainer excludes noul, dropping 113,594 canonical records from training |  |
| 2026-11-06 | #254 | open | The listwise trainer has no answerable mask and is safe on the canonical artifact only by accident |  |
| 2026-11-06 | #257 | closed | Enforce a mutation check for mask and exclusion tests in CI |  |
| 2026-11-07 | #263 | closed | JevBench admissibility: replace the point non-regression clause with a pre-specified non-inferiority margin |  |
| 2026-11-07 | #265 | closed | CI runs no tests on pull requests whose base is not main |  |
| 2026-11-08 | #273 | open | The hybrid backbone selects the fused CUDA kernel by package presence, not by input device |  |
| 2026-11-08 | #276 | open | The residual-arm register's status field has 31 values and cannot be queried |  |
| 2026-11-08 | #277 | closed | Zero-shot logit baseline for the 3-4B bidirectional-shortlist models |  |
| 2026-11-08 | #278 | closed | Screen the 3-4B shortlist on JevBench public (231 items), zero-shot |  |
| 2026-11-08 | #281 | open | The provenance block records no licence, so a non-commercial checkpoint reads as production-ready |  |
| 2026-11-08 | #282 | open | code_commit is resolved at write time, so a run can record a commit it never imported |  |
| 2026-11-08 | #288 | open | JevBench original tier: 24 items are a constant answer for every checkpoint tested |  |
| 2026-11-08 | #292 | closed | Screen the 3-4B shortlist on JevBench under the judgement readout |  |
| 2026-11-08 | #295 | closed | Does the 61.90-64.94 judgement band survive the 2507 releases? |  |
| 2026-11-08 | #298 | closed | Build an 800-item judgement-format development set and validate it against JevBench |  |
| 2026-11-08 | #299 | open | Feed the canonical corpus to the judgement objective |  |
| 2026-11-08 | #305 | closed | Does reranker training generalise across recipes? zerank-2 on the same Qwen3-4B base |  |
| 2026-11-08 | #306 | open | Wider reranker sweep: bge-minicpm-layerwise, Qwen3-VL-Reranker-2B, jina-reranker-m0 |  |
| 2026-11-08 | #307 | closed | Reranker size ladder 0.6B / 4B / 8B: is the 62-65% judgement band a capacity limit or a readout limit? |  |
| 2026-11-08 | #308 | closed | Instrument resolution: quote the detectable delta, and move every 1-3 pp claim to the held-out |  |
| 2026-11-08 | #310 | closed | The judgement readout ranks by log P(yes); research/35 and the provenance say log P(yes) - log P(no) |  |
| 2026-11-08 | #311 | closed | Dashboard publishes a mixed-readout JevBench table, mislabels zero-shot rows as trained heads, and cannot see logit_prov |  |
| 2026-11-08 | #314 | closed | Label trained heads with a project generation and record what produced each head result |  |
| 2026-11-08 | #318 | closed | bf16 collapses the polarity branch's yes/no gap on about 2% of dev rows and the tie answers no |  |
| 2026-11-08 | #319 | closed | Lead the landing page with what the project established, not with the Banking77 number |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-06 | #238 | merged | Add LoRA and a default-off base KL to the v4 trainers | #230 (part of) |
| 2026-11-06 | #239 | merged | Decision Index recon, v4 control adapter, and matched held-out spec (#236) | #236 (title-ref) |
| 2026-11-06 | #240 | closed | Withdrawn: documentation change not taken forward |  |
| 2026-11-06 | #241 | merged | Inventory the corpus on both machines and define the canonical set (#234) | #234 (closes), #234 (title-ref) |
| 2026-11-06 | #242 | merged | Data card, RACE correction, and source licence audit (#229 / #234) | #229 (title-ref), #234 (title-ref) |
| 2026-11-06 | #243 | merged | Add paired unanswerable synthesis for the answerable head (#237) | #237 (title-ref) |
| 2026-11-06 | #246 | merged | Audit the canonical corpus for duplicates and JevBench contamination (#235) | #235 (closes), #235 (title-ref) |
| 2026-11-06 | #248 | merged | Recover LoRA training throughput: skip the discarded prefix pass and keep fixed shapes |  |
| 2026-11-06 | #249 | merged | Emit the canonical corpus with unique ids, provenance, and a hash |  |
| 2026-11-06 | #250 | merged | Fold the LoRA adapter into the shared-prefix backbone (#245) | #245 (title-ref) |
| 2026-11-06 | #252 | merged | Record the nine-arm LoRA/KL scan gate outcomes |  |
| 2026-11-06 | #253 | merged | Mask the task loss on unanswerable rows; census the canonical corpus |  |
| 2026-11-06 | #255 | merged | Label the canonical corpus with the ten product abilities (#247) | #247 (closes), #247 (title-ref) |
| 2026-11-06 | #256 | merged | Record the corrected Decision Index suite, the held-out limit, and the freeze retrospective | #239 (follow-up to) |
| 2026-11-06 | #258 | merged | Generate score questions over existing states and report the #156 pilot | #156 (title-ref) |
| 2026-11-06 | #259 | merged | Score generation round two: state-carried rubrics and a level-0 cap | #258 (stacked on) |
| 2026-11-06 | #260 | merged | Annotate label kinds and enforce the label contract | #249 (stacked on) |
| 2026-11-06 | #261 | merged | Add mutation checks for masked and excluded paths (#257) | #257 (title-ref) |
| 2026-11-06 | #262 | merged | Build an abstention evaluation instrument (#237) | #237 (title-ref) |
| 2026-11-07 | #264 | merged | Train noul through the listwise path; record arm provenance |  |
| 2026-11-07 | #266 | merged | Run CI on pull requests into any base branch | #265 (closes) |
| 2026-11-07 | #267 | merged | Redraw the architecture figure for v4 |  |
| 2026-11-07 | #268 | merged | Give the listwise head the base distribution as per-option features |  |
| 2026-11-07 | #269 | merged | Land the canonical held-out definition and listwise training base |  |
| 2026-11-07 | #270 | merged | Apply --torch-seed in the listwise trainer | base `feat/canonical-support` |
| 2026-11-08 | #271 | merged | Score noul through the readout in listwise validation |  |
| 2026-11-08 | #272 | merged | Record the base-conditioned arm as failed at seed 0 |  |
| 2026-11-08 | #274 | merged | Append a correction for the unresolvable preregistration SHA |  |
| 2026-11-08 | #275 | merged | Commit the JevBench base probabilities for future gating |  |
| 2026-11-08 | #279 | merged | Embed run provenance in the logit baseline output |  |
| 2026-11-08 | #280 | closed | Embed provenance in logit eval output and pin the model revision | #277 (part of) |
| 2026-11-08 | #283 | merged | Redact absolute provenance paths before a result file is committed |  |
| 2026-11-08 | #284 | merged | Add zero-shot logit baselines for three checkpoints (#277) | #277 (closes), #277 (title-ref) |
| 2026-11-08 | #285 | merged | Score JevBench-public with a zero-shot logit readout (#278) | #278 (closes), #278 (title-ref) |
| 2026-11-08 | #286 | merged | Score a GDN hybrid through a cache it can hold |  |
| 2026-11-08 | #287 | merged | Commit zero-shot logit baselines with provenance (#277, #278) | #277 (title-ref), #278 (title-ref) |
| 2026-11-08 | #289 | merged | Record how little the top two JevBench rows differ |  |
| 2026-11-08 | #290 | merged | Widen the JevBench dead-cell finding to eight checkpoints, with the 195-question denominator | #277 (follow-up to) |
| 2026-11-08 | #291 | merged | Record the GDN kernel path and merge sharded runs safely |  |
| 2026-11-08 | #293 | merged | Add the judgement readout, the four-checkpoint rectangle and the base track (#292) | #292 (title-ref) |
| 2026-11-08 | #294 | merged | Document the judgement files and the commits they record |  |
| 2026-11-08 | #296 | merged | Record the two 2507 4B checkpoints and the provenanced Qwen3-4B control |  |
| 2026-11-08 | #297 | merged | Correct research/35 for multiplicity |  |
| 2026-11-08 | #300 | merged | Add the development-set sampler and loader (#298) | #298 (title-ref) |
| 2026-11-08 | #301 | merged | Let the dev-set sampler take targets, skip the authored block and cap clusters |  |
| 2026-11-08 | #302 | merged | Commit the 828-row development set and its manifest |  |
| 2026-11-08 | #303 | merged | Let the continuation eval load the development set too |  |
| 2026-11-08 | #304 | merged | Answer a criteria-less noul item at Answer: instead of judging the word yes |  |
| 2026-11-08 | #309 | merged | Record the four zerank-2 cells, judgement and continuation |  |
| 2026-11-08 | #312 | merged | State that the judgement readout ranks on log P(" yes") (#310) | #310 (closes), #310 (title-ref) |
| 2026-11-08 | #313 | merged | Label JevBench readouts and guard dashboard provenance (#311) | #311 (part of), #311 (title-ref) |
| 2026-11-08 | #315 | merged | Record trained-head generations and run identity (#314) | #314 (part of), #314 (title-ref) |
| 2026-11-08 | #316 | merged | Add the reranker ladder dev cells (#307) | #307 (title-ref) |
| 2026-11-08 | #317 | merged | Reranker size ladder on JevBench: 0.6B / 4B / 8B (#307) | #307 (part of), #307 (title-ref) |
| 2026-11-08 | #320 | merged | Adopt the gate policy: instruments have roles, the gate suggests |  |
| 2026-11-08 | #321 | merged | Lead the landing page with the project's findings (#319) | #319 (closes), #319 (title-ref) |

## P6 2026-11-09 to 2026-11-10 — Readout science: judgement, panel 2, code space, blocking, position

22 pull requests and 19 issues opened in this window.

Readout science. The judgement architecture (#324, #325) separates the
pretrained yes/no projection, the reading direction and a trained head. Panel 2
becomes a measured readout: zero-shot vocabulary codes (#326, #328, #330, #333),
calibration (#338, #344), prompt order (#336, #343), multi-layer reads (#337,
#345), content-addressed codes (#339, #346), the set-function head (#340, #349),
attention mass (#341, #351), the cascade (#342, #353) and order-invariant ties
(#347, #348). The code space is measured and closed: no tier extends the
52-letter alphabet (#354, #355), the string space is counted and rejected (#350,
#358), blocking is measured at K=150 (#356, #359) and the long-list tail is
decomposed (#357, #360). H2O-Lightning is scored on our instruments (#327,
#332) and the capacity ladder is recorded (#329, #331, #334).

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-09 | #324 | open | Judgement architecture: causal vs bidirectional readout, with and without a trained yes/no head |  |
| 2026-11-09 | #326 | closed | Panel 2 zero-shot: vocabulary code readout across four Qwen3-4B checkpoints |  |
| 2026-11-09 | #327 | closed | H2O-Lightning-4B on our instruments: fidelity, two-way contrast, and panel 2 latency |  |
| 2026-11-09 | #329 | closed | Readout x capacity: continuation, judgement and panel 2 across the Qwen3-Reranker ladder |  |
| 2026-11-09 | #335 | closed | Epic: readout designs beyond the six-panel taxonomy |  |
| 2026-11-09 | #336 | closed | A: prompt order as a readout parameter (#335) | #335 (part of) |
| 2026-11-09 | #337 | closed | G: multi-layer read for panel 2 (#335) | #335 (part of) |
| 2026-11-09 | #338 | closed | C: vocabulary-anchored calibration for panel 2 (#335) | #335 (part of) |
| 2026-11-09 | #339 | closed | B: content-addressed codes against panel 2's order sensitivity (#335) | #335 (part of) |
| 2026-11-09 | #340 | closed | D: set-function head over frozen option states (#335) | #335 (part of) |
| 2026-11-09 | #341 | closed | E: attention mass as a free set-level signal (#335) | #335 (part of) |
| 2026-11-09 | #342 | closed | F: cascade — set read, then re-read the close calls (#335) | #335 (part of) |
| 2026-11-09 | #347 | closed | Adopt order-invariant tie-breaking for panel-2, and count exact ties per cell | #335 (part of) |
| 2026-11-09 | #350 | open | Panel 2 silently drops items with more than 26 options; unify on the 84-symbol alphabet |  |
| 2026-11-09 | #354 | closed | Which code tiers anchor an answer: a paired alphabet test (#350) |  |
| 2026-11-10 | #356 | closed | Step 4: blocking readout for more than 52 options (#350) |  |
| 2026-11-10 | #357 | closed | Panel 2 ignores the tail of a long option list, on one dataset and not the other |  |
| 2026-11-10 | #361 | open | A v3 residual run whose name and metadata disagree |  |
| 2026-11-10 | #362 | open | Marker-token ablation: six trained heads |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-09 | #322 | merged | Gate #263: three-state interval screen replaces the point non-regression clause | #263 (closes), #263 (title-ref) |
| 2026-11-09 | #323 | merged | Put the gate margin on the scale the screen measures: 12 pp of cc, not 8 pp of raw |  |
| 2026-11-09 | #325 | merged | Judgement architecture: readout shape, direction and a trained head (#324) | #324 (part of), #324 (title-ref) |
| 2026-11-09 | #328 | merged | Panel 2 zero-shot: one-forward vocabulary-code readout on JevBench (#326) | #326 (title-ref) |
| 2026-11-09 | #330 | merged | Show the vocabulary-code readout on the dashboard (#326) | #326 (title-ref) |
| 2026-11-09 | #331 | merged | Readout by capacity: reranker and general lines on JevBench (#329) | #329 (title-ref) |
| 2026-11-09 | #332 | merged | Record H2O-Lightning-4B on our instruments (#327) | #327 (closes), #327 (title-ref) |
| 2026-11-09 | #333 | merged | Panel 2 against judgement on dev: sign agreement, size unresolved (#326) | #326 (title-ref) |
| 2026-11-09 | #334 | merged | Render the capacity ladder on the dashboard (#329) | #329 (title-ref) |
| 2026-11-09 | #343 | merged | Declare and measure the panel 2 prompt order (#336) | base `feature/326-dev-panel2`, #336 (closes), #335 (part of), #336 (title-ref) |
| 2026-11-09 | #344 | merged | Panel-2 calibration: per-type temperature on dev, evaluated on JevBench (#338) | #338 (closes), #338 (title-ref) |
| 2026-11-09 | #345 | merged | Declare and select among panel 2 read layers (#337) | base `feature/336-prompt-order`, #337 (closes), #335 (part of), #337 (title-ref) |
| 2026-11-09 | #346 | merged | Panel-2 order arms: content-addressed codes and a seeded mixture on JevBench (#339) | #339 (closes), #339 (title-ref) |
| 2026-11-09 | #348 | merged | Adopt natural-index tie-breaking and count exact ties per panel-2 cell (#347) | #347 (closes), #347 (title-ref) |
| 2026-11-09 | #349 | merged | Run arm D: a set-function head over frozen option states (#340) | #335 (part of), #340 (title-ref) |
| 2026-11-09 | #351 | merged | Read attention mass over option spans (#341) | #341 (closes), #335 (part of), #341 (title-ref) |
| 2026-11-09 | #352 | merged | Measure panel 2 under a content-addressed option order |  |
| 2026-11-09 | #353 | merged | Add the cascade rule and its dev-selected null (#342) | #342 (title-ref) |
| 2026-11-09 | #355 | merged | Measure which code tiers anchor an answer: none does (#354) | #354 (title-ref) |
| 2026-11-10 | #358 | merged | Count the string code space and reject the lexical tiers (#350, #354) | #350 (title-ref), #354 (title-ref) |
| 2026-11-10 | #359 | merged | Blocking readout frontier at K = 150 (#356) | #356 (title-ref) |
| 2026-11-10 | #360 | merged | Measure where the long-list tail comes from (#357) | #357 (title-ref) |

## P7 2026-11-11 to 2026-11-15 — Consolidation and the readout-training line

11 pull requests and 1 issues opened in this window.

Consolidation. JevBench v1.7.3 and the new readout instruments are published
(#363); canonical corpus v2 freezes the per-row near-duplicate policy (#364);
the panel-2 prompt-order and multi-layer arms are ported onto main (#365, #366).
The judgement objective trains on the frozen canonical corpus (#370).
Marker-token ablation (#362, #372) and the residual marker guard (#373, #374)
close the marker line. Three figures land: the long-list frontier (#367), the
architecture layout fix (#368) and the readout taxonomy (#369, #371).

### Issues opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-15 | #373 | closed | residual.pt does not store marker_token, so evaluation silently uses eos |  |

### Pull requests opened

| date | # | state | title | relations |
|---|---:|---|---|---|
| 2026-11-15 | #363 | merged | Publish the v1.7.3 JevBench reference and the new readout instruments |  |
| 2026-11-15 | #364 | merged | Freeze canonical corpus v2 with per-row near-duplicate removal |  |
| 2026-11-15 | #365 | merged | Port the panel-2 prompt-order and multi-layer-read arms onto main (#336, #337) | #336 (title-ref), #337 (title-ref) |
| 2026-11-15 | #366 | merged | Label prompt-order variants on the dashboard and pin the landing to the shipped cell |  |
| 2026-11-15 | #367 | merged | Add the long-list readout figure and its caption |  |
| 2026-11-15 | #368 | merged | Fix the architecture figure's text layout |  |
| 2026-11-15 | #369 | merged | Add the readout taxonomy figure |  |
| 2026-11-15 | #370 | merged | Train the judgement objective on the frozen canonical corpus |  |
| 2026-11-15 | #371 | merged | Fix the readout taxonomy's arrow routing |  |
| 2026-11-15 | #372 | merged | Marker-token ablation: five non-default markers unresolved (#362) | #362 (title-ref) |
| 2026-11-15 | #374 | merged | Record the option marker in residual.pt and guard its absence (#373) | #373 (closes), #373 (title-ref) |

## Dependency spine

The explicit parent and close links form a small number of chains. Read a chain
from its epic down; the research notes that report each link are named at the
end.

- **#1 probing epic** -> probes #2-#13 -> `research/02-architecture-analysis.md`.
- **#19 replication epic** -> steps #20-#24 -> #25, #26, #28, #31, #37 ->
  `research/04-model-replication.md`; training epic #73 -> #93, #95, #97, #98,
  #104, #106 -> `research/05`-`07`; backbone sweep #27 -> #33.
- **#50 synthetic pipeline** -> #68, #70 -> #69, #75, #107 -> `research/04`,
  and the data epic #106 -> #120, #121.
- **#229 canonical corpus** -> #234 -> #241 -> `research/25`, `research/29`;
  #235 -> #246 -> `research/235`, then `research/48`; #247 -> #255 ->
  `research/247`; #236 -> #239, #256 -> `research/26`-`28`, `research/31`
  (suite freeze); #237 -> #243, #262 -> `research/09-answerable-head-integration`,
  `research/30-abstention-eval-instrument`.
- **#230 adaptation ladder** -> #233 -> #238; #245 -> #250; #251 -> #264, #271;
  #254 -> #253.
- **#231 benchmark expansion** -> #236, #298, #307 -> #239, #300-#303, #316,
  #317.
- **#232 answerable head** -> #237 -> #243, #262.
- **#324 judgement architecture** -> #325 -> `research/38`.
- **#326 panel 2 zero-shot** -> #328, #330, #333 -> `research/39`, `research/46`.
- **#335 readout-design epic** -> #336-#342 -> #343-#353, #365 ->
  `research/39`-`42`.
- **#350 panel-2 alphabet** -> #354, #356, #357 -> #355, #358, #359, #360 ->
  `research/43`-`46`.
- **#362 marker ablation** -> #372; **#373** -> #374.

Stacked PRs (a non-`main` base) are the short chains inside #335 and the v4
trainers: #148 -> #158 -> #259 -> #260; #193 -> #195; #222 -> #224, #225, #227;
#336 -> #343; #337 -> #345; #338 -> #344; #339 -> #346; #341 -> #351; #347 ->
#348; #277 -> #284; #278 -> #285; #373 -> #374.

## How this record was built

`gh pr list --state all` and `gh issue list --state all` on 2026-11-15, parsed
for explicit relation phrases. Creation dates are used; merge dates differ by
minutes to days and are not shown. Titles are verbatim. No event is omitted:
the counts above equal the number of rows in the tables.

