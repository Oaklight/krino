# RL Training for the Krino Decision Model — Design

**Date:** 2026-10-27
**Status:** Design proposal. No reinforcement-learning code exists in the repository yet.
**Scope:** `model/training/`, `model/scripts/`, `model/configs/`, with small additions to `model/src/`.
**Related:** `research/04-model-replication.md` (Step 3 Phase B), `research/03-open-source-ecosystem.md` (RLCD landscape), `research/02-architecture-analysis.md` (RLCD training gap).

Line numbers refer to the working tree at the time of writing (2026-10-27). The design
assumes the frozen-backbone model. LoRA is not hypothetical — it ships in
`model/src/lora.py` and `--lora-layers` (#127) — and the places where an adapter changes
the answer, chiefly the reference policy in Section 4.2, are called out explicitly.

---

## 0. Summary

Krino's decision model is not a text generator. It is a **finite categorical policy**:
one forward pass produces logits over an enumerated answer set of size K (K=2 for noul,
K options for choice, L ordered levels for score). That changes which RL formulations
are appropriate and which are cargo cult.

The main conclusions:

1. **The natural formulation is a contextual bandit over a small discrete action set.**
   Because K is enumerable and at most 255, the expected reward can be computed exactly.
   Exact expected policy gradient is lower variance than REINFORCE, and cheaper than
   GRPO's group sampling whenever the reward is a local lookup rather than an external
   call. Use it as the default for local rewards.
2. **RL on gold labels cannot beat cross-entropy in expectation.** With a 0/1 correctness
   reward and exact expectation, the objective is `J(θ) = E_s π_θ(y|s)`, whose gradient is
   a positive scalar multiple of the negative CE gradient (the scalar is `π_θ(y|s)`), so it
   has the same optimum and direction. Sampled REINFORCE is an unbiased but higher-variance
   estimator of the same thing. The additive signals in this repo are proper scoring rules, teacher
   distributions, and (eventually) cost or preference data — not the labels themselves.
3. **The only dense reward currently available is teacher probability on synthetic data.**
   Benchmark JSONL sources carry hard labels but no `teacher_probs`; teacher probabilities
   exist only in the synthetic label files. The R2 mix uses at most 10,000 synthetic
   training items (`model/configs/r2_reranker4b.yaml:149-152`), so teacher reward is a
   small fraction of the data.
4. **Calibration is not an optional regularizer.** A 0/1 reward is invariant to how
   probability mass is distributed among wrong answers, so accuracy-only RL drives the
   policy to one-hot. KL to an SFT reference, an entropy floor, and a proper-scoring-rule
   reward are the mechanisms that preserve the product.
5. **Do the supervised calibration work first.** RPS for score, KL-to-teacher, and
   calibration metrics in the training loop are cheaper, more stable, and more likely to
   close the gap than online RL. The repository's own notes already flag this:
   "RLCD-style RL — may be unnecessary if teacher distillation closes the calibration
   gap" (`research/04-model-replication.md:397`). (RLCD is Reinforcement Learning for
   Calibrated Decisions.) Gate online RL on a reward source that is not already a
   supervised target.

The rest of this document argues each point, then gives a staged plan, a file/API sketch,
risk metrics, and an honest gap list.

---

## 1. What the model is, as a policy

### 1.1 Three primitives, one categorical distribution

| Primitive | Answer set | K | Order | Current head |
|---|---|---|---|---|
| noul | `{"false", "true"}` | 2 | ordered | sigmoid (`NoulHead`) or 2-option `UnifiedHead` |
| choice | `criteria` keys | 2–255 | unordered | `AttentionHead` → softmax |
| score | `criteria` levels | 2–10 | ordered | `AttentionHead` → softmax → expected value |

The forward paths are `DecisionModel.forward_noul` (`model/src/decision_model.py:305`),
`forward_choice` (`:323`), and `forward_score` (`:333`). Choice and score both call
`AttentionHead.forward`, which computes one logit per option
(`model/src/heads.py:186-221`; attention softmax at `:213`, option logits at `:217`).
`ScoreHead.predict` converts the level distribution to an expected value
(`model/src/heads.py:316-328`).

The `UnifiedHead` already treats all three primitives as the same operation: score a set
of options and softmax (`model/src/heads.py:331-464`). Noul becomes two options,
`"false"` and `"true"`, and the noul logit is their difference
(`model/src/heads.py:382-405`; `model/src/decision_model.py:305-321`).

This matters for the user's "score is the fundamental primitive" direction. At the
distribution level the claim is exact: every primitive is a categorical over an
enumerated set. The score primitive adds one thing — an **order relation** on the levels.
Noul is the L=2 ordered case; choice is the K-level unordered case. "Choice converts to
score" is exact only if the order is discarded; "score converts to choice" is exact via
argmax; "noul converts to score" is exact with two levels. The practical consequence is
that the project needs **one policy and one reward interface with an `ordered` flag**, not
three RL pipelines.

### 1.2 Why this is a contextual bandit, not an MDP

A decision episode has one state, one action, and one reward. There is no token-level
trajectory and no temporal credit assignment. The model's inference cost is one backbone
pass plus K option encodings; that is the efficiency advantage over autoregressive models.
Any RL formulation that samples text tokens would reintroduce autoregressive decoding and
destroy that advantage. The action space must stay at the typed head.

The backbone is frozen by default (`model/src/decision_model.py:192-193`) and its forward
pass runs under `torch.no_grad()` unless an adapter is present, which switches it to
`enable_grad` (`:258`). LoRA adapters change the reference-policy design (Section 4.2),
not the bandit formulation.

### 1.3 Efficiency budget

| Phase | Backbone passes per item | Head passes | Extra cost |
|---|---|---|---|
| Inference | 1 context pass + 1 batched option encode | 1 | none |
| SFT epoch | same as inference | 1 | none |
| RL, local reward, exact expectation | same as inference | 1 policy + 1 reference | K reward lookups; reference logits can be cached |
| RL, local reward, sampled REINFORCE | same as inference | 1 policy + 1 reference | G categorical draws from the same logits; no extra forward pass |
| RL, external reward | same as inference | 1 policy + 1 reference | G external calls; the only case where sampling saves money |

The efficiency argument for exact expectation is therefore not only statistical. It
keeps the training graph identical to inference, avoids a learned critic, and reuses the
same logits for every sampled answer. The only unavoidable extra cost is the reference
head pass for KL, which can be reduced to a cache lookup for a fixed training set.

### 1.4 One policy, four dispatch paths

The typed dispatch is currently duplicated in four places:

- `compute_loss` (`model/training/supervised.py:76-157`),
- the batched per-type functions (`:166-320`),
- `_eval_item` (`:562-627`),
- `_forward_and_extract` in the calibration trainer (`model/training/calibration.py:71-167`).

Adding a fifth copy for RL would be a maintenance hazard. The first refactor should be a
single `DecisionModel.forward_typed(state, question) -> (logits, keys, q_type)` that all
four paths (and RL) call. This is a small change with a large correctness payoff: the
current copies already differ in how they handle teacher probabilities, float score
labels, and augmented noul items.

---

## 2. Which RL formulations fit

### 2.1 Formal setup

For a state `s` (state text, instructions, option/level texts) let
`A(s) = {a_1, ..., a_K}` be the enumerated answer set. The policy is

```
π_θ(a_i | s) = softmax(z_θ(s))_i,   z_θ(s) ∈ R^K.
```

A reward `r(s, a)` is observed after choosing an answer. The objective is

```
J(θ) = E_{s~D} [ Σ_{a∈A(s)} π_θ(a|s) r(s,a) ].
```

This is a one-step objective. There is no discount factor, no value bootstrapping, and no
need for PPO-style ratio clipping. A baseline is still useful for variance reduction, but
because K is small the optimal baseline can be computed exactly.

### 2.2 Exact expected policy gradient (recommended default, for rewards that do not read the policy)

Because `A(s)` is enumerable, differentiate the expectation directly:

```
∇J(θ) = E_{s~D} [ Σ_a r(s,a) ∇π_θ(a|s) ]
      = E_{s~D} [ Σ_a π_θ(a|s) r(s,a) ∇log π_θ(a|s) ].
```

**This exchange is valid only when the reward does not depend on θ.** The general form
makes the dropped term explicit:

```
∇J = E_s [ Σ_a π_θ(a|s) ∇r(s,a) ]  +  E_s [ Σ_a r(s,a) ∇π_θ(a|s) ].
```

The first term vanishes for a reward that is a lookup -- a gold label, a teacher
probability, a fixed cost matrix -- which is the case this section covers, and it is why
the estimator costs one forward pass. It does **not** vanish for a reward computed from
the policy's own output. A log score `log π_θ(y|s)`, or a Brier score built from the same
logits, has `∇r ≠ 0`, and that term dominates rather than being a correction: measured on
a 4-way softmax with `r = log π(y)`, ‖∇J‖ is 6.4e-08 with the reward detached and 0.849
with it attached.

So a proper scoring rule belongs in the **loss**, not in this objective as a reward. With
`r = log π_θ(y|s)` and `J = Σ_a π r`, the objective is `J = π(y) log π(y)`, which is
stationary at `π(y) = e^-1` and decreases either side of it down to `-∞`; it is not
cross-entropy. Negating it is exactly cross-entropy. Anything of that shape should enter
as `-J` in the supervised loss, where the derivative is taken through `π` by construction.

For a 0/1 correctness reward `r(s,a) = 1[a = y]`, the objective collapses to
`J(θ) = E_s π_θ(y|s)`. Then
`∇J = E_s π_θ(y|s) ∇log π_θ(y|s) = -E_s π_θ(y|s) ∇(-log π_θ(y|s))`,
a positive scalar multiple of the negative cross-entropy gradient. Same optimum, same
direction, no new signal. This is the central negative result: **policy gradient on gold
labels is a reparameterization of CE.**
It can change optimization dynamics (temperature, clipping, exploration noise), but it
cannot change the optimum. The repository's benchmark labels are exactly this case.

The exact estimator is also the most efficient option for a "high performant" model:

- one backbone pass per item, unchanged from inference;
- no G-sample rollout;
- no learned critic (a critic would add parameters and, if kept, serving cost);
- the reward is a lookup (label or teacher probability), not a model call -- and, per
  above, not a function of the logits either.

### 2.3 REINFORCE with sampling

```
∇J ≈ (r(s,a) - b(s)) ∇log π_θ(a|s),   a ~ π_θ(·|s).
```

It is unbiased for a fixed reward and is the right tool when the reward can only be
observed for a sampled answer. In this repo that means an external judge or API call per
answer. If the reward is a local lookup, sampling only adds variance relative to
Section 2.2.

The optimal baseline is `b(s) = E_{a~π}[r(s,a)]`, which is exactly computable here. A
batch-mean or group-mean baseline is a noisy substitute. This is why GRPO is not needed
for the current action space.

One exploration caveat: sampling from `π_θ` does not explore answers outside the current
support. Temperature scaling or Dirichlet noise on the logits is what creates exploration.
With a 0/1 reward and a single gold answer, there is nothing to discover outside the label,
so exploration has no upside on the current benchmarks.

### 2.4 GRPO

GRPO samples G completions per prompt and uses the group mean and standard deviation as
the advantage:

```
A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)
L = -(1/G) Σ_i A_i log π_θ(a_i|s).
```

It is mechanically applicable here: a "completion" is one categorical draw. But it is
dominated for K ≤ 255:

- With K=2 or 4, G draws are mostly duplicates. The group mean is a Monte Carlo estimate
  of `E[r]`, which is exact and free.
- If all G rewards are equal — the common case for an easy item where every sample is
  correct, or a hard item where every sample is wrong — the group standard deviation is
  zero and the item contributes no gradient.
- Standard-deviation normalization amplifies noise when the policy is confident and the
  reward is binary.
- The PPO-style ratio and clipping in GRPO exist for off-policy updates. Single-step
  on-policy updates do not need them.

GRPO becomes reasonable only when reward evaluation is expensive enough that evaluating
all K answers is prohibitive, or if the project moves to text-space actions. The first
condition could arise with a paid external judge; the second should be rejected for
efficiency reasons.

### 2.5 DPO

DPO requires preference pairs `(s, a+, a-)` and a reference policy:

```
L_DPO = -log σ(β [log π(a+|s) - log π_ref(a+|s)
                 - log π(a-|s) + log π_ref(a-|s)]).
```

Where would pairs come from in this repo?

- **Gold label vs sampled wrong answer.** This is the only pair source the current data
  supports. It is degenerate: the preference is deterministic, so the Bradley-Terry model
  drives the log-probability margin to infinity. DPO would make the model more confident,
  not better calibrated. It also discards the probabilities of the other K-2 options,
  which is exactly the calibration signal.
- **Teacher distribution.** One can sample `a+ ~ q` and `a- ~ π`, or rank options by
  teacher probability. But this is a lossy version of distillation; KL to `q` uses the
  full distribution and is the direct estimator. `_teacher_loss` already implements that
  KL (`model/training/supervised.py:47-52`).
- **Human or LLM preferences.** The repository has none. `TypedQuestion` has no
  chosen/rejected fields (`data/format.py:31-39`) and there is no preference loader.
- **Rejection sampling from the model.** With a 0/1 reward and a single correct key, the
  best sample is the gold answer, so the pairs reduce to the first case. It would add
  signal only on tasks with multiple valid answers, which the current benchmarks do not
  have.

DPO is therefore not the right first stage. It needs a reference model (which the repo
does not have), it optimizes ranking rather than calibration, and the available pairs are
label-derived. It becomes reasonable only if genuine pairwise preference data is collected
later.

### 2.6 Proper-risk minimization: the option that actually matters

A proper scoring rule `S(π, y)` is minimized in expectation by the true conditional
distribution. For a distributional prediction there is no action to sample; the reward is
a function of the whole distribution:

```
J(θ) = E_{(s,y)~D} [ S(π_θ(·|s), y) ].
```

This is full-information, differentiable, and lower variance than any policy-gradient
estimator. It is not "RL" in the bandit sense, but it is what most of a calibration-focused
RLCD pipeline should be. The repo already uses the log score (cross-entropy) for noul and
choice, and CE or soft CE for score. What is missing:

- **RPS (ranked probability score)** for the ordered score primitive. CE treats levels as
  unordered classes; it cannot distinguish "very negative" from "neutral" when the truth
  is "negative". RPS penalizes distance and is strictly proper for ordinal outcomes.
  *Implemented:* `rps_loss` in `model/training/losses.py`, selected with
  `--score-loss rps` (default remains CE). What is still missing is RPS as an
  *evaluation* metric — score evaluation is argmax accuracy (`:611-624`) — and
  `score_mae` exists in `model/evaluation/accuracy.py:22` but is not called by the
  training loop.
- **Brier and MMCE in the multitask loop.** `brier_loss` and `mmce_loss` exist
  (`model/training/losses.py:18-48`, `:81-129`) and `CalibrationLossConfig` composes them
  (`model/training/calibration.py:170-193`), but that is a separate training loop
  (`:245-361`) that evaluates with CE (`:322`). The R2 multitask trainer does not use
  them.
- **KL to the teacher as a regularizer** is already implemented (`_teacher_loss`), but it
  is only applied where `teacher_probs` is present, which is the synthetic subset.

### 2.7 Comparison

| Formulation | What it needs | Fit to a small categorical action | Calibration | Verdict |
|---|---|---|---|---|
| Exact expected PG | reward per `(s,a)`, K enumerable | excellent | depends on reward | **default** |
| REINFORCE | sampler + baseline | good | depends on reward | only for expensive rewards |
| GRPO | G samples, group stats | poor for K≤10 | depends on reward | not first |
| DPO | pairs + reference | workable | poor (ranking) | only with genuine preferences |
| Proper-risk minimization | labels or soft targets | excellent | direct | **do first** |
| Text-space RL | autoregressive rollouts | wrong | n/a | reject |

### 2.8 The trap in "teacher probability as reward"

If the reward is the teacher's probability of the sampled answer,
`r(s,a) = q(a|s)`, then

```
J = Σ_a π(a) q(a),
```

which is maximized by a point mass on `argmax_a q(a)`. This destroys the distribution.
If instead the reward is the teacher's **log** probability,
`r(s,a) = log q(a|s)`, then

```
J = Σ_a π(a) log q(a) = -CE(π, q).
```

By Gibbs' inequality, `CE(π, q) ≥ H(q)` with equality at `π = q`; maximizing `J` therefore
recovers the teacher distribution. Teacher log-probability is distillation; teacher
probability is mode seeking. Any reward implementation must make this distinction
explicit, and a unit test should pin it.

---

## 3. Reward sources actually available

| Source | Where it lives | Coverage | Signal | Caveats |
|---|---|---|---|---|
| Gold labels | `TypedQuestion.label` (`data/format.py:31-39`); loaders in `data/pipeline.py`; `_eval_item` (`model/training/supervised.py:562-627`) | 26 benchmark sources plus synthetic (`model/configs/r2_reranker4b.yaml:29-168`) | 0/1 correctness, score MAE | already the SFT target; label noise; STS-B labels rounded (`data/pipeline.py:268`) |
| Calibration losses | `model/training/losses.py:18-48` (Brier), `:81-129` (MMCE), `:131-193` (focal) | all typed items | Brier, MMCE, focal | not wired into the R2 trainer; MMCE needs batches ≥2 |
| Teacher probabilities | `TypedQuestion.teacher_probs` (`data/format.py:39`); `_teacher_loss` (`model/training/supervised.py:47-52`); `_select_teacher_probs` (`data/pipeline.py:1062-1083`) | synthetic only; R2 caps at 10,000 train items (`model/configs/r2_reranker4b.yaml:149-152`) | KL / soft target | single teacher; missing keys filled with 0.0; teacher bias |
| Synthetic data | `load_synthetic` (`data/pipeline.py:1125-1169`); `SamplerConfig.synthetic_ratio` (`data/sampler.py:18-52`); `--synthetic-ratio` (`model/scripts/train_multitask.py:173`) | 154K items, 23 domains per the R2 config comment (`model/configs/r2_reranker4b.yaml:10`) | labels plus teacher probs | template artifacts; family-level split |
| Preferences | none | 0 | none | no fields, no loader |
| Cost matrix / decision utility | none | 0 | none | no field in `TypedQuestion` |
| Unlabeled verifiable outcomes | none | 0 | none | no environment or verifier |

### 3.1 Gold labels

The label is the SFT target and the obvious reward. It is verifiable and cheap. But it is
also the reason Section 2.2 matters: a 0/1 reward on a label is not a new signal. The
repository has already been bitten by label handling bugs (the noul truthiness bug fixed
in PR #39, recorded in the version history of `research/04-model-replication.md:13`), so a
reward implementation should reuse the same label-normalization code as `_eval_item`
rather than reimplementing it.

Score labels need special care. STS-B is rounded to integer levels at load time
(`data/pipeline.py:268`), SST-5 uses five levels (`data/pipeline.py:284-301`), and
`_score_target` interpolates float labels between adjacent levels
(`model/training/supervised.py:22-39`). A score reward should consume the same target
distribution, not re-round the label.

### 3.2 Calibration rewards

`losses.py` implements Brier, MMCE, and focal losses. `calibration.py` composes
`alpha * task_loss + beta * calibration_loss` (`model/training/calibration.py:170-193`)
and trains with a separate loop (`:245-361`). The evaluation module computes ECE and
Brier (`model/evaluation/calibration.py:8-45`). None of this is in the R2 multitask loop:
`eval_epoch_detailed` returns only loss and accuracy (`model/training/supervised.py:688-719`),
and ECE is computed only by separate eval scripts (`model/scripts/eval_trained.py:23`).

Two practical notes for RL:

- MMCE is a batch statistic (`model/training/losses.py:50-79`). It does not decompose
  into a per-item reward and is awkward in an online loop. Use Brier or log score as the
  per-item reward; keep MMCE for offline calibration SFT.
- ECE is not differentiable and is a population metric. Use it for evaluation, not as the
  training reward.

### 3.3 Teacher probabilities

The `teacher_probs` field is a flat `{option_key: probability}` dict
(`data/format.py:39`). For synthetic data it is assembled from per-domain label files by
`load_synthetic` (`data/pipeline.py:1125-1169`), and `_select_teacher_probs` picks one
teacher by priority: Jev, then Luna, then Qwen (`data/pipeline.py:1060-1083`). The
training loss is `KL(teacher || student)` against that flattened distribution (`model/training/supervised.py:47-52`).

Three limitations are important for RL:

1. **Coverage.** The teacher-probability files in this checkout are under
   `data/benchmarks/synthetic/` (`*_jev_labels.jsonl`). The 26 benchmark JSONL sources
   carry hard labels, not `teacher_probs`. Teacher reward therefore applies to a minority
   of the R2 mix.
2. **Missing keys are silently zero.** `_teacher_loss` builds the teacher tensor with
   `teacher_probs.get(k, 0.0)` and renormalizes (`model/training/supervised.py:49-50`).
   If the option keys and the teacher keys diverge, the target is quietly wrong. An RL
   reward must validate key coverage and raise, not fill.
3. **Single teacher, no ensemble.** The priority selector discards the other teachers.
   A multi-teacher KL (weighted average, or per-teacher heads) would be a richer signal,
   but it is not implemented.

The teacher is also not ground truth. The research notes record a fixed
compression-toward-middle distortion in Jev's probabilities
(`research/02-architecture-analysis.md:432`). A teacher-KL reward can teach the model to
imitate that distortion. Teacher reward should be weighted below gold reward where gold
exists, and validated on a gold canary.

### 3.4 Synthetic data

`load_synthetic` assembles items from families, variants, and label files on the fly
(`data/pipeline.py:1125-1169`). The sampler can mix synthetic and benchmark data by ratio
(`data/sampler.py:18-52`), and the R2 config uses 10,000 synthetic train items
(`model/configs/r2_reranker4b.yaml:149-152`). Synthetic data is the main source of teacher
probabilities, so it is also the main source of dense reward.

The risk is template exploitation. Synthetic families are generated from domain templates;
a policy trained with a reward on synthetic data can learn template artifacts rather than
decision quality. The split is family-level, not state- or template-level, and the code
itself notes that family-disjointness is not state-disjointness
(`data/pipeline.py:1086-1123`). A reward-hacking canary should be template-disjoint and,
where possible, from a different generator.

### 3.5 What is missing as a reward

There is no preference data, no cost matrix, no unlabeled state with a verifiable outcome,
and no interactive environment. Those are the signals that would justify online RL beyond
distillation. Until one of them exists, "RL" on this repo is a more expensive way to do
supervised calibration.

### 3.6 Reward design for the score-as-primitive direction

If score is the fundamental primitive, the reward should respect the order:

- **Score:** RPS or log score. RPS is the ordinal proper scoring rule; log score is the
  categorical proper scoring rule. Do not use MAE as the training reward: the L1 optimum
  can be a point mass at the conditional median, which destroys the distribution.
- **Noul:** Brier or log loss on the two-option categorical. The current sigmoid +
  BCE-with-logits is the log score; a Brier variant is available in
  `model/training/losses.py:36-48`.
- **Choice:** log score or Brier over K options. There is no ordinal structure, so RPS
  does not apply unless the options are ordered.

A single `RewardConfig` with an `ordered` flag can dispatch these without three code
paths.

---

## 4. Preserving calibration during RL

### 4.1 Why a 0/1 reward is calibration-blind

For a binary outcome, `E[r] = π(y|s)` depends only on the mass on the correct answer.
Among all distributions with the same argmax, the expected 0/1 reward is identical. RL on
this reward pushes mass toward the argmax until the distribution is one-hot; it has no
term that rewards matching the true conditional probability. Accuracy may stay flat while
ECE and Brier worsen, which is the classic RLHF calibration failure.

This is not a hypothetical for this repo: the score primitive is already the weakest
(`research/04-model-replication.md` reports score accuracy around 44–50% across models),
and the training loop does not measure calibration at all. Adding accuracy-only RL to that
setup would make the product worse in the dimension Jev is strongest.

### 4.2 KL to the SFT reference

```
L_KL = KL(π_θ(·|s) || π_ref(·|s)).
```

The reference anchors the distribution, prevents drift, and preserves the SFT model's
calibration. It also limits interference between question types when heads are shared
(`shared_attention` / `unified_head`, `model/src/decision_model.py:159-184`).

The repository has no reference-model lifecycle. `save_heads` writes only the non-backbone
state dict (`model/src/decision_model.py:382-397`), `load_heads` loads it into the same
model (`:399-402`), and the trainer saves `best_heads.pt` plus optimizer state
(`model/training/supervised.py:906-920`). There is no second model, no frozen copy, and no
serialized model config. To reconstruct a reference you need the backbone name and the
constructor arguments, which are not saved with the checkpoint.

With the current frozen backbone, the cheap implementation is a second `DecisionModel`
that **shares the same backbone object** and loads the SFT heads. The backbone is already
frozen (`model/src/decision_model.py:192-193`), so sharing is safe; only the head forward
passes differ. Even cheaper: compute the backbone hidden states once and apply both the
policy head and the reference head. The current `forward_*` API does not expose hidden
states, so this needs a small refactor (Section 6.2).

With a LoRA adapter, sharing the backbone is not enough: the reference must use the base
weights with the adapter disabled. That requires an enable/disable switch on the adapter
(or a separate base-weight copy), which `model/src/lora.py` does not provide yet — the
adapters are always active once injected.

### 4.3 Entropy regularization

```
L_ent = -β H(π_θ(·|s)).
```

Entropy prevents premature collapse to a point mass, but it does not anchor to the SFT
distribution. It can flatten probabilities and increase calibration error if weighted too
high. Use it as a small stabilizer, not as a substitute for KL. For a K-way categorical,
entropy is cheap to compute from the logits and should be logged every epoch.

### 4.4 Calibration-aware reward

A proper scoring rule reward directly measures the quality of the full distribution. It
is strictly proper, so the optimum is the true conditional distribution. This is the
strongest calibration signal and the one that should carry most of the weight. KL to the
reference and entropy are stabilizers; the proper scoring rule is the objective.

### 4.5 Temperature

The model's logits have an implicit temperature. RL can change their scale, so a policy
can become globally over- or under-confident even if its ranking is unchanged. A single
fitted temperature (as Laya does) can fix the global scale, but not per-item calibration.
The research notes record that Laya's ECE improvement comes primarily from temperature
fitting rather than from the RL training itself
(`research/03-open-source-ecosystem.md:108-118`). Temperature fitting is a useful
diagnostic and a cheap baseline, not a substitute for a calibration objective.

### 4.6 What breaks if this is skipped

- **Overconfidence.** Accuracy stays flat, mean confidence rises, ECE and Brier worsen.
- **Mode collapse.** Entropy falls, one answer dominates the prediction histogram, and
  the model becomes useless as a probability source.
- **Reference drift.** The policy moves far from the SFT model; other question types
  degrade if heads are shared.
- **Teacher-bias imitation.** A teacher-KL reward teaches Jev's compression and position
  bias, not truth.
- **Hidden by the serving API.** `predict` clamps noul to `[0.01, 0.99]` and rounds all
  probabilities to two decimals (`model/src/decision_model.py:351`, `:359`, `:370`).
  A collapsed model can look bounded at the API while its raw logits are extreme. Reward
  and diagnostics must use raw logits, not `predict`.

### 4.7 Starting defaults

These are starting points, not tuned values:

| Knob | Start | Rationale |
|---|---|---|
| `kl_coef` | 0.1 | strong enough to anchor, weak enough to learn |
| `entropy_coef` | 0.001–0.01 | collapse guard only |
| `reward.correctness` | 1.0 | gold where available |
| `reward.log_score` | 0.5 | dense calibration signal |
| `reward.rps` | 0.5 | ordinal score signal |
| `reward.teacher_kl` | 0.2 | regularizer, not the main objective |
| `temperature` (train) | 1.0 | match inference |
| `reference` | SFT checkpoint | must be reproducible |
| `seeds` | ≥3 | the repo has a history of single-seed conclusions that flipped (`research/06-head-size-sweep.md:17-60`) |

The repo's optimization experiments found that uniform learning rates beat per-type
learning rates and uncertainty weighting (`research/07-mtl-optimization.md:71-106`). RL
hyperparameters should be compared at matched learning rates and multiple seeds, not
inferred from one run.

---

## 5. Staged plan

### Stage 0 — Instrumentation and reference infrastructure

**Build**

- Add raw-probability and typed-dispatch APIs: `DecisionModel.forward_typed` and
  `predict_probs` (no rounding, no clamping).
- Add score metrics to `eval_epoch_detailed`: MAE, RPS, NLL, ECE, Brier, entropy,
  max-probability. `score_mae` already exists (`model/evaluation/accuracy.py:22`) and is
  used by `eval_reranker.py` and `eval_jev.py`, but the multitask trainer never imports
  it.
- Add `save_checkpoint` / `load_checkpoint` that persist the model constructor config
  (backbone name, rank, rival-aware, unified/shared head, max length) alongside the heads.
- Add a `ReferencePolicy` wrapper that shares the backbone and freezes the SFT heads.

**Measure**

- Reproduce the R2 baseline exactly.
- Report per-type and per-source accuracy, score MAE/RPS, ECE, Brier, entropy.
- Verify that the reference policy produces identical logits to the initial policy.

**Gate**

- Baseline reproduced within seed noise.
- Every metric the RL loop will use is computed by the training loop, not by a separate
  script.
- A reference checkpoint can be reconstructed from disk without guessing constructor
  arguments.

### Stage 1 — Calibration SFT (highest value, not RL)

**Build**

- Add RPS and a differentiable log-score helper to `model/training/losses.py`.
- Add a `loss:` block to the multitask config, mirroring `CalibrationLossConfig`
  (`model/training/calibration.py:46-60`): `task`, `brier`, `mmce`, `teacher_kl`, weights.
- Wire Brier/KL into `train_multitask` so the R2 pipeline can use them without switching
  to the separate calibration loop.
- Optionally enable LoRA (when it lands) so the backbone can adapt.

**Measure**

- Score MAE and RPS on SST-5, STS-B, Yelp.
- ECE and Brier for noul, choice, and score.
- Accuracy per type and source.
- Three seeds; paired bootstrap or McNemar for accuracy; bootstrap CI for ECE/Brier.

**Gate**

- Score RPS/MAE improves and ECE/Brier improve without an accuracy drop.
- If the gap does not move, stop and diagnose before building online RL. The most likely
  cause is data or loss, not the absence of RL.

**Why first**

- It is cheaper and more stable than online RL.
- It directly serves the score-as-primitive direction.
- It produces the reference model and metrics that RL needs.

### Stage 2 — Offline preference / reward-weighted optimization

**Build**

- A pair/weight constructor from teacher distributions or from model samples plus
  verifiable labels.
- Three comparable objectives: soft-label KL, reward-weighted CE / AWR, and DPO.
- A preference-data loader only if genuine pairwise data is collected.

**Measure**

- Same metrics as Stage 1, plus entropy and KL to the Stage 1 model.
- A canary set that is disjoint from the training templates.

**Gate**

- The offline objective beats Stage 1 on held-out calibration and accuracy.
- No entropy collapse, no ECE regression.
- If DPO does not beat soft-label KL, drop DPO. Expect it not to, for the reasons in
  Section 2.5.

### Stage 3 — Online RL with a new reward source

**Precondition**

A reward source that is not already a supervised target. Examples: a cost matrix
(false-positive vs false-negative costs), unlabeled states with a verifiable outcome, an
external judge, or an interactive environment. Without one, do not run this stage; it
would be CE with extra variance.

**Build**

- `sample_actions` (categorical sampling with temperature and a seeded generator).
- Exact expected policy gradient as the default; REINFORCE with an exact baseline as the
  fallback for expensive rewards.
- KL to the reference, entropy, reward clipping and normalization.
- Per-item reward logging (source, type, reward components, policy entropy, KL).

**Measure**

- Everything from Stage 1, plus:
  - reward mean/variance per source;
  - correlation between reward and held-out accuracy;
  - KL(π || π_ref) per epoch;
  - prediction histogram and max-probability distribution;
  - surface-form sensitivity (`model/evaluation/bias.py:59-100`) and ordering bias
    (`:12-56`);
  - a template-disjoint canary and a gold canary.

**Gate**

- Pre-registered success criteria, ≥3 seeds, paired tests.
- Accuracy or the targeted decision utility improves, and ECE/Brier do not regress beyond
  a pre-registered tolerance.
- No canary regression, no entropy collapse, no unbounded KL growth.

### Stage 4 — External teacher or online reward (optional)

**Build**

- A budgeted Jev API reward wrapper around `JevClient`
  (`probing/scripts/jev_client.py:58`), with caching, rate limiting, and a pinned model
  version (`jev-latest` vs `jev-preview`, `research/00-overview.md:63-64`).
- A gold-vs-teacher agreement check on a canary before using teacher reward at scale.

**Gate**

- Cost per point of improvement is acceptable.
- Teacher reward does not degrade gold accuracy or calibration.

### Stop rules

- If Stage 1 closes the calibration gap, skip Stage 3.
- If no non-label reward exists, do not run Stage 3.
- If DPO does not beat soft-label KL, drop DPO.
- If exact expectation and REINFORCE are within noise at equal compute, use exact
  expectation (simpler, lower variance).
- If the reward rises while held-out accuracy is flat, stop and audit reward hacking.

---

## 6. Concrete file and API sketch

### 6.1 New modules

**`model/training/rewards.py`** — per-item reward functions. Every function returns a
utility (higher is better), so one maximization loop can compose them. The losses in
`losses.py` are the negation of these utilities; that sign convention must be documented
and tested.

```python
"""Per-item rewards for the typed decision policy.

All functions return a utility (higher is better).  Losses in
``model/training/losses.py`` are the negation of these utilities.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RewardConfig:
    correctness: float = 1.0
    log_score: float = 0.0
    brier: float = 0.0
    rps: float = 0.0
    teacher_kl: float = 0.0
    clip: float | None = None
    normalize: str = "none"  # none | batch | group

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RewardConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def correctness_reward(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    """1 if argmax == target, else 0.  Non-differentiable; combine with
    exact expectation (differentiate through pi) or REINFORCE."""


def log_score_reward(logits: torch.Tensor, target_idx: int) -> torch.Tensor:
    """log π(target).  Differentiable; equals negative CE."""


def brier_reward(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Negative Brier score.  Differentiable."""


def rps_reward(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Negative ranked probability score.  Ordinal only."""


def teacher_kl_reward(
    logits: torch.Tensor,
    teacher_probs: dict[str, float],
    keys: list[str],
) -> torch.Tensor:
    """Negative KL(teacher || policy).  Raises on missing keys."""


def compose_rewards(
    logits: torch.Tensor,
    item: Any,
    keys: list[str],
    cfg: RewardConfig,
) -> torch.Tensor:
    """Weighted sum of the configured rewards for one item."""
```

**`model/training/rl.py`** — the RL loop. Mirror the structure of `train_multitask`
(`model/training/supervised.py:721-931`): a config dataclass, a per-epoch function, and a
top-level `train_rl`.

```python
from dataclasses import dataclass, field

import torch

from model.training.rewards import RewardConfig


@dataclass
class RLConfig:
    algorithm: str = "exact"  # exact | reinforce | group_reinforce | awr | dpo
    reward: RewardConfig = field(default_factory=RewardConfig)
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    temperature: float = 1.0
    group_size: int = 1
    baseline: str = "exact"  # none | batch_mean | group_mean | exact
    max_grad_norm: float = 1.0
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "RLConfig": ...


def sample_actions(
    logits: torch.Tensor,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one action index per row from softmax(logits / temperature)."""


def exact_policy_loss(
    logits: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """-Σ_a π(a) r(a) for one item, differentiated through the softmax."""


def reinforce_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """-(r - b) log π(a) for sampled actions."""


def kl_to_reference(
    logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(π || π_ref) for one item."""


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(π) for one item."""


def train_rl(
    model,
    reference,
    train_items: list,
    eval_items: list,
    sampler_config,
    cfg: RLConfig,
    epochs: int = 3,
    lr: float = 3e-4,
    checkpoint_dir=None,
    eval_every: int = 1,
    seed: int = 42,
) -> dict:
    """Online RL loop with per-item reward logging."""
```

**`model/training/reference.py`** — a frozen reference policy that shares the backbone.

```python
class ReferencePolicy:
    """Frozen SFT heads sharing the policy backbone.

    The backbone is already frozen in DecisionModel, so the reference
    does not need a second copy of it.  Only the head forward pass differs.
    """

    def __init__(self, model, heads_path):
        self.backbone = model.backbone
        self.heads = ...  # a second DecisionModel with the same backbone
        self.heads.load_heads(heads_path)
        for p in self.heads.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def logits(self, state, question):
        ...
```

For LoRA, this class must be able to disable the adapter for the reference forward. If
the adapter is implemented as an additive delta inside the backbone, add a module-level
`enabled` flag or a context manager; do not rely on a separate backbone copy, which
doubles memory.

**`model/scripts/train_rl.py`** — CLI. Mirror `model/scripts/train_multitask.py:133-173`.

```
python model/scripts/train_rl.py \
    --config model/configs/rl_reranker4b.yaml \
    --init-heads model/experiments/r2_reranker4b/best_heads.pt \
    --reference-heads model/experiments/r2_reranker4b/best_heads.pt \
    --algorithm exact \
    --reward-weights '{"correctness": 1.0, "log_score": 0.5, "rps": 0.5}' \
    --kl-coef 0.1 --entropy-coef 0.01 --epochs 3
```

Flags: `--config`, `--model`, `--init-heads`, `--reference-heads`, `--algorithm`,
`--reward-weights`, `--kl-coef`, `--entropy-coef`, `--temperature`, `--group-size`,
`--epochs`, `--lr`, `--seed`, `--device`, `--checkpoint-dir`, `--output`, `--eval-only`,
`--max-train`, `--max-eval`.

**`model/configs/rl_reranker4b.yaml`** — reuse the SFT `sources:` and `sampling:` blocks.

```yaml
model: Qwen/Qwen3-Reranker-4B
encoder: false
rank: 128
init_heads: model/experiments/r2_reranker4b/best_heads.pt
reference_heads: model/experiments/r2_reranker4b/best_heads.pt

rl:
  algorithm: exact
  reward:
    correctness: 1.0
    log_score: 0.5
    rps: 0.5
    teacher_kl: 0.2
  kl_coef: 0.1
  entropy_coef: 0.01
  temperature: 1.0
  group_size: 1
  baseline: exact

sources:
  # same 26 benchmark + synthetic blocks as r2_reranker4b.yaml
sampling:
  type_ratios: {noul: 1.0, choice: 1.0, score: 1.0}
  sampling_temperature: 0.7
  epoch_size: 15000
```

### 6.2 Changes to existing modules

- **`model/src/decision_model.py`**
  - Add `forward_typed(state, question) -> (logits, keys, q_type)` and
    `predict_probs(state, question) -> dict`. `predict` keeps its rounded output for API
    compatibility (`:254-291`); RL uses the raw path.
  - Add `save_checkpoint(path, config)` / `load_checkpoint(path)` that persist the
    constructor config with the heads. `save_heads` (`:293-298`) stays for compatibility.
- **`model/training/losses.py`**
  - Add `rps_loss` / `rps_reward`, `log_score_loss`, and an entropy helper. Keep the
    existing Brier/MMCE/focal functions.
- **`model/training/supervised.py`**
  - Factor the typed dispatch into `forward_typed` and use it in `compute_loss`,
    `_compute_*_batch`, and `_eval_item`.
  - Add calibration metrics to `eval_epoch_detailed`
    (`model/training/supervised.py:630-719`): ECE, Brier, RPS, entropy, max-probability.
  - Add a `loss:` config block to `train_multitask`
    (`model/training/supervised.py:721-931`), or a `loss_cfg` argument that mirrors
    `CalibrationLossConfig`.
- **`model/scripts/train_multitask.py`**
  - Add `--loss-config` / `--kl-coef` / `--brier-weight` flags if the calibration losses
    are wired into the multitask trainer.
- **`model/configs/`**
  - Add `rl_reranker4b.yaml`; optionally add a `loss:` block to the R2 configs.

### 6.3 Config keys

Top-level keys follow the existing config style (`model/configs/r2_reranker4b.yaml:14-27`):
`model`, `encoder`, `rank`, `epochs`, `lr`, `seed`, `sources`, `sampling`.

New `rl:` block:

| Key | Type | Meaning |
|---|---|---|
| `algorithm` | `exact \| reinforce \| group_reinforce \| awr \| dpo` | estimator |
| `reward.correctness` | float | weight on 0/1 gold correctness |
| `reward.log_score` | float | weight on log π(gold) |
| `reward.brier` | float | weight on negative Brier |
| `reward.rps` | float | weight on negative RPS (score only) |
| `reward.teacher_kl` | float | weight on negative teacher KL |
| `reward.clip` | float or null | per-item reward clip |
| `reward.normalize` | `none \| batch \| group` | reward normalization |
| `kl_coef` | float | weight on KL to the reference policy |
| `entropy_coef` | float | weight on negative entropy |
| `temperature` | float | sampling temperature |
| `group_size` | int | samples per item for `group_reinforce` |
| `baseline` | `none \| batch_mean \| group_mean \| exact` | advantage baseline |

### 6.4 Tests

All tests must run offline, following the existing style
(`pytest.importorskip("torch")`, mock backbone as in `tests/test_multitask_optim.py:11-52`).

- `tests/test_rewards.py`
  - `test_exact_expected_reward_equals_enumeration`
  - `test_log_score_reward_equals_negative_ce`
  - `test_brier_reward_matches_brier_loss`
  - `test_rps_hand_computed` (ordinal distance penalized; CE would not)
  - `test_teacher_log_score_is_distillation`
  - `test_teacher_probability_reward_is_mode_seeking` (documents the trap)
  - `test_teacher_kl_raises_on_missing_keys`
- `tests/test_rl_policy.py`
  - `test_sample_actions_shape_and_seed`
  - `test_temperature_one_matches_softmax`
  - `test_exact_expected_reward_matches_monte_carlo`
  - `test_kl_zero_when_policy_equals_reference`
- `tests/test_rl_trainer.py`
  - `test_one_epoch_runs_offline` with toy `TypedQuestion`s
  - `test_kl_zero_at_init`
  - `test_entropy_bonus_changes_objective`
  - `test_reward_logging_has_one_row_per_item`
- `tests/test_reference_policy.py`
  - `test_reference_logits_equal_initial_policy`
  - `test_reference_parameters_have_no_grad`

---

## 7. Risks and detection

| Risk | Mechanism | Metric | Detection / action |
|---|---|---|---|
| Reward hacking on synthetic templates | policy learns template artifacts rather than decision quality | per-source reward vs held-out accuracy; template-disjoint canary | if reward rises while canary accuracy is flat, stop; hold out templates |
| Calibration collapse | 0/1 reward is invariant to the distribution among wrong answers | ECE, Brier, reliability slope/intercept, mean confidence vs accuracy | ECE +0.02 absolute vs SFT, or confidence rising faster than accuracy, is a stop signal |
| Mode collapse | policy concentrates on one answer | entropy, max-probability, prediction histogram | entropy < 50% of SFT, or max-prob > 0.95 on > 80% of items |
| Reference drift | KL penalty too weak or reward too strong | KL to reference, mean absolute log-ratio | KL > 5 nats or > 10× its initial value |
| Teacher-bias imitation | teacher-KL reward teaches Jev's compression/position bias | agreement with teacher vs gold; position bias; surface-form R² | teacher reward up but gold accuracy down |
| Data leakage | family-level split is not state- or template-disjoint | train/eval overlap audit; canary from a different generator | any overlap on state text or template |
| Score reward hacking with MAE | L1 optimum can be a point mass at the median | RPS, log score, entropy | MAE improves while RPS/entropy collapse |
| Teacher-probability reward trap | `r=q(a)` is maximized by a point mass on argmax q | entropy, KL to teacher | entropy collapses while teacher probability rises |
| Missing teacher keys | `_teacher_loss` fills 0.0 and renormalizes (`model/training/supervised.py:49-50`) | key-coverage count; target-sum check | raise on missing keys; log coverage |
| Non-stationary teacher | Jev model version changes | reward mean/variance per source; version hash | pin the model version; cache responses |
| Numerical instability | bf16 logits, extreme rewards, log(0) | gradient norm, reward min/max, NaN count | clip rewards, use `log_softmax`, skip non-finite items |
| Shared-head interference | choice and score share attention (`model/src/decision_model.py:159-184`) | per-type accuracy/ECE before and after RL | evaluate all types every epoch; per-type KL if needed |
| Overfitting the RL state distribution | sampling only from the training pool | held-out source accuracy | hold out at least one benchmark source from RL |
| Compute blow-up | G samples × K options × external reward calls | items/s, reward calls/s, GPU memory | exact expectation for local rewards; cache reference logits |

The detection principle is simple: **log every component of the reward and every
calibration metric per epoch, and compare against the SFT reference on the same items.**
A single scalar reward curve is not enough to distinguish learning from hacking.

---

## 8. Honest gap list

What the repository does **not** have today, and what it would take.

| # | Gap | Why it blocks RL | Evidence | Fix |
|---|---|---|---|---|
| 1 | No action sampler | cannot form a policy-gradient estimate | `forward_*` return logits only (`model/src/decision_model.py:323-341`); no `torch.multinomial`/`Categorical` in `model/` | `sample_actions` |
| 2 | No raw-probability API | rewards computed from `predict` see rounded/clamped values | `predict` clamps noul to `[0.01,0.99]` and rounds (`model/src/decision_model.py:351`, `:359`, `:370`) | `predict_probs` |
| 3 | No reference policy | KL-to-SFT is impossible | `save_heads`/`load_heads` handle one model (`model/src/decision_model.py:382-402`); no frozen copy | `ReferencePolicy` |
| 4 | No model-config serialization | a reference cannot be reconstructed reliably | trainer saves `best_heads.pt` + optimizer only (`model/training/supervised.py:906-920`) | `save_checkpoint` with config |
| 5 | No reward buffer or per-item logging | cannot diagnose hacking or credit assignment | `train_epoch` accumulates losses only (`model/training/supervised.py:322-454`) | JSONL reward log |
| 6 | No KL-to-reference or entropy in the loop | calibration cannot be preserved | `losses.py` has Brier/MMCE/focal; `_teacher_loss` is KL to teacher (`model/training/supervised.py:47-52`) | add both |
| 7 | No RPS / ordinal loss | score is trained as unordered classes | score CE (`model/training/supervised.py:149`, `:316`); no RPS in `losses.py` | add RPS |
| 8 | Score is not evaluated continuously in training | cannot tell whether RL helps the primary primitive | `_eval_item` uses argmax (`model/training/supervised.py:611-624`); the multitask trainer never imports `score_mae` (`model/evaluation/accuracy.py:22`) | add MAE/RPS to `eval_epoch_detailed` |
| 9 | No calibration metrics in the multitask loop | calibration collapse would be invisible | `eval_epoch_detailed` returns loss/accuracy (`model/training/supervised.py:688-719`) | add ECE/Brier/entropy |
| 10 | `eval_trained.py` skips score | post-hoc evaluation cannot see the score primitive | `if q_type not in ("noul", "choice")` (`model/scripts/eval_trained.py:37`) | add score branch |
| 11 | No preference data | DPO has no pairs | `TypedQuestion` has no chosen/rejected (`data/format.py:31-39`) | collect only if needed |
| 12 | Teacher probs only on synthetic | dense reward covers a minority of the mix | teacher files are `data/benchmarks/synthetic/*_jev_labels.jsonl`; `load_synthetic` (`data/pipeline.py:1125-1169`) | expand coverage or treat as regularizer |
| 13 | Single teacher, no ensemble | reward inherits one teacher's bias | `_select_teacher_probs` priority (`data/pipeline.py:1062-1083`) | weighted multi-teacher KL |
| 14 | No cost matrix or decision utility | no reward that is not already a label | `TypedQuestion` fields (`data/format.py:31-39`) | add a cost field |
| 15 | No unlabeled verifiable outcomes or environment | online RL has nothing to explore against | no such code | new data source |
| 16 | No temperature parameter in the forward path | exploration and calibration scaling are fixed at T=1 | softmax calls in `model/src/heads.py:213`, `:324`, `:460` | add `temperature` argument |
| 17 | Choice/score contexts are encoded per item | RL throughput is limited | `_compute_choice_batch` / `_compute_score_batch` loop per item (`model/training/supervised.py:263-320`); only the noul path batches (`:219-237`) | batch contexts |
| 18 | No RL tests | regressions would be invisible | `tests/` has no RL tests | add the tests in Section 6.4 |
| 19 | No Jev API budget/rate control | external reward could be unbounded | `JevClient` (`probing/scripts/jev_client.py:58`) has no budget wrapper | add cache + budget |
| 20 | No distributed training | large RL sweeps are slow | single-GPU loops throughout | not critical for the first experiments |

The first six are blockers for any RL loop. Items 7–10 are blockers for doing RL on the
score primitive, which is the user's stated priority. Items 11–15 are the reason online
RL has no signal beyond distillation today.

---

## 9. Open questions

1. **Does RPS beat CE on the score benchmarks?** This is the cheapest high-value
   experiment and directly tests the "score is fundamental" premise. It needs no RL.
2. **Does KL-to-teacher improve calibration on synthetic without hurting accuracy?**
   The repo already lists this as an open question
   (`research/04-model-replication.md:420`).
3. **Is there a reward signal that is not a label?** If not, online RL is distillation
   with extra variance. The answer determines whether Stage 3 happens at all.
4. **Does exact expectation beat REINFORCE at equal compute?** Theory says yes for K≤255;
   measure it on a small sweep before committing to one estimator.
5. **Does LoRA change the SFT/RL trade-off?** A LoRA-adapted backbone can drift further
   from the reference, making the KL term more important.
6. **How much of Laya's calibration gain is temperature fitting rather than RL?** The
   research notes already flag this caveat
   (`research/03-open-source-ecosystem.md:108-118`). Reproduce the temperature-only
   baseline before attributing gains to RL.
7. **Does a unified head help or hurt RL across types?** Shared parameters mean a reward
   on one type can move the others; measure per-type KL and accuracy.

---

## 10. References

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for
  connectionist reinforcement learning. *Machine Learning*.
- Shao, Z. et al. (2024). DeepSeekMath: Pushing the limits of mathematical reasoning in
  open language models. arXiv:2402.03300. (GRPO.)
- Rafailov, R. et al. (2023). Direct preference optimization: Your language model is
  secretly a reward model. arXiv:2305.18290.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *JASA*.
- Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories.
  *Journal of Applied Meteorology*. (RPS.)
- Kumar, A. et al. (2018). Trainable calibrated output distributions for classification.
  arXiv:1806.02032. (MMCE.)
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to
  weigh losses for scene geometry and semantics. CVPR.
- Hu, E. J. et al. (2021). LoRA: Low-rank adaptation of large language models.
  arXiv:2106.09685.
- Repository: `research/03-open-source-ecosystem.md:94-118` (RLCD reproduction
  landscape), `research/04-model-replication.md:280-290` (LTR/RL table),
  `research/02-architecture-analysis.md:425-437` (RLCD training gap).
