# Controlled Jev black-box probes

This suite preregisters five tests before collecting new live data. It measures externally observable behavior; it does not claim that an HTTP response uniquely identifies an internal architecture.

## Running

Runtime code uses only the Python standard library and reads `TYPESAFE_API_KEY` from the process environment. It never loads `.env` files.

```bash
python probing/scripts/controlled/probe_1_output_serialization.py --mode smoke --seed 20260918 --dry-run
python probing/scripts/controlled/probe_2_option_interaction.py --mode full --seed 20260918 --output probing/results/controlled/probe_2.jsonl --confirm-live
python probing/scripts/controlled/probe_3_likelihood_sensitivity.py --analyze-only probing/results/controlled/probe_3.jsonl --reference-probabilities qwen-reference.json --output probing/results/controlled/probe_3-reanalysis.jsonl
```

All runners support `--mode smoke|full`, `--seed`, `--output`, and `--dry-run`. They print an estimated request count first. A live request additionally requires `--confirm-live`; no generated live results belong in commits. Full mode uses 100 semantic items where item-level inference matters and repeated randomized blocks for timing experiments. Smoke mode uses two or three items/cells where practical.

Each JSONL record contains the probe and version, preregistered prediction, seed, condition and item IDs, exact non-secret request payload and its SHA-256, parsed response, raw response SHA-256 and byte length, selected response headers, HTTP status, TTFB, total time, and errors. Authorization headers are never serialized.

Two-decimal probabilities represent intervals `[max(0,p-0.005), min(1,p+0.005)]`. Analyses separate definite changes (non-overlapping or sign-definite derived intervals) from effects that display rounding can hide.

## Preregistered probes

1. **Output serialization:** opaque question IDs and option keys across ASCII, Unicode, combining, emoji, and punctuation families; matched input padding; key length; latency against visible response bytes. Weak scaling disfavors full visible JSON generation but remains compatible with compact internal decoding followed by deterministic serialization.
2. **Option interaction:** pairwise baseline versus irrelevant, dominated, exact-duplicate, and paraphrase additions at balanced insertion positions. Strict IIA is tested using interval-censored `log(P(A)/P(B))`; duplicate-family mass is reported separately so softmax normalization is not mistaken for interaction.
3. **Likelihood sensitivity:** semantically equivalent option descriptions vary in length, grammaticality, and base-rate plausibility under balanced Latin-square assignments and opaque label classes. Jev collection is implemented. An optional JSON reference file supports later Qwen correlation/regression without importing torch.
4. **Execution topology:** randomized blocked factorial design over state length S, question count Q, option count K, and description length L, fitting S×Q, S×K, and Q×K features. Early/late decisive cues and reference remapping are included, but causal versus bidirectional processing is non-identifiable from HTTP behavior alone.
5. **Generation signatures:** many matched output entries vary output position, preceding-field complexity, and number of fields. Analysis records lexical fingerprints, exact-cent sums and possible correction patterns, plus smoothly interpolated evidence cases to look for heaping beyond unavoidable 0.01 rounding.

## Falsification matrix

The letters denote hypotheses, not conclusions:

- **A — joint/listwise scorer:** options are jointly represented and scored by a typed head.
- **B — causal LM direct readout:** a causal language model supplies token/logit-derived probabilities without visible text decoding.
- **C — bidirectional encoder plus typed heads:** a bidirectional representation feeds fixed output heads.
- **D — autoregressive constrained decoding:** probabilities/JSON are emitted through a constrained token-generation path.

| Observation | A | B | C | D |
|---|---|---|---|---|
| Definite strict-IIA shifts after irrelevant additions | Supports joint interaction; falsifies strict independent utility, not A broadly | Compatible if options share context | Compatible if options share context | Compatible if decoding couples fields |
| A:B odds stable but duplicate-family mass conserved | Compatible with independent utilities inside a listwise normalizer | Compatible | Compatible | Compatible; weak discriminator |
| Total latency grows with visible output bytes after matched input | Compatible if serialization dominates | Disfavored if readout and serialization are fixed-cost | Disfavored if typed heads and serialization are fixed-cost | Supports D, but network/server batching is a confound |
| TTFB flat while total time grows with output bytes | Weak evidence for streaming/generation after first byte | Disfavors one-shot response serialization | Disfavors one-shot response serialization | Supports D if responses actually stream |
| Surface wording likelihood predicts option mass under balancing | Compatible | Supports B | Compatible through pretrained representations | Supports D; tokenization can affect constrained generation |
| S×Q latency dominates while S×K is weak | Supports shared-state/per-question branches | Compatible with prefix reuse | Compatible with shared encoding plus heads | Compatible with cached-prefix decoding |
| Late cue outperforms early cue | Compatible with positional bias | Compatible with causal recency | Compatible with learned position effects | Compatible with causal decoding context |
| Stable last-field or maximum-entry rounding correction | Compatible with deterministic serializer | Compatible | Compatible | Supports D only if correction is not postprocessing |
| No position/lexical/timing signatures | Compatible | Compatible | Supports typed-head implementation weakly | Disfavors, but does not rule out, optimized D |

No single row confirms a hypothesis. Conclusions should update only when multiple preregistered contrasts jointly disfavor alternatives.

## Operational properties versus architecture

The suite can establish operational properties such as exact key preservation, probability quantization, strict-IIA violations, duplicate-family behavior, request/response scaling, and sensitivity to position or surface form. These are valid regardless of implementation.

The following pairs remain non-identifiable from these HTTP probes alone:

- causal attention versus bidirectional attention when the complete request is available before readout;
- a typed head versus constrained decoding followed by deterministic rounding when both return the same bytes and timing envelope;
- native model quantization versus API-layer rounding;
- shared-prefix KV caching versus another server-side batching scheme with the same scaling;
- RLCD-induced debiasing versus prompt templates, supervised post-training, calibration layers, or deterministic postprocessing;
- Qwen-family weights versus another model with similar tokenizer, language profile, and positional behavior.

Claims about exact backbones, attention masks, training objectives, or decoding algorithms therefore remain hypotheses unless supported by source disclosure, model artifacts, or a uniquely identifying intervention.
