# Canonical corpus: licence position per source group

Status: 2026-11-06. The audit is advisory; the policy below is a decision.
Nothing has been removed from the corpus. Counts are canonical records after id
dedup (RACE ids are suffixed; the passage id is the group), with synthetic items
attributed to the domain they were seeded from. Reproduce with
`scripts/corpus_licence_audit.py`.

## Policy decision: research use only

Decided 2026-11-06 by the project owner. The canonical corpus stays at 243,366
items and knowingly includes source groups whose terms are restrictive or
unresolved; the data cost real money to assemble and Krino has no commercial
goal to protect. Krino is research use only, and any release of model weights or
of the corpus must carry that restriction and point at this document. Revisit the
decision if the project gains attention or a commercial goal appears; the
manifest below is what the narrower builds will use then.

## Method

- Canonical assembly is the one in `research/25-corpus-inventory.md`: target 30K,
  residual 20K, typed-decisions, synthetic, then the remaining corpora, deduped by
  item id (RACE ids are suffixed before dedup).
- A synthetic item is built from a state drawn from its domain's seed dataset.
  `data/synthetic.py:_load_seed_states` calls the seed loader, and
  `data/synthetic_templates.py` maps each seeded domain to that loader. Every
  family in a seeded domain carries `_seeded: true`; the counts below confirm that
  the 14 seeded domains contain no unseeded families.
- Consequence: synthetic items inherit the licence of their seed dataset. A
  seeded domain is counted under its seed source below, not under "synthetic".
- The 9 pure-generation domains have no third-party seed. They are the `project`
  row.
- `out_of_scope` abstention records borrow a state from a different source than
  their question. Their `source` field names the question's dataset, so the
  attribution below is by question source. The state's source is not recorded.
  Rebuilding the abstention set after exclusions removes this ambiguity.
- Verdicts: **permissive** = commercial use allowed by the stated terms;
  **restrictive** = commercial use restricted or requires a separate agreement;
  **unresolved** = no licence stated, or the stated position is contradictory or
  cannot be relied on. Unresolved groups stay in the corpus; they are the list to
  work when the policy is revisited, not a blocker for research use. CC BY-SA is
  marked permissive with a share-alike note; the obligation it creates for model
  weights is not settled here.

## Table

| group | records | licence (as stated) | source read | verdict |
|---|---:|---|---|---|
| `anli` | 8,185 | CC BY-NC 4.0 | https://huggingface.co/datasets/facebook/anli; LICENSE "cc-4 Attribution-NonCommercial" | restrictive |
| `contractnli` | 11,181 | CC BY-NC-SA 4.0 | https://huggingface.co/datasets/kiddothe2b/contract-nli (`license: cc-by-nc-sa-4.0`) | restrictive |
| `apigen` | 16,081 | CC BY-NC 4.0; "This release is for research purposes only" | https://huggingface.co/datasets/Salesforce/APIGen-MT-5k | restrictive |
| `race` | 2,546 | "RACE dataset is available for non-commercial research purpose only." | https://huggingface.co/datasets/ehovy/race | restrictive |
| `agnews` | 1,118 | "...and any other non-commercial activity." | https://huggingface.co/datasets/fancyzhx/ag_news | restrictive |
| `mednli` | 9,422 | PhysioNet Credentialed Health Data License 1.5.0; only credentialed users who sign the DUA | https://physionet.org/content/mednli/1.0.0/ | restrictive |
| `yelp` | 1,070 | "intended for educational use"; HF tag `other`, `license_details: yelp-licence` | https://business.yelp.com/data/resources/open-dataset/ | restrictive |
| `multirc` | 6,184 | SuperGLUE: "it is our understanding that these licenses allow for their use and redistribution in a research context" | https://huggingface.co/datasets/aps/super_glue | restrictive |
| `hellaswag` | 2,571 | card claims MIT, but the GitHub repository is blocked under a DMCA notice (2026-09-14, WikiHow) | https://huggingface.co/datasets/Rowan/hellaswag; https://github.com/rowanz/hellaswag | unresolved |
| `sst2` | 1,876 | `license: unknown`; the `stanfordnlp/sentiment-treebank` repository has no LICENSE file | https://huggingface.co/datasets/stanfordnlp/sst2 | unresolved |
| `sst5` | 1,741 | no licence field; source is the Stanford Sentiment Treebank | https://huggingface.co/datasets/SetFit/sst5 | unresolved |
| `swag` | 1,890 | MIT in the release repository, which contains `data/`; the HF card still says `unknown` | https://github.com/rowanz/swagaf/blob/master/LICENSE | permissive |
| `piqa` | 1,425 | Academic Free License 3.0 | https://yonatanbisk.com/piqa/ | permissive |
| `logiqa` | 10,931 | no licence in the card or the repository | https://huggingface.co/datasets/lucasmccabe/logiqa | unresolved |
| `codesearchnet` | 7,468 | "each repository has its own license. Example-wise license information is not (yet) included" | https://huggingface.co/datasets/code-search-net/code_search_net | unresolved |
| `quality` | 2,380 | no licence file in the repository | https://github.com/emozilla/quality | unresolved |
| `2wiki` | 345 | no licence in the official repository; the `xanhho/2WikiMultihopQA` mirror states Apache-2.0 | https://github.com/Alab-NLP/2WikiMultiHopQA; https://huggingface.co/datasets/xanhho/2WikiMultihopQA | unresolved |
| `stsb` | 1,430 | GLUE card: "We refer users to the original licenses"; the STS-B mirror states `unknown` | https://huggingface.co/datasets/nyu-mll/glue; https://huggingface.co/datasets/mteb/stsbenchmark-sts | unresolved |
| `mnli` | 1,955 | MIT | https://huggingface.co/datasets/nyu-mll/multi_nli | permissive |
| `mmlu` | 9,109 | MIT | https://huggingface.co/datasets/cais/mmlu | permissive |
| `mmlu_pro` | 9,965 | MIT | https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | permissive |
| `commonsenseqa` | 10,404 | MIT | https://huggingface.co/datasets/tau/commonsense_qa | permissive |
| `boolq` | 8,397 | CC BY-SA 3.0 (share-alike) | https://huggingface.co/datasets/google/boolq | permissive |
| `fever` | 8,732 | CC BY-SA 3.0; GPL-3.0 (code) | https://huggingface.co/datasets/copenlu/fever_gold_evidence | permissive |
| `tabfact` | 8,976 | CC BY 4.0 | https://huggingface.co/datasets/wenhu/tab_fact | permissive |
| `banking77` | 2,979 | CC BY 4.0 | https://huggingface.co/datasets/legacy-datasets/banking77 | permissive |
| `arc` | 8,114 | CC BY-SA 4.0 (share-alike) | https://huggingface.co/datasets/allenai/ai2_arc | permissive |
| `hotpotqa` | 7,140 | CC BY-SA 4.0 (share-alike) | https://huggingface.co/datasets/hotpotqa/hotpot_qa | permissive |
| `musique` | 476 | CC BY 4.0 | https://github.com/StonyBrookNLP/musique/blob/main/LICENSE | permissive |
| `longbench` | 426 | MIT | https://github.com/THUDM/LongBench/blob/main/LICENSE | permissive |
| `drop` | 6,820 | CC BY-SA 4.0 (share-alike) | https://huggingface.co/datasets/ucinlp/drop | permissive |
| `winogrande` | 1,393 | Apache-2.0 | https://github.com/allenai/winogrande/blob/master/LICENSE | permissive |
| `typed_decisions` | 8,037 | Apache-2.0 | https://huggingface.co/datasets/LocalLLaMA/typed-decisions | permissive |
| `openjev_authored144` | 41 | project-authored; repository MIT, no third-party terms found | `scripts/build_authored_holdout.py` | permissive |
| `project` (9 pure-generation synthetic domains) | 62,558 | project-generated; repository MIT; generation-provider terms not audited | `data/synthetic.py` | permissive |

## Exclusion scenarios (contingency, not applied)

The owner decided on 2026-11-06 to keep every group (policy above). The counts
below are the fallback if that decision is revisited.

Current total: **243,366**. Restrictive groups total **55,787** records (22.92%,
8 groups); unresolved groups total **28,742** records (11.81%, 8 groups);
permissive groups total **158,837** records (65.27%, 19 groups).

`data/licence_manifest.json` records which builds may include each group, and
`scripts/build_canonical_corpus.py --build <name>` filters the canonical corpus
by it. Dry runs of the three builds reproduce 243,366 / 187,579 / 158,837 items.

| scenario | excluded | remaining | noul excluded | choice excluded | score excluded |
|---|---:|---:|---:|---:|---:|
| restrictive only | 55,787 | 187,579 | 27,118 | 24,717 | 3,952 |
| restrictive + unresolved | 84,529 | 158,837 | 38,181 | 36,892 | 9,456 |

Remaining per-type counts under each scenario:

| scenario | noul | choice | score |
|---|---:|---:|---:|
| current | 113,594 | 100,257 | 29,515 |
| restrictive only | 86,476 | 75,540 | 25,563 |
| restrictive + unresolved | 75,413 | 63,365 | 20,059 |

Excluding restrictive sources removes, among others, the whole `apigen` tool-call
corpus (16,081) and the ANLI-, ContractNLI- and MedNLI-seeded synthetic domains.
The two confirmed suspicions:

1. **MedNLI is not openly licensed.** PhysioNet states "Only credentialed users
   who sign the DUA can access the files", under the PhysioNet Credentialed Health
   Data License 1.5.0 and a required CITI "Data or Specimens Only Research"
   training. Redistribution of the derived rows needs separate clearance.
2. **Seeded synthetic items inherit the seed's terms.** The 14 seeded domains have
   no unseeded families, and their states come from the seed loader. ANLI (6,853
   synthetic items), ContractNLI (6,287) and MedNLI (6,744) therefore carry
   restrictive terms into the synthetic corpus, in addition to the sampled rows
   in the 20K/30K.

## Notes and limits

- HellaSwag's HF card claims MIT, but the repository is DMCA-blocked over WikiHow
  text. The MIT claim covers code; the data's position is not reliable. Marked
  unresolved rather than permissive.
- PIQA's project page states the Academic Free License 3.0. SWAG's release
  repository (`rowanz/swagaf`) is MIT and contains the `data/` directory; the
  AllenAI HF card's `unknown` predates that check. Both are marked permissive.
- CodeSearchNet is per-example GitHub code under per-repository licences that the
  dataset does not record. It cannot be cleared source-by-source from the corpus.
- AG News says "any other non-commercial activity"; RACE says "non-commercial
  research purpose only"; SuperGLUE says "research context". These are stated
  positions, not formal licences, and are marked restrictive.
- Yelp's page says "intended for educational use" rather than granting a licence.
- The `project` row covers 9 pure-generation domains. The generation provider's
  output terms were not audited; the repository's MIT licence covers the code, not
  necessarily the generated text.
- Counts are the current file attribution. The abstention set (6,000 records) is
  rebuilt from the remaining corpora after any exclusion, so its post-exclusion
  size will differ from a simple subtraction. The 2,000 `out_of_scope` records can
  carry a donor state from any source; rebuilding removes that ambiguity.
- The six head checkpoints already published on the model hub were trained on the
  earlier 20K corpus, which also carried restrictive sources. They are not clean
  under this policy either; no action is required today.
- No exclusion has been applied. The owner decided on 2026-11-06 to keep every
  group (policy above).
