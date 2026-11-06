# Decision Index public suite: licence position per upstream source

Source of truth: `docs/suite.md` at kit commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d`
(the kit's own words: "Licence notes are what the pinned sources state (or 'see repository
LICENSE' when only a licence file is present, and 'not stated' when none was found). Verify
before redistributing; none of these terms transfer to this repository's MIT licence.").

The rebuild is a local evaluation job. The suite stays in a private work directory on the
compute host; it is not committed, not uploaded, and not used for training. The code is
MIT; the data is not.

The rebuild fetches 44 upstream entries: the 37 of edition 0.1 (including ForecastBench,
which 0.3 retires at read time) and the seven added in 0.2. The generated Home appliance
benchmark needs no download.

| # | Benchmark | Upstream source | Licence position |
|---:|---|---|---|
| 1 | BFCL | gorilla-llm/gorilla | Apache-2.0 (repo LICENSE) |
| 2 | ToolRet | mangopy/ToolRet-Queries, ToolRet-Tools, tool-retrieval-benchmark | see repository LICENSE; aggregates upstream tool datasets with their own terms |
| 3 | API-Bank | AlibabaResearch/DAMO-ConvAI | see `api-bank/LICENSE` in the repo |
| 4 | BANKING77 | PolyAI-LDN/task-specific-datasets | CC BY 4.0 |
| 5 | CLINC150+OOS | clinc/oos-eval | CC BY 3.0 (per dataset card) |
| 6 | RouterBench | withmartian/routerbench | see repository LICENSE |
| 9 | Home appliance | generated locally, no download | MIT (this repository) |
| 10 | SGD/SGD-X | google-research-datasets/dstc8-schema-guided-dialogue | CC BY-SA 4.0 |
| 11 | ContractNLI | stanfordnlp/contract-nli | see repository LICENSE and dataset page terms |
| 12 | ANLI | facebook/anli | **CC BY-NC 4.0 (non-commercial)** |
| 20 | BPoMP | Zenodo record 7299879 | per the Zenodo record |
| 21 | Humicroedit | SemEval-2020 Task 7 dataset | SemEval-2020 Task 7 terms |
| 22 | POP909-CL | AndyWeasley2004/POP909-CL-Dataset | see repository LICENSE; POP909 underlying data is **research-use** |
| 23 | cfcolor | dgp.toronto.edu cfcolor | per the project page |
| 24 | MMLU | cais/mmlu | MIT |
| 25 | GPQA Diamond | idavidrein/gpqa | CC BY 4.0; authors ask not to publish the questions in plain text |
| 26 | ARC-Easy | allenai/ai2_arc | CC BY-SA 4.0 |
| 27 | ARC-Challenge | allenai/ai2_arc | CC BY-SA 4.0 |
| 28 | WinoGrande | allenai/winogrande | see dataset card |
| 29 | HellaSwag | Rowan/hellaswag | MIT |
| 30 | GSM8K | openai/gsm8k | MIT |
| 31 | ChessBench | google-deepmind/searchless_chess + GCS bag | see repository LICENSE (Apache-2.0 code) |
| 32 | MuSR | Zayne-sprague/MuSR | MIT |
| 33 | SATA-Bench | sata-bench/sata-bench | see repository LICENSE |
| 34 | SimpleBench | simple-bench/SimpleBench | MIT |
| 36 | BRIGHT | xlangai/BRIGHT | CC BY 4.0 |
| 37 | Amazon ESCI | amazon-science/esci-data | Apache-2.0 |
| 38 | ACOS | NUSTM/ACOS | **not stated in the repository** |
| 39 | FinEntity | yixuantt/FinEntity | **not stated in the repository** |
| 40 | iSarcasmEval | iabufarha/iSarcasmEval | see repository LICENSE |
| 41 | VAST | emilyallaway/zero-shot-stance | **not stated in the repository** |
| 42 | NLI4CT | ai-systems/Task-2-SemEval-2024 | SemEval-2024 Task 2 terms |
| 43 | CRUXEval | facebookresearch/cruxeval | MIT |
| 44 | CLadder | causalNLP/cladder | see repository LICENSE |
| 45 | HLE | cais/hle | MIT (per dataset card); **gated access** |
| 48 | ForecastBench | forecastingresearch/forecastbench-datasets | see repository LICENSE |
| 50 | Habermas Machine | google-deepmind/habermas_machine | see repository LICENSE |
| 57 | MMLU-Pro | TIGER-Lab/MMLU-Pro | MIT |
| 58 | BBH | suzgunmirac/BIG-Bench-Hard | MIT; **carries the BIG-bench canary, never train on it** |
| 59 | RAGTruth | ParticleMedia/RAGTruth | MIT annotations; contexts include CNN/Daily Mail, **MS MARCO (non-commercial)** and the **Yelp Open Dataset (no redistribution)**; rebuild locally, never republish the rows |
| 61 | HoVer | hover-nlp/hover + Wikipedia DB | CC BY-SA 4.0 (data), Wikipedia text CC BY-SA |
| 62 | When2Call | nvidia/When2Call | CC BY 4.0 |
| 64 | New Yorker | jmhessel/newyorker_caption_contest | CC BY 4.0 annotations; no cartoon images used |
| 56 | PhishNChips | AreLit/PhishNChips + anisselbd/jev-phishing-bench | per the dataset's `SOURCE_LICENSES.md` (project content MIT; URL sources cleared for academic use with attribution) |

## What this means for us

- **Non-commercial:** ANLI (CC BY-NC 4.0), RAGTruth's MS MARCO contexts, and POP909's
  underlying data. Use for internal evaluation only.
- **No redistribution:** RAGTruth's Yelp contexts. Keep the rebuilt suite private; do not
  upload it to a public Hub repo.
- **Share-alike:** SGD, ARC-Easy, ARC-Challenge and HoVer carry CC BY-SA 4.0 terms. If we
  ever publish derived rows, the share-alike terms apply to those rows.
- **Gated / canary:** HLE requires accepting the Hub terms; BBH carries the BIG-bench
  canary. Do not train on either.
- **Not stated:** ACOS, FinEntity and VAST state no licence in the repository. Treat them
  as "all rights reserved" until the maintainers clarify; evaluation use only.
- **Plain-text restriction:** GPQA asks that its questions not be published in plain text.
  Do not paste GPQA items into issues, PRs or reports.

The safe operating position is the one already in `research/26`: code MIT, data private,
evaluation only, no training, no republishing.
