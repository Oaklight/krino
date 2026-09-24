> 中文版本请见 [README_zh.md](README_zh.md)

# Probing

Black-box probing suite for TypeSafe AI's Jev decision model. All experiments observe external HTTP behavior — no model weights or internal access required.

## What this does

Sends structured requests to the Jev API and analyzes responses to infer operational properties: output quantization, option interaction effects, surface-form sensitivity, execution topology, and generation signatures.

## Structure

```
probing/
├── scripts/              # Probe implementations
│   ├── probe_01_*.py     #   Standard probes (13 experiments)
│   ├── probe_02_*.py
│   ├── ...
│   ├── controlled/       #   Preregistered controlled probes (5 experiments)
│   │   ├── probe_1_output_serialization.py
│   │   ├── probe_2_option_interaction.py
│   │   ├── probe_3_likelihood_sensitivity.py
│   │   ├── probe_4_execution_topology.py
│   │   └── probe_5_generation_signatures.py
│   ├── jev_client.py     #   Jev API client (stdlib only)
│   └── httpclient.py     #   HTTP client
├── results/              # Raw JSONL outputs
│   └── controlled/       #   Controlled probe outputs
└── controlled/
    └── README.md         # Preregistration: hypotheses, falsification matrix
```

## Running probes

All probe scripts use only the Python standard library. Set `TYPESAFE_API_KEY` in your environment.

```bash
# Standard probes
python probing/scripts/probe_01_token_accounting.py
python probing/scripts/probe_02_noul_precision.py

# Controlled probes (preregistered, with safety flags)
python probing/scripts/controlled/probe_1_output_serialization.py \
    --mode smoke --seed 20260918 --dry-run

# Live run (requires explicit confirmation)
python probing/scripts/controlled/probe_2_option_interaction.py \
    --mode full --seed 20260918 \
    --output probing/results/controlled/probe_2.jsonl \
    --confirm-live
```

Flags: `--mode smoke|full`, `--seed`, `--output`, `--dry-run`, `--confirm-live`.

## Key findings

- **21 probes, 5,620 API calls** across 3 rounds with rigor self-audits
- Architecture fingerprinting identified Qwen-family backbone, typed decision heads, and RLCD calibration
- Ordering bias is reduced but not eliminated
- Probabilities quantized to 0.01 resolution
- See [controlled/README.md](controlled/README.md) for the preregistered falsification matrix
