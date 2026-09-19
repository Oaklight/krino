# jev-explore

Exploration and experiments with [TypeSafe.ai](https://typesafe.ai)'s Jev — the first System One model.

## What is Jev?

Jev is a "System One" model that returns typed, probabilistic decisions instead of generated text. It evaluates a `state` against typed `questions` and returns structured `answers` with calibrated probabilities.

Three primitives:

| Type | Purpose | Returns |
|------|---------|---------|
| **Noul** | Yes/no probability | `noul` (0–1 float) |
| **Choice** | Pick one from a set | `choice`, `probabilities`, `confidence` |
| **Score** | Rate on ordered levels | `score`, `probabilities`, `legend`, `confidence` |

## Setup

```bash
cp .env.example .env
# Fill in your API key
```

## Controlled black-box suite

The [controlled probe suite](probing/controlled/README.md) preregisters five stdlib-only experiments and records raw HTTP evidence without credentials. The suite is implemented and unit tested; full live collection has not been run, and generated live results should remain uncommitted.

Start by inspecting the deterministic smoke schedule:

```bash
python probing/scripts/controlled/probe_1_output_serialization.py --mode smoke --seed 20260918 --dry-run
```

## Links

- [TypeSafe docs](https://docs.typesafe.ai)
- [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)
- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe)
