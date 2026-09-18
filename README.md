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

## Links

- [TypeSafe docs](https://docs.typesafe.ai)
- [Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)
- [System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python)
- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe)
