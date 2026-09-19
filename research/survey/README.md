# Related Projects Survey

Tracking open-source reproductions, alternative decision models, and community efforts related to TypeSafe's Jev and the System One paradigm.

## Reproduction Approaches

Three architectural strategies have emerged:

| Strategy | Examples | Pros | Cons |
|---|---|---|---|
| **Logit readout** (frozen LM) | OpenJev, LitJev, zhihz/openjev, SemIf | No training, fast setup | Uncalibrated, surface-form sensitive |
| **Trained heads/adapters** | NanoJev, Nimble, jevlike, jevbetter, kev | Calibration possible, customizable | Requires labeled data, GPU time |
| **Non-autoregressive** | razorback16/openjev (DiffusionGemma) | Architecturally closest to Jev's claims | Requires special models, less explored |

## Project Index

Detailed survey documents:

- [reproductions.md](reproductions.md) — Open Jev reproductions (NanoJev, OpenJev, jevlike, Nimble, etc.)
- [decision-models.md](decision-models.md) — Alternative decision/classification models and approaches
- [community.md](community.md) — Awesome lists, trackers, benchmarks, and community resources

## Sources

- [awesome-typesafe](https://github.com/AbdelStark/awesome-typesafe) — official community list
- [cobanov/awesome-jev](https://github.com/cobanov/awesome-jev) — 100+ projects cataloged
- [thevibeworks/awesome-typesafe-jev](https://github.com/thevibeworks/awesome-typesafe-jev) — alternative list
