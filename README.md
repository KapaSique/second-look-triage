# Second Look — a triage safety-net & audit system

Submission for the [Triagegeist](https://www.kaggle.com/competitions/triagegeist) hackathon
(Laitinen-Fredriksson Foundation). A clinically-grounded decision-support system for
emergency-department triage that prioritizes **safety, calibration, and honesty** over raw
accuracy — because on the provided synthetic data accuracy is trivially solved
(chief-complaint text predicts the ESI label with ~100% accuracy), while the real clinical
problem is catching the occult high-risk patient and behaving safely when data is missing.

> Full rationale: [`docs/superpowers/specs/2026-06-13-triagegeist-second-look-design.md`](docs/superpowers/specs/2026-06-13-triagegeist-second-look-design.md)

## What's here

| Path | Purpose |
|---|---|
| `src/` | Library modules: data prep, model core, red-flag NLP, decision policy, audit, explain |
| `kaggle/` | Kaggle kernel + dataset scaffolding (heavy compute runs on Kaggle GPU) |
| `notebooks/` | Source of the public end-to-end submission notebook |
| `app/` | Gradio interactive demo (deployed to a Hugging Face Space) |
| `writeup/` | Competition writeup + cover image |
| `figures/` | Generated figures |

## Data

Competition data is **not** redistributed here (prohibited by the competition rules). Download
it from the [competition data page](https://www.kaggle.com/competitions/triagegeist/data) into
`data/`.

## Status

🚧 Under active development (deadline 2026-06-21).
