# Second Look — Design Spec

**Competition:** [Triagegeist](https://www.kaggle.com/competitions/triagegeist) (Laitinen-Fredriksson Foundation) · judge-scored hackathon · $10,000 · deadline 2026-06-21 22:00 UTC
**Date:** 2026-06-13 · **Author:** team `ssstelmah` / `KapaSique`
**Status:** Approved design (brainstorming → spec)

---

## 1. What kind of competition this is (and why it dictates everything)

Triagegeist is **not a metric leaderboard**. There is no scored submission — confirmed: `kaggle competitions leaderboard triagegeist` returns *no results*, and `NOTE.md` states *"This is a Hackathon with no provided dataset."* Submissions are judged by a panel against a **100-point rubric**:

| Criterion | Points |
|---|---|
| Clinical Relevance | 25 |
| Technical Quality | 30 |
| Documentation & Writeup Quality | 20 |
| Insight & Findings | 15 |
| Novelty & Impact Potential | 10 |

**Implication:** raw predictive accuracy maps to at most part of the 30-point Technical bucket. The other 70 points reward clinical framing, honesty, insight, communication, and novelty. We win on the *whole package*, not on a classifier.

## 2. Key empirical findings from the provided synthetic data (these drive the design)

Light EDA on `train.csv` (80k rows, 40 cols) + `chief_complaints.csv` + `patient_history.csv` (25 comorbidity flags):

1. **The chief-complaint free text alone predicts the label with ~100% accuracy** (TF-IDF + logistic on a 25% holdout: acc 1.000, macro-F1 0.998). Vitals-only ≈ 0.73 acc. The synthetic generator assigned acuity essentially *from the complaint* (complaints literally name diagnoses: "acute NSTEMI", "ovarian torsion"). → **Predictive accuracy is a solved, non-differentiating objective. We say so, openly, and pivot to safety/calibration/robustness.**
2. **Informative missingness is a deployment trap.** Vitals (BP etc.) are missing for **0%** of acuity 1–3 but **~12%** of acuity 4–5. "No BP measured" is a near-perfect proxy for "not sick" *in this dataset only*. A naive model exploits it; in a real, chaotic ED, missingness has other causes → systematic undertriage risk. Centerpiece honest finding.
3. **Labels are near-deterministic in severity; no rater/site variability.** NEWS2 by acuity is cleanly monotonic (13.6 → 0.35). 50 nurses / 5 sites show essentially identical mean acuity (std 0.02). → We will **not** claim to have "found" undertriage or rater bias *in the data*; that would be dishonest. We audit our *model* and *deployment conditions* instead.
4. **Occult high-risk presentations exist and the text captures them.** Rare normal-vitals-but-urgent cases are textbook can't-miss: "chest pain with diaphoresis and arm radiation", "sepsis with altered mental status", "ovarian torsion with rigors". Justifies an NLP red-flag layer as an **independent, vitals-free safety channel**.
5. **NEWS2 alone caps at ~65% accuracy**; among normal-vitals patients acuity still spreads 3/4/5 (31/46/23%). Context + text genuinely refine triage → fusion modeling is justified, not gratuitous.

## 3. The winning thesis

> Accuracy is trivially solved here (text → 100%). The real, clinically meaningful problem in triage is **safety under messy, real-world conditions**: catching the occult high-risk patient, behaving safely when data is missing, being calibrated and honest about uncertainty, and not encoding bias. **Second Look** is a triage *safety-net & audit* system — and the only submission that honestly characterizes what the synthetic data can and cannot teach.

Product name: **Second Look** (a second pair of eyes on the triage decision).

## 4. Architecture — 7 isolated modules

Each module has one purpose, a clean interface, and is independently testable.

1. **`data_prep.py`** — load/merge the 4 tables; feature engineering (vitals, derived scores, comorbidity counts, temporal, *explicit honest missingness indicators*); text cleaning. Interface: `prepare(split) -> (X_tab, text, y)`.
2. **Data-forensics** (notebook section, uses `data_prep`) — reproduces findings §2 with figures. Honesty as a scored asset.
3. **`model_core.py`** — calibrated fusion classifier (tabular GBM + text). Headline = **calibration + uncertainty**, not accuracy. Explicit contrast: TF-IDF (memorizes synthetic phrases, 100% but brittle) vs **clinical transformer** (generalizes to paraphrases, GPU). Interface: `predict_proba(X) -> calibrated probs`.
4. **`redflag.py`** — curated clinical red-flag ontology (SAH/thunderclap, ACS/STEMI/NSTEMI, sepsis, stroke/FAST, AAA, ectopic/torsion, anaphylaxis, airway, …) + a transformer detector that generalizes beyond exact phrases. Vitals-independent. Interface: `flags(text) -> [RedFlag]`.
5. **`policy.py`** — cost-sensitive decision policy (undertriage cost ≫ overtriage). Operating-point selection; escalation = model ∨ red-flag ∨ early-warning. Reports **undertriage rate** at chosen points. Interface: `decide(probs, flags, ews) -> acuity, escalate, rationale`.
6. **`audit.py`** — (a) model fairness across age/sex/language/insurance; (b) **missingness stress-test** (drop vitals for sick patients, measure undertriage drift); (c) paraphrase-robustness of text channel; (d) calibration/reliability curves. Honest about synthetic-data limits + need for MIMIC/NHAMCS external validation.
7. **`explain.py`** + **`app.py`** — SHAP/attribution + which red-flags fired (per-patient "why"); Gradio demo on a Hugging Face Space surfacing ESI + calibrated probabilities + red-flag alerts + early-warning + **data-shift warning** when vitals are missing.

## 5. Rubric coverage map

- **Clinical Relevance (25):** safety-net framing, occult-risk, undertriage cost asymmetry, NEWS2 limitations, missingness-as-hazard — all grounded in emergency-medicine literature.
- **Technical Quality (30):** rigorous CV, calibration, fusion model, transformer-generalization study, fairness + robustness stress-tests, clean reproducible code, end-to-end notebook.
- **Documentation (20):** ≤2000-word writeup following the template; reproducible repo + README; runnable notebook.
- **Insight (15):** the text-determinism reveal, informative-missingness trap, occult cases, honest limits — communicated with appropriate uncertainty.
- **Novelty (10):** vitals-free red-flag safety channel; auditing the *model* not the (clean) data; honest synthetic-data forensics; working decision-support demo.

## 6. Compute & workflow (constraint: no heavy local compute; Mac stays free)

- **Kaggle GPU kernels** (user `ssstelmah`): transformer fine-tune/encode for the red-flag + generalization study; pushed via `kaggle kernels push` (script type, `enable_gpu=true`, private), mirroring the proven `neurogolf-2026` workflow.
- **Kaggle CPU kernel = the public submission notebook** (runs competition data natively, end-to-end, public at submission).
- **Mac (local):** CLI, file I/O, light reads, repo/demo assembly, writing. No training locally.
- **GitHub** (`KapaSique`, authed): repo + push, autonomous.
- **Hugging Face Space:** Gradio demo. Requires one-time user `hf auth login` (account auth is the user's; agent cannot enter tokens).

## 7. Deliverables (all four required for eligibility)

1. Public Kaggle Notebook — end-to-end, no errors.
2. Writeup ≤2000 words — clinical problem → methodology → results → limitations → reproducibility.
3. Cover image 560×280 px.
4. Public Project Link — **Hugging Face Space (Gradio demo) + GitHub repo**.

## 8. 8-day timeline

- **D1 (2026-06-13):** spec, repo init, EDA/forensics kernel (GPU), baselines.
- **D2–3:** predictive core + calibration; red-flag ontology + transformer (GPU).
- **D4:** decision policy + audit (fairness, missingness, robustness).
- **D5:** Gradio demo + HF Space (user HF login).
- **D6:** assemble end-to-end public notebook + figures.
- **D7:** writeup + cover image + README.
- **D8:** buffer, final verification, publish + submit (2 final submissions allowed).

## 9. Risks & mitigations

- *"text→100% looks like cheating"* → reframed as an honest discovery about the generator (a plus).
- *Scope vs 8 days* → modules isolated; trim audit depth/demo polish first; core + writeup are inviolable.
- *Transformer not needed for accuracy* → used for **generalization/robustness**, the deployment argument — not for accuracy.
- *Synthetic data has no real bias/variability* → stated openly; external validation flagged as future work.

## 10. Ethics & honesty stance

Every claim is grounded in evidence shown in the notebook. We never claim findings the data does not support (no fabricated undertriage/bias in the data). Synthetic-data limitations are stated plainly. This intellectual honesty is itself a scored asset under the rubric and the right thing to do for clinical AI.
