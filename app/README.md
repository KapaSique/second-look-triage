---
title: Second Look — ED Triage Safety Net
emoji: 🚑
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Triage safety-net decision support (synthetic demo)
---

# Second Look — ED Triage Safety Net

A "second pair of eyes" on an emergency-department triage decision, built for the
[Triagegeist](https://www.kaggle.com/competitions/triagegeist) hackathon.

Enter a free-text chief complaint and (optionally) vitals. Second Look returns:

- a **calibrated ESI estimate** (1 = most urgent … 5 = least),
- **red-flag alerts** from a curated clinical can't-miss ontology (vitals-independent),
- the **NEWS2** early-warning score computed from the vitals,
- a **cost-sensitive recommended acuity** that can only ever escalate, never silently undertriage,
- deranged-vital callouts and a plain-language rationale,
- a **data-shift warning** when vitals are missing.

**Why it exists:** on the synthetic Triagegeist data, chief-complaint text predicts the label
with ~100% accuracy — so accuracy is not a meaningful objective. The real problem is *safe*
behaviour: catching the occult high-risk patient and not collapsing when data is missing.

> ⚠️ Research demonstration on **synthetic** data. **Not a medical device.** Second Look
> supports, never replaces, clinician judgement.

Code & methodology: see the linked GitHub repository and Kaggle notebook.
