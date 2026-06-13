# Second Look — Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans (inline) to implement task-by-task. Steps use `- [ ]` checkboxes. Heavy compute (transformer train/encode) runs on **Kaggle GPU kernels** (`ssstelmah`), never locally. Local Python is for light dev/tests on already-downloaded data only.

**Goal:** Build *Second Look*, a triage safety-net & audit system, and ship all 4 required hackathon deliverables (public notebook, writeup ≤2000 words, cover image, HF Space + GitHub link).

**Architecture:** 7 isolated modules (`data_prep`, `model_core`, `redflag`, `policy`, `audit`, `explain`, `app`) + Kaggle kernels for heavy compute + a public end-to-end notebook that re-runs the pipeline + a Gradio demo. Accuracy is solved (text→100%); we win on safety/calibration/honesty/novelty against the 100-pt rubric.

**Tech Stack:** Python 3, pandas, scikit-learn, LightGBM, sentence-transformers / a clinical transformer (HF), SHAP, Gradio, matplotlib; Kaggle kernels API; GitHub (`gh`); Hugging Face Hub.

**Priority (if time runs short):** Phases 0,4,5 (forensics, notebook, writeup) are inviolable. Phases 2,3 (audit depth, demo polish) trim first.

---

## File structure (locked)

| File | Responsibility |
|---|---|
| `src/data_prep.py` | Load/merge 4 tables; feature engineering; honest missingness indicators; text clean. |
| `src/model_core.py` | Calibrated fusion classifier; TF-IDF vs transformer contrast; CV. |
| `src/redflag.py` | Clinical red-flag ontology + matcher (lexicon now, transformer-backed generalization). |
| `src/policy.py` | Cost-sensitive decision policy; undertriage metric; escalation logic. |
| `src/audit.py` | Fairness, missingness stress-test, paraphrase robustness, calibration metrics. |
| `src/explain.py` | SHAP + red-flag attribution; per-patient rationale. |
| `app/app.py` | Gradio demo. |
| `kaggle/eda_forensics/` | Kernel: EDA + baselines + figures (Phase 0). |
| `kaggle/transformer/` | Kernel (GPU): transformer generalization study (Phase 1). |
| `notebooks/second_look_submission.ipynb` | Public end-to-end notebook (Phase 4). |
| `writeup/writeup.md`, `writeup/cover.png` | Writeup + cover (Phase 5). |
| `tests/` | pytest for the pure-Python modules. |

**Kernel I/O contract:** every kernel writes `/kaggle/working/outputs/<name>.json` (metrics) + `figures/*.png`. Local fetches via `kaggle kernels output`. No kernel depends on a private dataset that isn't versioned first.

---

## Phase 0 — Foundations & forensics (D1) — CRITICAL PATH

### Task 0.1: `data_prep.py` — feature engineering with honest missingness
**Files:** Create `src/data_prep.py`; Test `tests/test_data_prep.py`

- [ ] **Step 1 — failing test** (`tests/test_data_prep.py`):
```python
import pandas as pd, numpy as np
from src.data_prep import engineer_features, MISSING_VITALS
def _row(**kw):
    base=dict(systolic_bp=120,diastolic_bp=80,heart_rate=80,respiratory_rate=16,
              temperature_c=37.0,spo2=98,gcs_total=15,pain_score=3,age=40,
              num_comorbidities=0,news2_score=1,shock_index=0.67)
    base.update(kw); return base
def test_missingness_indicator_added():
    df=pd.DataFrame([_row(systolic_bp=np.nan)])
    out=engineer_features(df)
    assert out["bp_missing"].iloc[0]==1
    assert "n_vitals_missing" in out.columns and out["n_vitals_missing"].iloc[0]>=1
def test_no_leak_of_outcome_columns():
    df=pd.DataFrame([_row()]); df["disposition"]="admitted"; df["ed_los_hours"]=3.0
    out=engineer_features(df)
    assert "disposition" not in out.columns and "ed_los_hours" not in out.columns
```
- [ ] **Step 2 — run, expect fail:** `cd /Users/artemcike/Documents/Kaggle/triagegeist && PYTHONPATH=. venv/bin/python -m pytest tests/test_data_prep.py -q` → FAIL (module missing).
- [ ] **Step 3 — implement** `engineer_features(df)`: add per-vital `*_missing` flags + `n_vitals_missing`; derived ratios already present (shock_index, news2); drop leakage cols (`disposition`,`ed_los_hours`) if present; return numeric+categorical frame. `MISSING_VITALS=['systolic_bp','diastolic_bp','heart_rate','respiratory_rate','temperature_c','spo2']`.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit:** `feat(data_prep): feature engineering with honest missingness indicators`

### Task 0.2: EDA/forensics Kaggle kernel (baselines + figures)
**Files:** Create `kaggle/eda_forensics/forensics.py`, `kaggle/eda_forensics/kernel-metadata.json`

- [ ] **Step 1 — write `forensics.py`** that, on Kaggle, attaches the competition data and computes/saves to `outputs/forensics.json` + `figures/`:
  - target distribution; missingness-by-acuity table + figure (the 0% vs 12% finding);
  - text-only vs vitals-only vs fusion accuracy + macro-F1 + **undertriage rate** (proper 5-fold CV, not a single split);
  - NEWS2-only ceiling; complaint cardinality (unique complaints, train/test overlap) to explain the 100%;
  - per-site / per-nurse mean acuity spread (variability ≈ 0 evidence);
  - occult-case extraction (acuity 1/2 with NEWS2<3) → table for writeup.
- [ ] **Step 2 — `kernel-metadata.json`** (script, `enable_gpu=false` here — CPU is enough; competition data source `triagegeist`). Mirror neurogolf metadata format.
- [ ] **Step 3 — push:** `kaggle kernels push -p kaggle/eda_forensics` ; poll status.
- [ ] **Step 4 — fetch + verify:** `kaggle kernels output ssstelmah/<slug> -p kaggle/eda_forensics/out` → assert `forensics.json` has `text_only_acc>=0.99`, `missing_by_acuity` present, CV numbers stable. Copy figures to `figures/`.
- [ ] **Step 5 — commit:** `feat(forensics): EDA + baselines kernel + figures`

---

## Phase 1 — Predictive core + red-flag layer (D2–3)

### Task 1.1: `redflag.py` — clinical ontology + matcher (TDD)
**Files:** Create `src/redflag.py`, `src/redflag_ontology.yaml`; Test `tests/test_redflag.py`

- [ ] **Step 1 — failing test:**
```python
from src.redflag import RedFlagMatcher
m=RedFlagMatcher()
def test_thunderclap_flags_sah():
    f=m.flags("thunderclap headache, worst of my life")
    assert any(x.category=="SAH" for x in f) and f[0].severity=="critical"
def test_acs_radiation():
    assert any(x.category=="ACS" for x in m.flags("chest pain with diaphoresis and arm radiation"))
def test_benign_no_flag():
    assert m.flags("contraception advice")==[]
```
- [ ] **Step 2 — run, expect fail.**
- [ ] **Step 3 — implement** ontology YAML (categories: SAH, ACS, STROKE, SEPSIS, AAA, ECTOPIC/TORSION, ANAPHYLAXIS, AIRWAY, GI_BLEED, MENINGITIS, …; each with trigger lexicon + severity + clinical note + ESI floor) and a matcher returning `RedFlag(category,severity,matched,esi_floor,note)`.
- [ ] **Step 4 — run, expect pass.**
- [ ] **Step 5 — commit:** `feat(redflag): clinical red-flag ontology + lexicon matcher`

### Task 1.2: `model_core.py` — calibrated fusion + CV (TDD on interface)
**Files:** Create `src/model_core.py`; Test `tests/test_model_core.py`
- [ ] Test: `fit/predict_proba` returns calibrated probs summing to 1, shape (n,5); `cross_validate` returns dict with `accuracy`,`macro_f1`,`undertriage_rate`,`ece` (calibration error). Implement with LightGBM on tabular + TF-IDF text (sparse), `CalibratedClassifierCV` or temperature scaling; compute ECE. Commit `feat(model_core): calibrated fusion classifier + CV metrics`.

### Task 1.3: transformer generalization study (Kaggle GPU kernel)
**Files:** Create `kaggle/transformer/train_encode.py`, `kernel-metadata.json` (`enable_gpu=true`)
- [ ] Fine-tune/encode chief complaints with a clinical/biomedical transformer (e.g. `emilyalsentzer/Bio_ClinicalBERT` or a sentence-transformer). Produce: (a) held-out acc to show it also separates classes; (b) **paraphrase-robustness test** — hand-craft paraphrases of red-flag complaints, show TF-IDF (memorized) fails on them while transformer/ontology generalize; save `outputs/transformer.json` + figure. Push GPU kernel, fetch, verify, commit.

---

## Phase 2 — Decision policy + audit (D4)

### Task 2.1: `policy.py` — cost-sensitive escalation (TDD)
- [ ] Test: given calibrated probs + red-flags + EWS, `decide(...)` never assigns acuity less urgent than a critical red-flag's `esi_floor`; undertriage-cost-weighted threshold lowers undertriage rate vs argmax. Implement; commit `feat(policy): cost-sensitive decision policy with red-flag floor`.

### Task 2.2: `audit.py` — fairness, missingness stress, calibration (TDD on metrics)
- [ ] Test metric fns: `group_undertriage_rates(df,preds,by)`; `missingness_stress(model,df)` (drop vitals for sick rows, measure undertriage drift); `reliability(probs,y)` → ECE/bins. Implement; commit `feat(audit): fairness + missingness stress-test + calibration metrics`.

### Task 2.3: audit Kaggle kernel — run full audit, save figures
- [ ] Kernel runs model_core + policy + audit on full data; saves fairness tables, missingness-drift figure, reliability diagram, confusion + undertriage matrix. Fetch, verify, commit.

---

## Phase 3 — Demo (D5)

### Task 3.1: `explain.py` + `app/app.py` Gradio
- [ ] `explain.py`: SHAP top-features + fired red-flags → rationale string. `app.py`: inputs (vitals + free-text complaint) → ESI, calibrated probs bar, red-flag alerts, EWS, rationale, **data-shift warning** if vitals blank. Smoke-test locally (light), commit.

### Task 3.2: Hugging Face Space deploy
- [ ] Install `huggingface_hub`; **(user one-time `hf auth login`)**; create Space `second-look-triage`, push `app/` + a small exported model artifact; verify it loads. Commit link to README.

---

## Phase 4 — Public submission notebook (D6) — INVIOLABLE

### Task 4.1: assemble `notebooks/second_look_submission.ipynb`
- [ ] End-to-end on Kaggle: data prep → forensics figures → fusion model + calibration → red-flag layer → decision policy → audit → conclusions, all narrated. Must **run top-to-bottom without errors** on Kaggle, public. Verify by running as a Kaggle kernel; fetch status=complete. Commit notebook source.

---

## Phase 5 — Writeup + cover + README (D7) — INVIOLABLE

### Task 5.1: `writeup/writeup.md` (≤2000 words, template sections)
- [ ] Sections: Clinical Problem Statement → Methodology → Results → Limitations → Reproducibility. Lead with the honest text-determinism finding + missingness trap + safety-net design. Cite ESI/NEWS2/undertriage literature. Word-count check ≤2000.
### Task 5.2: `writeup/cover.png` 560×280
- [ ] Generate a clean branded cover (matplotlib/SVG) showing the Second Look concept. Verify dimensions exactly 560×280.
### Task 5.3: finalize `README.md` with reproduction steps + links.

---

## Phase 6 — Verify, publish, submit (D8)

- [ ] Create public GitHub repo via `gh repo create` (KapaSique), push.
- [ ] Make Kaggle notebook public; confirm runs.
- [ ] HF Space live + linked.
- [ ] Create Writeup on Kaggle, attach notebook + cover + project link, select Track, **final confirm content with user, then submit** (≤2 final submissions). 
- [ ] Verification pass: all 4 required elements present; links public/no-login.

---

## Self-review notes
- Spec §4 modules → Tasks 0.1–3.1 (all covered). Spec §7 deliverables → Phases 4,5,6. Spec §2 findings → Phase 0 kernel. Spec §6 compute → kernels GPU/CPU split explicit. No placeholders in pure-Python tasks (test code given); kernel/notebook/writeup tasks use acceptance criteria by design (large scripts written at execution).
- Inter-task type consistency: `RedFlag(category,severity,matched,esi_floor,note)` used by redflag + policy + explain; `engineer_features` + `MISSING_VITALS` used across; kernel I/O contract uniform (`outputs/*.json`,`figures/*.png`).
