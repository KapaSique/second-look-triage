"""Second Look — interactive ED-triage safety-net demo (Gradio / Hugging Face Space).

A "second pair of eyes" on a triage decision. Given a free-text chief complaint and (optional)
vitals it returns: a calibrated ESI estimate, a cost-sensitive recommended acuity, red-flag
alerts from the clinical ontology, the NEWS2 early-warning score, deranged-vital callouts, a
plain-language rationale, and — crucially — a data-shift warning when vitals are missing.

Not a diagnostic device. Trained on synthetic data. Research demonstration only.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from redflag import RedFlagMatcher          # noqa: E402
from policy import TriagePolicy             # noqa: E402
from explain import explain, abnormal_vitals  # noqa: E402
from clinical import compute_derived        # noqa: E402

DEMO_COLS = ["chief_complaint_raw", "age", "sex", "arrival_mode",
             "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
             "temperature_c", "spo2", "gcs_total", "pain_score", "num_comorbidities",
             "news2_score", "shock_index"]
VITALS = ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature_c", "spo2"]

matcher = RedFlagMatcher()
policy = TriagePolicy()

_MODEL = None
def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(path):
        try:
            import joblib
            _MODEL = joblib.load(path)
        except Exception as e:  # pragma: no cover
            print("model load failed:", e); _MODEL = False
    else:
        _MODEL = False
    return _MODEL


def _heuristic_probs(news2: float, flags) -> np.ndarray:
    """Fallback when no trained model is available: a soft distribution from NEWS2/red-flags."""
    floor = min([f.esi_floor for f in flags], default=5)
    center = min(floor, 1 if news2 >= 7 else 2 if news2 >= 5 else 3 if news2 >= 1 else 4)
    p = np.array([np.exp(-abs((i + 1) - center)) for i in range(5)])
    return p / p.sum()


def _model_probs(d: dict):
    m = _load_model()
    if not m:
        return None
    row = {c: d.get(c, np.nan) for c in DEMO_COLS}
    try:
        return m.predict_proba(pd.DataFrame([row]))[0]
    except Exception as e:  # pragma: no cover
        print("predict failed:", e); return None


def assess(complaint, age, sex, arrival_mode, sbp, dbp, hr, rr, temp, spo2,
           gcs, pain, ncomorb, vitals_unmeasured):
    """Core assessment — importable & testable without launching the UI."""
    raw = dict(chief_complaint_raw=complaint or "", age=age, sex=sex, arrival_mode=arrival_mode,
               systolic_bp=sbp, diastolic_bp=dbp, heart_rate=hr, respiratory_rate=rr,
               temperature_c=temp, spo2=spo2, gcs_total=gcs, pain_score=pain,
               num_comorbidities=ncomorb)
    if vitals_unmeasured:
        for k in VITALS:
            raw[k] = np.nan
    d = compute_derived(raw)
    news2 = d["news2_score"]
    flags = matcher.flags(complaint or "")
    probs = _model_probs(d)
    if probs is None:
        probs = _heuristic_probs(news2, flags)
    decision = policy.decide(probs, redflags=flags, news2=news2)
    ex = explain(d, probs, decision, flags)

    prob_label = {f"ESI {i + 1}": float(probs[i]) for i in range(5)}

    badge = "🚨 ESCALATE" if decision.escalate else "✓ routine"
    md = [f"## Recommended triage: **ESI {decision.acuity}** &nbsp; {badge}"]
    md.append(f"*Model's most-likely ESI: {decision.base_acuity} · NEWS2: **{news2}** · "
              f"P(ESI≤2) = {decision.p_urgent:.0%}*")
    if vitals_unmeasured:
        md.append("> ⚠️ **Data-shift warning:** vitals were not measured. In this dataset "
                  "missing vitals correlate with *low* acuity — so a vitals-reliant model could "
                  "**undertriage** a sick patient here. Recommendation leans on the text channel.")
    if ex["red_flags"]:
        md.append("### 🚩 Red-flag alerts")
        for r in ex["red_flags"]:
            md.append(f"- {r}")
    if ex["abnormal_vitals"]:
        md.append("### 📉 Deranged vitals\n" + ", ".join(ex["abnormal_vitals"]))
    md.append(f"### 🧭 Why\n{decision.rationale}")
    md.append("\n<sub>Research demo on synthetic data — not a medical device. "
              "Second Look supports, never replaces, clinician judgement.</sub>")
    return "\n\n".join(md), prob_label


EXAMPLES = [
    ["chest pain with diaphoresis and arm radiation", 61, "M", "ambulance",
     128, 84, 96, 18, 36.9, 97, 15, 7, 2, False],
    ["thunderclap headache, worst of my life", 44, "F", "walk-in",
     134, 86, 78, 16, 37.0, 99, 15, 9, 0, False],
    ["needs a repeat prescription for blood pressure tablets", 58, "F", "walk-in",
     138, 88, 74, 15, 36.8, 99, 15, 0, 3, False],
    ["sepsis with altered mental status", 73, "M", "ambulance",
     92, 55, 122, 26, 38.9, 91, 13, 6, 5, False],
    ["ankle sprain after football, mild swelling", 25, "M", "walk-in",
     124, 78, 72, 14, 36.7, 99, 15, 4, 0, True],
]


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Second Look — ED triage safety net", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚑 Second Look\n**A triage safety-net & decision-support second opinion.** "
                    "Enter a chief complaint and vitals; get a calibrated ESI estimate, red-flag "
                    "alerts, NEWS2, and a safety-aware recommendation. *Synthetic-data research "
                    "demo — not for clinical use.*")
        with gr.Row():
            with gr.Column():
                complaint = gr.Textbox(label="Chief complaint (free text)", lines=2,
                                       placeholder="e.g. chest pain radiating to the left arm, sweaty")
                with gr.Row():
                    age = gr.Number(label="Age", value=55)
                    sex = gr.Radio(["M", "F", "Other"], label="Sex", value="M")
                    arrival = gr.Dropdown(["walk-in", "ambulance", "transfer", "police"],
                                          label="Arrival", value="walk-in")
                with gr.Row():
                    sbp = gr.Number(label="Systolic BP", value=120)
                    dbp = gr.Number(label="Diastolic BP", value=80)
                    hr = gr.Number(label="Heart rate", value=80)
                with gr.Row():
                    rr = gr.Number(label="Resp rate", value=16)
                    temp = gr.Number(label="Temp °C", value=37.0)
                    spo2 = gr.Number(label="SpO2 %", value=98)
                with gr.Row():
                    gcs = gr.Slider(3, 15, value=15, step=1, label="GCS")
                    pain = gr.Slider(0, 10, value=3, step=1, label="Pain")
                    ncomorb = gr.Number(label="# comorbidities", value=1)
                unmeasured = gr.Checkbox(label="Vitals not measured at triage (test missingness)")
                btn = gr.Button("Assess", variant="primary")
            with gr.Column():
                out_md = gr.Markdown()
                out_prob = gr.Label(num_top_classes=5, label="Calibrated ESI probability")
        inputs = [complaint, age, sex, arrival, sbp, dbp, hr, rr, temp, spo2, gcs, pain, ncomorb, unmeasured]
        btn.click(assess, inputs=inputs, outputs=[out_md, out_prob])
        gr.Examples(EXAMPLES, inputs=inputs)
    return demo


if __name__ == "__main__":
    build_ui().launch()
