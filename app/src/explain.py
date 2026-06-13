"""Clinically legible explanations for a single triage decision.

For an emergency physician, a list of deranged vitals + fired red flags + the decision
rationale is far more trustworthy than abstract SHAP values. So a per-patient explanation
surfaces:
  * abnormal vitals vs adult triage reference ranges (with direction),
  * fired red flags (with their clinical note),
  * the policy's escalation rationale,
  * the calibrated ESI probability distribution.

(Global feature importance via SHAP is reported once, at the model level, in the notebook —
not per patient, where it tends to mislead.)
"""
from __future__ import annotations
from typing import Dict, List
import numpy as np

# (low_ok, high_ok) adult triage reference ranges; outside -> flagged with direction.
VITAL_RANGES = {
    "spo2": (94, 100, "%"),
    "systolic_bp": (90, 180, "mmHg"),
    "heart_rate": (50, 100, "bpm"),
    "respiratory_rate": (10, 22, "/min"),
    "temperature_c": (36.0, 38.0, "°C"),
    "gcs_total": (15, 15, ""),
    "shock_index": (0.0, 0.9, ""),
    "pain_score": (0, 6, "/10"),
}
PRETTY = {"spo2": "SpO2", "systolic_bp": "Systolic BP", "heart_rate": "Heart rate",
          "respiratory_rate": "Respiratory rate", "temperature_c": "Temperature",
          "gcs_total": "GCS", "shock_index": "Shock index", "pain_score": "Pain"}


def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v))
    except Exception:
        return False


def abnormal_vitals(row: Dict) -> List[str]:
    """Return human-readable derangements for whichever vitals are present."""
    out: List[str] = []
    for k, (lo, hi, unit) in VITAL_RANGES.items():
        if k not in row:
            continue
        v = row[k]
        if _isnan(v):
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        name = PRETTY[k]
        if v < lo:
            tag = "critically low" if (k == "spo2" and v < 90) or (k == "systolic_bp" and v < 80) else "low"
            out.append(f"{name} {v:g}{unit} ({tag})")
        elif v > hi:
            tag = "high"
            if k == "respiratory_rate" and v >= 25:
                tag = "critically high"
            if k == "temperature_c" and v >= 39:
                tag = "high fever"
            if k == "shock_index" and v >= 1.0:
                tag = "elevated (haemodynamic concern)"
            out.append(f"{name} {v:g}{unit} ({tag})")
        elif k == "gcs_total" and v < 15:
            out.append(f"{name} {v:g} (reduced consciousness)")
    return out


def explain(row: Dict, probs, decision, redflags=None) -> Dict:
    probs = np.asarray(probs, dtype=float)
    rf = redflags or []
    red_lines = [f"{f.category} ({f.severity}): {f.note} [matched: '{f.matched}']" for f in rf]
    av = abnormal_vitals(row)
    summary_bits = []
    if decision.escalate:
        summary_bits.append("ESCALATED")
    summary_bits.append(f"recommended ESI {decision.acuity}")
    if rf:
        summary_bits.append(f"{len(rf)} red flag(s)")
    if av:
        summary_bits.append(f"{len(av)} deranged vital(s)")
    return {
        "recommended_esi": int(decision.acuity),
        "model_most_likely_esi": int(decision.base_acuity),
        "escalated": bool(decision.escalate),
        "probabilities": {i + 1: round(float(probs[i]), 3) for i in range(len(probs))},
        "abnormal_vitals": av,
        "red_flags": red_lines,
        "rationale": decision.rationale,
        "summary": " · ".join(summary_bits),
    }
