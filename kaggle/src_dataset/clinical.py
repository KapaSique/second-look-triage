"""Clinical calculators: NEWS2 and derived vitals.

The Gradio demo collects raw vitals; the model and policy expect derived fields
(`news2_score`, `shock_index`, ...). NEWS2 (National Early Warning Score 2) is the
standard UK deterioration score — implementing it from raw vitals makes the demo
clinically authentic and feeds the policy's early-warning floor. Reference: RCP NEWS2
(scale 1). Supplemental-oxygen and ACVPU are approximated (GCS<15 -> not Alert).
"""
from __future__ import annotations
from typing import Dict, Optional
import math


def _na(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _s_rr(rr):
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3


def _s_spo2(s):       # NEWS2 SpO2 scale 1
    if s >= 96: return 0
    if s >= 94: return 1
    if s >= 92: return 2
    return 3


def _s_temp(t):
    if t <= 35.0: return 3
    if t <= 36.0: return 1
    if t <= 38.0: return 0
    if t <= 39.0: return 1
    return 2


def _s_sbp(s):
    if s <= 90: return 3
    if s <= 100: return 2
    if s <= 110: return 1
    if s <= 219: return 0
    return 3


def _s_hr(h):
    if h <= 40: return 3
    if h <= 50: return 1
    if h <= 90: return 0
    if h <= 110: return 1
    if h <= 130: return 2
    return 3


def compute_news2(respiratory_rate=None, spo2=None, temperature_c=None,
                  systolic_bp=None, heart_rate=None, gcs_total: Optional[float] = 15,
                  on_oxygen: bool = False) -> int:
    """Sum NEWS2 over whichever components are present (missing components skipped)."""
    total = 0
    for v, fn in [(respiratory_rate, _s_rr), (spo2, _s_spo2), (temperature_c, _s_temp),
                  (systolic_bp, _s_sbp), (heart_rate, _s_hr)]:
        if not _na(v):
            total += fn(float(v))
    if not _na(gcs_total) and float(gcs_total) < 15:
        total += 3
    if on_oxygen:
        total += 2
    return int(total)


def compute_derived(d: Dict) -> Dict:
    """Return a copy of `d` with shock_index, MAP, pulse_pressure, bmi, news2_score added."""
    out = dict(d)
    sbp, dbp = out.get("systolic_bp"), out.get("diastolic_bp")
    hr, wt, ht = out.get("heart_rate"), out.get("weight_kg"), out.get("height_cm")
    if not _na(sbp) and not _na(hr) and float(sbp) > 0:
        out["shock_index"] = round(float(hr) / float(sbp), 3)
    if not _na(sbp) and not _na(dbp):
        out["mean_arterial_pressure"] = round((float(sbp) + 2 * float(dbp)) / 3, 1)
        out["pulse_pressure"] = round(float(sbp) - float(dbp), 1)
    if not _na(wt) and not _na(ht) and float(ht) > 0:
        out["bmi"] = round(float(wt) / (float(ht) / 100) ** 2, 1)
    out["news2_score"] = compute_news2(
        respiratory_rate=out.get("respiratory_rate"), spo2=out.get("spo2"),
        temperature_c=out.get("temperature_c"), systolic_bp=sbp, heart_rate=hr,
        gcs_total=out.get("gcs_total", 15))
    return out
