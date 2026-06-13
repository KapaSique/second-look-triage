import numpy as np
from src.explain import abnormal_vitals, explain
from src.redflag import RedFlag
from src.policy import Decision


def test_abnormal_vitals_flags_derangements():
    row = dict(spo2=88, heart_rate=130, systolic_bp=85, respiratory_rate=28,
               temperature_c=39.2, gcs_total=13, shock_index=1.4, pain_score=9)
    flags = abnormal_vitals(row)
    keys = " ".join(f.lower() for f in flags)
    assert "spo2" in keys and "respiratory" in keys and "gcs" in keys
    assert len(flags) >= 5


def test_normal_vitals_few_flags():
    row = dict(spo2=98, heart_rate=78, systolic_bp=120, respiratory_rate=16,
               temperature_c=36.8, gcs_total=15, shock_index=0.65, pain_score=2)
    assert len(abnormal_vitals(row)) == 0


def test_missing_vitals_skipped():
    row = dict(spo2=np.nan, heart_rate=np.nan)
    assert abnormal_vitals(row) == []


def test_explain_assembles_sections():
    row = dict(spo2=88, heart_rate=120, systolic_bp=95, respiratory_rate=26,
               temperature_c=38.5, gcs_total=14, shock_index=1.2, pain_score=8)
    probs = np.array([0.4, 0.3, 0.2, 0.06, 0.04])
    dec = Decision(acuity=1, base_acuity=2, escalate=True, rationale="red-flag [SAH] → floor ESI 1")
    flags = [RedFlag("SAH", "critical", "thunderclap headache", 1, "rule out SAH")]
    out = explain(row, probs, dec, flags)
    assert "abnormal_vitals" in out and "red_flags" in out and "rationale" in out
    assert out["recommended_esi"] == 1
    assert any("SAH" in r for r in out["red_flags"])
