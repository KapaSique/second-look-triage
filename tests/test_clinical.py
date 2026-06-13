import numpy as np
from src.clinical import compute_news2, compute_derived


def test_news2_normal_is_zero():
    assert compute_news2(respiratory_rate=16, spo2=98, temperature_c=37.0,
                         systolic_bp=120, heart_rate=75, gcs_total=15) == 0


def test_news2_deranged_example():
    # RR22(2)+SpO2 93(2)+temp38.5(1)+SBP100(2)+HR115(2)+alert(0) = 9
    assert compute_news2(respiratory_rate=22, spo2=93, temperature_c=38.5,
                         systolic_bp=100, heart_rate=115, gcs_total=15) == 9


def test_news2_reduced_consciousness_adds_three():
    base = compute_news2(respiratory_rate=16, spo2=98, temperature_c=37.0,
                         systolic_bp=120, heart_rate=75, gcs_total=15)
    red = compute_news2(respiratory_rate=16, spo2=98, temperature_c=37.0,
                        systolic_bp=120, heart_rate=75, gcs_total=12)
    assert red - base == 3


def test_news2_skips_missing_components():
    # only RR provided (22 -> 2); others None -> skipped
    assert compute_news2(respiratory_rate=22, spo2=None, temperature_c=None,
                         systolic_bp=None, heart_rate=None) == 2


def test_compute_derived_adds_fields():
    d = compute_derived(dict(systolic_bp=120, diastolic_bp=80, heart_rate=90,
                             respiratory_rate=16, spo2=98, temperature_c=37.0,
                             weight_kg=70, height_cm=175, gcs_total=15))
    assert abs(d["shock_index"] - 0.75) < 1e-6
    assert abs(d["pulse_pressure"] - 40) < 1e-6
    assert abs(d["mean_arterial_pressure"] - (120 + 2 * 80) / 3) < 1e-6
    assert "news2_score" in d and "bmi" in d
