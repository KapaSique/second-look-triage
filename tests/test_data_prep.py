import numpy as np
import pandas as pd

from src.data_prep import engineer_features, MISSING_VITALS, TARGET


def _row(**kw):
    base = dict(
        systolic_bp=120, diastolic_bp=80, heart_rate=80, respiratory_rate=16,
        temperature_c=37.0, spo2=98, gcs_total=15, pain_score=3, age=40,
        num_comorbidities=0, news2_score=1, shock_index=0.67,
    )
    base.update(kw)
    return base


def test_missingness_indicator_added():
    df = pd.DataFrame([_row(systolic_bp=np.nan)])
    out = engineer_features(df)
    assert out["systolic_bp_missing"].iloc[0] == 1
    assert "n_vitals_missing" in out.columns and out["n_vitals_missing"].iloc[0] >= 1
    assert out["any_vital_missing"].iloc[0] == 1


def test_no_missing_when_all_present():
    out = engineer_features(pd.DataFrame([_row()]))
    assert out["n_vitals_missing"].iloc[0] == 0
    assert out["any_vital_missing"].iloc[0] == 0


def test_pain_score_sentinel_becomes_missing():
    out = engineer_features(pd.DataFrame([_row(pain_score=-1)]))
    assert out["pain_missing"].iloc[0] == 1
    assert pd.isna(out["pain_score"].iloc[0])


def test_no_leak_of_outcome_columns():
    df = pd.DataFrame([_row()])
    df["disposition"] = "admitted"
    df["ed_los_hours"] = 3.0
    out = engineer_features(df)
    assert "disposition" not in out.columns
    assert "ed_los_hours" not in out.columns


def test_target_passed_through():
    df = pd.DataFrame([_row()])
    df[TARGET] = 3
    out = engineer_features(df)
    assert TARGET in out.columns and out[TARGET].iloc[0] == 3
