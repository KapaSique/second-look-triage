"""Data loading, merging, and feature engineering for Second Look.

Honest design choices baked in here:
  * Vital missingness is made EXPLICIT (`*_missing`, `n_vitals_missing`,
    `any_vital_missing`). In this synthetic dataset missingness is a near-perfect
    proxy for low acuity (vitals are skipped for low-acuity patients). Surfacing it
    lets the audit quantify the deployment hazard rather than hiding it.
  * Downstream OUTCOME columns (`disposition`, `ed_los_hours`) are dropped — they are
    not known at triage time and would be target leakage.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Vitals that can be missing at triage (per the data description).
MISSING_VITALS = [
    "systolic_bp", "diastolic_bp", "heart_rate",
    "respiratory_rate", "temperature_c", "spo2",
]
# Outcome columns available only in train and only AFTER the ED visit -> leakage.
LEAKAGE_COLS = ["disposition", "ed_los_hours"]
ID_COLS = ["patient_id"]
TARGET = "triage_acuity"


def load_and_merge(data_dir: str, split: str = "train") -> pd.DataFrame:
    """Load `{split}.csv` and left-join chief complaints + patient history."""
    base = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    cc = pd.read_csv(os.path.join(data_dir, "chief_complaints.csv"))
    cc = cc[["patient_id", "chief_complaint_raw"]]
    hx_path = os.path.join(data_dir, "patient_history.csv")
    out = base.merge(cc, on="patient_id", how="left")
    if os.path.exists(hx_path):
        out = out.merge(pd.read_csv(hx_path), on="patient_id", how="left")
    return out


def engineer_features(df: pd.DataFrame, drop_ids: bool = True) -> pd.DataFrame:
    """Return a model-ready feature frame with explicit missingness and no leakage.

    The target column (`triage_acuity`) is passed through untouched if present;
    callers separate X/y themselves. Free text (`chief_complaint_raw`) is also passed
    through for the text channel.
    """
    df = df.copy()

    # pain_score: -1 is an explicit missing sentinel -> flag + set NaN.
    if "pain_score" in df.columns:
        df["pain_missing"] = (df["pain_score"] == -1).astype(int)
        df.loc[df["pain_score"] == -1, "pain_score"] = np.nan

    # Per-vital missingness indicators + counts (the deployment-trap signal).
    present = [c for c in MISSING_VITALS if c in df.columns]
    for c in present:
        df[f"{c}_missing"] = df[c].isna().astype(int)
    if present:
        miss_cols = [f"{c}_missing" for c in present]
        df["n_vitals_missing"] = df[miss_cols].sum(axis=1)
        df["any_vital_missing"] = (df["n_vitals_missing"] > 0).astype(int)
    else:
        df["n_vitals_missing"] = 0
        df["any_vital_missing"] = 0

    # Cyclic encoding of arrival hour.
    if "arrival_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["arrival_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["arrival_hour"] / 24)

    # Comorbidity burden if patient history was merged.
    hx = [c for c in df.columns if c.startswith("hx_")]
    if hx:
        df["hx_count"] = df[hx].sum(axis=1)

    # Drop leakage outcomes and (optionally) ids.
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    if drop_ids:
        df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    return df
