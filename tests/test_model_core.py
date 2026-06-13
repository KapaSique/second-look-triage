import numpy as np
import pandas as pd

from src.model_core import (
    SecondLookModel, ACUITY_CLASSES, expected_calibration_error,
    undertriage_rate, cross_validate,
)


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    phrases = {1: "cardiac arrest unresponsive", 2: "chest pain diaphoresis",
               3: "abdominal pain moderate", 4: "ankle sprain minor", 5: "medication refill"}
    rows = []
    for i in range(n):
        a = (i % 5) + 1
        rows.append(dict(
            chief_complaint_raw=phrases[a],
            news2_score={1: 13, 2: 10, 3: 3, 4: 1, 5: 0}[a] + rng.normal(0, 0.5),
            shock_index=rng.normal(0.8, 0.1), spo2=rng.normal(97, 2),
            heart_rate=rng.normal(85, 10), respiratory_rate=rng.normal(17, 2),
            systolic_bp=rng.normal(120, 15), diastolic_bp=rng.normal(78, 10),
            temperature_c=rng.normal(37, 0.4), gcs_total=15, pain_score=rng.integers(0, 10),
            age=int(rng.integers(18, 90)), sex=rng.choice(["M", "F"]),
            arrival_mode=rng.choice(["walk-in", "ambulance"]),
            num_comorbidities=int(rng.integers(0, 8)),
            triage_acuity=a,
        ))
    df = pd.DataFrame(rows)
    return df, df["triage_acuity"].values


def test_predict_proba_shape_and_normalized():
    df, y = _toy()
    m = SecondLookModel(calibrate=False).fit(df, y)
    p = m.predict_proba(df)
    assert p.shape == (len(df), 5)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_predict_returns_valid_classes():
    df, y = _toy()
    m = SecondLookModel(calibrate=False).fit(df, y)
    preds = m.predict(df)
    assert set(np.unique(preds)).issubset(set(ACUITY_CLASSES))


def test_vitals_only_and_text_only_construct():
    df, y = _toy()
    SecondLookModel(use_text=False, calibrate=False).fit(df, y).predict_proba(df)
    SecondLookModel(use_tabular=False, calibrate=False).fit(df, y).predict_proba(df)


def test_undertriage_rate_known():
    # true urgent (1,2) predicted less urgent => undertriage
    yt = np.array([1, 2, 3, 4, 5])
    yp = np.array([3, 2, 3, 4, 5])  # the first (true 1 -> pred 3) is undertriage
    assert abs(undertriage_rate(yt, yp) - 50.0) < 1e-6  # 1 of 2 urgent undertriaged


def test_ece_confident_correct_low_overconfident_wrong_high():
    y = np.array([0, 1, 2, 3, 4] * 4)
    good = np.eye(5)[y] * 0.9 + 0.02                 # confident AND correct -> low ECE
    bad = np.eye(5)[(y + 1) % 5] * 0.9 + 0.02        # confident but WRONG -> high ECE
    assert expected_calibration_error(good, y) < expected_calibration_error(bad, y)


def test_cross_validate_keys():
    df, y = _toy()
    out = cross_validate(lambda: SecondLookModel(calibrate=False), df, y, n_splits=2)
    for k in ("accuracy", "macro_f1", "undertriage_rate", "ece"):
        assert k in out
