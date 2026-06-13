import numpy as np
import pandas as pd

from src.audit import (
    overtriage_rate, group_metrics, missingness_stress, reliability_table,
)


def test_overtriage_rate_known():
    yt = np.array([4, 5, 4, 1])
    yp = np.array([2, 5, 4, 1])  # true 4 -> pred 2 is overtriage; 1 of 2 low-acuity overtriaged
    assert abs(overtriage_rate(yt, yp) - 33.333) < 0.1  # 1 of 3 true-{4,5} overtriaged


def test_group_metrics_splits_by_column():
    df = pd.DataFrame({"sex": ["M", "M", "F", "F"]})
    yt = np.array([1, 2, 1, 2]); yp = np.array([3, 2, 1, 2])  # one M undertriaged
    g = group_metrics(df, yt, yp, "sex")
    assert set(g.keys()) == {"M", "F"}
    assert g["M"]["undertriage_pct"] > g["F"]["undertriage_pct"]
    assert g["M"]["n"] == 2


def test_missingness_stress_detects_degradation():
    # stub model: predicts the truth, UNLESS systolic_bp is missing -> outputs least-urgent (5)
    df = pd.DataFrame({"systolic_bp": [120.0, 110.0, 130.0, 100.0], "true": [1, 2, 4, 5]})

    def predict_fn(d):
        return np.where(d["systolic_bp"].isna().values, 5, d["true"].values)

    out = missingness_stress(predict_fn, df, df["true"].values, vitals=["systolic_bp"], target="urgent")
    assert out["baseline_undertriage"] == 0.0
    assert out["stressed_undertriage"] > out["baseline_undertriage"]
    assert out["delta"] > 0


def test_reliability_table_valid_ranges():
    y = np.array([0, 1, 2, 3, 4] * 6)
    probs = np.eye(5)[y] * 0.8 + 0.04
    rows = reliability_table(probs, y, n_bins=10)
    assert rows and all(0 <= r["conf"] <= 1 and 0 <= r["acc"] <= 1 for r in rows)
