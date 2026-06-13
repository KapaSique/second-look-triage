"""Honest audit of the MODEL and its deployment conditions.

The synthetic labels carry no injected bias or rater variability, so we never claim to have
"found" undertriage in the data. Instead we audit what we control and what a real deployment
would face:

  * group_metrics      — does OUR model's error (undertriage/overtriage) differ across
                         age/sex/language/insurance groups?
  * missingness_stress — the centerpiece deployment test: in this data "missing vitals" is a
                         proxy for low acuity, so what happens if vitals go missing for the
                         SICK (as they do in a chaotic real ED)? Does undertriage spike?
  * reliability_table  — calibration: do predicted confidences match observed accuracy?

These functions take plain arrays / a `predict_fn` callable so they are unit-testable without
a heavy fitted model; the audit kernel/notebook supplies the real model's predictions.
"""
from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np
import pandas as pd


def undertriage_rate(y_true, y_pred) -> float:
    yt = np.asarray(y_true); yp = np.asarray(y_pred); m = np.isin(yt, [1, 2])
    return float((yp[m] > yt[m]).mean() * 100) if m.sum() else 0.0


def overtriage_rate(y_true, y_pred) -> float:
    """% of truly low-acuity patients (ESI 4/5) assigned a MORE urgent (lower) level."""
    yt = np.asarray(y_true); yp = np.asarray(y_pred); m = np.isin(yt, [4, 5])
    return float((yp[m] < yt[m]).mean() * 100) if m.sum() else 0.0


def group_metrics(df: pd.DataFrame, y_true, y_pred, by: str) -> Dict[object, dict]:
    """Per-group n / undertriage% / overtriage% / accuracy for a grouping column."""
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    d = df.reset_index(drop=True)
    out: Dict[object, dict] = {}
    for g, pos in d.groupby(by).indices.items():
        pos = np.asarray(pos)
        out[g] = dict(
            n=int(len(pos)),
            undertriage_pct=undertriage_rate(yt[pos], yp[pos]),
            overtriage_pct=overtriage_rate(yt[pos], yp[pos]),
            accuracy=float((yt[pos] == yp[pos]).mean()),
        )
    return out


def missingness_stress(predict_fn: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                       y_true, vitals: List[str], target: str = "urgent") -> dict:
    """Drop `vitals` for the target rows and measure undertriage drift.

    target='urgent' simulates the dangerous real-ED case: a sick patient whose vitals were
    not captured. If the model leaned on the missingness-is-low-acuity shortcut, undertriage
    will spike.
    """
    yt = np.asarray(y_true)
    base = undertriage_rate(yt, np.asarray(predict_fn(df)))
    d = df.copy().reset_index(drop=True)
    mask = np.isin(yt, [1, 2]) if target == "urgent" else np.ones(len(d), dtype=bool)
    for v in vitals:
        if v in d.columns:
            d.loc[mask, v] = np.nan
    stressed = undertriage_rate(yt, np.asarray(predict_fn(d)))
    return dict(baseline_undertriage=round(base, 3), stressed_undertriage=round(stressed, 3),
                delta=round(stressed - base, 3))


def reliability_table(probs, y_true_idx, n_bins: int = 10) -> List[dict]:
    """Binned confidence vs accuracy for a reliability diagram. y_true_idx are 0-based indices."""
    probs = np.asarray(probs); y = np.asarray(y_true_idx)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); rows: List[dict] = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            rows.append(dict(bin_lo=float(bins[i]), bin_hi=float(bins[i + 1]),
                             conf=float(conf[m].mean()), acc=float(correct[m].mean()),
                             n=int(m.sum())))
    return rows
