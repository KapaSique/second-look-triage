"""Calibrated fusion classifier for ESI acuity — the predictive engine.

Design notes:
  * On this synthetic data the chief-complaint text predicts acuity ~perfectly, so the
    point of this module is NOT to chase accuracy. It is to be (a) a clean, calibrated,
    probabilistic engine for the demo/policy/audit, and (b) the substrate for the honest
    "text memorizes vs vitals is uncertain" contrast.
  * Logistic regression over a ColumnTransformer (numeric + one-hot + TF-IDF) is chosen
    deliberately: natively probabilistic, easy to calibrate, sparse-friendly, and light
    enough to run inside a Hugging Face Space — no gradient-boosting/native-lib dependency.
  * High-cardinality identifiers (`site_id`, `triage_nurse_id`) are excluded — the forensics
    showed they carry no signal, and using nurse identity to triage would be indefensible.
"""
from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

try:
    from src.data_prep import engineer_features, TARGET
except ImportError:  # flat import when src/ is shipped as a Kaggle utility dataset
    from data_prep import engineer_features, TARGET

ACUITY_CLASSES: List[int] = [1, 2, 3, 4, 5]
TEXT_COL = "chief_complaint_raw"
EXCLUDE = {TARGET, TEXT_COL, "site_id", "triage_nurse_id"}


class SecondLookModel:
    def __init__(self, use_text: bool = True, use_tabular: bool = True,
                 calibrate: bool = True, calib_method: str = "sigmoid",
                 max_features: int = 20000):
        self.use_text = use_text
        self.use_tabular = use_tabular
        self.calibrate = calibrate
        self.calib_method = calib_method
        self.max_features = max_features
        self.pipe = None

    def _build(self, feats: pd.DataFrame) -> ColumnTransformer:
        transformers = []
        if self.use_tabular:
            num = [c for c in feats.columns if c not in EXCLUDE and is_numeric_dtype(feats[c])]
            cat = [c for c in feats.columns if c not in EXCLUDE and not is_numeric_dtype(feats[c]) and c != TEXT_COL]
            if num:
                transformers.append(("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
            if cat:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat))
        if self.use_text and TEXT_COL in feats.columns:
            transformers.append(("txt", TfidfVectorizer(max_features=self.max_features,
                                                         ngram_range=(1, 2)), TEXT_COL))
        if not transformers:
            raise ValueError("No features selected (use_text/use_tabular both off or columns absent).")
        return ColumnTransformer(transformers, remainder="drop")

    def fit(self, df: pd.DataFrame, y) -> "SecondLookModel":
        feats = engineer_features(df)
        ct = self._build(feats)
        base = Pipeline([("ct", ct), ("clf", LogisticRegression(max_iter=2000))])
        if self.calibrate:
            self.pipe = CalibratedClassifierCV(base, method=self.calib_method, cv=3)
        else:
            self.pipe = base
        self.pipe.fit(feats, np.asarray(y))
        self._classes = list(self.pipe.classes_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        feats = engineer_features(df)
        proba = self.pipe.predict_proba(feats)
        full = np.zeros((len(feats), 5), dtype=float)
        for j, c in enumerate(self._classes):
            full[:, ACUITY_CLASSES.index(int(c))] = proba[:, j]
        s = full.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return full / s

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        idx = self.predict_proba(df).argmax(axis=1)
        return np.array([ACUITY_CLASSES[i] for i in idx])


# ---- metrics ----
def undertriage_rate(y_true, y_pred) -> float:
    """% of truly urgent patients (ESI 1/2) assigned a LESS urgent (higher) level."""
    yt = np.asarray(y_true); yp = np.asarray(y_pred)
    m = np.isin(yt, [1, 2])
    return float((yp[m] > yt[m]).mean() * 100) if m.sum() else 0.0


def expected_calibration_error(probs, y_true_idx, n_bins: int = 10) -> float:
    """ECE over max-probability confidence. `y_true_idx` are 0-based class column indices."""
    probs = np.asarray(probs); y = np.asarray(y_true_idx)
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0; n = len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n
    return float(ece)


def cross_validate(model_factory: Callable[[], SecondLookModel], df: pd.DataFrame, y,
                   n_splits: int = 5) -> Dict[str, float]:
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs, f1s, uts, eces = [], [], [], []
    for tr, te in skf.split(df, y):
        m = model_factory().fit(df.iloc[tr], y[tr])
        proba = m.predict_proba(df.iloc[te])
        pred = np.array([ACUITY_CLASSES[i] for i in proba.argmax(1)])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        uts.append(undertriage_rate(y[te], pred))
        true_idx = np.array([ACUITY_CLASSES.index(int(v)) for v in y[te]])
        eces.append(expected_calibration_error(proba, true_idx))
    return dict(accuracy=float(np.mean(accs)), macro_f1=float(np.mean(f1s)),
                undertriage_rate=float(np.mean(uts)), ece=float(np.mean(eces)))
