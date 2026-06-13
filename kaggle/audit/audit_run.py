"""Second Look — full audit + demo-model export (Kaggle kernel).

Fits the calibrated Second Look model, then audits it honestly:
  * holdout metrics (accuracy, macro-F1, undertriage, ECE),
  * calibration reliability diagram,
  * fairness: undertriage/overtriage by sex / age_group / language / insurance,
  * the CENTERPIECE deployment test — missingness stress: blank vitals for truly urgent
    patients and watch undertriage. A VITALS-ONLY model exploits the "missing=low-acuity"
    shortcut and undertriages; the TEXT-FUSION model resists. This quantifies the
    informative-missingness hazard.
  * top text tokens per acuity class (what drives the text channel).

Also exports a compact calibrated model (`model.pkl`) for the Gradio demo.

Imports the same tested `src` modules (shipped as the `second-look-src` dataset).
Outputs -> /kaggle/working/outputs/audit.json , figures/*.png , model.pkl
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import joblib

SRC = glob.glob("/kaggle/input/**/data_prep.py", recursive=True)
sys.path.append(os.path.dirname(SRC[0]))
from data_prep import load_and_merge          # noqa
from model_core import SecondLookModel, ACUITY_CLASSES, expected_calibration_error, undertriage_rate
from audit import group_metrics, missingness_stress, reliability_table, overtriage_rate

OUT = "/kaggle/working/outputs"; FIG = "/kaggle/working/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
DATA = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])

df = load_and_merge(DATA, "train")
y = df["triage_acuity"].values
from sklearn.model_selection import train_test_split
tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=y)
tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
ytr, yte = y[tr_idx], y[te_idx]
res = {}

# ---- calibrated fusion model: holdout metrics + calibration ----
print("fitting calibrated fusion (holdout)...")
model = SecondLookModel(calibrate=True, max_features=20000).fit(tr, ytr)
proba = model.predict_proba(te)
pred = np.array([ACUITY_CLASSES[i] for i in proba.argmax(1)])
true_idx = np.array([ACUITY_CLASSES.index(int(v)) for v in yte])
from sklearn.metrics import accuracy_score, f1_score
res["holdout"] = dict(
    accuracy=round(float(accuracy_score(yte, pred)), 4),
    macro_f1=round(float(f1_score(yte, pred, average="macro")), 4),
    undertriage_pct=round(undertriage_rate(yte, pred), 3),
    overtriage_pct=round(overtriage_rate(yte, pred), 3),
    ece=round(expected_calibration_error(proba, true_idx), 4),
)
print("holdout:", res["holdout"])

# reliability diagram
rel = reliability_table(proba, true_idx, n_bins=10); res["reliability"] = rel
plt.figure(figsize=(4.2, 4.2))
plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
if rel:
    plt.plot([r["conf"] for r in rel], [r["acc"] for r in rel], "o-", color="#2c6fbb")
plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.xlim(0, 1); plt.ylim(0, 1)
plt.title(f"Reliability (ECE={res['holdout']['ece']:.3f})"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/05_reliability.png", dpi=130); plt.close()

# ---- fairness by group ----
res["fairness"] = {}
for col in ["sex", "age_group", "language", "insurance_type"]:
    if col in te.columns:
        gm = group_metrics(te, yte, pred, col)
        res["fairness"][col] = {str(k): v for k, v in gm.items()}
# fairness figure (undertriage by sex + age_group)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, col in zip(ax, ["sex", "age_group"]):
    if col in res["fairness"]:
        gm = res["fairness"][col]
        keys = list(gm.keys()); vals = [gm[k]["undertriage_pct"] for k in keys]
        a.bar(keys, vals, color="#8e44ad"); a.set_title(f"Undertriage % by {col}")
        a.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{FIG}/06_fairness.png", dpi=130); plt.close()

# ---- confusion matrix ----
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, pred, labels=ACUITY_CLASSES, normalize="true")
plt.figure(figsize=(4.6, 4))
plt.imshow(cm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(fraction=0.046)
plt.xticks(range(5), ACUITY_CLASSES); plt.yticks(range(5), ACUITY_CLASSES)
plt.xlabel("predicted ESI"); plt.ylabel("true ESI"); plt.title("Confusion (row-normalized)")
for i in range(5):
    for j in range(5):
        plt.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                 color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/07_confusion.png", dpi=130); plt.close()

# ---- missingness stress: vitals-only (vulnerable) vs fusion (robust) ----
print("missingness stress...")
VIT = ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature_c", "spo2"]
vit_model = SecondLookModel(use_text=False, calibrate=False, max_features=1).fit(tr, ytr)
fus_model = SecondLookModel(calibrate=False, max_features=20000).fit(tr, ytr)
res["missingness_stress"] = {
    "vitals_only": missingness_stress(lambda d: vit_model.predict(d), te, yte, VIT, "urgent"),
    "fusion": missingness_stress(lambda d: fus_model.predict(d), te, yte, VIT, "urgent"),
}
print("stress:", res["missingness_stress"])
ms = res["missingness_stress"]
labels = ["vitals-only", "fusion (text)"]
base = [ms["vitals_only"]["baseline_undertriage"], ms["fusion"]["baseline_undertriage"]]
strs = [ms["vitals_only"]["stressed_undertriage"], ms["fusion"]["stressed_undertriage"]]
x = np.arange(2); w = 0.38
plt.figure(figsize=(6, 3.6))
plt.bar(x - w/2, base, w, label="vitals present", color="#27ae60")
plt.bar(x + w/2, strs, w, label="vitals blanked for the sick", color="#c0392b")
plt.xticks(x, labels); plt.ylabel("undertriage % (true ESI 1/2)")
plt.title("Informative-missingness stress test"); plt.legend()
plt.tight_layout(); plt.savefig(f"{FIG}/08_missingness_stress.png", dpi=130); plt.close()

# ---- top text tokens per class (uncalibrated fusion for interpretability) ----
try:
    ct = fus_model.pipe.named_steps["ct"]; clf = fus_model.pipe.named_steps["clf"]
    names = np.array(ct.get_feature_names_out())
    txt_mask = np.array([n.startswith("txt__") for n in names])
    coef = clf.coef_
    top = {}
    for ci, cls in enumerate(clf.classes_):
        order = np.argsort(coef[ci][txt_mask])[::-1][:8]
        top[int(cls)] = [names[txt_mask][o].replace("txt__", "") for o in order]
    res["top_tokens_per_class"] = top
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    for ax, cls in zip(axes, sorted(top)):
        toks = top[cls][::-1]
        ax.barh(range(len(toks)), range(1, len(toks)+1), color="#2c6fbb")
        ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=7)
        ax.set_title(f"ESI {cls}"); ax.set_xticks([])
    plt.suptitle("Top chief-complaint tokens per acuity (text channel)")
    plt.tight_layout(); plt.savefig(f"{FIG}/09_top_tokens.png", dpi=130); plt.close()
except Exception as e:
    res["top_tokens_error"] = str(e)[:200]

# ---- export compact calibrated model for the demo (FIXED minimal schema) ----
# The Gradio demo can only realistically collect a handful of fields, so we train the
# exported model on an explicit, demo-controllable schema (text + key vitals + 2 derived).
DEMO_COLS = ["chief_complaint_raw", "age", "sex", "arrival_mode",
             "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
             "temperature_c", "spo2", "gcs_total", "pain_score", "num_comorbidities",
             "news2_score", "shock_index"]
print("fitting compact demo model on ALL data (fixed schema)...")
demo = SecondLookModel(calibrate=True, max_features=8000).fit(df[DEMO_COLS], y)
joblib.dump(demo, "/kaggle/working/model.pkl")
res["demo_cols"] = DEMO_COLS
res["demo_model_bytes"] = os.path.getsize("/kaggle/working/model.pkl")

with open(f"{OUT}/audit.json", "w") as f:
    json.dump(res, f, indent=2, default=str)
print("DONE", json.dumps({k: res[k] for k in ["holdout", "missingness_stress"]}, indent=2))
