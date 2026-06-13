"""Second Look — data forensics & honest baselines (Kaggle CPU kernel).

Produces the evidence base for the writeup/notebook:
  1. target distribution
  2. INFORMATIVE MISSINGNESS by acuity  (the deployment-trap centerpiece)
  3. 5-fold CV baselines: text-only vs vitals-only vs fusion (acc, macroF1, undertriage)
  4. NEWS2-only ceiling
  5. why text -> ~100%: complaint cardinality + determinism + train/test overlap
  6. nurse/site variability (~0)
  7. occult high-risk cases (urgent but normal vitals)

Outputs -> /kaggle/working/outputs/forensics.json  and  /kaggle/working/figures/*.png
"""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp

OUT = "/kaggle/working/outputs"; FIG = "/kaggle/working/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

MISSING_VITALS = ["systolic_bp","diastolic_bp","heart_rate","respiratory_rate","temperature_c","spo2"]
res = {}

train = pd.read_csv(find("train.csv"))
test  = pd.read_csv(find("test.csv"))
cc    = pd.read_csv(find("chief_complaints.csv"))[["patient_id","chief_complaint_raw"]]
train = train.merge(cc, on="patient_id", how="left")
test  = test.merge(cc, on="patient_id", how="left")
y = train["triage_acuity"].values
res["n_train"], res["n_test"] = len(train), len(test)

# ---- 1. target distribution ----
dist = train["triage_acuity"].value_counts().sort_index()
res["target_distribution"] = {int(k): int(v) for k, v in dist.items()}
res["target_pct"] = {int(k): round(v*100, 2) for k, v in (dist/len(train)).items()}
plt.figure(figsize=(5,3.2))
plt.bar(dist.index.astype(str), dist.values, color="#2c6fbb")
plt.xlabel("ESI triage_acuity (1=most urgent)"); plt.ylabel("patients"); plt.title("Acuity distribution (train)")
plt.tight_layout(); plt.savefig(f"{FIG}/01_target_distribution.png", dpi=130); plt.close()

# ---- 2. informative missingness by acuity ----
miss_by_ac = {}
for v in MISSING_VITALS:
    if v in train:
        miss_by_ac[v] = (train.groupby("triage_acuity")[v].apply(lambda s: s.isna().mean()*100)).round(2).to_dict()
res["missing_pct_by_acuity"] = {k: {int(a): float(p) for a, p in d.items()} for k, d in miss_by_ac.items()}
# figure: systolic_bp missing rate by acuity
bp = train.assign(m=train["systolic_bp"].isna()).groupby("triage_acuity")["m"].mean()*100
plt.figure(figsize=(5,3.2))
plt.bar(bp.index.astype(str), bp.values, color="#c0392b")
plt.xlabel("triage_acuity"); plt.ylabel("% systolic_bp MISSING")
plt.title("Informative missingness:\nvitals skipped only for low-acuity")
plt.tight_layout(); plt.savefig(f"{FIG}/02_informative_missingness.png", dpi=130); plt.close()

# ---- helper metrics ----
def undertriage_rate(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp); m = np.isin(yt, [1, 2])
    return float((yp[m] > yt[m]).mean()*100) if m.sum() else 0.0

def cv_eval(make_Xtr_Xte, y, n=5):
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=0)
    accs, f1s, uts = [], [], []
    idx = np.arange(len(y))
    for tr, te in skf.split(idx, y):
        Xtr, Xte = make_Xtr_Xte(tr, te)
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(Xtr, y[tr]); p = clf.predict(Xte)
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
        uts.append(undertriage_rate(y[te], p))
    return dict(acc=round(np.mean(accs),4), acc_std=round(np.std(accs),4),
               macro_f1=round(np.mean(f1s),4), undertriage_pct=round(np.mean(uts),3))

txt = train["chief_complaint_raw"].fillna("").values
NUM = [c for c in ["news2_score","shock_index","spo2","gcs_total","heart_rate","respiratory_rate",
                   "systolic_bp","diastolic_bp","temperature_c","age","pain_score","num_comorbidities",
                   "num_prior_ed_visits_12m","bmi"] if c in train.columns]
Xnum_all = train[NUM].replace(-1, np.nan).values

def make_text(tr, te):
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1,2)).fit(txt[tr])
    return tf.transform(txt[tr]), tf.transform(txt[te])
def make_vitals(tr, te):
    imp = SimpleImputer().fit(Xnum_all[tr]); sc = StandardScaler().fit(imp.transform(Xnum_all[tr]))
    return sc.transform(imp.transform(Xnum_all[tr])), sc.transform(imp.transform(Xnum_all[te]))
def make_fusion(tr, te):
    tf = TfidfVectorizer(max_features=5000, ngram_range=(1,2)).fit(txt[tr])
    imp = SimpleImputer().fit(Xnum_all[tr]); sc = StandardScaler().fit(imp.transform(Xnum_all[tr]))
    A = sp.hstack([sc.transform(imp.transform(Xnum_all[tr])), tf.transform(txt[tr])]).tocsr()
    B = sp.hstack([sc.transform(imp.transform(Xnum_all[te])), tf.transform(txt[te])]).tocsr()
    return A, B

print("CV: text-only ..."); res.setdefault("baselines", {})["text_only"]   = cv_eval(make_text, y)
print("CV: vitals-only ..."); res["baselines"]["vitals_only"] = cv_eval(make_vitals, y)
print("CV: fusion ...");      res["baselines"]["fusion"]      = cv_eval(make_fusion, y)

# baseline comparison figure
labels = ["text_only","vitals_only","fusion"]
accs = [res["baselines"][l]["acc"] for l in labels]
uts  = [res["baselines"][l]["undertriage_pct"] for l in labels]
fig, ax = plt.subplots(1, 2, figsize=(8,3.2))
ax[0].bar(labels, accs, color=["#27ae60","#e67e22","#2c6fbb"]); ax[0].set_ylim(0,1.05); ax[0].set_title("Accuracy (5-fold)")
ax[1].bar(labels, uts, color=["#27ae60","#e67e22","#2c6fbb"]); ax[1].set_title("Undertriage % (true ESI 1/2)")
for a in ax: a.tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(f"{FIG}/03_baselines.png", dpi=130); plt.close()

# ---- 4. NEWS2-only ceiling ----
d = train.dropna(subset=["news2_score"])
sk = StratifiedKFold(5, shuffle=True, random_state=0); accs=[]
for tr, te in sk.split(d, d["triage_acuity"]):
    m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(d.iloc[tr][["news2_score"]], d.iloc[tr]["triage_acuity"])
    accs.append(accuracy_score(d.iloc[te]["triage_acuity"], m.predict(d.iloc[te][["news2_score"]])))
res["news2_only_acc"] = round(float(np.mean(accs)), 4)

# ---- 5. why text->100%: complaint cardinality / determinism / overlap ----
g = train.groupby("chief_complaint_raw")["triage_acuity"].nunique()
res["complaint_analysis"] = {
    "n_unique_train": int(train["chief_complaint_raw"].nunique()),
    "pct_complaints_single_acuity": round(float((g == 1).mean()*100), 2),
    "pct_train_rows_deterministic_complaint": round(float(train["chief_complaint_raw"].map(g).eq(1).mean()*100), 2),
    "test_complaints_seen_in_train_pct": round(float(test["chief_complaint_raw"].isin(set(train["chief_complaint_raw"])).mean()*100), 2),
}

# ---- 6. nurse/site variability ----
def spread(col):
    gg = train.groupby(col)["triage_acuity"].agg(["mean","count"]); gg = gg[gg["count"] >= 50]
    return dict(n=int(len(gg)), mean_min=round(float(gg["mean"].min()),3),
                mean_max=round(float(gg["mean"].max()),3), mean_std=round(float(gg["mean"].std()),4))
res["variability"] = {c: spread(c) for c in ["site_id","triage_nurse_id"] if c in train.columns}

# ---- 7. occult high-risk: acuity 1/2 with NEWS2<3 ----
occ = train[(train["triage_acuity"].isin([1,2])) & (train["news2_score"] < 3)]
res["occult"] = {
    "n": int(len(occ)),
    "pct_of_urgent": round(float(len(occ)/max(1, train["triage_acuity"].isin([1,2]).sum())*100), 3),
    "examples": occ["chief_complaint_raw"].dropna().head(15).tolist(),
}

with open(f"{OUT}/forensics.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE. Key numbers:")
print(json.dumps({k: res[k] for k in ["baselines","news2_only_acc","complaint_analysis","variability","occult"]}, indent=2)[:2500])
