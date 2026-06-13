"""External validity on REAL ED data — NHAMCS 2019-2022 (CDC, public).

Tests whether the two patterns we found in the synthetic Triagegeist data also appear in
real US emergency-department visits:

  (1) INFORMATIVE MISSINGNESS — are triage vitals recorded LESS often for lower-acuity
      patients?  (synthetic: systolic-BP missing 0% for ESI 1-3 vs ~12% for ESI 4-5)
  (2) OCCULT HIGH-RISK — do genuinely high-acuity patients sometimes present with
      normal-range vitals? (motivates a vitals-independent red-flag channel)

NHAMCS variables (raw CDC .dta, read with convert_categoricals=False):
  IMMEDR = triage immediacy 1=Immediate .. 5=Non-urgent; <=0 / >5 = no-triage/unknown -> dropped
  vitals: BPSYS, BPDIAS, PULSE, RESPR, TEMPF, POPCT  (CDC blanks coded negative -> missing)

Unweighted exploratory analysis (we study a data-RECORDING pattern, not national estimates).
Outputs -> /kaggle/working/outputs/nhamcs.json  +  /kaggle/working/figures/*.png
"""
import os, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/kaggle/working/outputs"; FIG = "/kaggle/working/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
VITALS = ["BPSYS", "BPDIAS", "PULSE", "RESPR", "TEMPF", "POPCT"]
res = {}

# ---- load all NHAMCS ED .dta files, raw numeric codes ----
files = sorted(glob.glob("/kaggle/input/**/ed20*.dta", recursive=True))
print("found dta:", files)
frames = []
for f in files:
    try:
        d = pd.read_stata(f, convert_categoricals=False)
        d.columns = [c.upper() for c in d.columns]
        frames.append(d[[c for c in (["IMMEDR"] + VITALS) if c in d.columns]].assign(_yr=os.path.basename(f)))
    except Exception as e:
        print("skip", f, repr(e)[:100])
assert frames, "no NHAMCS .dta loaded"
df = pd.concat(frames, ignore_index=True, sort=False)
res["n_visits_total"] = int(len(df))
res["years"] = sorted(df["_yr"].unique().tolist())

# ---- valid triage rows: IMMEDR 1..5 ----
df = df[df["IMMEDR"].between(1, 5)].copy()
res["n_with_triage_1_5"] = int(len(df))
res["immedr_distribution"] = {int(k): int(v) for k, v in df["IMMEDR"].value_counts().sort_index().items()}

# ---- (1) informative missingness: a vital is "not recorded" if NaN or <=0 (CDC blanks negative) ----
present_vitals = [v for v in VITALS if v in df.columns]
for v in present_vitals:
    df[v + "_miss"] = (df[v].isna() | (df[v] <= 0)).astype(int)

miss_by_immedr = {}
for v in present_vitals:
    s = df.groupby("IMMEDR")[v + "_miss"].mean().mul(100).round(2)
    miss_by_immedr[v] = {int(k): float(val) for k, val in s.items()}
res["missing_pct_by_immedr"] = miss_by_immedr

# any-vital-missing by acuity + correlation (does missingness rise with less-urgent acuity?)
df["any_vital_miss"] = df[[v + "_miss" for v in present_vitals]].max(axis=1)
anymiss = df.groupby("IMMEDR")["any_vital_miss"].mean().mul(100).round(2)
res["any_vital_missing_pct_by_immedr"] = {int(k): float(v) for k, v in anymiss.items()}
# Spearman corr between acuity level (1..5, higher=less urgent) and per-row missingness
res["spearman_acuity_vs_anymiss"] = float(
    pd.Series(df["IMMEDR"]).corr(df["any_vital_miss"], method="spearman"))
# headline: BP missing in urgent (1-2) vs non-urgent (4-5)
bp = "BPSYS"
if bp in present_vitals:
    res["bp_missing_urgent_1_2_pct"] = round(float(df.loc[df.IMMEDR.isin([1, 2]), bp + "_miss"].mean() * 100), 2)
    res["bp_missing_nonurgent_4_5_pct"] = round(float(df.loc[df.IMMEDR.isin([4, 5]), bp + "_miss"].mean() * 100), 2)

# ---- (2) occult high-risk: high-acuity (IMMEDR 1-2) with all-normal RECORDED vitals ----
def normal_mask(d):
    ok = pd.Series(True, index=d.index)
    rng = {"BPSYS": (90, 140), "PULSE": (60, 100), "RESPR": (12, 20), "POPCT": (95, 100)}
    for v, (lo, hi) in rng.items():
        if v in d.columns:
            rec = (d[v] > 0) & d[v].notna()
            ok &= (~rec) | (d[v].between(lo, hi))   # recorded vitals all in normal range
    # require at least BP+pulse recorded so "normal" is meaningful
    have = pd.Series(True, index=d.index)
    for v in ["BPSYS", "PULSE"]:
        if v in d.columns:
            have &= (d[v] > 0) & d[v].notna()
    return ok & have

urg = df[df.IMMEDR.isin([1, 2])]
occ = urg[normal_mask(urg)]
res["occult"] = {
    "n_urgent_1_2": int(len(urg)),
    "n_urgent_with_all_normal_vitals": int(len(occ)),
    "pct_urgent_with_normal_vitals": round(float(len(occ) / max(1, len(urg)) * 100), 2),
}

# ---- figures ----
# fig A: NHAMCS BP-missing by IMMEDR
if bp in present_vitals:
    s = df.groupby("IMMEDR")[bp + "_miss"].mean().mul(100)
    plt.figure(figsize=(5, 3.3))
    plt.bar(s.index.astype(int).astype(str), s.values, color="#c0392b")
    plt.xlabel("NHAMCS triage immediacy (1=Immediate .. 5=Non-urgent)")
    plt.ylabel("% systolic-BP NOT recorded")
    plt.title("Real ED data (NHAMCS 2019-22):\nvitals recorded less for lower-acuity")
    plt.tight_layout(); plt.savefig(f"{FIG}/10_nhamcs_missingness.png", dpi=130); plt.close()

# fig B: synthetic vs NHAMCS side-by-side (synthetic numbers from our forensics)
synth = {1: 0.0, 2: 0.0, 3: 0.0, 4: 12.15, 5: 11.84}
if bp in present_vitals:
    nh = df.groupby("IMMEDR")[bp + "_miss"].mean().mul(100).reindex([1, 2, 3, 4, 5])
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    ax[0].bar([str(i) for i in range(1, 6)], [synth[i] for i in range(1, 6)], color="#2c6fbb")
    ax[0].set_title("Synthetic Triagegeist"); ax[0].set_xlabel("ESI acuity"); ax[0].set_ylabel("% systolic-BP missing")
    ax[1].bar([str(i) for i in range(1, 6)], [nh.get(i, np.nan) for i in range(1, 6)], color="#c0392b")
    ax[1].set_title("Real NHAMCS 2019-22"); ax[1].set_xlabel("triage immediacy")
    fig.suptitle("Vitals recorded non-randomly by acuity — real ED data is U-shaped", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{FIG}/11_synthetic_vs_nhamcs.png", dpi=130); plt.close()

with open(f"{OUT}/nhamcs.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2)[:2500])
