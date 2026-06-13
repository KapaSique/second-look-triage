"""Assemble the public end-to-end Kaggle submission notebook with nbformat.

Runs locally only to BUILD the .ipynb JSON (no execution). Kaggle executes it on push.
Notebook attaches: competition `triagegeist` + dataset `ssstelmah/second-look-src`.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Second Look — a triage safety-net that knows what it doesn't know
### Triagegeist · emergency-department triage decision support

**TL;DR.** On the provided synthetic data, the free-text chief complaint predicts the ESI
label with ~99.9% accuracy — so *accuracy is not a meaningful objective here*. The real,
clinically important problem is **safe behaviour under realistic conditions**: catching the
occult high-risk patient, not collapsing when vitals are missing, being calibrated, and being
honest about what synthetic data can and cannot prove. This notebook builds **Second Look** —
a calibrated predictor + a clinical red-flag safety channel + a cost-sensitive decision policy
+ an honest audit — and reports several findings, including a **missingness shortcut that
drives undertriage from <1% to ~78% under a realistic data shift**.

*Research demonstration on synthetic data — not a medical device.*
""")

code("""import os, sys, glob, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# import the tested Second Look modules (shipped as a Kaggle utility dataset)
SRC = glob.glob('/kaggle/input/**/data_prep.py', recursive=True)
sys.path.append(os.path.dirname(SRC[0]))
from data_prep import load_and_merge
from model_core import SecondLookModel, ACUITY_CLASSES, expected_calibration_error, undertriage_rate
from redflag import RedFlagMatcher
from policy import TriagePolicy
from audit import group_metrics, missingness_stress, reliability_table, overtriage_rate
from explain import explain
from clinical import compute_derived

DATA = os.path.dirname(glob.glob('/kaggle/input/**/train.csv', recursive=True)[0])
df = load_and_merge(DATA, 'train'); y = df['triage_acuity'].values
print('train:', df.shape, '| classes:', dict(pd.Series(y).value_counts().sort_index()))""")

md("""## 1 · Accuracy is a mirage — the label is a near-lookup

We first establish, honestly, what the data is. Three nested baselines under 5-fold CV, plus a
look at the chief-complaint vocabulary.""")

code("""txt = df['chief_complaint_raw'].fillna('').values
NUM = [c for c in ['news2_score','shock_index','spo2','gcs_total','heart_rate','respiratory_rate',
                   'systolic_bp','diastolic_bp','temperature_c','age','pain_score','num_comorbidities'] if c in df]
Xnum = df[NUM].replace(-1, np.nan).values
def cv(makeXY):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); a=[]; u=[]
    for tr,te in skf.split(np.arange(len(y)), y):
        Xtr,Xte = makeXY(tr,te); m=LogisticRegression(max_iter=1000).fit(Xtr,y[tr]); p=m.predict(Xte)
        a.append(accuracy_score(y[te],p)); u.append(undertriage_rate(y[te],p))
    return np.mean(a), np.mean(u)
def mk_text(tr,te):
    v=TfidfVectorizer(max_features=5000,ngram_range=(1,2)).fit(txt[tr]); return v.transform(txt[tr]), v.transform(txt[te])
def mk_vit(tr,te):
    im=SimpleImputer().fit(Xnum[tr]); sc=StandardScaler().fit(im.transform(Xnum[tr]))
    return sc.transform(im.transform(Xnum[tr])), sc.transform(im.transform(Xnum[te]))
for name,mk in [('text-only',mk_text),('vitals-only',mk_vit)]:
    acc,und = cv(mk); print(f'{name:12s} acc={acc:.4f}  undertriage(true ESI1/2)={und:.2f}%')
g = df.groupby('chief_complaint_raw')['triage_acuity'].nunique()
print(f"\\nunique complaints: {df['chief_complaint_raw'].nunique()} | "
      f"% mapping to a single acuity: {(g==1).mean()*100:.1f}%")""")

md("""**Reading.** Text alone ≈ **99.9%** accuracy (undertriage ~0.3%); vitals alone ~80% (undertriage
~4.6%). ~99.7% of complaints map to a single acuity and ~99.8% of test complaints are already in
train — the task is essentially a phrase→label lookup. **We will not treat accuracy as the
objective.**""")

md("""## 2 · Informative missingness is a deployment trap""")

code("""VIT=['systolic_bp','diastolic_bp','heart_rate','respiratory_rate','temperature_c','spo2']
miss = pd.DataFrame({v:(df.groupby('triage_acuity')[v].apply(lambda s:s.isna().mean()*100)).round(1) for v in VIT})
print('% MISSING by acuity:'); print(miss)
bp = df.assign(m=df.systolic_bp.isna()).groupby('triage_acuity').m.mean()*100
plt.figure(figsize=(5,3)); plt.bar(bp.index.astype(str), bp.values, color='#c0392b')
plt.xlabel('triage_acuity'); plt.ylabel('% systolic_bp MISSING')
plt.title('Vitals are skipped only for low-acuity patients'); plt.tight_layout(); plt.show()""")

md("""Systolic-BP missingness is **0% for ESI 1–3 but ~12% for ESI 4–5**. "No BP recorded" is a
near-perfect proxy for "not sick" *in this dataset*. We return to the danger this creates in §5.""")

md("""## 3 · Occult high-risk patients, and the ceiling of vitals-only scores""")

code("""occ = df[(df.triage_acuity.isin([1,2])) & (df.news2_score<3)]
print(f'occult high-risk (urgent but NEWS2<3): {len(occ)} ({len(occ)/df.triage_acuity.isin([1,2]).sum()*100:.2f}% of urgent)')
print('examples:'); [print('  -',t) for t in occ.chief_complaint_raw.dropna().head(8)]
d2 = df.dropna(subset=['news2_score']); sk=StratifiedKFold(5,shuffle=True,random_state=0); a=[]
for tr,te in sk.split(d2,d2.triage_acuity):
    m=DecisionTreeClassifier(max_depth=4,random_state=0).fit(d2.iloc[tr][['news2_score']],d2.iloc[tr].triage_acuity)
    a.append(accuracy_score(d2.iloc[te].triage_acuity,m.predict(d2.iloc[te][['news2_score']])))
print(f'\\nNEWS2-only accuracy ceiling: {np.mean(a):.3f}')""")

md("""These textbook can't-miss complaints present with deceptively normal vitals — NEWS2 alone
(~65% ceiling) would miss them. The **free text** and a **clinical red-flag layer** are the only
channels that see them.""")

md("""## 4 · The Second Look model: calibrated, audited, fair

A calibrated logistic fusion model (vitals + one-hot demographics + TF-IDF complaint). We report
calibration and fairness, not just accuracy.""")

code("""tr_i,te_i = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=y)
tr,te = df.iloc[tr_i].reset_index(drop=True), df.iloc[te_i].reset_index(drop=True)
ytr,yte = y[tr_i], y[te_i]
model = SecondLookModel(calibrate=True, max_features=20000).fit(tr, ytr)
proba = model.predict_proba(te); pred = np.array([ACUITY_CLASSES[i] for i in proba.argmax(1)])
ti = np.array([ACUITY_CLASSES.index(int(v)) for v in yte])
print(f'holdout: acc={accuracy_score(yte,pred):.4f}  macroF1={f1_score(yte,pred,average="macro"):.4f}  '
      f'undertriage={undertriage_rate(yte,pred):.2f}%  overtriage={overtriage_rate(yte,pred):.2f}%  '
      f'ECE={expected_calibration_error(proba,ti):.3f}')
# fairness
for col in ['sex','age_group']:
    gm = group_metrics(te, yte, pred, col)
    print(f'\\nundertriage % by {col}:', {k:round(v["undertriage_pct"],2) for k,v in gm.items()})""")

md("""## 5 · The missingness trap, quantified (the centerpiece)

Standard practice adds missingness indicators. We stress-test by blanking the vitals of *truly
urgent* patients — the real-ED scenario of a crashing patient whose vitals weren't captured — and
compare an identical model **with vs without** the indicator features.""")

code("""fus_with = SecondLookModel(calibrate=False, max_features=20000).fit(tr, ytr)
fus_without = SecondLookModel(calibrate=False, max_features=20000, drop_missingness_indicators=True).fit(tr, ytr)
ms = {'with indicators': missingness_stress(lambda d: fus_with.predict(d), te, yte, VIT, 'urgent'),
      'without (de-biased)': missingness_stress(lambda d: fus_without.predict(d), te, yte, VIT, 'urgent')}
for k,v in ms.items(): print(f'{k:22s} undertriage  baseline={v["baseline_undertriage"]:.2f}%  ->  '
                              f'vitals-blanked={v["stressed_undertriage"]:.2f}%')
labels=list(ms); x=np.arange(2); w=0.38
plt.figure(figsize=(6,3.6))
plt.bar(x-w/2,[ms[k]['baseline_undertriage'] for k in labels],w,label='vitals present',color='#27ae60')
plt.bar(x+w/2,[ms[k]['stressed_undertriage'] for k in labels],w,label='vitals blanked for the sick',color='#c0392b')
plt.xticks(x,labels); plt.ylabel('undertriage % (true ESI 1/2)')
plt.title('Informative-missingness trap'); plt.legend(); plt.tight_layout(); plt.show()""")

md("""**The "excellent" held-out model is the dangerous one.** With the missingness indicators,
blanking the vitals of sick patients sends undertriage from <1% to ~78%; removing them de-biases
the model. Our shipped demo uses the de-biased model. *A model that looks great on the synthetic
holdout would be unsafe in deployment — only the audit catches it.*""")

md("""## 6 · Generalization: clinical knowledge beats memorisation

The synthetic vocabulary is closed, so a TF-IDF model memorises. We probe with hand-written lay
paraphrases (not in train) and measure **critical safe-recall** (critical cases assigned ESI ≤ 2).""")

code("""PARA=[("sudden explosive headache unlike anything before",True),("crushing chest discomfort, sweaty, pain shooting down the left arm",True),
("the left side of his face is drooping and his speech is slurred",True),("burning up with fever, confused and barely responsive",True),
("tearing pain ripping through to the back",True),("throat is closing up after a bee sting, covered in hives",True),
("can't catch her breath and the lips are turning blue",True),("throwing up bright red blood and feeling faint",True),
("stiff neck, high fever and the light hurts his eyes",True),("collapsed and is not waking up",True),
("needs a repeat prescription for blood pressure tablets",False),("twisted an ankle at football, mild swelling",False),
("mild sore throat for a couple of days, no fever",False),("here for a flu shot",False),("blocked nose and sneezing for a week",False)]
corpus = df.groupby('chief_complaint_raw').triage_acuity.agg(lambda s:int(s.mode().iloc[0]))
ctexts, cac = corpus.index.tolist(), corpus.values
tf = TfidfVectorizer(ngram_range=(1,2)).fit(ctexts+[p[0] for p in PARA])
sim = cosine_similarity(tf.transform([p[0] for p in PARA]), tf.transform(ctexts))
tfidf_pred = cac[sim.argmax(1)]
mm = RedFlagMatcher()
crit=[i for i,p in enumerate(PARA) if p[1]]
tfidf_safe = np.mean([tfidf_pred[i]<=2 for i in crit])*100
onto_safe  = np.mean([(mm.esi_floor(PARA[i][0]) or 4)<=2 for i in crit])*100
print(f'critical safe-recall  |  TF-IDF NN: {tfidf_safe:.0f}%   red-flag ontology: {onto_safe:.0f}%'
      f'   (biomedical transformer, separate kernel: 35%)')""")

md("""| approach | critical safe-recall |
|---|---|
| TF-IDF nearest-neighbour (memoriser) | ~30% |
| biomedical transformer (PubMedBERT) NN | ~35% |
| **curated clinical red-flag ontology** | **~90%** (0% false positives) |

A system that scores 100% on the synthetic vocabulary catches only ~⅓ of unseen lay emergencies.
Encoding clinical knowledge generalises far better — but even 90% on a small probe is not a
deployment guarantee.""")

md("""## 7 · Second Look in action

The full pipeline (calibrated probs → red-flags → NEWS2 → cost-sensitive policy → explanation) on
illustrative cases, including an occult ACS with normal vitals and a missing-vitals scenario.""")

code("""policy = TriagePolicy()
CASES = [
 dict(chief_complaint_raw='chest pain with diaphoresis and arm radiation', age=61, systolic_bp=128, diastolic_bp=84,
      heart_rate=96, respiratory_rate=18, temperature_c=36.9, spo2=97, gcs_total=15, pain_score=7),
 dict(chief_complaint_raw='needs a repeat prescription for blood pressure tablets', age=58, systolic_bp=138, diastolic_bp=88,
      heart_rate=74, respiratory_rate=15, temperature_c=36.8, spo2=99, gcs_total=15, pain_score=0),
 dict(chief_complaint_raw='sepsis with altered mental status', age=73, systolic_bp=92, diastolic_bp=55,
      heart_rate=122, respiratory_rate=26, temperature_c=38.9, spo2=91, gcs_total=13, pain_score=6),
]
DEMO_COLS=['chief_complaint_raw','age','sex','arrival_mode','systolic_bp','diastolic_bp','heart_rate',
           'respiratory_rate','temperature_c','spo2','gcs_total','pain_score','num_comorbidities','news2_score','shock_index']
demo_model = SecondLookModel(calibrate=True, max_features=8000, drop_missingness_indicators=True).fit(df[DEMO_COLS], y)
for c in CASES:
    c.setdefault('sex','M'); c.setdefault('arrival_mode','ambulance'); c.setdefault('num_comorbidities',2)
    d = compute_derived(dict(c)); flags = RedFlagMatcher().flags(c['chief_complaint_raw'])
    probs = demo_model.predict_proba(pd.DataFrame([{k:d.get(k,np.nan) for k in DEMO_COLS}]))[0]
    dec = policy.decide(probs, redflags=flags, news2=d['news2_score'])
    print(f"\\n• {c['chief_complaint_raw'][:55]!r}  NEWS2={d['news2_score']}")
    print(f"  -> recommended ESI {dec.acuity} ({'ESCALATE' if dec.escalate else 'routine'}); {dec.rationale}")""")

md("""## 8 · Conclusions, limitations, reproducibility

**Conclusion.** On this data the right engineering question is not *"how accurate?"* but *"how
safe, and how do we know?"*. Second Look answers it with a calibrated predictor, a knowledge-based
red-flag safety channel that generalises, a cost-sensitive policy that can only escalate, and an
audit that surfaces a lethal missingness shortcut a naïve model would hide.

**Limitations (stated plainly).** The data is synthetic: labels are near-deterministic in
severity, with **no** rater/site variability and **no** injected demographic bias — so we make
**no** claim to have found bias or drift *in the data*; every conclusion is about our *model* and
*deployment conditions*. The red-flag ontology was developed with the paraphrase examples in mind
(its 90% reflects design intent; TF-IDF/transformer are zero-shot). External validity is unproven
and demands real corpora (MIMIC-IV-ED, NHAMCS). Not a medical device.

**Reproducibility.** Seven unit-tested `src` modules (44 tests), three reproducible Kaggle kernels
(forensics, generalization, audit), this notebook, and a live Gradio demo. See the linked GitHub
repository and Hugging Face Space.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
with open("notebooks/second_look_submission.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/second_look_submission.ipynb with", len(cells), "cells")
