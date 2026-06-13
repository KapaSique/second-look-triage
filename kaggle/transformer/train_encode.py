"""Second Look — generalization study (Kaggle GPU kernel).

Question that matters for deployment: the synthetic chief-complaint strings are a closed
vocabulary (4949 phrases; 99.8% of test phrases were seen in train), so a TF-IDF lexicon
"solves" the task by memorization. But a real triage nurse types FREE TEXT that paraphrases
the same presentation in unseen words. Does the signal survive paraphrasing?

We test a held-out set of hand-written clinical paraphrases (same meaning, different words,
NOT in train) and compare a TF-IDF nearest-neighbour (memoriser) against a biomedical
sentence-transformer nearest-neighbour (semantic). Metric that counts clinically:
  * safe-urgent recall: of critical paraphrases (expected ESI<=2), how many are assigned <=2
  * mean absolute acuity error
This is the honest argument for why a semantic model — not the 100%-accurate lexicon — is
what a real deployment needs.

Outputs -> /kaggle/working/outputs/transformer.json , figures/04_generalization.png
"""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
os.system("pip -q install sentence-transformers >/dev/null 2>&1")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUT = "/kaggle/working/outputs"; FIG = "/kaggle/working/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

def find(name):
    h = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not h: raise FileNotFoundError(name)
    return h[0]

train = pd.read_csv(find("train.csv"))[["patient_id", "triage_acuity"]]
cc = pd.read_csv(find("chief_complaints.csv"))[["patient_id", "chief_complaint_raw"]]
train = train.merge(cc, on="patient_id", how="left").dropna(subset=["chief_complaint_raw"])
# unique complaint -> majority acuity (99.7% are single-acuity anyway)
uni = train.groupby("chief_complaint_raw")["triage_acuity"].agg(lambda s: int(s.mode().iloc[0]))
corpus = uni.index.tolist(); corpus_ac = uni.values

# Hand-written paraphrases NOT present verbatim in train: (text, expected_acuity, critical?)
PARAPHRASES = [
    ("sudden explosive headache unlike anything before", 2, True),
    ("crushing chest discomfort, sweaty, pain shooting down the left arm", 2, True),
    ("the left side of his face is drooping and his speech is slurred", 1, True),
    ("burning up with fever, confused and barely responsive, looks septic", 2, True),
    ("tearing pain ripping through to the back", 1, True),
    ("throat is closing up after a bee sting, covered in hives", 1, True),
    ("can't catch her breath and the lips are turning blue", 1, True),
    ("severe lower belly pain, she is pregnant and just fainted", 2, True),
    ("throwing up bright red blood and feeling faint", 2, True),
    ("stiff neck, high fever and the light hurts his eyes", 2, True),
    ("collapsed and is not waking up", 1, True),
    ("sugar very high, breathing deep and fast, drowsy", 2, True),
    ("sudden weakness down one arm and leg this morning", 1, True),
    ("massive nosebleed that won't stop, soaking through towels", 2, True),
    ("baby is floppy and not feeding, very hot", 2, True),
    ("needs a repeat prescription for blood pressure tablets", 5, False),
    ("twisted an ankle at football, a bit swollen, can still walk", 4, False),
    ("small paper cut on the finger, barely bleeding", 5, False),
    ("mild sore throat for a couple of days, no fever", 4, False),
    ("wants advice about contraception options", 5, False),
    ("blocked nose and sneezing for a week", 5, False),
    ("routine wound dressing change", 4, False),
    ("mild lower back ache after gardening", 4, False),
]
para_text = [p[0] for p in PARAPHRASES]
para_exp = np.array([p[1] for p in PARAPHRASES])
para_crit = np.array([p[2] for p in PARAPHRASES])

def nn_predict(sim):  # sim: (n_para, n_corpus) -> acuity of best match
    return corpus_ac[sim.argmax(axis=1)]

def score(pred, tag):
    mae = float(np.abs(pred - para_exp).mean())
    exact = float((pred == para_exp).mean() * 100)
    crit = para_crit
    safe = float((pred[crit] <= 2).mean() * 100)          # critical assigned urgent?
    overall_safe = float(((pred <= 2) == (para_exp <= 2)).mean() * 100)
    print(f"[{tag}] exact={exact:.1f}%  MAE={mae:.2f}  critical-safe-recall={safe:.1f}%  urgent-agree={overall_safe:.1f}%")
    return dict(exact_pct=round(exact, 1), mae=round(mae, 2),
                critical_safe_recall_pct=round(safe, 1), urgent_agreement_pct=round(overall_safe, 1))

res = {"n_paraphrases": len(PARAPHRASES), "n_critical": int(para_crit.sum())}

# ---- TF-IDF memoriser ----
tf = TfidfVectorizer(ngram_range=(1, 2)).fit(corpus + para_text)
sim_tf = cosine_similarity(tf.transform(para_text), tf.transform(corpus))
res["tfidf"] = score(nn_predict(sim_tf), "TF-IDF")

# ---- biomedical sentence-transformer ----
emb_ok = True
try:
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res["device"] = dev
    name = "pritamdeka/S-PubMedBert-MS-MARCO"
    try:
        st = SentenceTransformer(name, device=dev)
    except Exception:
        name = "sentence-transformers/all-MiniLM-L6-v2"; st = SentenceTransformer(name, device=dev)
    res["embed_model"] = name
    E_corpus = st.encode(corpus, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    E_para = st.encode(para_text, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sim_emb = E_para @ E_corpus.T
    res["transformer"] = score(nn_predict(sim_emb), f"EMB:{name.split('/')[-1]}")
except Exception as e:
    emb_ok = False; res["transformer_error"] = str(e)[:300]
    print("EMB failed:", e)

# ---- figure ----
if emb_ok:
    cats = ["exact_pct", "critical_safe_recall_pct", "urgent_agreement_pct"]
    labels = ["Exact acuity", "Critical safe-recall", "Urgent agreement"]
    tfv = [res["tfidf"][c] for c in cats]; emv = [res["transformer"][c] for c in cats]
    x = np.arange(len(cats)); w = 0.38
    plt.figure(figsize=(7, 3.6))
    plt.bar(x - w/2, tfv, w, label="TF-IDF (memorises)", color="#e67e22")
    plt.bar(x + w/2, emv, w, label="Sentence-transformer (semantic)", color="#27ae60")
    plt.xticks(x, labels, rotation=10); plt.ylabel("%"); plt.ylim(0, 105)
    plt.title("Generalization to unseen paraphrases of red-flag presentations")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/04_generalization.png", dpi=130); plt.close()

with open(f"{OUT}/transformer.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE", json.dumps(res, indent=2)[:1500])
