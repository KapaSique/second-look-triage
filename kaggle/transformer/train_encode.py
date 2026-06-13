"""Second Look — generalization study (Kaggle GPU kernel).

Why this matters for deployment: the synthetic chief-complaint strings are a closed
vocabulary (4949 phrases; 99.8% of test phrases seen in train), so a TF-IDF lexicon
"solves" the task by memorisation. A real triage nurse types FREE TEXT that paraphrases
the same presentation in unseen words. Does the signal survive paraphrasing?

We hold out hand-written clinical paraphrases (same meaning, different words, NOT in train)
and compare a TF-IDF nearest-neighbour (memoriser) vs a biomedical transformer mean-pooled
embedding nearest-neighbour (semantic). Clinically meaningful metric:
  * critical safe-recall: of critical paraphrases (expected ESI<=2), how many get <=2.

IMPORTANT: we use the PRE-INSTALLED torch + transformers on the Kaggle GPU image and only
download model weights. We do NOT `pip install sentence-transformers`, because that pulls a
torch wheel incompatible with the Kaggle GPU (cudaErrorNoKernelImageForDevice).

Outputs -> /kaggle/working/outputs/transformer.json , figures/04_generalization.png
"""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
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
uni = train.groupby("chief_complaint_raw")["triage_acuity"].agg(lambda s: int(s.mode().iloc[0]))
corpus = uni.index.tolist(); corpus_ac = uni.values
print(f"corpus unique complaints: {len(corpus)}")

# Hand-written paraphrases NOT present verbatim in train: (text, expected_acuity, critical)
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
    ("a massive nosebleed that will not stop, soaking through towels", 2, True),
    ("the baby is floppy and not feeding, very hot to touch", 2, True),
    ("he took a whole bottle of pills on purpose", 2, True),
    ("car crash at high speed, trapped, bleeding heavily", 1, True),
    ("severe allergic reaction, face and tongue swelling fast", 1, True),
    ("worst abdominal pain of my life, belly rigid", 2, True),
    ("not breathing properly, wheezing badly, exhausted", 2, True),
    ("needs a repeat prescription for blood pressure tablets", 5, False),
    ("twisted an ankle at football, a bit swollen, can still walk", 4, False),
    ("small paper cut on the finger, barely bleeding", 5, False),
    ("mild sore throat for a couple of days, no fever", 4, False),
    ("wants advice about contraception options", 5, False),
    ("blocked nose and sneezing for a week", 5, False),
    ("routine wound dressing change", 4, False),
    ("mild lower back ache after gardening", 4, False),
    ("here for a flu shot", 5, False),
    ("itchy mild rash on the arm for a few days", 4, False),
    ("wants results of a routine blood test", 5, False),
    ("mild headache, took paracetamol, feeling better", 4, False),
    ("stubbed toe, sore but walking fine", 5, False),
    ("dry cough for two weeks, otherwise well", 4, False),
    ("asking about travel vaccinations", 5, False),
    ("mild earache, no fever", 4, False),
]
para_text = [p[0] for p in PARAPHRASES]
para_exp = np.array([p[1] for p in PARAPHRASES])
para_crit = np.array([p[2] for p in PARAPHRASES])

def nn_predict(sim):
    return corpus_ac[sim.argmax(axis=1)]

def score(pred, tag):
    mae = float(np.abs(pred - para_exp).mean())
    exact = float((pred == para_exp).mean() * 100)
    safe = float((pred[para_crit] <= 2).mean() * 100)
    agree = float(((pred <= 2) == (para_exp <= 2)).mean() * 100)
    print(f"[{tag}] exact={exact:.1f}%  MAE={mae:.2f}  critical-safe-recall={safe:.1f}%  urgent-agree={agree:.1f}%")
    return dict(exact_pct=round(exact, 1), mae=round(mae, 2),
                critical_safe_recall_pct=round(safe, 1), urgent_agreement_pct=round(agree, 1))

res = {"n_paraphrases": len(PARAPHRASES), "n_critical": int(para_crit.sum())}

# ---- TF-IDF memoriser ----
tf = TfidfVectorizer(ngram_range=(1, 2)).fit(corpus + para_text)
sim_tf = cosine_similarity(tf.transform(para_text), tf.transform(corpus))
res["tfidf"] = score(nn_predict(sim_tf), "TF-IDF")

# ---- biomedical transformer (preinstalled torch + transformers; mean pooling) ----
emb_ok = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    name = "pritamdeka/S-PubMedBert-MS-MARCO"
    try:
        tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
    except Exception:
        name = "sentence-transformers/all-MiniLM-L6-v2"
        tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
    res["embed_model"] = name
    # Probe the GPU with one tiny forward; some Kaggle GPUs (arch) are unsupported by the
    # preinstalled torch (cudaErrorNoKernelImageForDevice). Fall back to CPU -> guaranteed result.
    dev = "cpu"
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
            _t = tok(["test"], return_tensors="pt").to("cuda")
            with torch.no_grad():
                model(**_t)
            dev = "cuda"
        except Exception as ce:
            model = model.to("cpu"); dev = "cpu"; res["cuda_fallback"] = str(ce)[:120]
    res["device"] = dev

    def encode(texts, bs=128, max_len=64):
        outs = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            with torch.no_grad():
                hid = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hid * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=1)
            outs.append(emb.cpu().numpy())
        return np.vstack(outs)

    E_corpus = encode(corpus); E_para = encode(para_text)
    res["transformer"] = score(nn_predict(E_para @ E_corpus.T), f"EMB:{name.split('/')[-1]}")
except Exception as e:
    emb_ok = False; res["transformer_error"] = str(e)[:300]; print("EMB failed:", e)

if emb_ok:
    cats = ["exact_pct", "critical_safe_recall_pct", "urgent_agreement_pct"]
    labels = ["Exact acuity", "Critical safe-recall", "Urgent agreement"]
    tfv = [res["tfidf"][c] for c in cats]; emv = [res["transformer"][c] for c in cats]
    x = np.arange(len(cats)); w = 0.38
    plt.figure(figsize=(7, 3.6))
    plt.bar(x - w/2, tfv, w, label="TF-IDF (memorises)", color="#e67e22")
    plt.bar(x + w/2, emv, w, label="Biomedical transformer (semantic)", color="#27ae60")
    plt.xticks(x, labels, rotation=8); plt.ylabel("%"); plt.ylim(0, 105)
    plt.title("Generalization to unseen paraphrases of red-flag presentations")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/04_generalization.png", dpi=130); plt.close()

with open(f"{OUT}/transformer.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE", json.dumps(res, indent=2)[:1500])
