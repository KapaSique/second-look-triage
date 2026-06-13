# Second Look — a triage safety-net that knows what it doesn't know

A clinically-grounded decision-support system for emergency-department triage, built for the
[**Triagegeist**](https://www.kaggle.com/competitions/triagegeist) hackathon
(Laitinen-Fredriksson Foundation).

**🔗 Links** · [Live demo (Hugging Face Space)](https://huggingface.co/spaces/KapaSique/second-look-triage) · [Kaggle notebook](https://www.kaggle.com/code/ssstelmah/second-look-triage-safety-net) · [Design spec](docs/superpowers/specs/2026-06-13-triagegeist-second-look-design.md)

> ⚠️ Research demonstration on **synthetic** data. **Not a medical device.** Second Look
> supports, never replaces, clinician judgement.

---

## The one idea

On the provided synthetic data, the free-text chief complaint predicts the ESI label with
**~99.9% accuracy** — so *accuracy is not a meaningful objective here*. The real, clinically
important problem is **safe behaviour under realistic conditions**: catching the occult
high-risk patient, not collapsing when data is missing, being calibrated, and being honest about
what synthetic data can and cannot prove. Second Look is built around that.

## Findings (all reproduced by the notebook & kernels)

| # | Finding | Evidence |
|---|---|---|
| 1 | Accuracy is a mirage — the label is a near phrase→ESI lookup | text-only **acc 0.9994** (5-fold); 4,949 complaints, 99.7% single-acuity, 99.8% of test seen in train |
| 2 | **Informative missingness is a deployment trap** | blanking the vitals of the *sick* drives fusion-model undertriage **0.84% → 78%**; de-biasing → 15% |
| 3 | Occult high-risk patients exist; vitals miss them | NEWS2-only accuracy ceiling **65%**; free text + red-flags catch them |
| 4 | **Clinical knowledge generalises; memorisation doesn't** | critical safe-recall — TF-IDF 30% · PubMedBERT 35% · **red-flag ontology 90%** (0% FP) |
| 5 | Calibrated & audited | holdout acc 0.992, undertriage 2.8%, ECE 0.106; fairness reported across sex/age/language/insurance |

<p align="center">
<img src="figures/02_informative_missingness.png" width="32%"> <img src="figures/08_missingness_stress.png" width="32%"> <img src="figures/04_generalization.png" width="32%">
</p>

## The system (7 unit-tested modules — `src/`)

- `data_prep` — load/merge + feature engineering with **explicit, honest** missingness indicators
- `model_core` — calibrated fusion classifier (vitals + demographics + TF-IDF), CV/ECE/undertriage
- `redflag` — vitals-independent clinical can't-miss **ontology** + matcher
- `policy` — cost-sensitive decision policy; red-flag & NEWS2 floors can only **escalate**
- `audit` — fairness, missingness stress-test, calibration
- `clinical` — NEWS2 calculator + derived vitals
- `explain` + `app/` — per-patient explanation + Gradio demo

## Reproduce

```bash
pip install -r requirements.txt        # or use the Kaggle notebook directly
pip install pytest && PYTHONPATH=. pytest -q     # 44 tests
```

Heavy compute runs as reproducible **Kaggle kernels** (`kaggle/`): `forensics`, `generalization`,
`audit` — each imports the same tested `src` modules (shipped as a Kaggle utility dataset) and
regenerates every number and figure. The public notebook (`notebooks/`) runs the whole pipeline
end-to-end. Competition data is **not** redistributed here (per the competition rules) — download
it from the [competition data page](https://www.kaggle.com/competitions/triagegeist/data).

## Honesty & limitations

The data is synthetic: labels are near-deterministic in severity, with **no** rater/site
variability and **no** injected demographic bias — so we make **no** claim to have found bias or
drift *in the data*. Every conclusion is about our *model* and *deployment conditions*. The
red-flag ontology was developed with the paraphrase probe in mind (its 90% reflects design
intent; TF-IDF/transformer are zero-shot). External validity is unproven and demands real corpora
(MIMIC-IV-ED, NHAMCS).

## License

Code: MIT. Synthetic competition data is **not** included (non-commercial research license, no
redistribution).
