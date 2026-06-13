# Second Look: a triage safety-net that knows what it doesn't know

**Subtitle:** When 100% accuracy is the wrong target — building (and honestly auditing) decision support for emergency triage.

---

## Clinical problem statement

Emergency-department triage assigns each arriving patient an acuity level (the Emergency Severity Index, ESI 1–5 [1]) that governs how long they safely wait. The dangerous error is **undertriage** — labelling a sick patient as low-acuity. Undertriage delays care for time-critical conditions (STEMI, sepsis, stroke, subarachnoid haemorrhage) and is a documented, asymmetric patient-safety hazard [2,3]: missing one decompensating patient outweighs over-triaging many well ones. The Triagegeist brief names two concrete failure modes — **inter-rater variability** [2] and **systematic undertriage of certain populations** [3] — as the problems worth solving.

We therefore did **not** set out to build "the most accurate ESI classifier." We set out to build a *safety net*: a system that flags the occult high-risk patient, behaves sanely when data is missing, is honest about its uncertainty, and — crucially — is honest about what the available data can and cannot prove. We call it **Second Look**: a second pair of eyes on the triage decision, never a replacement for the clinician.

## Methodology

**1. Data forensics before modelling.** We first interrogated the provided synthetic dataset (80k train / 20k test; vitals, demographics, 25 comorbidity flags, free-text chief complaints, NEWS2/shock-index, ESI label). Three findings reshaped the whole project (Results). The headline: the free-text chief complaint **alone predicts the ESI label with ~99.9% accuracy** under 5-fold cross-validation. Accuracy is therefore a solved, non-discriminating objective on this data — so we pivoted to calibration, safety behaviour, robustness and fairness.

**2. The Second Look system** has five composable, unit-tested modules (44 passing tests):

- *Calibrated fusion model* (`model_core`): logistic regression over a `ColumnTransformer` (median-imputed vitals + one-hot demographics + TF-IDF of the complaint), wrapped in probability calibration. Chosen over gradient boosting deliberately — natively probabilistic, calibratable, and light enough to run inside the public demo.
- *Clinical red-flag ontology* (`redflag`): a curated, vitals-independent lexicon of can't-miss presentations (SAH, ACS, stroke, sepsis, AAA/dissection, anaphylaxis, airway, torsion/ectopic, meningitis, GI-bleed, major trauma, …), each with an ESI floor and clinical note. It encodes *knowledge*, not corpus statistics.
- *Cost-sensitive decision policy* (`policy`): combines the calibrated probabilities, the red-flag ESI floor, and the NEWS2 early-warning score. Red flags and high NEWS2 can only ever **escalate**, never silently lower urgency; the probability-mass operating point is tunable along an undertriage/overtriage trade-off.
- *Honest audit* (`audit`): fairness across sex/age/language/insurance, a missingness stress-test, and calibration reliability.
- *Explanation + demo* (`explain`, Gradio): per-patient deranged-vital callouts, fired red flags, NEWS2, rationale, and a data-shift warning — deployed as a public Hugging Face Space.

**3. Compute.** All computation ran on Kaggle (forensics, generalization study, full audit); the modelling here is light by nature (the signal is text-deterministic), so GPU was not the bottleneck — an honest observation we prefer to a gratuitous training run. Heavy assets (model artefact, figures) are produced by reproducible Kaggle kernels that import the same tested `src` modules.

## Results

**Finding 1 — Accuracy is a mirage; the label is a near-lookup.** Text-only: **acc 0.9994, macro-F1 0.998** (5-fold). Vitals-only: acc 0.803. The complaint vocabulary is closed — 4,949 unique complaints, **99.7%** mapping to a single acuity, and **99.8%** of test complaints already appear in train. A TF-IDF model "wins" by memorising. We report this openly: chasing leaderboard-style accuracy here would be self-deception.

**Finding 2 — Informative missingness is a deployment trap.** Vitals are missing **only for low-acuity patients**: systolic-BP missingness is 0% for ESI 1–3 but ~12% for ESI 4–5 (respiratory rate 0→9%; temperature only for ESI 5). "No BP recorded" is thus a near-perfect proxy for "not sick" — *in this dataset*. A model that uses missingness indicators (standard practice) weaponises this shortcut — that the *presence* of a measurement is itself predictive of outcome is a documented EHR hazard [5]. Our stress test blanks the vitals of genuinely urgent patients (the real-ED scenario of a crashing patient whose vitals weren't captured): undertriage of the fusion model jumps from **0.84% to 78.0%**. Removing the missingness indicators cuts this sharply — stressed undertriage falls to **15.3%** (an ~80% reduction) — though residual vulnerability remains because the model still partly leans on vitals, which is itself an honest signal that this needs real-data validation, not a one-line fix. The "excellent" held-out model is the dangerous one; only the audit catches it. Our shipped demo model is the de-biased version.

**Finding 3 — Occult high-risk patients exist, and free text is the only channel that sees them.** 0.21% of urgent patients present with deceptively normal vitals — but textbook can't-miss complaints ("chest pain with diaphoresis and arm radiation", "sepsis with altered mental status", "ovarian torsion with rigors"). NEWS2 [4] alone caps at **65%** accuracy and would miss them; the text and the red-flag ontology catch them.

**Finding 4 — Generalization: knowledge beats memorisation on unseen language.** We hand-wrote 36 paraphrases (20 critical) of red-flag presentations in lay language *not* present in training, and measured *critical safe-recall* (fraction of critical cases assigned ESI ≤ 2):

| Approach | Critical safe-recall |
|---|---|
| TF-IDF nearest-neighbour (memoriser) | 30% |
| Biomedical transformer (PubMedBERT) nearest-neighbour | 35% |
| **Curated clinical red-flag ontology** | **90%** (0% false positives on benign) |

Two honest lessons: (a) a system that scores 100% on the synthetic vocabulary catches only ~⅓ of unseen lay descriptions of emergencies; (b) encoding clinical knowledge generalises far better than memorising or embedding the synthetic corpus — but even 90% on a small, hand-built probe is not a deployment guarantee.

**Finding 5 — Calibration and fairness.** On a 20% holdout the calibrated model reaches acc 0.992, undertriage 2.8%, ECE 0.106. Undertriage is similar across sex (F 2.6%, M 2.9%) but elevated for the small "Other" group (5.1%, n=396) — a finding we surface rather than bury. Because the synthetic generator injected no group bias, this reflects sampling/representation, not learned discrimination — exactly the kind of caveat the audit is for.

**Finding 6 — The patterns hold on real ED data (NHAMCS).** We tested our two clinical claims on **43,921 real triaged visits** from the CDC's National Hospital Ambulatory Medical Care Survey (NHAMCS 2019–2022 [6]). *Occult high-risk is real:* **26.7%** of high-acuity (immediacy 1–2) patients present with all-normal recorded vitals — vitals alone would miss one in four. *Informative missingness is real, with a sharper twist:* vitals-recording is **U-shaped** in acuity — missing most for both the *least* urgent (skipped) and the *most* urgent (immediacy 1: 27% of visits miss ≥1 vital, deferred during resuscitation), vs ~11% in the middle. The synthetic generator encoded only the low-acuity half; a model that learns "missing ⇒ less sick" would therefore **undertriage the crashing patient** in deployment — turning our simulated stress-test (Finding 2) into a documented, real-world hazard.

## Limitations

We are deliberately forthright:

- **The data is synthetic.** Labels are near-deterministic in severity; there is **no** inter-rater variability (nurse/site mean-acuity SD ≈ 0.01–0.02) and **no** injected undertriage or demographic bias. We therefore make **no** claim to have "discovered" bias or rater drift in the data — doing so would be dishonest. Every conclusion is about our *model* and *deployment conditions*, not the data's social validity.
- **The red-flag ontology was developed with the paraphrase examples in mind**, so its 90% reflects design intent; the TF-IDF and transformer numbers are zero-shot. The probe is illustrative, not a benchmark.
- **External validity is only partially tested.** We corroborated the occult-risk and informative-missingness patterns on real NHAMCS data (Finding 6), but full validation of the *safe-triage system* demands richer clinical corpora with linked outcomes (MIMIC-IV-ED). We treat this as the essential next step, not a footnote.
- **Not a medical device.** Second Look is a decision-support second opinion and a research demonstration.

## Reproducibility

A public GitHub repository contains the seven `src` modules, **44 unit tests**, the Kaggle kernels (forensics, generalization, audit) that regenerate every number and figure, the public end-to-end Kaggle notebook, and the Gradio app. Heavy compute is packaged as reproducible Kaggle kernels importing the same tested code; the design spec and implementation plan are included. **Links:** [Kaggle notebook](https://www.kaggle.com/code/ssstelmah/second-look-triage-safety-net) · [GitHub](https://github.com/KapaSique/second-look-triage) · [Live demo](https://huggingface.co/spaces/KapaSique/second-look-triage).

## References

1. Gilboy N, Tanabe P, Travers D, Rosenau AM. *Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4.* Agency for Healthcare Research and Quality, Rockville, MD.
2. Hinson JS, Martinez DA, Cabral S, et al. Triage Performance in Emergency Medicine: A Systematic Review. *Ann Emerg Med.* 2019;74(1):140–152.
3. Platts-Mills TF, Travers D, Biese K, et al. At risk of undertriage? Testing the performance and accuracy of the Emergency Severity Index in older emergency department patients. *Ann Emerg Med.* 2012;60(3):317–325.
4. Royal College of Physicians. *National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS.* London: RCP; 2017.
5. Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. *BMJ.* 2018;361:k1479.
6. CDC / National Center for Health Statistics. *National Hospital Ambulatory Medical Care Survey (NHAMCS), 2019–2022 Emergency Department public-use files.*

---

*Second Look's contribution is not a number on a leaderboard that does not exist. It is a clinically-framed, honestly-audited demonstration that on this data the right engineering question is not "how accurate?" but "how safe, and how do we know?" — and a working tool that answers it.*
