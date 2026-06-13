# Second Look: a triage safety-net that knows what it doesn't know

**Subtitle:** When 100% accuracy is the wrong target — building (and honestly auditing) decision support for emergency triage.

---

## Clinical problem statement

Emergency-department triage assigns each arriving patient an acuity level (the Emergency Severity Index, ESI 1–5) that governs how long they safely wait. The dangerous error is **undertriage** — labelling a sick patient as low-acuity. Undertriage delays care for time-critical conditions (STEMI, sepsis, stroke, subarachnoid haemorrhage) and is a documented, asymmetric patient-safety hazard: missing one decompensating patient outweighs over-triaging many well ones. The Triagegeist brief names two concrete failure modes — **inter-rater variability** and **systematic undertriage of certain populations** — as the problems worth solving.

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

**Finding 2 — Informative missingness is a deployment trap.** Vitals are missing **only for low-acuity patients**: systolic-BP missingness is 0% for ESI 1–3 but ~12% for ESI 4–5 (respiratory rate 0→9%; temperature only for ESI 5). "No BP recorded" is thus a near-perfect proxy for "not sick" — *in this dataset*. A model that uses missingness indicators (standard practice) weaponises this shortcut. Our stress test blanks the vitals of genuinely urgent patients (the real-ED scenario of a crashing patient whose vitals weren't captured): undertriage of the fusion model jumps from **0.84% to 78.0%**. Removing the missingness indicators de-biases the model — stressed undertriage stays at **[WITHOUT-INDICATORS]%**. The "excellent" held-out model is the dangerous one; the honest audit is what catches it. Our shipped demo model is the de-biased version.

**Finding 3 — Occult high-risk patients exist, and free text is the only channel that sees them.** 0.21% of urgent patients present with deceptively normal vitals — but textbook can't-miss complaints ("chest pain with diaphoresis and arm radiation", "sepsis with altered mental status", "ovarian torsion with rigors"). NEWS2 alone caps at **65%** accuracy and would miss them; the text and the red-flag ontology catch them.

**Finding 4 — Generalization: knowledge beats memorisation on unseen language.** We hand-wrote 36 paraphrases (20 critical) of red-flag presentations in lay language *not* present in training, and measured *critical safe-recall* (fraction of critical cases assigned ESI ≤ 2):

| Approach | Critical safe-recall |
|---|---|
| TF-IDF nearest-neighbour (memoriser) | 30% |
| Biomedical transformer (PubMedBERT) nearest-neighbour | 35% |
| **Curated clinical red-flag ontology** | **90%** (0% false positives on benign) |

Two honest lessons: (a) a system that scores 100% on the synthetic vocabulary catches only ~⅓ of unseen lay descriptions of emergencies; (b) encoding clinical knowledge generalises far better than memorising or embedding the synthetic corpus — but even 90% on a small, hand-built probe is not a deployment guarantee.

**Finding 5 — Calibration and fairness.** On a 20% holdout the calibrated model reaches acc 0.992, undertriage 2.8%, ECE 0.106. Undertriage is similar across sex (F 2.6%, M 2.9%) but elevated for the small "Other" group (5.1%, n=396) — a finding we surface rather than bury. Because the synthetic generator injected no group bias, this reflects sampling/representation, not learned discrimination — exactly the kind of caveat the audit is for.

## Limitations

We are deliberately forthright:

- **The data is synthetic.** Labels are near-deterministic in severity; there is **no** inter-rater variability (nurse/site mean-acuity SD ≈ 0.01–0.02) and **no** injected undertriage or demographic bias. We therefore make **no** claim to have "discovered" bias or rater drift in the data — doing so would be dishonest. Every conclusion is about our *model* and *deployment conditions*, not the data's social validity.
- **The red-flag ontology was developed with the paraphrase examples in mind**, so its 90% reflects design intent; the TF-IDF and transformer numbers are zero-shot. The probe is illustrative, not a benchmark.
- **External validity is unproven.** Safe free-text triage demands validation on real, diverse clinical corpora (MIMIC-IV-ED, NHAMCS). We treat this as the essential next step, not a footnote.
- **Not a medical device.** Second Look is a decision-support second opinion and a research demonstration.

## Reproducibility

A public GitHub repository contains the seven `src` modules, **44 unit tests**, the Kaggle kernels (forensics, generalization, audit) that regenerate every number and figure, the public end-to-end Kaggle notebook, and the Gradio app. Heavy compute is packaged as reproducible Kaggle kernels importing the same tested code; the design spec and implementation plan are included. Links: [Notebook] · [GitHub] · [Live demo].

---

*Second Look's contribution is not a number on a leaderboard that does not exist. It is a clinically-framed, honestly-audited demonstration that on this data the right engineering question is not "how accurate?" but "how safe, and how do we know?" — and a working tool that answers it.*
