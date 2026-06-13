"""Clinical red-flag ontology + matcher — the vitals-independent safety channel.

Rationale: in this dataset the chief-complaint text carries the true acuity signal,
and some time-critical presentations ("thunderclap headache", "chest pain with
diaphoresis") can present with deceptively normal vitals (the *occult high-risk*
patient). A curated red-flag layer screens the free text for can't-miss diagnoses
independently of vitals/early-warning scores, and enforces an ESI floor so the
decision policy can never silently undertriage a flagged presentation.

The ontology is intentionally embedded in Python (no external file / yaml dependency)
so it runs unchanged inside a Kaggle kernel and the Gradio demo. Each entry maps to a
real emergency-medicine can't-miss category with an ESI floor and a one-line note.

`severity`:  'critical' (immediate, ESI 1-2) | 'high' (urgent screen, ESI<=3).
A lexicon matcher gives a transparent, auditable baseline; Phase 1 adds a transformer
to generalize to paraphrases the lexicon misses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class RedFlag:
    category: str
    severity: str       # 'critical' | 'high'
    matched: str        # the term/combo that fired
    esi_floor: int      # most-urgent ESI this presentation should be capped at (1=most urgent)
    note: str           # clinical rationale (shown in the demo)


# any_of: any single phrase fires.  combos: every term in a sublist must co-occur.
ONTOLOGY: List[Dict[str, Any]] = [
    {"category": "SAH", "severity": "critical", "esi_floor": 1,
     "note": "Thunderclap/sudden severe headache — rule out subarachnoid haemorrhage.",
     "any_of": ["thunderclap headache", "worst headache of my life", "worst headache of life",
                "sudden severe headache", "subarachnoid"]},
    {"category": "ACS", "severity": "critical", "esi_floor": 2,
     "note": "Features of acute coronary syndrome — immediate ECG and troponin.",
     "any_of": ["nstemi", "stemi", "acute coronary", "myocardial infarction", "unstable angina"],
     "combos": [["chest pain", "diaphoresis"], ["chest pain", "arm radiation"],
                ["chest pain", "radiation to arm"], ["chest pain", "radiating"],
                ["chest pain", "jaw"], ["chest pain", "sweating"], ["chest pain", "short of breath"]]},
    {"category": "ACS", "severity": "high", "esi_floor": 3,
     "note": "Undifferentiated chest pain — screen for cardiac cause.",
     "any_of": ["chest pain", "chest pressure", "chest tightness"]},
    {"category": "STROKE", "severity": "critical", "esi_floor": 1,
     "note": "Possible acute stroke — activate stroke pathway, time-critical.",
     "any_of": ["facial droop", "slurred speech", "hemiparesis", "aphasia", "stroke",
                "sudden vision loss", "one-sided weakness", "weakness one side", "fast positive"],
     "combos": [["sudden", "weakness"], ["sudden", "numbness"]]},
    {"category": "SEPSIS", "severity": "critical", "esi_floor": 2,
     "note": "Possible sepsis — lactate, cultures, early antibiotics.",
     "any_of": ["sepsis", "septic shock", "septic"],
     "combos": [["fever", "altered mental status"], ["fever", "hypotension"],
                ["fever", "rigors"], ["infection", "confusion"]]},
    {"category": "AAA", "severity": "critical", "esi_floor": 1,
     "note": "Possible ruptured abdominal aortic aneurysm.",
     "any_of": ["abdominal aortic aneurysm", "aaa rupture", "pulsatile mass"],
     "combos": [["tearing", "back pain"], ["tearing", "abdominal"]]},
    {"category": "AORTIC_DISSECTION", "severity": "critical", "esi_floor": 1,
     "note": "Possible aortic dissection — tearing pain radiating to back.",
     "any_of": ["aortic dissection"], "combos": [["tearing", "chest pain"], ["ripping", "chest"]]},
    {"category": "TORSION_ECTOPIC", "severity": "critical", "esi_floor": 2,
     "note": "Time-critical surgical emergency (torsion / ectopic).",
     "any_of": ["ovarian torsion", "testicular torsion", "ectopic pregnancy", "ectopic"],
     "combos": [["pelvic pain", "pregnant"]]},
    {"category": "ANAPHYLAXIS", "severity": "critical", "esi_floor": 1,
     "note": "Possible anaphylaxis — airway risk, IM adrenaline.",
     "any_of": ["anaphylaxis", "throat swelling", "angioedema", "tongue swelling"],
     "combos": [["rash", "difficulty breathing"], ["hives", "throat"]]},
    {"category": "AIRWAY", "severity": "critical", "esi_floor": 1,
     "note": "Airway / breathing compromise.",
     "any_of": ["stridor", "respiratory distress", "cyanosis", "unable to speak",
                "gasping", "choking", "apnoea", "apnea"]},
    {"category": "MENINGITIS", "severity": "critical", "esi_floor": 2,
     "note": "Possible meningitis — fever, neck stiffness, photophobia.",
     "any_of": ["meningitis"],
     "combos": [["neck stiffness", "fever"], ["fever", "photophobia"], ["petechial", "fever"]]},
    {"category": "GI_BLEED", "severity": "high", "esi_floor": 2,
     "note": "Significant GI bleeding.",
     "any_of": ["haematemesis", "hematemesis", "melena", "melaena", "coffee ground",
                "gi bleed", "gastrointestinal bleed"]},
    {"category": "DKA", "severity": "high", "esi_floor": 2,
     "note": "Possible diabetic ketoacidosis / metabolic emergency.",
     "any_of": ["dka", "diabetic ketoacidosis", "kussmaul"]},
    {"category": "OB_EMERGENCY", "severity": "critical", "esi_floor": 1,
     "note": "Obstetric emergency.",
     "any_of": ["postpartum haemorrhage", "postpartum hemorrhage", "eclampsia",
                "placental abruption", "cord prolapse"]},
    {"category": "MAJOR_TRAUMA", "severity": "critical", "esi_floor": 1,
     "note": "High-energy / major trauma mechanism.",
     "any_of": ["high-speed mva", "multiple injuries", "vascular compromise", "amputation",
                "penetrating", "gunshot", "stab wound"]},
    {"category": "SUICIDE_OD", "severity": "high", "esi_floor": 2,
     "note": "Self-harm / overdose — safety and medical risk.",
     "any_of": ["suicidal", "suicide attempt", "overdose", "self-harm", "intentional ingestion"]},
    {"category": "HAEMOPTYSIS", "severity": "high", "esi_floor": 2,
     "note": "Significant haemoptysis.",
     "any_of": ["haemoptysis significant", "massive haemoptysis", "coughing up blood"]},
]


def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).lower().replace("，", ",").strip()


class RedFlagMatcher:
    """Transparent lexicon matcher over the embedded clinical ontology."""

    def __init__(self, ontology: Optional[List[Dict[str, Any]]] = None):
        self.ontology = ontology if ontology is not None else ONTOLOGY

    def flags(self, text: str) -> List[RedFlag]:
        t = _norm(text)
        if not t:
            return []
        hits: List[RedFlag] = []
        for e in self.ontology:
            matched = None
            for term in e.get("any_of", []):
                if term in t:
                    matched = term
                    break
            if matched is None:
                for combo in e.get("combos", []):
                    if all(term in t for term in combo):
                        matched = " + ".join(combo)
                        break
            if matched is not None:
                hits.append(RedFlag(e["category"], e["severity"], matched, e["esi_floor"], e["note"]))
        return self._dedup_sort(hits)

    @staticmethod
    def _dedup_sort(hits: List[RedFlag]) -> List[RedFlag]:
        # keep the most severe (lowest esi_floor) flag per category
        best: Dict[str, RedFlag] = {}
        for h in hits:
            cur = best.get(h.category)
            if cur is None or h.esi_floor < cur.esi_floor:
                best[h.category] = h
        out = list(best.values())
        out.sort(key=lambda r: (0 if r.severity == "critical" else 1, r.esi_floor))
        return out

    def esi_floor(self, text: str) -> Optional[int]:
        """Most urgent ESI floor implied by any red flag, else None."""
        fs = self.flags(text)
        return min((f.esi_floor for f in fs), default=None)
