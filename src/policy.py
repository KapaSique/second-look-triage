"""Cost-sensitive triage decision policy — the safety-net "brain".

Combines three independent channels and resolves them toward patient safety:

  1. the calibrated model's probability over ESI 1-5,
  2. red-flag esi_floor from the free-text safety channel (`redflag.py`),
  3. an early-warning floor from NEWS2 (vitals-based deterioration score).

Triage error is asymmetric: *undertriaging* a sick patient (sending them away) can be
fatal, while *overtriaging* costs time and resources. Rather than an opaque cost matrix
(which, with a strong undertriage penalty, collapses to "label everyone ESI 1"), we use
interpretable probability-mass thresholds — an expected-cost operating point that the audit
sweeps into an undertriage/overtriage trade-off curve. A red flag or a high NEWS2 can only
ever raise urgency, never lower it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

ACUITY = [1, 2, 3, 4, 5]


@dataclass
class Decision:
    acuity: int            # final recommended ESI (1=most urgent)
    base_acuity: int       # model's most-likely ESI before safety floors
    escalate: bool         # True if a safety channel raised urgency / critical flag present
    rationale: str         # human-readable explanation
    floors: List[str] = field(default_factory=list)
    p_urgent: float = 0.0  # P(ESI<=2)


class TriagePolicy:
    def __init__(self, tau_urgent: float = 0.50, tau_critical: float = 0.35,
                 news2_urgent: int = 7, news2_elevated: int = 5):
        # thresholds are tunable operating points (audited via trade-off curve)
        self.tau_urgent = tau_urgent
        self.tau_critical = tau_critical
        self.news2_urgent = news2_urgent
        self.news2_elevated = news2_elevated

    def decide(self, probs, redflags: Optional[list] = None,
               news2: Optional[float] = None) -> Decision:
        p = np.asarray(probs, dtype=float)
        p = p / p.sum() if p.sum() else p
        base = ACUITY[int(p.argmax())]
        acuity = base
        floors: List[str] = []
        reasons: List[str] = []

        p_crit = float(p[0])              # P(ESI 1)
        p_urgent = float(p[0] + p[1])     # P(ESI 1 or 2)

        # 1. probability-mass safety floors (cost-sensitive operating point)
        if p_urgent >= self.tau_urgent and acuity > 2:
            acuity = 2; floors.append("p_urgent")
            reasons.append(f"P(ESI≤2)={p_urgent:.0%} ≥ {self.tau_urgent:.0%} → floor ESI 2")
        if p_crit >= self.tau_critical and acuity > 1:
            acuity = 1; floors.append("p_critical")
            reasons.append(f"P(ESI 1)={p_crit:.0%} ≥ {self.tau_critical:.0%} → floor ESI 1")

        # 2. red-flag floor (vitals-independent text channel)
        critical_flag = False
        if redflags:
            rf_floor = min(f.esi_floor for f in redflags)
            critical_flag = any(getattr(f, "severity", "") == "critical" for f in redflags)
            if rf_floor < acuity:
                cats = ", ".join(sorted({f.category for f in redflags if f.esi_floor == rf_floor}))
                acuity = rf_floor; floors.append("red-flag")
                reasons.append(f"red-flag [{cats}] → floor ESI {rf_floor}")

        # 3. NEWS2 early-warning floor
        if news2 is not None:
            if news2 >= self.news2_urgent and acuity > 2:
                acuity = 2; floors.append("news2")
                reasons.append(f"NEWS2={news2:g} ≥ {self.news2_urgent} → floor ESI 2")
            elif news2 >= self.news2_elevated and acuity > 3:
                acuity = 3; floors.append("news2")
                reasons.append(f"NEWS2={news2:g} ≥ {self.news2_elevated} → floor ESI 3")

        escalate = (acuity < base) or critical_flag
        if not reasons:
            reasons.append(f"model most-likely ESI {base}; no safety floor triggered")
        rationale = "; ".join(reasons)
        return Decision(acuity=int(acuity), base_acuity=int(base), escalate=bool(escalate),
                        rationale=rationale, floors=floors, p_urgent=round(p_urgent, 4))
