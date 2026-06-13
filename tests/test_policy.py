import numpy as np
from src.policy import TriagePolicy, Decision
from src.redflag import RedFlag

pol = TriagePolicy()


def test_low_acuity_no_escalation():
    d = pol.decide([0.0, 0.0, 0.02, 0.08, 0.90])
    assert d.acuity == 5 and d.escalate is False


def test_uncertain_mass_on_urgent_lowers_acuity_below_argmax():
    # argmax is ESI 3, but a third of the mass is on ESI 1/2 -> safety floor pulls it more urgent
    d = pol.decide([0.30, 0.25, 0.45, 0.0, 0.0])
    assert d.base_acuity == 3
    assert d.acuity <= 2 and d.acuity < d.base_acuity


def test_critical_redflag_enforces_floor():
    flags = [RedFlag("STROKE", "critical", "facial droop", 1, "possible acute stroke")]
    d = pol.decide([0.0, 0.0, 0.0, 0.2, 0.8], redflags=flags)
    assert d.acuity <= 1 and d.escalate is True
    assert "red-flag" in d.rationale.lower() or "stroke" in d.rationale.lower()


def test_news2_floor():
    d = pol.decide([0.0, 0.0, 0.0, 0.7, 0.3], news2=8)
    assert d.acuity <= 2 and d.escalate is True


def test_returns_decision_with_rationale():
    d = pol.decide([0.1, 0.2, 0.7, 0.0, 0.0])
    assert isinstance(d, Decision) and isinstance(d.rationale, str) and d.rationale
