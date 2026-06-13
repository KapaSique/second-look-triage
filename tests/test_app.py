"""Smoke test for the demo's core logic (no Gradio server, no model.pkl needed)."""
import importlib.util
import os

_APP = os.path.join(os.path.dirname(__file__), "..", "app", "app.py")
spec = importlib.util.spec_from_file_location("second_look_app", _APP)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_assess_returns_markdown_and_probs():
    md, probs = app.assess("ankle sprain, mild swelling", 30, "M", "walk-in",
                           124, 78, 72, 14, 36.7, 99, 15, 3, 0, False)
    assert isinstance(md, str) and "ESI" in md
    assert isinstance(probs, dict) and len(probs) == 5
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_occult_acs_escalates():
    # textbook ACS with deceptively normal vitals -> red flag must escalate
    md, probs = app.assess("chest pain with diaphoresis and arm radiation", 60, "M",
                           "ambulance", 128, 84, 96, 18, 36.9, 97, 15, 6, 2, False)
    assert "ESCALATE" in md
    assert "ACS" in md


def test_missingness_warning_shown():
    md, _ = app.assess("feeling unwell", 70, "F", "walk-in",
                       120, 80, 80, 16, 37.0, 98, 15, 4, 3, True)
    assert "Data-shift warning" in md
