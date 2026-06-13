from src.redflag import RedFlagMatcher, RedFlag

m = RedFlagMatcher()


def test_thunderclap_flags_sah():
    f = m.flags("thunderclap headache, worsening with movement")
    assert any(x.category == "SAH" for x in f)
    assert f[0].severity == "critical"


def test_acs_radiation_combo():
    f = m.flags("chest pain with diaphoresis and arm radiation, since 1h")
    assert any(x.category == "ACS" for x in f)


def test_acs_diagnosis_term():
    assert any(x.category == "ACS" for x in m.flags("acute NSTEMI with fever"))


def test_sepsis_ams():
    assert any(x.category == "SEPSIS" for x in m.flags("sepsis with altered mental status, constant"))


def test_torsion():
    assert any(x.category in ("TORSION_ECTOPIC",) for x in m.flags("ovarian torsion with rigors"))


def test_benign_no_flag():
    assert m.flags("contraception advice, intermittent") == []
    assert m.flags("general health question") == []


def test_critical_sorted_first_and_has_esi_floor():
    f = m.flags("high-speed MVA multiple injuries with vascular compromise")
    assert f, "expected at least one flag"
    assert f[0].severity == "critical"
    assert 1 <= f[0].esi_floor <= 2


def test_dedup_one_flag_per_category():
    # chest pain + diaphoresis triggers both the ACS screen and the ACS combo;
    # the matcher must collapse to a single ACS flag (most severe kept).
    cats = [x.category for x in m.flags("chest pain with diaphoresis")]
    assert cats.count("ACS") == 1


def test_redflag_is_frozen_dataclass():
    rf = RedFlag("ACS", "critical", "nstemi", 2, "note")
    assert rf.category == "ACS" and rf.esi_floor == 2


def test_lay_language_variants():
    assert any(x.category == "GI_BLEED" for x in m.flags("throwing up bright red blood and feeling faint"))
    assert any(x.category == "AIRWAY" for x in m.flags("can't catch her breath and the lips are turning blue"))
    assert any(x.category == "UNRESPONSIVE" for x in m.flags("collapsed and is not waking up"))
    assert any(x.category == "MENINGITIS" for x in m.flags("stiff neck, high fever and the light hurts his eyes"))


def test_benign_lay_phrases_still_clean():
    for t in ["here for a flu shot", "mild earache, no fever", "asking about travel vaccinations",
              "stubbed toe, sore but walking fine", "blocked nose and sneezing for a week"]:
        assert m.flags(t) == [], f"false flag on benign: {t!r}"
