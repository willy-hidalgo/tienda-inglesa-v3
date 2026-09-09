from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_v1294_version_and_risk_settings():
    s=(ROOT/"settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in s
    assert 'V1294_VALUE_SEC23_MAX_VOLUME_SHARE: float = 0.25' in s
    assert 'V1294_VALUE_SEC23_MAX_SEGMENT_VOLUME_SHARE: float = 0.08' in s

def test_v1294_sec1_frozen_and_sec23_strict_budget():
    src=(ROOT/"app/forecasting/leaf_v12.py").read_text(encoding="utf-8")
    assert 'if section_id != "23" and section_strong' in src
    assert 'if used + w <= cap + 1e-9' in src
    assert 'V1294_VALUE_SEC23_MAX_VOLUME_SHARE' in src
    assert 'V1294_VALUE_SEC23_MAX_SEGMENT_VOLUME_SHARE' in src
    assert '_high_guard' in src
    assert '_allocation_weight' in src

def test_v1294_metadata_semantics_and_report():
    validate=(ROOT/"app/forecasting/validate_v11.py").read_text(encoding="utf-8")
    report=(ROOT/"app/forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert 'seg_rows = target.filter(pl.col("v1293_segment_selector_enabled_value").fill_null(False))' in validate
    assert 'budget duro de volumen Sec23' in validate
    assert '=== 27) v12.9.7 SEC23 VOLUME-RISK CONTROL ===' in report

def test_v1294_impact_semantics_high_is_high():
    src=(ROOT/"app/forecasting/leaf_v12.py").read_text(encoding="utf-8")
    assert '1.0 = highest-volume leaf' in src
    assert '_pct[_order] = (float(profile.height) - np.arange' in src
