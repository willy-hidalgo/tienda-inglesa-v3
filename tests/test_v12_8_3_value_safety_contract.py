from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_v12_8_3_value_safety_settings_and_policy_contract():
    settings = _text("settings.py")
    leaf = _text("app/forecasting/leaf_v12.py")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_VALUE_PORTFOLIO_SAFETY_ENABLED: bool = True' in settings
    assert 'V12_VALUE_ALL_MIN_RECENT_CONFIRMATIONS: int = 2' in settings
    assert 'V12_VALUE_ALL_BIAS_COVERAGE_GRID: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65)' in settings
    assert 'V12_VALUE_META_LEAF_MIN_UTILITY_GAIN: float = 0.0025' in settings
    assert 'def _calibrate_value_safety_policy' in leaf
    assert 'policy_y = _calibrate_meta_policy(qty_scores, "y")' in leaf
    assert 'policy_v = _calibrate_value_walkforward_policy(block_scores)' in leaf
    assert 'v12_value_safety_dominance_pass' in leaf
    assert 'v12_value_safety_recent_confirmations' in leaf
    assert 'v12_value_safety_bias_coverage_pass' in leaf
    assert 'v12_value_safety_meta_margin_pass' in leaf
    assert 'insufficient_closed_evidence' in leaf

def test_v12_8_3_value_safety_validator_acceptance_dashboard_contract():
    validator = _text("app/forecasting/validate_v11.py")
    report = _text("app/forecasting_acceptance_report.py")
    artifacts = _text("app/dashboard_artifacts.py")
    dashboard = _text("app/dashboard.py")
    data = _text("app/dashboard_data.py")
    assert 'ARTIFACT_VERSION = 20' in artifacts
    assert 'v12_value_safety_reason' in artifacts
    assert 'walk-forward' in validator
    assert '=== 22) v12.9.7 VALUE PORTFOLIO SAFETY (LEGACY DIAGNÓSTICO) ===' in report
    assert 'v12.9.7 · Value Portfolio Safety (legacy diagnostic)' in dashboard
    assert 'Cargar diagnóstico avanzado v11/v12' in dashboard
    assert 'def _official_oos_metrics_from_artifact' in data
