from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_v12_8_version_meta_and_target_specific_contract():
    settings = _text("settings.py")
    leaf = _text("app/forecasting/leaf_v12.py")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_REQUIRE_JOINT_TARGET_WIN: bool = False' in settings
    assert 'V12_META_SELECTOR_ENABLED: bool = True' in settings
    assert 'V12_META_SELECTOR_RECENCY_WEIGHTS: tuple[float, float, float] = (0.55, 0.30, 0.15)' in settings
    assert 'V12_META_SELECTOR_THRESHOLDS: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65)' in settings
    assert 'V12_META_SELECTOR_ADAPTIVE_POLICY: bool = True' in settings
    assert '_fit_meta_selector' in leaf
    assert '_calibrate_meta_policy' in leaf
    assert 'v12_meta_portfolio_mode_y' in leaf
    assert 'v12_meta_threshold_y' in leaf
    assert 'v12_meta_bias_guard_pass_y' in leaf
    assert 'v12_meta_probability_y' in leaf
    assert 'v12_meta_probability_value' in leaf
    assert 'np.linalg.solve' in leaf

def test_v12_8_restores_uncalibrated_baseline_and_keeps_challenger():
    settings = _text("settings.py")
    leaf = _text("app/forecasting/leaf_v12.py")
    assert 'V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER: bool = False' in settings
    assert 'v12_sku_forecast_calibrated_challenger_y' in leaf
    assert '.otherwise(pl.col("v12_sku_forecast_uncalibrated_y"))' in leaf
    assert 'v12_sku_level_calibration_applied_y' in leaf

def test_v12_8_dashboard_and_acceptance_contract():
    artifacts = _text("app/dashboard_artifacts.py")
    dashboard = _text("app/dashboard.py")
    report = _text("app/forecasting_acceptance_report.py")
    assert 'ARTIFACT_VERSION = 20' in artifacts
    assert '"v12_meta_probability_y"' in artifacts
    assert 'Diagnóstico de modelo v12.9.12' in dashboard
    assert 'P(v12 gana) mediana' in dashboard
    assert '=== 21) v12.9.7 META-SELECTOR + PORTFOLIO MODE ===' in report
