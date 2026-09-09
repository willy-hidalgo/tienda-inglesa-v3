from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_v12_7_version_and_calibration_contract():
    settings = _text("settings.py")
    leaf = _text("app/forecasting/leaf_v12.py")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert "V12_SKU_LEVEL_CALIBRATION_ENABLED" in settings
    assert "V12_SKU_LEVEL_CALIBRATION_PRIOR_BLOCKS" in settings
    assert "V12_SKU_LEVEL_CALIBRATION_CLIP" in settings
    assert "v12_sku_forecast_uncalibrated_y" in leaf
    assert "v12_sku_level_calibration_factor_y" in leaf
    assert "v12_sku_forecast_calibrated_challenger_y" in leaf
    assert "V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER" in settings
    assert "V12_SELECTOR_BIAS_WEIGHT" in settings
    assert "v12_validation_utility_improvement_y" in leaf


def test_acceptance_has_calibration_and_oracle_diagnostics():
    report = _text("app/forecasting_acceptance_report.py")
    assert "=== 19) v12.8 CALIBRACIÓN SKU-TOTAL — CHALLENGER NO APLICADO POR DEFECTO ===" in report
    assert "=== 20) ORACLE DIAGNÓSTICO v11/v12 POR LEAF" in report
    assert "NO CAUSAL" in report


def test_dashboard_v12_7_contract():
    artifacts = _text("app/dashboard_artifacts.py")
    data = _text("app/dashboard_data.py")
    dashboard = _text("app/dashboard.py")
    backend = _text("app/backend.py")
    assert "ARTIFACT_VERSION = 20" in artifacts
    assert '"v11_yhat_raw_before_v12"' in artifacts
    assert '"v12_candidate_yhat_raw"' in artifacts
    assert "model_compare" in data
    assert "Oracle v11/v12 (no causal)" in data
    assert '"in_ds"' in backend and '"oos_ds"' in backend and '"fcst_ds"' in backend
    assert "Artefactos desactualizados" in dashboard
    assert "Diagnóstico de modelo v12.9.12" in dashboard
    assert "actual + in-sample + OOS + forecast-only" in dashboard
