from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v1292_version_and_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert "V1292_VALUE_EXPECTED_IMPACT_ENABLED: bool = True" in settings
    assert "V1292_VALUE_PORTFOLIO_MAX_VOLUME_SHARE" in settings
    assert "V1292_VALUE_TOP02_MIN_GAIN" in settings


def test_v1292_uses_strict_temporal_oof_calibration():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'fit_df = train.filter(pl.col("_eg_label_block") > label_block)' in src
    assert 'valid_df = train.filter(pl.col("_eg_label_block") == label_block)' in src
    assert "bucket_gain" in src
    assert "v1292_raw_expected_gain_value" in src
    assert "v1292_calibrated_gain_value" in src
    assert "confidence = np.clip(confidence, 0.0, 1.0)" in src


def test_v1292_has_asymmetric_risk_and_volume_budget():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert "v1292_bucket_loss_rate_value" in src
    assert "v1292_leaf_p10_gain_value" in src
    assert "v1292_impact_percentile_value" in src
    assert "v1292_risk_score_value" in src
    assert "v1292_calibrated_gain_value" in src
    assert "diagnostics and never authorize production" in src


def test_v1292_validator_scopes_sparse_metadata_and_checks_ranges():
    validate = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert 'expected-impact fields are diagnostic only in v12.9.7' in validate
    assert 'pl.col("v1291_expected_gain_confidence_value") > 1.0' in validate
    assert "selector jerárquico" in validate


def test_v1292_acceptance_has_calibration_and_risk_audit():
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert "=== 24) v12.9.2 VALUE CALIBRATED EXPECTED-IMPACT SELECTOR (DIAGNÓSTICO) ===" in report
    assert "OOF calibration buckets (raw -> calibrated -> realized)" in report
    assert "=== 25) TOP 50 VALUE ERROR CONTRIBUTORS — v12.9.2 RISK AUDIT (DIAGNÓSTICO) ===" in report
