from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v1291_version_and_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert "V1291_VALUE_EXPECTED_GAIN_ENABLED: bool = True" in settings
    assert "V1291_VALUE_EXPECTED_GAIN_MIN_GAIN" in settings
    assert "V1291_VALUE_EXPECTED_GAIN_PORTFOLIO_MIN_EXPECTED_GAIN" in settings


def test_expected_gain_is_continuous_volume_weighted_and_causal():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert "def _fit_value_expected_gain_selector" in src
    assert 'pl.col("_utility_improvement_v").cast(pl.Float64).alias("_eg_target")' in src
    assert 'pl.col("_den_v").cast(pl.Float64).alias("_eg_weight")' in src
    assert "feature_blocks = tuple(" in src
    assert "label_block + 1" in src
    assert "V1291_VALUE_EXPECTED_GAIN_L2" in src
    assert "w = w / max(float(w.mean()), 1e-12)" in src
    assert 'score_blocks = tuple(range(1, min(max_block, feature_n) + 1))' in src


def test_expected_gain_only_owns_value_at_28d_and_qty_remains_frozen():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'qty_scores = block_scores.filter(pl.col("_validation_block") <= qty_blocks)' in src
    assert 'getattr(settings, "RLS_BLOCK_DAYS", 28)' in src
    assert 'def _fit_value_expected_gain_selector' in src
    assert 'diagnostics and never authorize production' in src
    assert '_adaptive_preselect_expr("y", mode_y, threshold_y).alias("_preselect_y")' in src


def test_expected_gain_has_conservative_guards_and_reporting():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    validate = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert "V1292_VALUE_MIN_CONFIDENCE" in src
    assert "v1292_calibrated_gain_value" in src
    assert "v1291_expected_gain_portfolio_pass_value" in src
    assert "segment_ok_v" in validate
    assert "=== 24) v12.9.2 VALUE CALIBRATED EXPECTED-IMPACT SELECTOR" in report
    assert "=== 25) TOP 50 VALUE ERROR CONTRIBUTORS" in report
    assert "OOF calibration buckets" in report
