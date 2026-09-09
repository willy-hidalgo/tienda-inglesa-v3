from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_4_version_and_sku_total_ensemble_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_SKU_SELECTION_BLOCKS: int = 3' in settings
    assert 'V12_SKU_SELECTION_MIN_BLOCKS: int = 2' in settings
    assert 'V12_SKU_RECENT_SCORE_WEIGHT: float = 0.25' in settings


def test_v12_4_sku_total_ensemble_is_causal_and_contains_anchor_models():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert '"v11_sum"' in src
    assert '"recent_weekday"' in src
    assert '"same_weekday_4"' in src
    assert '"lag28"' in src
    assert '"annual_scaled"' in src
    assert '"annual_blend"' in src
    assert '(pl.col("ds") - pl.duration(days=28)).alias("_lag28_ds")' in src
    assert '(pl.col("ds") - pl.duration(days=35)).alias("_lag35_ds")' in src
    assert '(pl.col("ds") - pl.duration(days=42)).alias("_lag42_ds")' in src
    assert '(pl.col("ds") - pl.duration(days=49)).alias("_lag49_ds")' in src
    assert 'start = origin - dt.timedelta(days=block_days * block_idx)' in src


def test_v12_4_exposes_sku_model_diagnostics_and_validator_checks_them():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    for col in (
        "v12_sku_model_y", "v12_sku_model_value",
        "v12_sku_validation_wmape_y", "v12_sku_validation_wmape_value",
    ):
        assert col in src
        assert col in validator
    assert '=== 16) CALIDAD OOS DEL ENSAMBLE DE TOTAL SKU v12.8 ===' in report
