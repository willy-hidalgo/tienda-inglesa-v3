from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_5_version_and_share_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_SHARE_RECENT_DAYS: int = 28' in settings
    assert 'V12_SHARE_STABLE_DAYS: int = 84' in settings
    assert 'V12_SHARE_RECENT_WEIGHT: float = 0.70' in settings


def test_v12_5_allocation_uses_28x84_not_weekday_share():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'alias("_share_28x84_y")' in src
    assert 'alias("_share_28x84_v")' in src
    assert '.then(pl.col("_share_28x84_y"))' in src
    assert '.then(pl.col("_share_28x84_v"))' in src
    # DOW share remains available for occurrence diagnostics, but it cannot be
    # the magnitude weight of the production allocation anymore.
    alloc = src[src.index('alias("_share_28x84_y")'):src.index('alias("v12_store_share_y")')]
    assert '.then(pl.col("_share_blend_y"))' not in alloc
    assert '.then(pl.col("_share_blend_v"))' not in alloc


def test_v12_5_keeps_true_hurdle_and_normalization_contract():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'v12_occurrence_gate_open_joint' in src
    assert 'pl.col("_open_count_y")' in src
    assert 'pl.col("_open_count_v")' in src
    assert '1.0 / pl.col("_open_count_y").cast(pl.Float64)' in src
    assert '1.0 / pl.col("_open_count_v").cast(pl.Float64)' in src
    assert 'v12_sku_forecast_y") * pl.col("v12_store_share_y' in src
    assert 'v12_sku_forecast_value") * pl.col("v12_store_share_value' in src


def test_v12_5_acceptance_reports_share_contract():
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert '=== 17) SHARE v12.8 — RECENCIA 28D×84D ===' in report
