from pathlib import Path
import datetime as dt
import settings


def test_update_block_options_and_paths():
    assert settings.APP_VERSION == "12.9.12"
    assert settings.UPDATE_BLOCK_OPTIONS == (1, 7, 14, 28)
    assert settings.METRIC_HORIZON_DAYS == 28
    for d in settings.UPDATE_BLOCK_OPTIONS:
        assert settings.update_block_forecast_path(d).name == "forecast.parquet"
        assert f"block_{d:02d}d" in str(settings.update_block_forecast_path(d))


def test_oos_is_fixed_28_while_update_cadence_changes():
    old = settings.RLS_BLOCK_DAYS
    try:
        for d in (1, 7, 14, 28):
            settings.RLS_BLOCK_DAYS = d
            hz = settings.section_horizons("1", dt.date(2024, 1, 1), dt.date(2025, 12, 31))
            assert (hz["test_end"] - hz["test_start"]).days + 1 == 28
            assert (hz["forecast_end"] - hz["forecast_start"]).days + 1 == 28
            assert hz["update_block_days"] == d
            assert ((hz["train_end"] - hz["train_start"]).days + 1) % d == 0
    finally:
        settings.RLS_BLOCK_DAYS = old


def test_dashboard_and_cli_are_precalculated_not_runtime_fit():
    root = Path(__file__).resolve().parents[1]
    dashboard = (root / "app" / "dashboard.py").read_text(encoding="utf-8")
    forecasts = (root / "app" / "forecasts.py").read_text(encoding="utf-8")
    artifacts = (root / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")
    assert '"Bloque de actualización"' in dashboard
    assert "scenario_paths" in dashboard
    assert "--all-update-blocks" in forecasts
    assert "--update-block-days" in forecasts
    assert "ARTIFACT_VERSION = 20" in artifacts


def test_leaf_horizon_invariant_uses_metric_horizon_not_update_cadence():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert 'metric_horizon_days = int(getattr(settings, "METRIC_HORIZON_DAYS", 28))' in runner
    assert 'pl.col("_n_dates") != metric_horizon_days' in runner
    assert 'exactly {metric_horizon_days} dates' in runner


def test_leaf_level_invariant_is_rowwise_and_multicadence_safe():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "Multi-cadence-safe v11 identity" in runner
    assert 'alias("_expected_v11_yhat_raw")' in runner
    assert 'alias("_expected_v11_valuehat_raw")' in runner
    assert 'pl.col("ses_level_y")' in runner
    assert 'pl.col("driver_factor_y")' in runner
    assert 'pl.col("sku_seasonal_multiplier_y").fill_null(1.0)' in runner
    assert "mean forecast / SES must match" not in runner
