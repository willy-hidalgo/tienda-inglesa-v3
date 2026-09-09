from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_leaf_level_invariant_includes_sku_seasonal_multiplier():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert 'sku_seasonal_multiplier_y' in src
    assert '_expected_forecast_factor_y' in src
    assert 'parent level factor × selected SKU seasonal multiplier' in src
    assert '_n_seasonal_multiplier_y' in src


def test_offline_validator_uses_final_leaf_formula():
    src = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert '* pl.col("sku_seasonal_multiplier_y")' in src
    assert '* pl.col("sku_seasonal_multiplier_value")' in src
    assert 'run_status_errors(path)' in src


def test_cli_marks_failed_runs_so_stale_forecast_cannot_be_audited():
    src = (ROOT / "app" / "forecasts.py").read_text(encoding="utf-8")
    assert 'mark_running(config.out_dir)' in src
    assert 'mark_success(config.out_dir / "forecast.parquet")' in src
    assert 'except BaseException as exc:' in src
    assert 'mark_failed(config.out_dir, exc)' in src


def test_acceptance_report_rejects_stale_dashboard_artifacts():
    src = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert 'run_status_errors(forecast)' in src
    assert 'artifacts.artifacts_match_source(forecast, index)' in src
    assert 'No se puede ejecutar aceptación' in src
