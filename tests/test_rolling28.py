"""Contratos de cadencia y horizonte v13."""
from __future__ import annotations

import datetime as dt
import polars as pl

import settings


def test_update_block_options_and_fixed_oos():
    assert settings.UPDATE_BLOCK_OPTIONS == (1, 7, 14, 28)
    assert settings.METRIC_HORIZON_DAYS == 28


def test_section_horizons_oos_and_forecast_only_are_28_days():
    hz = settings.section_horizons(
        "1", first_data=dt.date(2024, 5, 3), last_actual=dt.date(2026, 4, 30)
    )
    assert (hz["test_end"] - hz["test_start"]).days + 1 == 28
    assert (hz["forecast_end"] - hz["forecast_start"]).days + 1 == 28
    assert hz["forecast_start"] == hz["test_end"] + dt.timedelta(days=1)


def test_metrics_rolling28_excludes_y_zero():
    from app import backend
    df = pl.DataFrame({
        "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
        "y": [100.0, 0.0],
        "yhat28": [80.0, 45.0],
    })
    m = backend.metrics_rolling28(df)
    assert abs(m["wmape_28"] - 0.2) < 1e-9
    assert m["n"] == 1
