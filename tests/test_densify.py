"""Panel denso + N puntos ranking = spine."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from app.forecasting.panel import densify_section_panel
import backend
import settings


def test_densify_same_n_dates_per_uid():
    df = pl.DataFrame(
        {
            "unique_id": ["1", "1", "1||s1", "1||s1"],
            "ds": [
                dt.date(2024, 5, 1),
                dt.date(2024, 5, 3),
                dt.date(2024, 5, 1),
                dt.date(2024, 5, 2),
            ],
            "y": [10.0, 20.0, 5.0, 0.0],
            "value": [100.0, 200.0, 50.0, 0.0],
            "seccion": ["1"] * 4,
        }
    )
    out = densify_section_panel(df, dt.date(2024, 5, 1), dt.date(2024, 5, 3))
    assert out.height == 6
    n_by = out.group_by("unique_id").agg(pl.len().alias("n"))
    assert n_by["n"].unique().to_list() == [3]
    row = out.filter(
        (pl.col("unique_id") == "1") & (pl.col("ds") == dt.date(2024, 5, 2))
    )
    assert row["y"][0] == 0.0


def test_ranking_n_points_constant_spine():
    df = pl.DataFrame(
        {
            "unique_id": ["1||a"] * 3 + ["1||b"] * 2,
            "ds": [
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
                dt.date(2024, 1, 1),
                dt.date(2024, 1, 2),
            ],
            "y": [10.0, 0.0, 20.0, 5.0, 0.0],
            "yhat": [9.0, 1.0, 18.0, 4.0, 1.0],
            "period_type": ["in_sample"] * 5,
        }
    )
    # forzar spine = 10 para todos
    w = backend.wmape_por_id(["1||a", "1||b"], df, n_fechas_spine=10)
    assert set(w["n_points"].to_list()) == {10}
    # a tiene 2 ventas, b tiene 1
    assert w.filter(pl.col("unique_id") == "1||a")["n_with_sales"][0] == 2
    assert w.filter(pl.col("unique_id") == "1||b")["n_with_sales"][0] == 1


def test_chart_series_includes_zeros():
    df = pl.DataFrame(
        {
            "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)],
            "y": [10.0, 0.0, 5.0],
            "yhat": [9.0, 1.0, 4.0],
            "period_type": ["in_sample"] * 3,
        }
    )
    hist, _ = backend.split_hist_forecast(
        df,
        test_end=dt.date(2024, 1, 3),
        forecast_start=dt.date(2024, 1, 4),
        forecast_end=dt.date(2024, 1, 10),
        has_period=True,
    )
    assert hist.height == 3
    assert 0.0 in hist["y"].to_list()


def test_metrics_ignore_zero_rows_not_dropped():
    df = pl.DataFrame(
        {
            "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "y": [10.0, 0.0],
            "yhat": [8.0, 1.0],
        }
    )
    w, b, n = backend.calcular_metricas(df)
    assert n == 1
    assert df.height == 2
