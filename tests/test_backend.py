"""Tests del backend y dashboard_data (sin Streamlit)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import backend
import settings
from dashboard_data import prepare_dashboard_state


def _sample_forecast() -> pl.DataFrame:
    rows = []
    base = dt.date(2025, 11, 1)
    for i in range(40):
        d = base + dt.timedelta(days=i)
        period = "out_sample" if d >= dt.date(2025, 12, 1) else "in_sample"
        if d > dt.date(2025, 12, 7):
            period = "forecast_only"
            y, yhat = 0.0, 12.0
        else:
            y, yhat = 10.0 + i % 5, 9.0 + i % 4
        rows.append(
            {
                "unique_id": "23",
                "ds": d,
                "y": y,
                "yhat": yhat,
                "value": y * 10,
                "valuehat": yhat * 10,
                "period_type": period,
                "sku_desc": "",
                "store_name": "",
                "seccion": "23",
                "train_end": dt.date(2025, 11, 30),
                "test_start": dt.date(2025, 12, 1),
                "test_end": dt.date(2025, 12, 7),
                "forecast_start": dt.date(2025, 12, 8),
                "forecast_end": dt.date(2026, 2, 1),
            }
        )
        # child series for ranking
        rows.append(
            {
                **{k: v for k, v in rows[-1].items() if k != "unique_id"},
                "unique_id": "23||00155",
                "store_name": "DELSOL",
            }
        )
    return pl.DataFrame(rows)


def test_detail_view_excludes_horizon_cols():
    df = _sample_forecast()
    out = backend.detail_view(df)
    for c in backend.HORIZON_COLUMNS:
        assert c not in out.columns
    assert "abs_error" in out.columns
    assert set(out.columns) <= set(backend.DETAIL_COLUMNS)


def test_wmape_excluye_y_cero():
    df = pl.DataFrame(
        {
            "unique_id": ["1||a", "1||a", "1||b"],
            "y": [10.0, 0.0, 20.0],
            "yhat": [8.0, 5.0, 18.0],
            "period_type": ["out_sample"] * 3,
        }
    )
    w = backend.wmape_por_id(["1||a", "1||b"], df)
    r = w.filter(pl.col("unique_id") == "1||a")
    assert abs(r["wmape"][0] - 0.2) < 1e-9
    assert r["n_points"][0] == 1


def test_ranking_wmape_table_columns():
    base = pl.DataFrame(
        {
            "unique_id": ["1||00122", "1||00154"],
            "wmape": [0.1, 0.2],
            "sum_y": [100.0, 50.0],
            "n_points": [10, 5],
        }
    )
    desc = {"1||00122": "CENTRAL", "1||00154": "HIPER"}
    t = backend.ranking_wmape_table(base, "1", desc)
    assert list(t.columns) == [
        "Código",
        "Descripción",
        "wMAPE (%)",
        "N puntos",
        "unique_id",
    ]
    assert t.height == 2


def test_build_chart_series_keys():
    df = _sample_forecast().filter(pl.col("unique_id") == "23")
    ch = backend.build_chart_series(
        df,
        test_end=dt.date(2025, 12, 7),
        forecast_start=dt.date(2025, 12, 8),
        forecast_end=dt.date(2026, 2, 1),
        cutoff=dt.date(2025, 11, 30),
        has_period=True,
    )
    assert "hist_ds" in ch and "fcst_yhat" in ch
    assert "cutoff" in ch


def test_prepare_dashboard_state():
    df = _sample_forecast()
    view = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="23",
        cutoff_date=dt.date(2025, 11, 30),
        candidatos=["23", "23||00155"],
        nombre_nivel="seccion",
    )
    assert view.selected_id == "23"
    assert "in" in view.metrics and "out" in view.metrics
    assert view.detail.height > 0
    assert "train_end" not in view.detail.columns
    assert view.chart["hist_ds"] is not None


def test_format_detail_display():
    df = pl.DataFrame(
        {
            "unique_id": ["1"],
            "ds": [dt.date(2024, 1, 1)],
            "y": [12345.2],
            "yhat": [1000.9],
            "value": [999999.4],
            "valuehat": [0.4],
            "abs_error": [234.6],
        }
    )
    out = backend.format_detail_display(df)
    assert out["y"][0] == "12,345"
    assert out["yhat"][0] == "1,001"
    assert out["value"][0] == "999,999"
    assert out["valuehat"][0] == "0"
    assert out["abs_error"][0] == "235"
