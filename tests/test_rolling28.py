"""Compatibilidad de métricas 28d y contrato expanding-28 vigente."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import settings


def test_expanding_28_settings():
    assert settings.RLS_BLOCK_DAYS == 28
    assert settings.RLS_FIT_MODE == "expanding_28"
    assert settings.METRICS_MODE == "rolling_28"


def test_metrics_rolling28_backend():
    import backend

    df = pl.DataFrame(
        {
            "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "y": [100.0, 50.0],
            "yhat": [90.0, 40.0],
            "yhat28": [80.0, 45.0],
        }
    )
    m = backend.metrics_rolling28(df)
    # |100-80|+|50-45| / 150 = 25/150
    assert abs(m["wmape_28"] - 25 / 150) < 1e-9
    assert m["n"] == 2


def test_metrics_rolling28_absent_column_returns_zero():
    """Sin columna yhat28 no debe fallar; devuelve métricas vacías."""
    import backend

    df = pl.DataFrame(
        {
            "ds": [dt.date(2024, 1, 1)],
            "y": [100.0],
            "yhat": [90.0],
        }
    )
    m = backend.metrics_rolling28(df)
    assert m == {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}


def test_metrics_rolling28_excludes_y_zero():
    import backend

    df = pl.DataFrame(
        {
            "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "y": [100.0, 0.0],
            "yhat28": [80.0, 45.0],
        }
    )
    m = backend.metrics_rolling28(df)
    # solo la fila con y!=0: |100-80|/100 = 0.2
    assert abs(m["wmape_28"] - 0.2) < 1e-9
    assert m["n"] == 1
