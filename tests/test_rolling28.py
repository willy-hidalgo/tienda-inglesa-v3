"""Tests rolling 28d (métricas y helpers; fit RLS se mockea si no hay rls_opt)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import settings
from forecasts import RLSForecastRunner


def test_rolling_horizon_setting():
    assert settings.ROLLING_HORIZON_DAYS == 28


def test_first_monday_helper():
    # 2024-05-01 was Wednesday → next Monday 2024-05-06
    assert RLSForecastRunner._first_monday_on_or_after(dt.date(2024, 5, 1)) == dt.date(
        2024, 5, 6
    )
    # already Monday
    assert RLSForecastRunner._first_monday_on_or_after(dt.date(2024, 5, 6)) == dt.date(
        2024, 5, 6
    )


def test_compute_wmape28():
    df = pl.DataFrame(
        {
            "unique_id": ["1", "1", "1", "1"],
            "y": [10.0, 0.0, 20.0, 30.0],
            "yhat28": [8.0, 5.0, 18.0, 25.0],
        }
    )
    w = RLSForecastRunner.compute_wmape28(df)
    # excluye y=0 → |10-8|+|20-18|+|30-25| / (10+20+30) = 9/60 = 0.15
    assert w.height == 1
    assert abs(w["wmape_28"][0] - 0.15) < 1e-9
    # bias = (8+18+25 - 10-20-30)/60 = (51-60)/60 = -0.15
    assert abs(w["bias_28"][0] - (-0.15)) < 1e-9
    assert w["n_points_28"][0] == 3


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
