"""Regresión del fix de log-space en la derivación SKU+tienda.

El modelo RLS se ajusta sobre log1p(y). Por lo tanto, para reconstruir el
pronóstico fuera-del-spine (out_sample / forecast_only) se debe operar en
log-space:

    log(y) = intercept + efecto_drivers + residuo_SES
    yhat   = expm1(intercept + efecto_drivers + residuo_SES)

Históricamente el código restaba el efecto (log-space) de la serie lineal (y),
produciendo un residuo incoherente y pronósticos 0/negativos en OOS/forecast
— la causa del ranking SKU+tienda vacío. Este test protege la corrección.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from app.forecasting.runner import RLSForecastRunner  # noqa: E402


def _ses_non_causal(values, alpha):
    """ewm_mean(adjust=False): s_t = alpha*x_t + (1-alpha)*s_{t-1}, s_0 = x_0."""
    s = []
    prev = None
    for x in values:
        prev = x if prev is None else alpha * x + (1 - alpha) * prev
        s.append(prev)
    return s


def test_derive_sku_store_logspace_reconstruction_oos_and_forecast():
    """OOS/forecast-only se reconstruyen con expm1(intercept+efecto+SES(log))."""
    runner = RLSForecastRunner(
        driver_cols=["intercept", "driver_a"], rmse_error=0.2
    )
    # driver_cols_price deriva excluyendo los drivers de precio; aquí idéntico.
    runner._driver_cols_price = ["intercept", "driver_a"]

    driver_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    n_train = 5
    n_fcst = 2
    dates = [
        dt.date(2024, 1, 1) + dt.timedelta(days=i)
        for i in range(n_train + n_fcst)
    ]
    period = ["in_sample"] * n_train + ["forecast_only"] * n_fcst
    # forecast_only no tiene actuals (y=0, value=0).
    y = [11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0]
    value = [v * 10 for v in y]

    panel = pl.DataFrame(
        {
            "unique_id": ["1||T:X||S:Y"] * len(dates),
            "ds": dates,
            "y": y,
            "value": value,
            "intercept": [1] * len(dates),
            "driver_a": driver_a,
            "period_type": period,
        }
    )

    # Coeficientes log: [intercept, driver_a]
    coef_y = np.array([1.5, 0.5])
    coef_p = np.array([15.0, 5.0])
    section_coefs = {"1": (coef_y, coef_p)}
    store_coefs = {"1||T:X": (coef_y, coef_p)}

    out = runner.derive_sku_store_forecasts(
        panel, "1", section_coefs, store_coefs, alpha=0.1
    ).sort("ds")

    intercept = 1.5
    effect = [0.5 * d for d in driver_a]  # coef_driver_a * driver_a
    expected_in = [
        round(float(np.expm1(intercept + e)))
        for e in effect[:n_train]
    ]
    log_resid = [
        np.log1p(yy) - (intercept + e)
        for yy, e in zip(y[:n_train], effect[:n_train])
    ]
    ses_last = _ses_non_causal(log_resid, 0.1)[-1]
    expected_fcst = [
        round(float(np.expm1(intercept + e + ses_last)))
        for e in effect[n_train:]
    ]

    got_in = out.filter(pl.col("period_type") == "in_sample")["yhat"].to_list()
    got_fcst = out.filter(pl.col("period_type") == "forecast_only")["yhat"].to_list()

    assert got_in == expected_in
    assert got_fcst == expected_fcst
    assert all(v > 0 for v in got_fcst), "pronósticos no deben ser 0/negativos"
