"""
Tests del nuevo modelo jerárquico: RLS SOLO a nivel sección
(`fit_and_predict_sections`) + método derivado sin RLS para tienda/sku/
tienda+sku (`compute_derived_forecasts` / `_apply_causal_ses`).

Estos tests son el equivalente de regresión más crítico del cambio de
arquitectura: verifican, con referencias calculadas a mano en Python puro,
que el efecto de drivers + SES causal producen exactamente el `yhat`
esperado — sin depender de que el fit de RLS "adivine" el resultado
correcto (los coeficientes se inyectan directamente donde no requieren
numba/rls_opt, para poder testear la lógica de `compute_derived_forecasts`
de forma aislada y determinística).
"""
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

from forecasts import RLSForecastRunner

try:
    from rls_opt import RecursiveLeastSquaresRegression
except ImportError:  # pragma: no cover
    RecursiveLeastSquaresRegression = None


def _manual_causal_ses(values: list[float], alpha: float) -> list[float]:
    """Referencia en Python puro de la SES causal: s(t)=alpha*x(t)+(1-alpha)*s(t-1);
    yhat(t) = s(t-1), yhat(0) = s(0) (sin historia previa)."""
    s = []
    out = []
    prev = None
    for x in values:
        if prev is None:
            s_t = x
        else:
            s_t = alpha * x + (1 - alpha) * prev
        out.append(prev if prev is not None else s_t)
        s.append(s_t)
        prev = s_t
    return out


def test_apply_causal_ses_matches_manual_recursion():
    values = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0]
    alpha = 0.3
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * len(values),
            "ds": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(values))],
            "_x": values,
        }
    )
    out = RLSForecastRunner._apply_causal_ses(df, "_x", "_x_hat", alpha)
    expected = _manual_causal_ses(values, alpha)
    assert out.sort("ds")["_x_hat"].to_list() == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_apply_causal_ses_two_series_independent():
    """El estado de una serie no debe mezclarse con el de otra."""
    vals_a = [10.0, 20.0, 5.0]
    vals_b = [100.0, 50.0, 80.0]
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(3)]
    df = pl.DataFrame(
        {
            "unique_id": ["a"] * 3 + ["b"] * 3,
            "ds": dates + dates,
            "_x": vals_a + vals_b,
        }
    )
    out = RLSForecastRunner._apply_causal_ses(df, "_x", "_x_hat", 0.2)
    got_a = out.filter(pl.col("unique_id") == "a").sort("ds")["_x_hat"].to_list()
    got_b = out.filter(pl.col("unique_id") == "b").sort("ds")["_x_hat"].to_list()
    assert got_a == pytest.approx(_manual_causal_ses(vals_a, 0.2))
    assert got_b == pytest.approx(_manual_causal_ses(vals_b, 0.2))


def test_apply_causal_ses_forecast_only_holds_state_constant():
    """Los períodos forecast_only no actualizan el estado; se les asigna
    constante el último estado suavizado de la parte 'actual'."""
    actual_vals = [10.0, 12.0, 8.0, 15.0]
    alpha = 0.25
    n_actual = len(actual_vals)
    n_forecast = 3
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_actual + n_forecast)]
    period_types = ["in_sample"] * n_actual + ["forecast_only"] * n_forecast
    # valores "fake" en la zona forecast (no deben influir en el resultado)
    values = actual_vals + [999.0, -999.0, 0.0]

    df = pl.DataFrame({"unique_id": ["a"] * len(dates), "ds": dates, "_x": values, "period_type": period_types})
    out = RLSForecastRunner._apply_causal_ses(df, "_x", "_x_hat", alpha).sort("ds")

    # estado final tras la última observación actual (recursión completa, no shifteada)
    s = None
    for x in actual_vals:
        s = x if s is None else alpha * x + (1 - alpha) * s
    forecast_hat = out.filter(pl.col("period_type") == "forecast_only")["_x_hat"].to_list()
    assert forecast_hat == pytest.approx([s] * n_forecast, rel=1e-9, abs=1e-9)

    # y la parte actual coincide con la referencia causal de siempre
    actual_hat = out.filter(pl.col("period_type") != "forecast_only")["_x_hat"].to_list()
    assert actual_hat == pytest.approx(_manual_causal_ses(actual_vals, alpha), rel=1e-9, abs=1e-9)


def test_compute_derived_forecasts_matches_manual_reference():
    """
    Referencia manual completa del procedimiento (efecto + SES) para un
    nodo tienda+sku, con coeficientes de sección inyectados directamente
    (no requiere ajustar RLS real): confirma que `compute_derived_forecasts`
    hace exactamente lo descripto en README § Modelo jerárquico.
    """
    runner = RLSForecastRunner(
        driver_cols=["intercept", "driver_a", "driver_b"],
        rmse_error=0.2,
    )
    runner._driver_cols_price = ["intercept", "driver_a", "driver_b"]

    n = 10
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(0)
    driver_a = rng.uniform(0, 1, n).tolist()
    driver_b = rng.integers(0, 2, n).astype(float).tolist()
    y = (10 + rng.normal(0, 1, n)).tolist()
    value = [v * 10 for v in y]

    panel = pl.DataFrame(
        {
            "unique_id": ["1||T:X||S:Y"] * n,
            "ds": dates,
            "y": y,
            "value": value,
            "intercept": [1] * n,
            "driver_a": driver_a,
            "driver_b": driver_b,
            "period_type": ["in_sample"] * n,
        }
    )

    coef_y = np.array([5.0, 2.0, -1.5])  # [intercept, driver_a, driver_b]
    coef_p = np.array([50.0, 20.0, -15.0])
    coefs = {"1": (coef_y, coef_p)}

    alpha = 0.1
    out = runner.compute_derived_forecasts(panel, coefs, alpha=alpha).sort("ds")

    # Referencia manual
    effect_y = [2.0 * a + -1.5 * b for a, b in zip(driver_a, driver_b)]
    effect_v = [20.0 * a + -15.0 * b for a, b in zip(driver_a, driver_b)]
    y_neto = [yy - e for yy, e in zip(y, effect_y)]
    v_neto = [vv - e for vv, e in zip(value, effect_v)]
    y_neto_hat = _manual_causal_ses(y_neto, alpha)
    v_neto_hat = _manual_causal_ses(v_neto, alpha)
    expected_yhat = [round(h + e) for h, e in zip(y_neto_hat, effect_y)]
    expected_valuehat = [round(h + e, 2) for h, e in zip(v_neto_hat, effect_v)]
    expected_effect_y = effect_y

    assert out["yhat"].to_list() == pytest.approx(expected_yhat, rel=1e-6, abs=1e-6)
    assert out["valuehat"].to_list() == pytest.approx(expected_valuehat, rel=1e-6, abs=1e-6)
    assert out["driver_effect"].to_list() == pytest.approx(expected_effect_y, rel=1e-9, abs=1e-9)


def test_compute_derived_forecasts_ignores_section_only_ids():
    runner = RLSForecastRunner(driver_cols=["intercept", "driver_a"], rmse_error=0.2)
    runner._driver_cols_price = ["intercept", "driver_a"]
    panel = pl.DataFrame(
        {
            "unique_id": ["1", "1"],  # nodo de sección, sin "||"
            "ds": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "y": [10.0, 12.0],
            "value": [100.0, 120.0],
            "intercept": [1, 1],
            "driver_a": [0.5, 0.6],
            "period_type": ["in_sample", "in_sample"],
        }
    )
    out = runner.compute_derived_forecasts(panel, {"1": (np.array([1.0, 1.0]), np.array([1.0, 1.0]))}, alpha=0.1)
    assert out.height == 0


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_fit_and_predict_sections_only_fits_given_ids():
    """`fit_and_predict_sections` debe ignorar cualquier id que no esté en
    `section_ids` (aunque esté presente en el DataFrame de train)."""
    runner = RLSForecastRunner(driver_cols=["intercept", "driver_a"], rmse_error=0.2)
    n = 30
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(1)
    train = pl.concat(
        [
            pl.DataFrame(
                {
                    "unique_id": [uid] * n,
                    "ds": dates,
                    "y": (10 + rng.normal(0, 1, n)).tolist(),
                    "value": (100 + rng.normal(0, 5, n)).tolist(),
                    "intercept": [1] * n,
                    "driver_a": rng.uniform(0, 1, n).tolist(),
                }
            )
            for uid in ("1", "1||T:A||S:X")  # el 2do NO debe ajustarse
        ],
        how="vertical",
    )
    targets = {"in_sample": train}
    res_df, coefs = runner.fit_and_predict_sections(train, targets, ["1"], desc="test")
    assert set(coefs.keys()) == {"1"}
    assert res_df.height > 0
    assert set(res_df["unique_id"].unique().to_list()) == {"1"}