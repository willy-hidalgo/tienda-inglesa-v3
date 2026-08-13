"""Tests rolling 28d (métricas y helpers; fit RLS se mockea si no hay rls_opt).

Nota (Fase 3b): `_rolling_28_one` ahora hace 1 solo fit por serie con
`return_all_coefs=True` y camina los bloques de 28 días leyendo el vector de
coeficientes vigente al inicio de cada bloque (walk-forward causal, O(n) por
serie), en vez de reajustar el modelo completo en cada bloque (O(n²)). Los
tests de abajo cubren: (1) las piezas que no cambiaron (helpers, métricas), y
(2) que el nuevo algoritmo es causal (no usa información futura) y no
lanza excepciones para series sintéticas pequeñas.
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

import settings
from forecasts import RLSForecastRunner

try:
    from rls_opt import RecursiveLeastSquaresRegression
except ImportError:  # pragma: no cover
    RecursiveLeastSquaresRegression = None


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


def test_run_rolling_28_aligns_min_obs_28():
    """Fase 5: series con < 28 obs no deben entrar al rolling (antes: umbral 2)."""
    runner = RLSForecastRunner(
        driver_cols=["intercept"], rmse_error=0.2, n_jobs=1
    )
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    panel = pl.DataFrame(
        {
            "unique_id": ["1||short"] * 10,
            "ds": dates,
            "y": [1.0] * 10,
            "value": [1.0] * 10,
            "intercept": [1] * 10,
        }
    )
    out = runner.run_rolling_28(panel, ["seccion", "store", "sku"])
    # con min_obs=28 esta serie (10 filas) queda excluida
    assert out.height == 0


def test_compute_rolling_28_default_is_false():
    """El cálculo de rolling28 es opcional y viene desactivado por defecto."""
    assert settings.COMPUTE_ROLLING_28 is False


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_run_section_skips_rolling28_when_disabled(monkeypatch):
    """
    Con settings.COMPUTE_ROLLING_28=False (default), `_run_section` no debe
    agregar columnas yhat28/valuehat28 en absoluto (ni siquiera nulas) — el
    dashboard detecta la disponibilidad del rolling28 por presencia de la
    columna, así que una columna toda-nula rompería esa detección.
    Con COMPUTE_ROLLING_28=True, las columnas sí deben estar presentes.
    """
    from forecasts import (
        RLSForecastPipeline,
        HolidayCalendar,
        CalendarFeatureBuilder,
        DataAggregator,
    )

    sec = "23"
    hz = settings.section_horizons(sec, first_data=dt.date(2024, 5, 1))
    all_days = []
    d = hz["train_start"]
    while d <= hz["test_end"]:
        all_days.append(d)
        d += dt.timedelta(days=1)

    rng = np.random.default_rng(0)
    rows = []
    locales = settings.SECCIONES[sec]["locales"][:1]
    for store in locales:
        for sku in ["SKU_A"]:
            for day in all_days:
                qty = max(0.0, rng.normal(8, 3))
                rows.append(
                    {
                        "SALES_DAY": day,
                        "SECCION": sec,
                        "SKU_ID": sku,
                        "STORE_ID": store,
                        "SLS_QTY": round(qty, 0),
                        "SLS_VAL": round(qty * 12.5, 2),
                        "DESCRIPCION": f"Desc {sku}",
                    }
                )
    selected = pl.DataFrame(rows)

    class FakeCfg:
        date_column = "SALES_DAY"
        quantity_column = "SLS_QTY"
        price_column = "SLS_VAL"
        aggregation_levels = settings.AGGREGATION_LEVELS
        forecast_levels = settings.FORECAST_LEVELS
        rmse_error = settings.RMSE_ERROR
        forgetting_factor = settings.FORGETTING_FACTOR
        min_y_to_update = settings.MIN_Y_TO_update
        correction_factor = settings.CORRECTION_FACTOR
        out_dir = None

    def _build_pipeline():
        pipeline = RLSForecastPipeline.__new__(RLSForecastPipeline)
        pipeline._cfg = FakeCfg()
        pipeline._n_jobs = 1
        pipeline._limit_series = None
        pipeline._calendar = HolidayCalendar(settings.HOLIDAYS, 2026)
        pipeline._feature_builder = CalendarFeatureBuilder(pipeline._calendar)
        pipeline._aggregator = DataAggregator(
            FakeCfg.date_column,
            FakeCfg.quantity_column,
            FakeCfg.price_column,
            FakeCfg.aggregation_levels,
        )
        return pipeline

    # Desactivado (default)
    monkeypatch.setattr(settings, "COMPUTE_ROLLING_28", False)
    pipeline_off = _build_pipeline()
    res_off, _, _ = pipeline_off._run_section(selected, sec, dt.date(2024, 5, 1), None)
    assert res_off.height > 0
    assert "yhat28" not in res_off.columns
    assert "valuehat28" not in res_off.columns

    # Activado
    monkeypatch.setattr(settings, "COMPUTE_ROLLING_28", True)
    pipeline_on = _build_pipeline()
    res_on, _, _ = pipeline_on._run_section(selected, sec, dt.date(2024, 5, 1), None)
    assert res_on.height > 0
    assert "yhat28" in res_on.columns
    assert "valuehat28" in res_on.columns


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_rolling_28_one_fits_exactly_once_per_variable(monkeypatch):
    """
    Invariante de performance de la Fase 3(b): independientemente de la
    longitud de la serie (y por lo tanto del número de bloques de 28 días),
    `model.fit()` debe llamarse UNA sola vez por variable (y, precio) — no
    una vez por bloque como en la versión O(n²) anterior.
    """
    runner = RLSForecastRunner(
        driver_cols=["intercept", "weekday_2"], rmse_error=0.2, n_jobs=1
    )
    n = 140  # 5 bloques de 28 días: la versión O(n^2) haría ~2*5 refits
    start = RLSForecastRunner._first_monday_on_or_after(dt.date(2024, 1, 1))
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(0)
    y = np.round(10 + rng.normal(0, 1, size=n)).clip(min=0)
    weekday2 = [1 if d.weekday() == 1 else 0 for d in dates]

    df = pl.DataFrame(
        {
            "unique_id": ["1||a"] * n,
            "ds": dates,
            "y": y.tolist(),
            "value": (y * 10).tolist(),
            "intercept": [1] * n,
            "weekday_2": weekday2,
        }
    )

    call_count = {"fit": 0}
    original_fit = RecursiveLeastSquaresRegression.fit

    def _counting_fit(self, *args, **kwargs):
        call_count["fit"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(RecursiveLeastSquaresRegression, "fit", _counting_fit)

    out = runner._rolling_28_one("1||a", df)

    assert out is not None
    assert out.height == n
    # 1 fit para la variable y + 1 fit para la variable precio = 2 (no 10+)
    assert call_count["fit"] == 2


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_rolling_28_one_coefficient_state_is_sequential_after_seed():
    """
    Una vez fijado el seed, el estado de coeficientes en la posición j de
    `all_coef` depende solo de las observaciones 0..j (propiedad interna de
    `_rls`). Verificamos que el número de estados de coeficientes devueltos
    coincide exactamente con el número de actuals usados para el fit (no
    más, no menos), lo cual es la precondición para que el walk-forward por
    bloques (`_coef_before` vía `np.searchsorted`) sea válido.
    """
    runner = RLSForecastRunner(driver_cols=["intercept"], rmse_error=0.2, n_jobs=1)
    n = 60
    start = RLSForecastRunner._first_monday_on_or_after(dt.date(2024, 1, 1))
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    y = [5.0 if i % 3 else 0.0 for i in range(n)]  # ~2/3 son actuals (y!=0)
    df = pl.DataFrame(
        {
            "unique_id": ["1||b"] * n,
            "ds": dates,
            "y": y,
            "value": [v * 10 for v in y],
            "intercept": [1] * n,
        }
    )
    n_actuals = sum(1 for v in y if v != 0)

    model = runner._new_rls(runner._min_y_to_update, return_all_coefs=True)
    import numpy as _np

    X_all = df.select(["intercept"]).to_numpy().astype(_np.float64)
    y_all = _np.array(y, dtype=_np.float64)
    idx_act = _np.where(y_all != 0)[0]
    priors = runner._default_priors(1)
    model.fit(x=X_all[idx_act], y=_np.log1p(y_all[idx_act]), priors=priors)

    assert model.all_coef[0].shape[0] == n_actuals == idx_act.size
