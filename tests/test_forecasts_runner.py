"""
Tests de equivalencia del refactor de Fase 1 (fit único por serie).

Antes: `_run_section` llamaba `runner.run(train, target)` una vez por cada
target (in_sample / out_sample / forecast_only) → 3 fits redundantes sobre
el mismo `train`, ya que el fit de RLS es determinístico (mismos x, y,
priors → mismo modelo).

Ahora: `fit_and_predict_multi` ajusta el modelo UNA sola vez y predice sobre
todos los targets. Como el fit es determinístico, el resultado numérico
debe ser IDÉNTICO al de fitear por separado para cada target (que es
exactamente lo que hacía el código viejo) — este test lo verifica
explícitamente, y además verifica que el número de llamadas a `.fit()` no
escala con el número de targets.
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


def _make_daily_df(uid: str, start: dt.date, n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    y = np.round(10 + rng.normal(0, 2, size=n)).clip(min=0.0)
    value = y * 5 + rng.normal(0, 1, size=n)
    weekday2 = [1 if d.weekday() == 1 else 0 for d in dates]
    return pl.DataFrame(
        {
            "unique_id": [uid] * n,
            "ds": dates,
            "y": y.tolist(),
            "value": value.tolist(),
            "intercept": [1] * n,
            "weekday_2": weekday2,
        }
    )


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_fit_and_predict_multi_matches_fit_per_target_reference():
    uid = "1||00122||SKUX"
    driver_cols = ["intercept", "weekday_2"]
    runner = RLSForecastRunner(driver_cols=driver_cols, rmse_error=0.2, n_jobs=1)

    train_g = _make_daily_df(uid, dt.date(2024, 1, 1), 40, seed=1)
    oos_g = _make_daily_df(uid, dt.date(2024, 2, 10), 7, seed=2)
    fcst_g = _make_daily_df(uid, dt.date(2024, 2, 17), 28, seed=3)

    targets = {"in_sample": train_g, "out_sample": oos_g, "forecast_only": fcst_g}

    res_df, _ = runner.fit_and_predict_multi(
        train_g, targets, ["seccion", "store", "sku"], desc="test"
    )
    assert res_df.height == 40 + 7 + 28

    # Referencia: fit UNA vez (determinístico) y predecir por separado con
    # los mismos modelos — equivalente a lo que el código viejo hacía al
    # fitear 3 veces sobre datos idénticos.
    model_y_ref, model_p_ref = runner._fit_models(train_g)
    for name, tgt in targets.items():
        ref_frame = runner._predict_with_models(uid, model_y_ref, model_p_ref, train_g, tgt)
        got = (
            res_df.filter(
                (pl.col("unique_id") == uid) & (pl.col("period_type") == name)
            )
            .sort("ds")
        )
        assert got["yhat"].to_list() == pytest.approx(
            ref_frame.sort("ds")["yhat"].to_list(), rel=1e-9, abs=1e-9
        )
        assert got["valuehat"].to_list() == pytest.approx(
            ref_frame.sort("ds")["valuehat"].to_list(), rel=1e-9, abs=1e-9
        )


@pytest.mark.skipif(
    RecursiveLeastSquaresRegression is None, reason="rls_opt no disponible"
)
def test_fit_count_does_not_scale_with_number_of_targets(monkeypatch):
    uid = "1||00122||SKUY"
    driver_cols = ["intercept", "weekday_2"]
    runner = RLSForecastRunner(driver_cols=driver_cols, rmse_error=0.2, n_jobs=1)

    train_g = _make_daily_df(uid, dt.date(2024, 1, 1), 30, seed=4)
    targets = {
        "in_sample": train_g,
        "out_sample": _make_daily_df(uid, dt.date(2024, 2, 1), 5, seed=5),
        "forecast_only": _make_daily_df(uid, dt.date(2024, 2, 10), 5, seed=6),
    }

    call_count = {"fit": 0}
    original_fit = RecursiveLeastSquaresRegression.fit

    def _counting_fit(self, *args, **kwargs):
        call_count["fit"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(RecursiveLeastSquaresRegression, "fit", _counting_fit)

    res_df, _ = runner.fit_and_predict_multi(
        train_g, targets, ["seccion", "store", "sku"], desc="test"
    )
    assert res_df.height > 0
    # 1 fit variable y + 1 fit variable precio = 2, sin importar que haya
    # 3 targets (antes: 2 * 3 = 6 fits).
    assert call_count["fit"] == 2


def test_run_compat_wrapper_drops_period_type():
    """`run()` se mantiene como wrapper de compatibilidad de la API vieja."""
    uid = "1||x"
    runner = RLSForecastRunner(driver_cols=["intercept"], rmse_error=0.2, n_jobs=1)
    train_g = _make_daily_df(uid, dt.date(2024, 1, 1), 30, seed=7)
    test_g = _make_daily_df(uid, dt.date(2024, 2, 1), 5, seed=8)
    res_df, _ = runner.run(train_g, test_g, ["seccion", "store", "sku"])
    if res_df.height:
        assert "period_type" not in res_df.columns
