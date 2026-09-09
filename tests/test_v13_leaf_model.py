"""Pruebas del único modelo productivo SKU+Tienda de v13."""
from __future__ import annotations

import datetime as dt
import math

import polars as pl

from app.forecasting.leaf_ses_rls import _positive_median_levels, build_leaf_forecasts


def _leaf_rows(start: dt.date, n: int, *, uid: str = "1||T:00154||S:455436") -> pl.DataFrame:
    rows = []
    for i in range(n):
        y = 10.0 if i < 28 else 20.0
        # pico deliberado: la mediana no debe desplazarse.
        if i == 5:
            y = 500.0
        rows.append({
            "unique_id": uid,
            "ds": start + dt.timedelta(days=i),
            "y": y,
            "value": 10.0 * y,
            "seccion": "1",
            "store_name": "HIPERPREC",
            "sku_desc": "DULCE CREMA DE LECHE 250G CONA",
        })
    return pl.DataFrame(rows)


def _parent_rows(start: dt.date, end: dt.date) -> pl.DataFrame:
    rows = []
    d = start
    while d <= end:
        period = "in_sample" if d < dt.date(2024, 4, 1) else "out_sample"
        # forecast-only se identifica por fecha más adelante en el fixture real;
        # para el cálculo de calidad solo importan los bloques observados.
        for uid, yhat, effect in [
            ("1", 70.0, math.log(1.02)),
            ("1||T:00154", 95.0, math.log(1.10)),
        ]:
            rows.append({
                "unique_id": uid,
                "ds": d,
                "y": 100.0,
                "yhat": yhat,
                "value": 1000.0,
                "valuehat": 10.0 * yhat,
                "period_type": period,
                "rls_metric_eligible": True,
                "driver_effect": effect,
                "driver_effect_value": effect,
            })
        d += dt.timedelta(days=1)
    return pl.DataFrame(rows)


def test_positive_initial_level_is_median_not_calendar_mean():
    start = dt.date(2024, 1, 1)
    uid = "1||T:00154||S:455436"
    df = pl.DataFrame({
        "unique_id": [uid] * 6,
        "ds": [start + dt.timedelta(days=i) for i in range(6)],
        "y": [0.0, 10.0, 10.0, 500.0, 10.0, 0.0],
        "value": [0.0, 100.0, 100.0, 5000.0, 100.0, 0.0],
    })
    got = _positive_median_levels(df, warmup_days=28).row(0, named=True)
    assert got["initial_level_y"] == 10.0
    assert got["initial_level_value"] == 100.0
    assert got["warmup_positive_days"] == 4


def test_same_ses_rls_family_generates_insample_oos_and_forecast_only():
    start = dt.date(2024, 1, 1)
    train = _leaf_rows(start, 84)
    test_start = start + dt.timedelta(days=84)
    test_end = test_start + dt.timedelta(days=27)
    fc_start = test_end + dt.timedelta(days=1)
    fc_end = fc_start + dt.timedelta(days=27)
    oos = _leaf_rows(test_start, 28)
    parents = _parent_rows(start, fc_end)
    parents = parents.with_columns(
        pl.when(pl.col("ds") >= fc_start)
        .then(pl.lit("forecast_only"))
        .when(pl.col("ds") >= test_start)
        .then(pl.lit("out_sample"))
        .otherwise(pl.lit("in_sample"))
        .alias("period_type"),
        pl.when(pl.col("ds") >= fc_start).then(False).otherwise(pl.col("rls_metric_eligible")).alias("rls_metric_eligible"),
    )
    hz = {
        "train_start": start,
        "train_end": start + dt.timedelta(days=83),
        "test_start": test_start,
        "test_end": test_end,
        "forecast_start": fc_start,
        "forecast_end": fc_end,
    }
    out = build_leaf_forecasts(
        train_leaves=train,
        oos_leaves=oos,
        parent_forecasts=parents,
        section_id="1",
        horizons=hz,
    )
    assert set(out["period_type"].unique().to_list()) == {"in_sample", "out_sample", "forecast_only"}
    post = out.filter(pl.col("ds") > pl.col("leaf_warmup_end"))
    assert post.height > 0
    assert post.filter(~pl.col("modelo_seleccionado").str.starts_with("SES+RLS(")).height == 0
    assert post.filter(~pl.col("parent_model_y").is_in(["store", "section"])).height == 0
    # La misma identidad algebraica se usa en todos los períodos.
    ident = ((pl.col("ses_level_y").log1p() + pl.col("driver_effect")).exp() - 1.0).clip(lower_bound=0.0)
    max_diff = post.select((pl.col("yhat_raw") - ident).abs().max()).item()
    assert float(max_diff or 0.0) < 1e-8


def test_negative_parent_effect_respects_nonnegative_raw_identity():
    """La identidad auditada incluye el piso físico 0 antes del redondeo."""
    start = dt.date(2024, 1, 1)
    train = _leaf_rows(start, 84)
    test_start = start + dt.timedelta(days=84)
    test_end = test_start + dt.timedelta(days=27)
    fc_start = test_end + dt.timedelta(days=1)
    fc_end = fc_start + dt.timedelta(days=27)
    oos = _leaf_rows(test_start, 28)

    # Un efecto muy negativo hace que la transformación algebraica sin piso
    # sea negativa para niveles leaf pequeños. Producción debe almacenar 0.
    parents = _parent_rows(start, fc_end).with_columns(
        pl.lit(math.log(0.01)).alias("driver_effect"),
        pl.lit(math.log(0.01)).alias("driver_effect_value"),
        pl.when(pl.col("ds") >= fc_start).then(pl.lit("forecast_only"))
        .when(pl.col("ds") >= test_start).then(pl.lit("out_sample"))
        .otherwise(pl.lit("in_sample")).alias("period_type"),
        pl.when(pl.col("ds") >= fc_start).then(False).otherwise(True).alias("rls_metric_eligible"),
    )
    hz = {
        "train_start": start,
        "train_end": start + dt.timedelta(days=83),
        "test_start": test_start,
        "test_end": test_end,
        "forecast_start": fc_start,
        "forecast_end": fc_end,
    }
    out = build_leaf_forecasts(
        train_leaves=train, oos_leaves=oos, parent_forecasts=parents,
        section_id="1", horizons=hz,
    )
    post = out.filter(pl.col("ds") > pl.col("leaf_warmup_end"))
    assert post.filter(pl.col("yhat_raw") == 0.0).height > 0
    ident = (
        (pl.col("ses_level_y").clip(lower_bound=0.0).log1p() + pl.col("driver_effect"))
        .exp()
        .sub(1.0)
        .clip(lower_bound=0.0)
    )
    max_diff = post.select((pl.col("yhat_raw") - ident).abs().max()).item()
    assert float(max_diff or 0.0) < 1e-8


def test_ses_state_uses_history_beyond_warmup():
    start = dt.date(2024, 1, 1)
    train = _leaf_rows(start, 84)
    test_start = start + dt.timedelta(days=84)
    test_end = test_start + dt.timedelta(days=27)
    fc_start = test_end + dt.timedelta(days=1)
    fc_end = fc_start + dt.timedelta(days=27)
    oos = _leaf_rows(test_start, 28)
    parents = _parent_rows(start, fc_end).with_columns(
        pl.when(pl.col("ds") >= fc_start).then(pl.lit("forecast_only"))
        .when(pl.col("ds") >= test_start).then(pl.lit("out_sample"))
        .otherwise(pl.lit("in_sample")).alias("period_type"),
        pl.when(pl.col("ds") >= fc_start).then(False).otherwise(True).alias("rls_metric_eligible"),
    )
    hz = {
        "train_start": start,
        "train_end": start + dt.timedelta(days=83),
        "test_start": test_start,
        "test_end": test_end,
        "forecast_start": fc_start,
        "forecast_end": fc_end,
    }
    out = build_leaf_forecasts(
        train_leaves=train, oos_leaves=oos, parent_forecasts=parents,
        section_id="1", horizons=hz,
    )
    init = float(out["initial_level_y"][0])
    oos_level = float(out.filter(pl.col("period_type") == "out_sample")["ses_level_y"][0])
    assert init == 10.0
    assert oos_level > init


def test_parent_selection_uses_prior_positive_wmape_and_prefers_better_store():
    start = dt.date(2024, 1, 1)
    train = _leaf_rows(start, 84)
    test_start = start + dt.timedelta(days=84)
    test_end = test_start + dt.timedelta(days=27)
    fc_start = test_end + dt.timedelta(days=1)
    fc_end = fc_start + dt.timedelta(days=27)
    oos = _leaf_rows(test_start, 28)
    parents = _parent_rows(start, fc_end).with_columns(
        pl.when(pl.col("ds") >= fc_start).then(pl.lit("forecast_only"))
        .when(pl.col("ds") >= test_start).then(pl.lit("out_sample"))
        .otherwise(pl.lit("in_sample")).alias("period_type"),
        pl.when(pl.col("ds") >= fc_start).then(False).otherwise(True).alias("rls_metric_eligible"),
    )
    hz = {
        "train_start": start, "train_end": start + dt.timedelta(days=83),
        "test_start": test_start, "test_end": test_end,
        "forecast_start": fc_start, "forecast_end": fc_end,
    }
    out = build_leaf_forecasts(
        train_leaves=train, oos_leaves=oos, parent_forecasts=parents,
        section_id="1", horizons=hz,
    )
    # Después de existir historia cerrada, tienda tiene wMAPE 5% y sección 30%.
    scored = out.filter((pl.col("ds") > pl.col("leaf_warmup_end")) & pl.col("parent_wmape_y").is_not_null())
    assert scored.height > 0
    assert scored.filter(pl.col("parent_model_y") != "store").height == 0
