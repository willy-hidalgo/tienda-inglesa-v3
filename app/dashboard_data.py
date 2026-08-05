"""
Capa de datos del dashboard: prepara un ViewModel listo para renderizar.
Sin Streamlit ni Plotly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import polars as pl

try:
    from app import backend
except ImportError:  # pragma: no cover
    import backend  # type: ignore

import settings


@dataclass
class DashboardView:
    """Todo lo que el dashboard necesita pintar."""

    selected_id: str
    label: str
    seccion: str
    unidad: str
    freq: str
    horizons: dict[str, dt.date | None]
    ranking_wmape: pl.DataFrame
    ranking_rotacion: pl.DataFrame
    ranking_n_series: int
    metrics: dict[str, dict[str, float | int]]
    chart: dict[str, Any]
    detail: pl.DataFrame
    ds_min: dt.date | None
    ds_max: dt.date | None
    cutoff: dt.date
    nombre_nivel: str
    has_value_cols: bool = False


def prepare_dashboard_state(
    res_df: pl.DataFrame,
    *,
    unidad: str,
    freq: str,
    selected_id: str,
    cutoff_date: dt.date,
    candidatos: list[str],
    nombre_nivel: str,
    label_map: dict[str, str] | None = None,
    desc_map: dict[str, str] | None = None,
) -> DashboardView:
    """
    Calcula rankings, métricas, series de gráfico y detalle
    a partir del DataFrame de forecasts y la selección de UI.
    """
    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    has_period = "period_type" in cols

    unit_df = backend.prepare_unit_df(res_df, unidad, has_value)
    if label_map is None or desc_map is None:
        label_map, desc_map = backend.build_label_maps(unit_df)

    df_daily = backend.filter_series(unit_df, selected_id)
    seccion = selected_id.split("||")[0]
    horizons = backend.resolve_horizons(df_daily, seccion, cols)

    train_end = horizons["train_end"]
    test_end = horizons["test_end"]
    fcst_start = horizons["forecast_start"]
    fcst_end = horizons["forecast_end"]

    ds_min, ds_max = backend.ds_range(df_daily)
    cutoff = cutoff_date
    if train_end and ds_min and ds_max and not (ds_min <= cutoff <= ds_max):
        cutoff = ds_min

    df_view = backend.aggregate_temporal(df_daily, freq)

    tabla_base = backend.wmape_por_id(candidatos, unit_df)
    ranking_wmape = backend.ranking_wmape_table(tabla_base, selected_id, desc_map)
    ranking_rot = backend.ranking_rotacion_table(
        tabla_base, selected_id, desc_map, unidad
    )

    metrics = backend.metrics_in_out_total(
        df_view, cutoff, test_end or cutoff
    )

    chart = backend.build_chart_series(
        df_view,
        test_end or cutoff,
        fcst_start or (cutoff + dt.timedelta(days=1)),
        fcst_end or (cutoff + dt.timedelta(days=28)),
        cutoff,
        has_period,
    )

    detail = backend.detail_view(df_view)
    label = label_map.get(selected_id, settings.display_label(selected_id))

    return DashboardView(
        selected_id=selected_id,
        label=label,
        seccion=seccion,
        unidad=unidad,
        freq=freq,
        horizons=horizons,
        ranking_wmape=ranking_wmape,
        ranking_rotacion=ranking_rot,
        ranking_n_series=tabla_base.height,
        metrics=metrics,
        chart=chart,
        detail=detail,
        ds_min=ds_min,
        ds_max=ds_max,
        cutoff=cutoff,
        nombre_nivel=nombre_nivel,
        has_value_cols=has_value,
    )
