"""
Capa de datos del dashboard: prepara un ViewModel listo para renderizar.
Sin Streamlit ni Plotly.

Modelo de filtros (reemplaza la navegación jerárquica anterior): Sección es
obligatoria; Tienda y SKU son dos filtros INDEPENDIENTES al mismo nivel,
cualquiera puede estar vacío, uno solo, o ambos a la vez. El nodo actual se
resuelve vía `settings.make_unique_id(seccion, store, sku)`.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
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
    seccion: str
    store: str | None
    sku: str | None
    node_kind: str  # "seccion" | "tienda" | "sku" | "tienda_sku"
    label: str
    unidad: str
    freq: str
    horizons: dict[str, dt.date | None]
    ranking_tiendas: pl.DataFrame
    ranking_skus: pl.DataFrame
    n_spine: int
    metrics: dict[str, dict[str, float | int]]
    metrics_28: dict[str, float | int]
    has_rolling28: bool
    store_context: str | None
    chart: dict[str, Any]
    detail: pl.DataFrame
    ds_min: dt.date | None
    ds_max: dt.date | None
    cutoff: dt.date
    has_value_cols: bool = False


def _node_kind(store: str | None, sku: str | None) -> str:
    if store is not None and sku is not None:
        return "tienda_sku"
    if sku is not None:
        return "sku"
    if store is not None:
        return "tienda"
    return "seccion"


def prepare_dashboard_state(
    res_df: pl.DataFrame,
    *,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
    all_ids: list[str] | None = None,
    label_map: dict[str, str] | None = None,
    desc_map: dict[str, str] | None = None,
) -> DashboardView:
    """
    Calcula rankings, métricas, series de gráfico y detalle a partir del
    DataFrame de forecasts y la selección de filtros (sección obligatoria,
    tienda y sku independientes y opcionales).
    """
    selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
    node_kind = _node_kind(store, sku)

    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    has_period = "period_type" in cols

    unit_df = backend.prepare_unit_df(res_df, unidad, has_value)
    if label_map is None or desc_map is None:
        label_map, desc_map = backend.build_label_maps(unit_df)
    if all_ids is None:
        all_ids = res_df["unique_id"].unique().to_list()

    df_daily = backend.filter_series(unit_df, selected_id)
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

    # N puntos totales = longitud del spine de la sección (settings), igual
    # para todos los nodos de esa sección.
    hz_spine = settings.section_horizons(seccion)
    n_spine = (hz_spine["forecast_end"] - hz_spine["train_start"]).days + 1
    candidatos = [
        uid for uid in all_ids if uid == seccion or uid.startswith(f"{seccion}||")
    ]
    n_data = backend.spine_n_fechas(unit_df, candidatos)
    if n_data > n_spine:
        n_spine = n_data

    tabla_base = backend.wmape_por_id(candidatos, unit_df, n_fechas_spine=n_spine)

    # Ranking de tiendas: si hay SKU elegido, compara tienda+sku entre
    # tiendas para ese SKU; si no, compara tiendas puras.
    ranking_tiendas = backend.ranking_table(
        tabla_base,
        seccion=seccion,
        axis="store",
        fixed_peer=sku,
        exclude=store,
        desc_map=desc_map,
        unidad=unidad,
    )
    # Ranking de SKU: si hay tienda elegida, compara tienda+sku entre SKU
    # para esa tienda; si no, compara SKU puros (todas las tiendas).
    ranking_skus = backend.ranking_table(
        tabla_base,
        seccion=seccion,
        axis="sku",
        fixed_peer=store,
        exclude=sku,
        desc_map=desc_map,
        unidad=unidad,
    )

    metrics = backend.metrics_in_out_total(
        df_view, cutoff, test_end or cutoff
    )
    metrics_28 = backend.metrics_rolling28(df_view)

    # Rolling28 es opcional y (desde el modelo jerárquico RLS-sección) solo
    # se calcula para el nodo de sección. Se detecta por presencia de datos
    # no-nulos en el nodo ACTUALMENTE seleccionado, no solo por la columna
    # existir en el dataset.
    has_rolling28 = (
        "yhat28" in df_daily.columns
        and df_daily.filter(pl.col("yhat28").is_not_null()).height > 0
    )

    # Contexto de tienda: solo cuando AMBOS filtros (tienda y sku) están
    # activos a la vez — un SKU sin tienda es, por diseño, un agregado
    # cross-tienda y no tiene "una" tienda que mostrar.
    store_context: str | None = None
    if store is not None and sku is not None:
        store_only_id = settings.make_unique_id(seccion, store=store)
        store_context = (label_map or {}).get(
            store_only_id, settings.display_label(store_only_id)
        )

    chart = backend.build_chart_series(
        df_view,
        test_end or cutoff,
        fcst_start or (cutoff + dt.timedelta(days=1)),
        fcst_end or (cutoff + dt.timedelta(days=28)),
        cutoff,
        has_period,
    )

    detail = backend.format_detail_display(backend.detail_view(df_view))
    label = label_map.get(selected_id, settings.display_label(selected_id))

    return DashboardView(
        selected_id=selected_id,
        seccion=seccion,
        store=store,
        sku=sku,
        node_kind=node_kind,
        label=label,
        unidad=unidad,
        freq=freq,
        horizons=horizons,
        ranking_tiendas=ranking_tiendas,
        ranking_skus=ranking_skus,
        n_spine=n_spine,
        metrics=metrics,
        metrics_28=metrics_28,
        has_rolling28=has_rolling28,
        store_context=store_context,
        chart=chart,
        detail=detail,
        ds_min=ds_min,
        ds_max=ds_max,
        cutoff=cutoff,
        has_value_cols=has_value,
    )
