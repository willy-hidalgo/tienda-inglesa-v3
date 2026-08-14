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


def _aggregate_pure_sku(
    unit_df: pl.DataFrame, seccion: str, sku: str
) -> pl.DataFrame:
    """Serie diaria de un SKU puro = suma de hojas sku+tienda de ese SKU.

    El pipeline no materializa nodos «SKU puro»; el dashboard los sintetiza
    al vuelo cuando el usuario elige SKU sin tienda.
    """
    prefix = f"{seccion}||"
    suffix = f"||S:{sku}"
    leaves = unit_df.filter(
        pl.col("unique_id").str.starts_with(prefix)
        & pl.col("unique_id").str.ends_with(suffix)
        & pl.col("unique_id").str.contains(r"\|\|T:", literal=False)
    )
    if leaves.height == 0:
        return leaves
    aggs = [pl.col("y").sum(), pl.col("yhat").sum()]
    if "yhat28" in leaves.columns:
        aggs.append(pl.col("yhat28").sum())
    if "value" in leaves.columns:
        aggs.append(pl.col("value").sum())
    if "valuehat" in leaves.columns:
        aggs.append(pl.col("valuehat").sum())
    if "valuehat28" in leaves.columns:
        aggs.append(pl.col("valuehat28").sum())
    extra = [
        c
        for c in (
            "period_type",
            "sku_desc",
            "store_name",
            "seccion",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "forecast_start",
            "forecast_end",
        )
        if c in leaves.columns
    ]
    for c in extra:
        aggs.append(pl.col(c).first())
    out = (
        leaves.group_by("ds")
        .agg(aggs)
        .with_columns(pl.lit(settings.make_unique_id(seccion, sku=sku)).alias("unique_id"))
        .sort("ds")
    )
    return out


class DashboardService:
    """Servicio/Façade que prepara el ViewModel del dashboard.

    Encapsula el "orquestador" de backend + dashboard_data detrás de un único
    punto de entrada con inyección de dependencias:

      - `res_df` / `unidad` / `freq` / `cutoff_date` se inyectan en el
        constructor (contexto global del panel cargado);
      - `prepare(seccion, store, sku, ...)` recibe la selección del usuario y
        los artefactos cacheados (label/desc maps, `tabla_base`, `n_spine`).

    Patrones aplicados:
      - **Façade**: una sola API (`prepare`) frente a las múltiples funciones
        de `backend` (agregación temporal, wmape, rankings, métricas, chart…).
      - **Service Layer / DI**: las dependencias no se resuelven dentro del
        método; se pasan desde afuera, lo que lo hace testeable y desacoplado
        de Streamlit.
    """

    def __init__(
        self,
        res_df: pl.DataFrame,
        *,
        unidad: str,
        freq: str,
        cutoff_date: dt.date,
    ):
        self._res_df = res_df
        self._unidad = unidad
        self._freq = freq
        self._cutoff_date = cutoff_date
        cols = set(res_df.columns)
        self._has_value = "value" in cols and "valuehat" in cols
        self._has_period = "period_type" in cols
        self._unit_df = backend.prepare_unit_df(res_df, unidad, self._has_value)
        # Derivados globales calculados una sola vez y reutilizados entre
        # preparaciones del mismo panel (misma instancia del servicio).
        self._label_map: dict[str, str] | None = None
        self._desc_map: dict[str, str] | None = None
        self._all_ids: list[str] | None = None

    def labels(self) -> tuple[dict[str, str], dict[str, str]]:
        """Label/desc maps del panel (cálculo perezoso y cacheado)."""
        if self._label_map is None or self._desc_map is None:
            self._label_map, self._desc_map = backend.build_label_maps(self._unit_df)
        return self._label_map, self._desc_map

    def all_ids(self) -> list[str]:
        """Unique_ids del panel (cálculo perezoso y cacheado)."""
        if self._all_ids is None:
            self._all_ids = self._res_df["unique_id"].unique().to_list()
        return self._all_ids

    def prepare(
        self,
        seccion: str,
        store: str | None,
        sku: str | None,
        *,
        all_ids: list[str] | None = None,
        label_map: dict[str, str] | None = None,
        desc_map: dict[str, str] | None = None,
        tabla_base: pl.DataFrame | None = None,
        n_spine: int | None = None,
    ) -> DashboardView:
        """Construye el `DashboardView` para la selección (sección obligatoria,
        tienda/sku independientes y opcionales)."""
        selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
        node_kind = _node_kind(store, sku)

        if label_map is None or desc_map is None:
            label_map, desc_map = self.labels()
        if all_ids is None:
            all_ids = self.all_ids()

        # SKU puro (sin tienda): el parquet no tiene esa serie → agregar hojas.
        if store is None and sku is not None:
            df_daily = _aggregate_pure_sku(self._unit_df, seccion, sku)
        else:
            df_daily = backend.filter_series(self._unit_df, selected_id)

        # Si aún no hay filas (nodo inexistente), horizons caen a settings.
        horizons = backend.resolve_horizons(
            df_daily, seccion, set(self._res_df.columns)
        )

        train_end = horizons["train_end"]
        test_end = horizons["test_end"]
        fcst_start = horizons["forecast_start"]
        fcst_end = horizons["forecast_end"]

        ds_min, ds_max = backend.ds_range(df_daily)
        cutoff = self._cutoff_date
        if train_end and ds_min and ds_max and not (ds_min <= cutoff <= ds_max):
            cutoff = ds_min

        df_view = backend.aggregate_temporal(df_daily, self._freq)

        # N puntos totales = longitud del spine de la sección (settings), igual
        # para todos los nodos de esa sección.
        if n_spine is None or tabla_base is None:
            hz_spine = settings.section_horizons(seccion)
            n_spine_calc = (hz_spine["forecast_end"] - hz_spine["train_start"]).days + 1
            candidatos = [
                uid
                for uid in all_ids
                if uid == seccion or uid.startswith(f"{seccion}||")
            ]
            n_data = backend.spine_n_fechas(self._unit_df, candidatos)
            if n_data > n_spine_calc:
                n_spine_calc = n_data
            if n_spine is None:
                n_spine = n_spine_calc
            if tabla_base is None:
                tabla_base = backend.wmape_por_id(
                    candidatos, self._unit_df, n_fechas_spine=n_spine
                )

        # Ranking de tiendas: si hay SKU elegido, compara tienda+sku entre
        # tiendas para ese SKU; si no, compara tiendas puras.
        ranking_tiendas = backend.ranking_table(
            tabla_base,
            seccion=seccion,
            axis="store",
            fixed_peer=sku,
            exclude=store,
            desc_map=desc_map,
            unidad=self._unidad,
        )
        # Ranking de SKU: si hay tienda elegida, hojas de esa tienda; si no,
        # agrega hojas por SKU (no existen nodos SKU puro en el parquet).
        ranking_skus = backend.ranking_table(
            tabla_base,
            seccion=seccion,
            axis="sku",
            fixed_peer=store,
            exclude=sku,
            desc_map=desc_map,
            unidad=self._unidad,
        )

        metrics = backend.metrics_in_out_total(df_view, cutoff, test_end or cutoff)
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
            self._has_period,
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
            unidad=self._unidad,
            freq=self._freq,
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
            has_value_cols=self._has_value,
        )


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
    tabla_base: pl.DataFrame | None = None,
    n_spine: int | None = None,
) -> DashboardView:
    """
    Calcula rankings, métricas, series de gráfico y detalle a partir del
    DataFrame de forecasts y la selección de filtros (sección obligatoria,
    tienda y sku independientes y opcionales).

    Función de conveniencia (backward-compatible) que delega en
    `DashboardService` — la fachada orientada a objetos.

    `tabla_base` / `n_spine` opcionales: si vienen precalculados a nivel
    sección (cache), se reutilizan y se evita el coste de wmape_por_id en
    cada cambio de tienda/SKU.
    """
    service = DashboardService(
        res_df, unidad=unidad, freq=freq, cutoff_date=cutoff_date
    )
    return service.prepare(
        seccion,
        store,
        sku,
        all_ids=all_ids,
        label_map=label_map,
        desc_map=desc_map,
        tabla_base=tabla_base,
        n_spine=n_spine,
    )
