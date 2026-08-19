"""
Capa de datos del dashboard: prepara un ViewModel listo para renderizar.
Sin Streamlit ni Plotly.

Modo rápido (artefactos precalculados):
  - rankings y métricas salen de metrics.parquet
  - series del gráfico salen de series.parquet filtrado por unique_id
  - cero wmape_por_id / scans del panel completo en el hot path

Modo legacy (sin artefactos): mantiene prepare_dashboard_state original
sobre el DataFrame completo (lento; solo fallback).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

try:
    from app import backend
    from app import dashboard_artifacts as artifacts
except ImportError:  # pragma: no cover
    import backend  # type: ignore
    import dashboard_artifacts as artifacts  # type: ignore

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


def _parse_date(v: str | None) -> dt.date | None:
    if not v:
        return None
    if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.datetime):
        return v.date()
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def _horizons_from_index(
    index: dict[str, Any], seccion: str, df_daily: pl.DataFrame
) -> dict[str, dt.date | None]:
    raw = (index.get("horizons_by_sec") or {}).get(seccion) or {}
    hz = {k: _parse_date(v) for k, v in raw.items()}
    cols = set(df_daily.columns)
    if df_daily.height:
        resolved = backend.resolve_horizons(df_daily, seccion, cols)
        for k, v in resolved.items():
            if v is not None:
                hz[k] = v
    if not hz.get("train_end"):
        fallback = settings.section_horizons(seccion)
        hz = {
            k: fallback.get(k)
            for k in (
                "train_end",
                "test_start",
                "test_end",
                "forecast_start",
                "forecast_end",
            )
        }
    return hz


def _ranking_from_metrics(
    metrics: pl.DataFrame,
    *,
    seccion: str,
    unidad: str,
    axis: str,
    fixed_peer: str | None,
    exclude: str | None,
    desc_map: dict[str, str],
    n_spine: int,
) -> pl.DataFrame:
    """Ranking barato sobre metrics.parquet (misma semántica que ranking_table)."""
    label_rot = "Rotación ($)" if unidad.startswith("Valor") else "Rotación (unid.)"
    empty = pl.DataFrame(
        schema={
            "Código": pl.Utf8,
            "Descripción": pl.Utf8,
            "wMAPE (%)": pl.Utf8,
            label_rot: pl.Utf8,
            "N puntos": pl.UInt32,
            "% ≠0": pl.Utf8,
            "unique_id": pl.Utf8,
        }
    )
    base = metrics.filter(
        (pl.col("seccion") == seccion) & (pl.col("unidad") == unidad)
    )
    if base.height == 0:
        # La unidad seleccionada es parte del contrato del dashboard.
        # Nunca reutilizar métricas de otra unidad: eso desincroniza ranking,
        # KPIs y gráfico. Si faltan artefactos de la unidad, devolver vacío y
        # obligar a regenerarlos.
        return empty

    # Defensa: metrics mal generados no deben duplicar filas del ranking
    if "unidad" in base.columns:
        base = base.unique(subset=["unique_id", "unidad"], keep="first")
    else:
        base = base.unique(subset=["unique_id"], keep="first")

    if "store" not in base.columns or "sku" not in base.columns:
        uids = base["unique_id"].to_list()
        stores, skus = [], []
        for u in uids:
            p = settings.split_unique_id(u)
            stores.append(p.get("store"))
            skus.append(p.get("sku"))
        base = base.with_columns(
            pl.Series("store", stores),
            pl.Series("sku", skus),
        )

    # Normalizar códigos a Utf8 para comparaciones estables
    if "store" in base.columns:
        base = base.with_columns(pl.col("store").cast(pl.Utf8))
    if "sku" in base.columns:
        base = base.with_columns(pl.col("sku").cast(pl.Utf8))

    if axis == "store":
        # Siempre y solo nodos tienda puros (store presente, sku ausente).
        tabla = base.filter(
            pl.col("store").is_not_null() & pl.col("sku").is_null()
        )
        if exclude is not None:
            tabla = tabla.filter(pl.col("store") != str(exclude))
        code_col = "store"
    else:
        # Ranking SKU
        if fixed_peer is not None:
            # Tienda seleccionada → SOLO hojas sku+tienda de ESA tienda
            peer = str(fixed_peer)
            tabla = base.filter(
                (pl.col("store") == peer) & pl.col("sku").is_not_null()
            )
            if tabla.height == 0:
                # Fallback por patrón de unique_id (por si store no parseó bien)
                needle = f"||T:{peer}||"
                tabla = base.filter(
                    pl.col("unique_id").str.contains(needle, literal=True)
                    & (
                        pl.col("sku").is_not_null()
                        | pl.col("unique_id").str.contains("||S:", literal=True)
                    )
                )
                if tabla.height and (
                    "sku" not in tabla.columns
                    or tabla.filter(pl.col("sku").is_not_null()).height == 0
                ):
                    tabla = tabla.with_columns(
                        pl.col("unique_id")
                        .str.extract(r"\|\|S:([^|]+)", 1)
                        .alias("sku")
                    )
            if exclude is not None:
                tabla = tabla.filter(pl.col("sku") != str(exclude))
            code_col = "sku"
        else:
            # Sin tienda: preferir nodos SKU puro; si no, agregar hojas
            pure = base.filter(
                pl.col("sku").is_not_null() & pl.col("store").is_null()
            )
            if exclude is not None:
                pure = pure.filter(pl.col("sku") != str(exclude))
            if pure.height > 0:
                tabla = pure
                code_col = "sku"
            else:
                leaves = base.filter(
                    pl.col("store").is_not_null() & pl.col("sku").is_not_null()
                )
                if exclude is not None:
                    leaves = leaves.filter(pl.col("sku") != str(exclude))
                if leaves.height == 0:
                    return empty
                tabla = (
                    leaves.group_by("sku")
                    .agg(
                        (pl.col("wmape") * pl.col("sum_y")).sum().alias("_sum_abs"),
                        pl.col("sum_y").sum().alias("sum_y"),
                        pl.col("n_with_sales").sum().alias("n_with_sales"),
                        pl.col("n_points").first().alias("n_points"),
                    )
                    .with_columns(
                        pl.when(pl.col("sum_y") != 0)
                        .then(pl.col("_sum_abs") / pl.col("sum_y"))
                        .otherwise(0.0)
                        .alias("wmape"),
                        pl.concat_str(
                            [pl.lit(seccion), pl.lit("||S:"), pl.col("sku")]
                        ).alias("unique_id"),
                    )
                    .drop("_sum_abs")
                )
                code_col = "sku"

    # Solo filas con rotación y wMAPE > 0; orden ASCENDENTE por wMAPE (mejor primero)
    tabla = tabla.filter((pl.col("sum_y") > 0) & (pl.col("wmape") > 0))
    if tabla.height == 0:
        return empty
    if code_col in tabla.columns:
        # Un código por fila; keep first tras sort ASC = menor wMAPE
        tabla = tabla.sort("wmape", descending=False).unique(
            subset=[code_col], keep="first", maintain_order=True
        )
    else:
        tabla = tabla.sort("wmape", descending=False)

    codes = tabla[code_col].to_list()
    uids2 = tabla["unique_id"].to_list()
    n_with_sales = tabla["n_with_sales"].to_list()
    pct = [
        (float(nw) / float(n_spine) * 100) if n_spine else 0.0
        for nw in n_with_sales
    ]
    descriptions: list[str] = []
    for u, code in zip(uids2, codes):
        d = desc_map.get(u, "")
        if not d and axis == "sku":
            d = next(
                (desc_map[k] for k in desc_map if f"||S:{code}" in k and desc_map[k]),
                "",
            )
        descriptions.append(d)

    return pl.DataFrame(
        {
            "Código": [str(c) for c in codes],
            "Descripción": descriptions,
            "wMAPE (%)": [f"{w * 100:,.2f}" for w in tabla["wmape"].to_list()],
            label_rot: [f"{v:,.2f}" for v in tabla["sum_y"].to_list()],
            "N puntos": [int(x) for x in n_with_sales],
            "% ≠0": [f"{p:,.1f}" for p in pct],
            "unique_id": uids2,
        }
    )


def prepare_dashboard_state_fast(
    *,
    index: dict[str, Any],
    metrics: pl.DataFrame,
    unidad: str,
    freq: str,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff_date: dt.date,
    forecast_path: str | Path | None = None,
    df_daily: pl.DataFrame | None = None,
) -> DashboardView:
    """Path rápido: index + metrics en memoria, 1 serie (pre-cargada o leída)."""
    selected_id = settings.make_unique_id(seccion, store=store, sku=sku)
    node_kind = _node_kind(store, sku)
    label_map: dict[str, str] = index.get("label_map") or {}
    desc_map: dict[str, str] = index.get("desc_map") or {}
    has_value = bool(index.get("has_value"))
    n_spine = int((index.get("n_spine_by_sec") or {}).get(seccion, 0) or 0)

    if df_daily is None:
        fpath = Path(forecast_path) if forecast_path else None
        df_daily = artifacts.load_series(
            selected_id,
            fpath,
            unidad=unidad,
            has_value=has_value,
        )

    horizons = _horizons_from_index(index, seccion, df_daily)
    test_end = horizons.get("test_end")
    fcst_start = horizons.get("forecast_start")
    fcst_end = horizons.get("forecast_end")
    train_end = horizons.get("train_end")

    ds_min, ds_max = backend.ds_range(df_daily)
    cutoff = cutoff_date
    if train_end and ds_min and ds_max and not (ds_min <= cutoff <= ds_max):
        cutoff = ds_min

    df_view = backend.aggregate_temporal(df_daily, freq)
    has_period = "period_type" in df_view.columns

    ranking_tiendas = _ranking_from_metrics(
        metrics,
        seccion=seccion,
        unidad=unidad,
        axis="store",
        fixed_peer=sku,
        exclude=store,
        desc_map=desc_map,
        n_spine=n_spine or 1,
    )
    ranking_skus = _ranking_from_metrics(
        metrics,
        seccion=seccion,
        unidad=unidad,
        axis="sku",
        fixed_peer=store,
        exclude=sku,
        desc_map=desc_map,
        n_spine=n_spine or 1,
    )

    # Métricas in/out/total siempre bottom-up desde hojas del alcance.
    # La serie del gráfico sigue siendo el nodo seleccionado (df_view).
    fpath = Path(forecast_path) if forecast_path else None
    if store is not None and sku is not None:
        # Hoja: la propia serie diaria (bottom-up trivial de una hoja)
        metrics_io = backend.metrics_in_out_total(
            df_daily, cutoff, test_end or cutoff
        )
    else:
        leaves = artifacts.load_leaves_for_scope(
            seccion,
            store=store,
            sku=sku,
            forecast_path=fpath,
            unidad=unidad,
            has_value=has_value,
        )
        if leaves.height:
            metrics_io = backend.metrics_in_out_bottom_up(
                leaves,
                seccion=seccion,
                store=store,
                sku=sku,
                cutoff=cutoff,
                test_end=test_end or cutoff,
            )
        else:
            metrics_io = backend.metrics_in_out_total(
                df_view, cutoff, test_end or cutoff
            )
    metrics_28 = backend.metrics_rolling28(df_view)

    has_rolling28 = (
        "yhat28" in df_daily.columns
        and df_daily.filter(pl.col("yhat28").is_not_null()).height > 0
    )

    store_context: str | None = None
    if store is not None and sku is not None:
        store_only_id = settings.make_unique_id(seccion, store=store)
        store_context = label_map.get(
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
        metrics=metrics_io,
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


def _aggregate_pure_sku(
    unit_df: pl.DataFrame, seccion: str, sku: str
) -> pl.DataFrame:
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
    return (
        leaves.group_by("ds")
        .agg(aggs)
        .with_columns(
            pl.lit(settings.make_unique_id(seccion, sku=sku)).alias("unique_id")
        )
        .sort("ds")
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
    """Legacy: cálculos sobre el DataFrame completo (solo si no hay artefactos)."""
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

    if store is None and sku is not None:
        df_daily = _aggregate_pure_sku(unit_df, seccion, sku)
    else:
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

    if n_spine is None or tabla_base is None:
        hz_spine = settings.section_horizons(seccion)
        n_spine_calc = (hz_spine["forecast_end"] - hz_spine["train_start"]).days + 1
        candidatos = [
            uid for uid in all_ids if uid == seccion or uid.startswith(f"{seccion}||")
        ]
        n_data = backend.spine_n_fechas(unit_df, candidatos)
        if n_data > n_spine_calc:
            n_spine_calc = n_data
        if n_spine is None:
            n_spine = n_spine_calc
        if tabla_base is None:
            # Rankings reportan wMAPE out-of-sample
            tabla_base = backend.wmape_por_id(
                candidatos,
                unit_df,
                n_fechas_spine=n_spine,
                period_types=["out_sample"],
            )

    ranking_tiendas = backend.ranking_table(
        tabla_base,
        seccion=seccion,
        axis="store",
        fixed_peer=sku,
        exclude=store,
        desc_map=desc_map,
        unidad=unidad,
    )
    ranking_skus = backend.ranking_table(
        tabla_base,
        seccion=seccion,
        axis="sku",
        fixed_peer=store,
        exclude=sku,
        desc_map=desc_map,
        unidad=unidad,
    )

    # Ranking ya es bottom-up vía wmape_por_id. Métricas in/out del nodo:
    # siempre bottom-up desde hojas del alcance seleccionado.
    metrics = backend.metrics_in_out_bottom_up(
        unit_df,
        seccion=seccion,
        store=store,
        sku=sku,
        cutoff=cutoff,
        test_end=test_end or cutoff,
    )
    metrics_28 = backend.metrics_rolling28(df_view)

    has_rolling28 = (
        "yhat28" in df_daily.columns
        and df_daily.filter(pl.col("yhat28").is_not_null()).height > 0
    )

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
