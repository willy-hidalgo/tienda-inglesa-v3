"""
Backend de cálculos para el Forecast Explorer.
El dashboard solo visualiza; aquí viven métricas, rankings y preparaciones.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl

import settings

# Columnas visibles en la tabla de detalle del dashboard
DETAIL_COLUMNS = [
    "unique_id",
    "ds",
    "y",
    "yhat",
    "value",
    "valuehat",
    "period_type",
    "sku_desc",
    "store_name",
    "seccion",
    "abs_error",
]

# Columnas de horizonte que nunca se muestran en detalle
HORIZON_COLUMNS = {
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "forecast_start",
    "forecast_end",
}

PAGE_SIZE = 5


def prepare_unit_df(df: pl.DataFrame, unidad: str, has_price: bool) -> pl.DataFrame:
    if unidad == "Unidades" or not has_price:
        return df
    return df.with_columns(
        pl.col("value").alias("y"),
        pl.col("valuehat").alias("yhat"),
    )


def aggregate_temporal(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    if freq == "Diario" or df.height == 0:
        out = df
        if "abs_error" not in out.columns and "y" in out.columns and "yhat" in out.columns:
            out = out.with_columns(
                (pl.col("y") - pl.col("yhat")).abs().alias("abs_error")
            )
        return out
    period = "1w" if freq == "Semanal" else "1mo"
    aggs = [pl.col("y").sum(), pl.col("yhat").sum()]
    if "value" in df.columns:
        aggs.append(pl.col("value").sum())
    if "valuehat" in df.columns:
        aggs.append(pl.col("valuehat").sum())
    if "period_type" in df.columns:
        aggs.append(pl.col("period_type").max())
    for col in ("sku_desc", "store_name", "seccion", "unique_id"):
        if col in df.columns:
            aggs.append(pl.col(col).first())
    return (
        df.with_columns(pl.col("ds").dt.truncate(period).alias("ds"))
        .group_by("ds")
        .agg(aggs)
        .sort("ds")
        .with_columns((pl.col("y") - pl.col("yhat")).abs().alias("abs_error"))
    )


def filter_series(df: pl.DataFrame, unique_id: str) -> pl.DataFrame:
    return df.filter(pl.col("unique_id") == unique_id).sort("ds")


def opciones_nivel(all_ids: list[str], prefix: str | None) -> list[str]:
    if prefix is None:
        return sorted(i for i in all_ids if "||" not in i)
    depth = prefix.count("||") + 1
    pref = prefix + "||"
    return sorted(
        i for i in all_ids if i.startswith(pref) and i.count("||") == depth
    )


def build_label_maps(df: pl.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    cols = [c for c in ["unique_id", "sku_desc", "store_name"] if c in df.columns]
    meta = df.select(cols).unique(subset=["unique_id"])
    uids = meta["unique_id"].to_list()
    descs = (
        meta["sku_desc"].to_list()
        if "sku_desc" in meta.columns
        else [""] * len(uids)
    )
    snames = (
        meta["store_name"].to_list()
        if "store_name" in meta.columns
        else [""] * len(uids)
    )
    label_map: dict[str, str] = {}
    desc_map: dict[str, str] = {}
    for uid, desc, sname in zip(uids, descs, snames):
        d = desc if desc else None
        s = sname if sname else None
        label_map[uid] = settings.display_label(uid, d, s)
        desc_map[uid] = settings.ranking_description(uid, d, s)
    return label_map, desc_map


def wmape_por_id(ids: list[str], df: pl.DataFrame) -> pl.DataFrame:
    """
    WMAPE = Σ|y−ŷ| / Σ|y|  (excluye y==0 y forecast_only).
    Incluye n_points y sum_y.
    """
    schema = {
        "unique_id": pl.Utf8,
        "wmape": pl.Float64,
        "sum_y": pl.Float64,
        "n_points": pl.UInt32,
    }
    if not ids:
        return pl.DataFrame(schema=schema)
    scored = df.filter(pl.col("unique_id").is_in(ids))
    if "period_type" in scored.columns:
        scored = scored.filter(pl.col("period_type") != "forecast_only")
    scored = scored.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    if scored.height == 0:
        return pl.DataFrame(schema=schema)
    return (
        scored.group_by("unique_id")
        .agg(
            pl.col("y").sum().alias("sum_y"),
            (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_error"),
            pl.len().alias("n_points"),
        )
        .filter(pl.col("sum_y") != 0)
        .with_columns((pl.col("sum_abs_error") / pl.col("sum_y")).alias("wmape"))
        .select(["unique_id", "wmape", "sum_y", "n_points"])
    )


def ranking_wmape_table(
    tabla_base: pl.DataFrame,
    selected_id: str,
    label_map: dict[str, str],
    desc_map: dict[str, str],
) -> pl.DataFrame:
    """Tabla de ranking wMAPE lista para mostrar (todas las filas ordenadas)."""
    tabla = (
        tabla_base.filter(
            (pl.col("wmape") != 0) & (pl.col("unique_id") != selected_id)
        )
        .sort("wmape")
    )
    if tabla.height == 0:
        return pl.DataFrame(
            schema={
                "Código": pl.Utf8,
                "Descripción": pl.Utf8,
                "wMAPE (%)": pl.Utf8,
                "N puntos": pl.UInt32,
                "unique_id": pl.Utf8,
            }
        )
    uids = tabla["unique_id"].to_list()
    return pl.DataFrame(
        {
            "Código": [settings.ranking_code(u) for u in uids],
            "Descripción": [desc_map.get(u, "") for u in uids],
            "wMAPE (%)": [f"{w * 100:,.2f}" for w in tabla["wmape"].to_list()],
            "N puntos": tabla["n_points"].to_list(),
            "unique_id": uids,
        }
    )


def ranking_rotacion_table(
    tabla_base: pl.DataFrame,
    selected_id: str,
    desc_map: dict[str, str],
    unidad: str,
) -> pl.DataFrame:
    label_rot = "Rotación ($)" if unidad.startswith("Valor") else "Rotación (unid.)"
    tabla = (
        tabla_base.filter(pl.col("unique_id") != selected_id)
        .sort("sum_y", descending=True)
    )
    if tabla.height == 0:
        return pl.DataFrame(
            schema={
                "Código": pl.Utf8,
                "Descripción": pl.Utf8,
                label_rot: pl.Utf8,
                "N puntos": pl.UInt32,
                "unique_id": pl.Utf8,
            }
        )
    uids = tabla["unique_id"].to_list()
    return pl.DataFrame(
        {
            "Código": [settings.ranking_code(u) for u in uids],
            "Descripción": [desc_map.get(u, "") for u in uids],
            label_rot: [f"{v:,.2f}" for v in tabla["sum_y"].to_list()],
            "N puntos": tabla["n_points"].to_list(),
            "unique_id": uids,
        }
    )


def calcular_metricas(df: pl.DataFrame) -> tuple[float, float, int]:
    """WMAPE, BIAS (fracción), n_points. Excluye y==0."""
    if df.height == 0:
        return 0.0, 0.0, 0
    scored = df.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    if scored.height == 0:
        return 0.0, 0.0, 0
    if "abs_error" not in scored.columns:
        scored = scored.with_columns(
            (pl.col("y") - pl.col("yhat")).abs().alias("abs_error")
        )
    sum_y = float(scored["y"].sum())
    if sum_y == 0:
        return 0.0, 0.0, 0
    wmape = float(scored["abs_error"].sum()) / abs(sum_y)
    bias = float((scored["yhat"] - scored["y"]).sum()) / sum_y
    return wmape, bias, scored.height


def split_hist_forecast(
    df_view: pl.DataFrame,
    test_end: dt.date,
    forecast_start: dt.date,
    forecast_end: dt.date,
    has_period: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if has_period and "period_type" in df_view.columns:
        hist = df_view.filter(pl.col("period_type") != "forecast_only")
        fcst = df_view.filter(pl.col("period_type") == "forecast_only")
        return hist, fcst
    hist = df_view.filter(
        (pl.col("ds").cast(pl.Date) <= test_end)
        & (pl.col("y").is_not_null())
        & (pl.col("y") != 0)
    )
    fcst = df_view.filter(
        (pl.col("ds").cast(pl.Date) >= forecast_start)
        & (pl.col("ds").cast(pl.Date) <= forecast_end)
    )
    return hist, fcst


def detail_view(df: pl.DataFrame) -> pl.DataFrame:
    """Selecciona solo columnas de detalle permitidas."""
    cols = [c for c in DETAIL_COLUMNS if c in df.columns]
    if "abs_error" not in cols and "y" in df.columns and "yhat" in df.columns:
        df = df.with_columns(
            (pl.col("y") - pl.col("yhat")).abs().alias("abs_error")
        )
        cols = [c for c in DETAIL_COLUMNS if c in df.columns]
    return df.select(cols)


def meta_date_from_df(
    df: pl.DataFrame, col: str, fallback: Any, available_cols: set[str]
) -> dt.date | None:
    if col in available_cols and df.height:
        vals = df[col].drop_nulls()
        if vals.len():
            v = vals[0]
            return v.date() if isinstance(v, dt.datetime) else v
    if isinstance(fallback, dt.datetime):
        return fallback.date()
    return fallback


def section_horizons_for(
    seccion: str, first_data: dt.date | None
) -> dict:
    return settings.section_horizons(seccion, first_data)
