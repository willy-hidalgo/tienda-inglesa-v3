"""
Backend de cálculos (sin Streamlit / sin Plotly).
El dashboard solo visualiza lo que prepara dashboard_data.
"""
from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from typing import Any

import polars as pl

import settings

DETAIL_COLUMNS = [
    "unique_id",
    "ds",
    "y",
    "yhat",
    "yhat28",
    "value",
    "valuehat",
    "valuehat28",
    "period_type",
    "driver_effect",
    "driver_effect_value",
    "sku_desc",
    "store_name",
    "seccion",
    "abs_error",
]

HORIZON_COLUMNS = {
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "forecast_start",
    "forecast_end",
}


# ─────────────────────────────────────────────────────────────────────────────
# Carga
# ─────────────────────────────────────────────────────────────────────────────
def load_forecast_parquet(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    schema = pl.scan_parquet(path).collect_schema()
    return (
        pl.scan_parquet(path)
        .select(list(schema.names()))
        .with_columns(pl.col("ds").cast(pl.Date))
        .collect()
    )


def load_forecast_bytes(data: bytes, name: str) -> pl.DataFrame:
    if name.endswith(".csv"):
        df = pl.read_csv(io.BytesIO(data), try_parse_dates=True)
    else:
        df = pl.read_parquet(io.BytesIO(data))
    if "ds" in df.columns:
        df = df.with_columns(pl.col("ds").cast(pl.Date))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Unidad / agregación / serie
# ─────────────────────────────────────────────────────────────────────────────
def prepare_unit_df(df: pl.DataFrame, unidad: str, has_value: bool) -> pl.DataFrame:
    if unidad == "Unidades" or not has_value:
        return df
    exprs = [
        pl.col("value").alias("y"),
        pl.col("valuehat").alias("yhat"),
    ]
    if "valuehat28" in df.columns:
        exprs.append(pl.col("valuehat28").alias("yhat28"))
    return df.with_columns(exprs)


def aggregate_temporal(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    if freq == "Diario" or df.height == 0:
        out = df
        if (
            "abs_error" not in out.columns
            and "y" in out.columns
            and "yhat" in out.columns
        ):
            out = out.with_columns(
                (pl.col("y") - pl.col("yhat")).abs().alias("abs_error")
            )
        return out
    period = "1w" if freq == "Semanal" else "1mo"
    aggs: list = [pl.col("y").sum(), pl.col("yhat").sum()]
    if "yhat28" in df.columns:
        aggs.append(pl.col("yhat28").sum())
    if "valuehat28" in df.columns:
        aggs.append(pl.col("valuehat28").sum())
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


# ─────────────────────────────────────────────────────────────────────────────
# Filtros independientes (tienda / sku) — reemplaza la navegación jerárquica
# ─────────────────────────────────────────────────────────────────────────────
def secciones_disponibles(all_ids: list[str]) -> list[str]:
    """Ids de nivel sección (sin tienda ni sku)."""
    out = set()
    for uid in all_ids:
        p = settings.split_unique_id(uid)
        if p["store"] is None and p["sku"] is None:
            out.add(p["seccion"])
    return sorted(out)


def all_stores_in_section(all_ids: list[str], seccion: str) -> list[str]:
    """Todas las tiendas de la sección (nodo tienda, sin filtro de sku)."""
    out = set()
    for uid in all_ids:
        p = settings.split_unique_id(uid)
        if p["seccion"] == seccion and p["store"] is not None and p["sku"] is None:
            out.add(p["store"])
    return sorted(out)


def all_skus_in_section(all_ids: list[str], seccion: str) -> list[str]:
    """Todos los SKU de la sección.

    El pipeline solo materializa hojas sku+tienda (y nodos sección/tienda).
    No existen nodos «SKU puro»; se recolectan los SKU desde cualquier
    unique_id que los contenga (típicamente profundidad 2).
    """
    out = set()
    for uid in all_ids:
        p = settings.split_unique_id(uid)
        if p["seccion"] == seccion and p["sku"] is not None:
            out.add(p["sku"])
    return sorted(out)


def stores_for_sku(all_ids: list[str], seccion: str, sku: str) -> list[str]:
    """Tiendas donde existe el SKU dado (nodo tienda+sku) — para acotar el
    select de Tienda cuando el usuario elegió SKU primero."""
    out = set()
    for uid in all_ids:
        p = settings.split_unique_id(uid)
        if p["seccion"] == seccion and p["sku"] == sku and p["store"] is not None:
            out.add(p["store"])
    return sorted(out)


def skus_for_store(all_ids: list[str], seccion: str, store: str) -> list[str]:
    """SKU que existen en la tienda dada (nodo tienda+sku) — para acotar el
    select de SKU cuando el usuario elegió Tienda primero."""
    out = set()
    for uid in all_ids:
        p = settings.split_unique_id(uid)
        if p["seccion"] == seccion and p["store"] == store and p["sku"] is not None:
            out.add(p["sku"])
    return sorted(out)


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


# ─────────────────────────────────────────────────────────────────────────────
# Métricas / rankings
# ─────────────────────────────────────────────────────────────────────────────
def spine_n_fechas(df: pl.DataFrame, unique_ids: list[str] | None = None) -> int:
    """Nº de fechas distintas del panel (spine denso)."""
    if df.height == 0 or "ds" not in df.columns:
        return 0
    if unique_ids:
        ids_df = pl.DataFrame({"unique_id": unique_ids})
        sub = df.join(ids_df, on="unique_id", how="semi")
        if sub.height == 0:
            sub = df
    else:
        sub = df
    return int(sub.select(pl.col("ds").n_unique()).item())


def _is_leaf_expr() -> pl.Expr:
    """Hoja sku+tienda: exactamente 2 separadores '||' (sec||T:x||S:y)."""
    return pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2


def _scored_leaves(
    df: pl.DataFrame,
    *,
    period_types: list[str] | None = None,
) -> pl.DataFrame:
    """
    Hojas sku+tienda con venta.

    period_types:
      - None → excluye solo forecast_only (in_sample + out_sample; compat métricas in/out)
      - lista → filtra a esos period_type (p.ej. ["out_sample"] para rankings OOS)

    Si no hay columna period_type, no se filtra por periodo.
    Columnas extra: _abs_error, _store_uid, _seccion, _sku, _sku_uid.
    """
    if df.height == 0 or "y" not in df.columns or "yhat" not in df.columns:
        return df.head(0)

    leaves = df.filter(_is_leaf_expr())
    if (
        "rls_metric_eligible" in leaves.columns
        and str(getattr(settings, "METRICS_MODE", "rolling_28")).lower() == "rolling_28"
    ):
        leaves = leaves.filter(pl.col("rls_metric_eligible") == True)
    if "period_type" in leaves.columns:
        if period_types is not None:
            leaves = leaves.filter(pl.col("period_type").is_in(list(period_types)))
        else:
            leaves = leaves.filter(pl.col("period_type") != "forecast_only")
    leaves = leaves.filter(
        pl.col("y").is_not_null()
        & pl.col("yhat").is_not_null()
        & pl.col("y").is_finite()
        & pl.col("yhat").is_finite()
        & (pl.col("y") != 0)
    )
    if leaves.height == 0:
        return leaves

    return leaves.with_columns(
        (pl.col("y") - pl.col("yhat")).abs().alias("_abs_error"),
        # "1||T:00063||S:127360" → store_uid "1||T:00063", seccion "1", sku "127360"
        pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
        pl.col("unique_id").str.split("||").list.get(0).alias("_seccion"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("_sku"),
    ).with_columns(
        pl.concat_str(
            [pl.col("_seccion"), pl.lit("||S:"), pl.col("_sku")]
        ).alias("_sku_uid")
    )


def wmape_bottom_up(
    df: pl.DataFrame,
    *,
    n_fechas_spine: int | None = None,
    period_types: list[str] | None = None,
) -> pl.DataFrame:
    """
    WMAPE bottom-up desde hojas sku+tienda (única fuente de verdad).

    Definición (igual para hoja / tienda / sección / SKU puro):
        abs_err_i = |y_i − ŷ_i|   en cada fila de hoja con y ≠ 0
        wMAPE     = Σ abs_err / Σ |y|

    - Hoja sku+tienda: suma sobre sus propias filas.
    - Tienda: suma de abs_err y de y de **todas** las hojas de esa tienda
      (no usa el yhat del nodo tienda agregado).
    - Sección: idem sobre todas las hojas de la sección.
    - SKU puro (sec||S:sku): suma sobre hojas de ese SKU en todas las tiendas.

    n_with_sales = nº de **días distintos** con al menos una observación y≠0
    en el alcance (si varias hojas sku+tienda comparten la misma fecha, cuenta 1).
    Sin columna `ds`, cae a contar filas (compat tests / paneles mínimos).

    period_types: ver `_scored_leaves`. Rankings usan `["out_sample"]`.
    """
    schema = {
        "unique_id": pl.Utf8,
        "wmape": pl.Float64,
        "sum_y": pl.Float64,
        "n_points": pl.UInt32,
        "n_with_sales": pl.UInt32,
    }
    leaves = _scored_leaves(df, period_types=period_types)
    if leaves.height == 0:
        return pl.DataFrame(schema=schema)

    n_fechas_spine = int(
        n_fechas_spine
        if n_fechas_spine is not None
        else (
            leaves.select(pl.col("ds").n_unique()).item()
            if "ds" in leaves.columns
            else 0
        )
    )
    n_spine_lit = pl.lit(n_fechas_spine).cast(pl.UInt32)

    # Días distintos con venta; sin ds → filas (compat)
    if "ds" in leaves.columns:
        n_sales_expr = pl.col("ds").n_unique().cast(pl.UInt32).alias("n_with_sales")
    else:
        n_sales_expr = pl.len().cast(pl.UInt32).alias("n_with_sales")

    def _agg_level(group_key: str) -> pl.DataFrame:
        out = (
            leaves.group_by(group_key)
            .agg(
                pl.col("_abs_error").sum().alias("sum_abs_error"),
                pl.col("y").sum().alias("sum_y"),
                n_sales_expr,
            )
            .with_columns(
                pl.when(pl.col("sum_y") != 0)
                .then(pl.col("sum_abs_error") / pl.col("sum_y").abs())
                .otherwise(0.0)
                .alias("wmape"),
                n_spine_lit.alias("n_points"),
            )
        )
        if group_key != "unique_id":
            out = out.rename({group_key: "unique_id"})
        return out.select(["unique_id", "wmape", "sum_y", "n_points", "n_with_sales"])

    out_leaf = _agg_level("unique_id")
    out_store = _agg_level("_store_uid")
    out_sec = _agg_level("_seccion")
    out_sku = _agg_level("_sku_uid")

    return pl.concat(
        [out_leaf, out_store, out_sec, out_sku], how="diagonal_relaxed"
    )


def wmape_por_id(
    ids: list[str] | None,
    df: pl.DataFrame,
    *,
    n_fechas_spine: int | None = None,
    fill_missing: bool = True,
    period_types: list[str] | None = None,
) -> pl.DataFrame:
    """
    WMAPE bottom-up (ver `wmape_bottom_up`).

    Si se pasa `ids`, filtra el resultado a esos unique_id y opcionalmente
    completa faltantes con wMAPE=0.

    Rankings del dashboard: period_types=["out_sample"].
    """
    schema = {
        "unique_id": pl.Utf8,
        "wmape": pl.Float64,
        "sum_y": pl.Float64,
        "n_points": pl.UInt32,
        "n_with_sales": pl.UInt32,
    }
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    if ids is not None and len(ids) == 0:
        return pl.DataFrame(schema=schema)

    agg = wmape_bottom_up(
        df, n_fechas_spine=n_fechas_spine, period_types=period_types
    )
    if ids is None:
        return agg if agg.height else pl.DataFrame(schema=schema)

    ids_df = pl.DataFrame({"unique_id": ids}).unique()
    agg = agg.join(ids_df, on="unique_id", how="semi")

    if not fill_missing:
        return agg if agg.height else pl.DataFrame(schema=schema)

    n_spine = int(n_fechas_spine or 0)
    n_spine_lit = pl.lit(n_spine).cast(pl.UInt32)
    if agg.height == 0:
        return ids_df.with_columns(
            pl.lit(0.0).alias("wmape"),
            pl.lit(0.0).alias("sum_y"),
            n_spine_lit.alias("n_points"),
            pl.lit(0).cast(pl.UInt32).alias("n_with_sales"),
        )
    missing = ids_df.join(agg.select("unique_id"), on="unique_id", how="anti")
    if missing.height == 0:
        return agg
    extra = missing.with_columns(
        pl.lit(0.0).alias("wmape"),
        pl.lit(0.0).alias("sum_y"),
        n_spine_lit.alias("n_points"),
        pl.lit(0).cast(pl.UInt32).alias("n_with_sales"),
    )
    return pl.concat([agg, extra], how="diagonal_relaxed")


def wmape_all_ids(
    df: pl.DataFrame,
    *,
    n_fechas_spine: int | None = None,
    period_types: list[str] | None = None,
) -> pl.DataFrame:
    """WMAPE bottom-up de todos los nodos derivables de las hojas."""
    return wmape_bottom_up(
        df, n_fechas_spine=n_fechas_spine, period_types=period_types
    )


def wmape_scope_from_leaves(
    df: pl.DataFrame,
    *,
    seccion: str,
    store: str | None = None,
    sku: str | None = None,
) -> dict[str, float | int]:
    """
    wMAPE de un alcance (sección / tienda / sku+tienda / sku puro) calculado
    siempre bottom-up desde hojas. Usado por métricas in/out del dashboard.
    """
    leaves = _scored_leaves(df)
    if leaves.height == 0:
        return {"wmape": 0.0, "sum_y": 0.0, "sum_abs_error": 0.0, "n": 0}

    leaves = leaves.filter(pl.col("_seccion") == str(seccion))
    if store is not None and sku is not None:
        uid = f"{seccion}||T:{store}||S:{sku}"
        leaves = leaves.filter(pl.col("unique_id") == uid)
    elif store is not None:
        leaves = leaves.filter(pl.col("_store_uid") == f"{seccion}||T:{store}")
    elif sku is not None:
        leaves = leaves.filter(pl.col("_sku") == str(sku))

    if leaves.height == 0:
        return {"wmape": 0.0, "sum_y": 0.0, "sum_abs_error": 0.0, "n": 0}

    sum_y = float(leaves["y"].sum())
    sum_err = float(leaves["_abs_error"].sum())
    wmape = (sum_err / abs(sum_y)) if sum_y != 0 else 0.0
    return {
        "wmape": wmape,
        "sum_y": sum_y,
        "sum_abs_error": sum_err,
        "n": leaves.height,
    }

def ranking_table(
    tabla_base: pl.DataFrame,
    *,
    seccion: str,
    axis: str,
    fixed_peer: str | None,
    exclude: str | None,
    desc_map: dict[str, str],
    unidad: str,
) -> pl.DataFrame:
    """
    Tabla de ranking para UN eje (`axis="store"` o `axis="sku"`), con
    filtro cruzado según el otro eje (`fixed_peer`):

    - `axis="store"`, `fixed_peer=None`   → todas las tiendas de la sección
      (nodos tienda puros, que sí materializa el pipeline).
    - `axis="store"`, `fixed_peer=<sku>`  → tiendas donde existe ese SKU
      (comparando hojas tienda+sku entre tiendas, para ese SKU fijo).
    - `axis="sku"`, `fixed_peer=None`     → SKU de la sección agregando las
      hojas sku+tienda (el pipeline NO genera nodos SKU puro).
    - `axis="sku"`, `fixed_peer=<store>`  → SKU que existen en esa tienda
      (hojas tienda+sku de esa tienda).
    - `exclude`: código del nodo actualmente seleccionado en ESTE eje (se
      omite de su propio ranking).

    Columnas: Código | Descripción | wMAPE (%) | Rotación | N puntos | % ≠0
    - wMAPE (%) es el de **out-of-sample** cuando `tabla_base` se construyó
      con period_types=["out_sample"] (path dashboard / artefactos).
    - "N puntos" = nº de **días distintos** con al menos una observación y≠0
      en el alcance (varias hojas sku+tienda el mismo día cuentan 1). No es
      la longitud del spine (esa se muestra aparte — ver `DashboardView.n_spine`).
    - "% ≠0" = N puntos / longitud del spine, como cobertura de venta.
    """
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
    if tabla_base.height == 0:
        return empty

    # Parseo vectorizado del unique_id (evita loop Python por fila).
    # Formato: "sec" | "sec||T:store" | "sec||S:sku" | "sec||T:store||S:sku"
    enriched = tabla_base.with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("_sec"),
        pl.when(pl.col("unique_id").str.contains(r"\|\|T:", literal=False))
        .then(pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1))
        .otherwise(None)
        .alias("_store"),
        pl.when(pl.col("unique_id").str.contains(r"\|\|S:", literal=False))
        .then(pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1))
        .otherwise(None)
        .alias("_sku"),
    ).filter(pl.col("_sec") == seccion)

    enriched = enriched.with_columns(
        pl.col("_store").cast(pl.Utf8),
        pl.col("_sku").cast(pl.Utf8),
    )

    if axis == "store":
        # Siempre y solo nodos tienda puros (store presente, sku ausente).
        tabla = enriched.filter(
            pl.col("_store").is_not_null() & pl.col("_sku").is_null()
        )
        if exclude is not None:
            tabla = tabla.filter(pl.col("_store") != str(exclude))
        code_expr = pl.col("_store")
        uid_expr = pl.col("unique_id")
        code_col = "_store"
    else:  # axis == "sku"
        if fixed_peer is not None:
            # Tienda seleccionada → SOLO hojas sku+tienda de ESA tienda
            peer = str(fixed_peer)
            tabla = enriched.filter(
                (pl.col("_store") == peer) & pl.col("_sku").is_not_null()
            )
            if tabla.height == 0:
                needle = f"||T:{peer}||"
                tabla = enriched.filter(
                    pl.col("unique_id").str.contains(needle, literal=True)
                    & pl.col("_sku").is_not_null()
                )
            if exclude is not None:
                tabla = tabla.filter(pl.col("_sku") != str(exclude))
            code_expr = pl.col("_sku")
            uid_expr = pl.col("unique_id")
            code_col = "_sku"
        else:
            pure = enriched.filter(
                pl.col("_sku").is_not_null() & pl.col("_store").is_null()
            )
            if exclude is not None:
                pure = pure.filter(pl.col("_sku") != str(exclude))
            if pure.height > 0:
                tabla = pure
                code_expr = pl.col("_sku")
                uid_expr = pl.col("unique_id")
                code_col = "_sku"
            else:
                leaves = enriched.filter(
                    pl.col("_store").is_not_null() & pl.col("_sku").is_not_null()
                )
                if exclude is not None:
                    leaves = leaves.filter(pl.col("_sku") != str(exclude))
                if leaves.height == 0:
                    return empty
                tabla = (
                    leaves.group_by("_sku")
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
                            [pl.lit(seccion), pl.lit("||S:"), pl.col("_sku")]
                        ).alias("unique_id"),
                    )
                    .drop("_sum_abs")
                )
                code_expr = pl.col("_sku")
                uid_expr = pl.col("unique_id")
                code_col = "_sku"

    # Solo filas con rotación y wMAPE > 0. Para SKU se exige además una
    # cobertura mínima de días con actual distinto de cero.
    tabla = tabla.filter((pl.col("sum_y") > 0) & (pl.col("wmape") > 0))
    if axis == "sku":
        min_nonzero = int(getattr(settings, "RANKING_SKU_MIN_NONZERO_POINTS", 15))
        tabla = tabla.filter(pl.col("n_with_sales") >= min_nonzero)
    if tabla.height == 0:
        return empty
    tabla = tabla.sort("wmape", descending=False).unique(
        subset=[code_col], keep="first", maintain_order=True
    )

    codes = tabla.select(code_expr.alias("Código"))["Código"].to_list()
    uids2 = tabla.select(uid_expr.alias("unique_id"))["unique_id"].to_list()
    n_points = tabla["n_points"].to_list()
    n_with_sales = tabla["n_with_sales"].to_list()
    pct = [
        (float(nw) / float(np_) * 100) if np_ else 0.0
        for nw, np_ in zip(n_with_sales, n_points)
    ]
    descriptions: list[str] = []
    for u, code in zip(uids2, codes):
        d = desc_map.get(u, "")
        if not d and axis == "sku":
            prefix_hit = next(
                (desc_map[k] for k in desc_map if f"||S:{code}" in k and desc_map[k]),
                "",
            )
            d = prefix_hit
        descriptions.append(d)

    return pl.DataFrame(
        {
            "Código": codes,
            "Descripción": descriptions,
            "wMAPE (%)": [f"{w * 100:,.2f}" for w in tabla["wmape"].to_list()],
            label_rot: [f"{v:,.2f}" for v in tabla["sum_y"].to_list()],
            "N puntos": [int(x) for x in n_with_sales],
            "% ≠0": [f"{p:,.1f}" for p in pct],
            "unique_id": uids2,
        }
    )


def calcular_metricas(df: pl.DataFrame) -> tuple[float, float, int]:
    """
    wMAPE = Σ|y−ŷ| / Σ|y| sobre filas con y≠0 y ŷ finito.
    Misma exclusión que rankings / `_scored_leaves`.
    Si `df` trae varias hojas, es bottom-up (suma de abs_err de cada fila hoja).
    """
    if df.height == 0 or "y" not in df.columns or "yhat" not in df.columns:
        return 0.0, 0.0, 0
    metric_df = df
    if (
        "rls_metric_eligible" in metric_df.columns
        and str(getattr(settings, "METRICS_MODE", "rolling_28")).lower() == "rolling_28"
    ):
        metric_df = metric_df.filter(pl.col("rls_metric_eligible") == True)
    scored = metric_df.filter(
        pl.col("y").is_not_null()
        & pl.col("yhat").is_not_null()
        & pl.col("y").is_finite()
        & pl.col("yhat").is_finite()
        & (pl.col("y") != 0)
    )
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
    # n = días distintos con venta si hay ds; si no, filas (compat)
    if "ds" in scored.columns:
        n = int(scored.select(pl.col("ds").n_unique()).item())
    else:
        n = int(scored.height)
    return wmape, bias, n


def metrics_in_out_total(
    df_view: pl.DataFrame, cutoff: dt.date, test_end: dt.date
) -> dict[str, dict[str, float | int]]:
    """
    Métricas in / out / total. Si `df_view` son hojas (varios unique_id),
    el wMAPE es bottom-up: Σ|y−ŷ| / Σ|y| sobre todas las filas hoja del tramo.

    Definición OOS alineada con rankings:
      - Si existe `period_type` → in_sample / out_sample / (ambos para total)
      - Si no → fallback por fechas: in < cutoff; out en [cutoff, test_end]
    """
    has_period = "period_type" in df_view.columns
    if has_period:
        df_in = df_view.filter(pl.col("period_type") == "in_sample")
        df_out = df_view.filter(pl.col("period_type") == "out_sample")
        df_tot = df_view.filter(
            pl.col("period_type").is_in(["in_sample", "out_sample"])
        )
    else:
        df_in = df_view.filter(pl.col("ds").cast(pl.Date) < cutoff)
        df_out = df_view.filter(
            (pl.col("ds").cast(pl.Date) >= cutoff)
            & (pl.col("ds").cast(pl.Date) <= test_end)
        )
        df_tot = df_view.filter(pl.col("ds").cast(pl.Date) <= test_end)

    out: dict[str, dict[str, float | int]] = {}
    for key, frame in (("in", df_in), ("out", df_out), ("total", df_tot)):
        w, b, n = calcular_metricas(frame)
        out[key] = {"wmape": w, "bias": b, "n": n}
    return out


def metrics_in_out_bottom_up(
    unit_df: pl.DataFrame,
    *,
    seccion: str,
    store: str | None,
    sku: str | None,
    cutoff: dt.date,
    test_end: dt.date,
) -> dict[str, dict[str, float | int]]:
    """
    in/out/total wMAPE bottom-up para un alcance (sección / tienda / hoja / sku).
    Parte siempre de hojas sku+tienda.
    """
    leaves = _scored_leaves(unit_df)
    if leaves.height == 0:
        z = {"wmape": 0.0, "bias": 0.0, "n": 0}
        return {"in": z, "out": dict(z), "total": dict(z)}

    leaves = leaves.filter(pl.col("_seccion") == str(seccion))
    if store is not None and sku is not None:
        leaves = leaves.filter(
            pl.col("unique_id") == f"{seccion}||T:{store}||S:{sku}"
        )
    elif store is not None:
        leaves = leaves.filter(pl.col("_store_uid") == f"{seccion}||T:{store}")
    elif sku is not None:
        leaves = leaves.filter(pl.col("_sku") == str(sku))

    if leaves.height == 0:
        z = {"wmape": 0.0, "bias": 0.0, "n": 0}
        return {"in": z, "out": dict(z), "total": dict(z)}

    return metrics_in_out_total(leaves, cutoff, test_end)


# ─────────────────────────────────────────────────────────────────────────────
# Horizontes / splits de gráfico / detalle
# ─────────────────────────────────────────────────────────────────────────────
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


def resolve_horizons(
    df_daily: pl.DataFrame, seccion: str, available_cols: set[str]
) -> dict[str, dt.date | None]:
    first_d = df_daily["ds"].min() if df_daily.height else settings.FECHAS_TRAIN[0]
    if isinstance(first_d, dt.datetime):
        first_d = first_d.date()
    hz = settings.section_horizons(seccion, first_d)
    train_end = meta_date_from_df(df_daily, "train_end", hz["train_end"], available_cols)
    test_start = meta_date_from_df(
        df_daily, "test_start", hz["test_start"], available_cols
    )
    test_end = meta_date_from_df(df_daily, "test_end", hz["test_end"], available_cols)
    fcst_start = meta_date_from_df(
        df_daily,
        "forecast_start",
        hz.get("forecast_start", hz["test_end"] + dt.timedelta(days=1)),
        available_cols,
    )
    if fcst_start is None or (
        isinstance(test_end, dt.date) and fcst_start <= test_end
    ):
        fcst_start = test_end + dt.timedelta(days=1) if test_end else fcst_start
    fcst_end = meta_date_from_df(
        df_daily, "forecast_end", hz["forecast_end"], available_cols
    )
    return {
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "forecast_start": fcst_start,
        "forecast_end": fcst_end,
    }


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
    # Incluir y=0 en el panel/gráfico; el filtro de ceros es solo para métricas
    hist = df_view.filter(pl.col("ds").cast(pl.Date) <= test_end)
    fcst = df_view.filter(
        (pl.col("ds").cast(pl.Date) >= forecast_start)
        & (pl.col("ds").cast(pl.Date) <= forecast_end)
    )
    return hist, fcst


def build_chart_series(
    df_view: pl.DataFrame,
    test_end: dt.date,
    forecast_start: dt.date,
    forecast_end: dt.date,
    cutoff: dt.date,
    has_period: bool,
) -> dict[str, Any]:
    """Series listas para Plotly (sin importar plotly)."""
    hist, fcst = split_hist_forecast(
        df_view, test_end, forecast_start, forecast_end, has_period
    )
    def _col(df, name):
        if name in df.columns and df.height:
            return df[name].to_list()
        return []

    return {
        "hist_ds": hist["ds"].to_list() if hist.height else [],
        "hist_y": hist["y"].to_list() if hist.height else [],
        "hist_yhat": hist["yhat"].to_list() if hist.height else [],
        "hist_yhat28": _col(hist, "yhat28"),
        "fcst_ds": fcst["ds"].to_list() if fcst.height else [],
        "fcst_yhat": fcst["yhat"].to_list() if fcst.height else [],
        "fcst_yhat28": _col(fcst, "yhat28"),
        "cutoff": cutoff,
        "test_end": test_end,
    }


def detail_view(df: pl.DataFrame) -> pl.DataFrame:
    cols = [c for c in DETAIL_COLUMNS if c in df.columns]
    if "abs_error" not in cols and "y" in df.columns and "yhat" in df.columns:
        df = df.with_columns(
            (pl.col("y") - pl.col("yhat")).abs().alias("abs_error")
        )
        cols = [c for c in DETAIL_COLUMNS if c in df.columns]
    return df.select(cols)


def ds_range(df: pl.DataFrame) -> tuple[dt.date | None, dt.date | None]:
    if df.height == 0:
        return None, None
    mn, mx = df["ds"].min(), df["ds"].max()
    if isinstance(mn, dt.datetime):
        mn = mn.date()
    if isinstance(mx, dt.datetime):
        mx = mx.date()
    return mn, mx



def format_detail_display(df: pl.DataFrame) -> pl.DataFrame:
    """
    Formatea columnas numéricas para la UI:
      12345.2 → "12,345"  (entero con separador de miles)
    El resto de columnas se deja igual.
    """
    if df.height == 0:
        return df
    num_cols = [
        c
        for c in (
            "y", "yhat", "yhat28", "value", "valuehat", "valuehat28",
            "driver_effect", "driver_effect_value", "abs_error",
        )
        if c in df.columns
    ]
    if not num_cols:
        return df

    def _fmt(v: float | None) -> str:
        if v is None:
            return ""
        try:
            if v != v:  # NaN
                return ""
            return f"{round(float(v)):,}"
        except (TypeError, ValueError):
            return ""

    exprs = [
        pl.col(c)
        .map_elements(_fmt, return_dtype=pl.Utf8)
        .alias(c)
        for c in num_cols
    ]
    return df.with_columns(exprs)


def metrics_rolling28(df_view: pl.DataFrame) -> dict[str, float | int]:
    """WMAPE/BIAS de yhat28 vs y sobre el df agregado visible."""
    if df_view.height == 0 or "yhat28" not in df_view.columns:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    scored = df_view.filter(
        pl.col("y").is_not_null()
        & (pl.col("y") != 0)
        & pl.col("yhat28").is_not_null()
    )
    if scored.height == 0:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    sum_y = float(scored["y"].sum())
    if sum_y == 0:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    ae = float((scored["y"] - scored["yhat28"]).abs().sum())
    bias = float((scored["yhat28"] - scored["y"]).sum()) / sum_y
    return {"wmape_28": ae / abs(sum_y), "bias_28": bias, "n": scored.height}
