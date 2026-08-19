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
        meta["sku_desc"].to_list() if "sku_desc" in meta.columns else [""] * len(uids)
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
    sub = df
    if unique_ids:
        sub = df.filter(pl.col("unique_id").is_in(unique_ids))
        if sub.height == 0:
            sub = df
    return int(sub["ds"].n_unique())


def wmape_por_id(
    ids: list[str],
    df: pl.DataFrame,
    *,
    n_fechas_spine: int | None = None,
) -> pl.DataFrame:
    """
    WMAPE = Σ|y−ŷ|/Σ|y| (excluye y==0 y forecast_only del cálculo).
    n_points = longitud del spine (igual para todos si el panel está denso).
    n_with_sales = conteo y≠0 (informativo).
    """
    schema = {
        "unique_id": pl.Utf8,
        "wmape": pl.Float64,
        "sum_y": pl.Float64,
        "n_points": pl.UInt32,
        "n_with_sales": pl.UInt32,
    }
    if not ids:
        return pl.DataFrame(schema=schema)

    base = df.filter(pl.col("unique_id").is_in(ids))
    if "period_type" in base.columns:
        base_m = base.filter(pl.col("period_type") != "forecast_only")
    else:
        base_m = base

    if n_fechas_spine is None:
        n_fechas_spine = spine_n_fechas(base_m if base_m.height else base, ids)
    n_fechas_spine = int(n_fechas_spine or 0)

    scored = base_m.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    if scored.height == 0:
        # devolver todos los ids con wmape null/0 y n_points = spine
        return pl.DataFrame(
            {
                "unique_id": ids,
                "wmape": [0.0] * len(ids),
                "sum_y": [0.0] * len(ids),
                "n_points": [n_fechas_spine] * len(ids),
                "n_with_sales": [0] * len(ids),
            }
        ).filter(pl.col("unique_id").is_in(ids))

    agg = (
        scored.group_by("unique_id")
        .agg(
            pl.col("y").sum().alias("sum_y"),
            (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_error"),
            pl.len().alias("n_with_sales"),
        )
        .filter(pl.col("sum_y") != 0)
        .with_columns((pl.col("sum_abs_error") / pl.col("sum_y")).alias("wmape"))
        .with_columns(pl.lit(n_fechas_spine).cast(pl.UInt32).alias("n_points"))
        .select(["unique_id", "wmape", "sum_y", "n_points", "n_with_sales"])
    )
    # ids sin ventas: aún así n_points = spine
    missing = [i for i in ids if i not in set(agg["unique_id"].to_list())]
    if missing:
        extra = pl.DataFrame(
            {
                "unique_id": missing,
                "wmape": [0.0] * len(missing),
                "sum_y": [0.0] * len(missing),
                "n_points": [n_fechas_spine] * len(missing),
                "n_with_sales": [0] * len(missing),
            }
        )
        agg = pl.concat([agg, extra], how="diagonal_relaxed")
    return agg


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
    - "N puntos" = cantidad de puntos con venta (y≠0), no la longitud del
      spine (que es igual para todas las filas y se muestra aparte como
      una etiqueta única — ver `DashboardView.n_spine`).
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

    if axis == "store":
        if fixed_peer is not None:
            # Hojas tienda+sku del SKU fijo
            tabla = enriched.filter(
                (pl.col("_sku") == fixed_peer) & pl.col("_store").is_not_null()
            )
            if exclude is not None:
                tabla = tabla.filter(pl.col("_store") != exclude)
            code_expr = pl.col("_store")
            uid_expr = pl.col("unique_id")
        else:
            # Nodos tienda puros (depth 1, sin SKU)
            tabla = enriched.filter(
                pl.col("_store").is_not_null() & pl.col("_sku").is_null()
            )
            if exclude is not None:
                tabla = tabla.filter(pl.col("_store") != exclude)
            code_expr = pl.col("_store")
            uid_expr = pl.col("unique_id")
    else:  # axis == "sku"
        if fixed_peer is not None:
            # Hojas tienda+sku de la tienda fija
            tabla = enriched.filter(
                (pl.col("_store") == fixed_peer) & pl.col("_sku").is_not_null()
            )
            if exclude is not None:
                tabla = tabla.filter(pl.col("_sku") != exclude)
            code_expr = pl.col("_sku")
            uid_expr = pl.col("unique_id")
        else:
            # Agregar hojas sku+tienda → ranking por SKU (no hay nodos SKU puro)
            leaves = enriched.filter(
                pl.col("_store").is_not_null() & pl.col("_sku").is_not_null()
            )
            if exclude is not None:
                leaves = leaves.filter(pl.col("_sku") != exclude)
            if leaves.height == 0:
                return empty
            # WMAPE agregado ≈ Σ(wmape_i * sum_y_i) / Σ sum_y_i
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
                    # unique_id virtual de SKU puro (para click → filtro SKU)
                    pl.concat_str(
                        [pl.lit(seccion), pl.lit("||S:"), pl.col("_sku")]
                    ).alias("unique_id"),
                )
                .drop("_sum_abs")
            )
            code_expr = pl.col("_sku")
            uid_expr = pl.col("unique_id")

    # Mostrar filas con ventas (sum_y>0); wmape==0 se mantiene al final
    # (antes se filtraba wmape!=0 y desaparecían series con error nulo o
    # sin ventas que igual interesan para rotación).
    tabla = tabla.filter(pl.col("sum_y") > 0).sort("wmape")
    if tabla.height == 0:
        return empty

    codes = tabla.select(code_expr.alias("Código"))["Código"].to_list()
    uids2 = tabla.select(uid_expr.alias("unique_id"))["unique_id"].to_list()
    n_points = tabla["n_points"].to_list()
    n_with_sales = tabla["n_with_sales"].to_list()
    pct = [
        (float(nw) / float(np_) * 100) if np_ else 0.0
        for nw, np_ in zip(n_with_sales, n_points)
    ]
    # Descripción: preferir la del unique_id; si es SKU virtual, buscar
    # cualquier hoja con ese SKU en desc_map.
    descriptions: list[str] = []
    for u, code in zip(uids2, codes):
        d = desc_map.get(u, "")
        if not d and axis == "sku" and fixed_peer is None:
            # Buscar primera hoja que contenga este SKU
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


def metrics_in_out_total(
    df_view: pl.DataFrame, cutoff: dt.date, test_end: dt.date
) -> dict[str, dict[str, float | int]]:
    df_in = df_view.filter(pl.col("ds").cast(pl.Date) < cutoff)
    df_out = df_view.filter(
        (pl.col("ds").cast(pl.Date) >= cutoff)
        & (pl.col("ds").cast(pl.Date) <= test_end)
        & (pl.col("y").is_not_null())
        & (pl.col("y") != 0)
    )
    df_tot = df_view.filter(
        (pl.col("ds").cast(pl.Date) <= test_end)
        & (pl.col("y").is_not_null())
        & (pl.col("y") != 0)
    )
    out: dict[str, dict[str, float | int]] = {}
    for key, frame in (("in", df_in), ("out", df_out), ("total", df_tot)):
        w, b, n = calcular_metricas(frame)
        out[key] = {"wmape": w, "bias": b, "n": n}
    return out


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
    train_end = meta_date_from_df(
        df_daily, "train_end", hz["train_end"], available_cols
    )
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
    if fcst_start is None or (isinstance(test_end, dt.date) and fcst_start <= test_end):
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
        df = df.with_columns((pl.col("y") - pl.col("yhat")).abs().alias("abs_error"))
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
            "y",
            "yhat",
            "yhat28",
            "value",
            "valuehat",
            "valuehat28",
            "driver_effect",
            "driver_effect_value",
            "abs_error",
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
        except TypeError, ValueError:
            return ""

    exprs = [
        pl.col(c).map_elements(_fmt, return_dtype=pl.Utf8).alias(c) for c in num_cols
    ]
    return df.with_columns(exprs)


def metrics_rolling28(df_view: pl.DataFrame) -> dict[str, float | int]:
    """WMAPE/BIAS de yhat28 vs y sobre el df agregado visible."""
    if df_view.height == 0 or "yhat28" not in df_view.columns:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    scored = df_view.filter(
        pl.col("y").is_not_null() & (pl.col("y") != 0) & pl.col("yhat28").is_not_null()
    )
    if scored.height == 0:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    sum_y = float(scored["y"].sum())
    if sum_y == 0:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    ae = float((scored["y"] - scored["yhat28"]).abs().sum())
    bias = float((scored["yhat28"] - scored["y"]).sum()) / sum_y
    return {"wmape_28": ae / abs(sum_y), "bias_28": bias, "n": scored.height}
