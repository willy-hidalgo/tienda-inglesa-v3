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
_FORECAST_LOAD_COLS = [
    "unique_id",
    "ds",
    "y",
    "yhat",
    "yhat28",
    "value",
    "valuehat",
    "valuehat28",
    "period_type",
    "sku_desc",
    "store_name",
    "seccion",
    "train_end",
    "test_start",
    "test_end",
    "forecast_start",
    "forecast_end",
]


def load_forecast_parquet(path: str | Path) -> pl.DataFrame:
    """Carga columnar mínima (evita traer drivers/features al dashboard)."""
    path = Path(path)
    available = set(pl.scan_parquet(path).collect_schema().names())
    cols = [c for c in _FORECAST_LOAD_COLS if c in available]
    if "unique_id" not in cols or "ds" not in cols:
        cols = list(available)
    return (
        pl.scan_parquet(path)
        .select(cols)
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

    need = ["unique_id", "y", "yhat"]
    if "period_type" in df.columns:
        need.append("period_type")
    slim = df.select([c for c in need if c in df.columns])
    base = slim.filter(pl.col("unique_id").is_in(ids))
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

def ranking_wmape_table(
    tabla_base: pl.DataFrame,
    selected_id: str,
    desc_map: dict[str, str],
) -> pl.DataFrame:
    tabla = tabla_base.filter(
        (pl.col("wmape") != 0) & (pl.col("unique_id") != selected_id)
    ).sort("wmape")
    empty = pl.DataFrame(
        schema={
            "Código": pl.Utf8,
            "Descripción": pl.Utf8,
            "wMAPE (%)": pl.Utf8,
            "N puntos": pl.UInt32,
            "unique_id": pl.Utf8,
        }
    )
    if tabla.height == 0:
        return empty
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
    tabla = tabla_base.filter(pl.col("unique_id") != selected_id).sort(
        "sum_y", descending=True
    )
    empty = pl.DataFrame(
        schema={
            "Código": pl.Utf8,
            "Descripción": pl.Utf8,
            label_rot: pl.Utf8,
            "N puntos": pl.UInt32,
            "unique_id": pl.Utf8,
        }
    )
    if tabla.height == 0:
        return empty
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
        for c in ("y", "yhat", "yhat28", "value", "valuehat", "valuehat28", "abs_error")
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
