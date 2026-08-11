"""
Backend de cálculos de visualización (sin Streamlit / sin Plotly).

El pipeline (app/forecasts.py) precalcula forecasts/wmape en parquet; este
módulo es la ÚNICA capa de cálculo del dashboard:

  - `build_dashboard_context`: pre-agregado por (archivo, unidad) — unit_df
    (y/value alias-eados), mapas de etiquetas (vectorizados en polars) y
    wmape/rotación/n puntos por unique_id. Se construye UNA vez por
    archivo+unidad y se cachea; cada interacción del dashboard solo trabaja
    sobre subconjuntos (O(candidatos) para rankings, O(serie seleccionada)
    para gráfico/métricas) — no vuelve a tocar el panel completo.
  - `prepare_dashboard_state` / `prepare_dashboard_state_from_context`:
    ViewModel listo para renderizar.
"""
from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass
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


def opciones_nivel(all_ids: list[str], prefix: str | None) -> list[str]:
    if prefix is None:
        return sorted(i for i in all_ids if "||" not in i)
    depth = prefix.count("||") + 1
    pref = prefix + "||"
    return sorted(
        i for i in all_ids if i.startswith(pref) and i.count("||") == depth
    )


def build_label_maps(df: pl.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """
    Mapas unique_id → (etiqueta, descripción) vectorizados en Polars.

    Evita el loop Python previo sobre todas las series (costoso a nivel SKU
    con miles de unique_ids). El contrato de salida es idéntico al de
    `settings.display_label` / `settings.ranking_description`.
    """
    cols = [c for c in ["unique_id", "sku_desc", "store_name"] if c in df.columns]
    if not cols:
        return {}, {}
    meta = df.select(cols).unique(subset=["unique_id"])

    parts = meta.with_columns(
        pl.col("unique_id").str.split_exact("||", 2).alias("_p")
    ).unnest("_p")

    has_desc = "sku_desc" in parts.columns
    has_sname = "store_name" in parts.columns
    if has_desc:
        parts = parts.with_columns(
            pl.col("sku_desc").replace("", None).alias("_desc")
        )
    if has_sname:
        parts = parts.with_columns(
            pl.col("store_name").replace("", None).alias("_sname")
        )

    # depth: 0 sección · 1 tienda · 2 SKU
    parts = parts.with_columns(
        pl.when(pl.col("field_1").is_null())
        .then(0)
        .when(pl.col("field_2").is_null())
        .then(1)
        .otherwise(2)
        .alias("_depth")
    )

    label_expr = (
        pl.when(pl.col("_depth") == 0).then(pl.format("Sección {}", pl.col("field_0")))
    )
    if has_sname:
        label_expr = label_expr.when(pl.col("_depth") == 1).then(
            pl.when(pl.col("_sname").is_not_null())
            .then(pl.format("{} — {}", pl.col("field_1"), pl.col("_sname")))
            .otherwise(pl.col("field_1"))
        )
    else:
        label_expr = label_expr.when(pl.col("_depth") == 1).then(pl.col("field_1"))
    if has_desc:
        label_expr = label_expr.when(pl.col("_depth") == 2).then(
            pl.when(pl.col("_desc").is_not_null())
            .then(pl.format("{} — {}", pl.col("field_2"), pl.col("_desc")))
            .otherwise(pl.col("field_2"))
        )
    else:
        label_expr = label_expr.when(pl.col("_depth") == 2).then(pl.col("field_2"))
    label_expr = label_expr.otherwise(pl.col("unique_id"))

    desc_expr = pl.when(pl.col("_depth") == 0).then(
        pl.format("Sección {}", pl.col("field_0"))
    )
    if has_sname:
        desc_expr = desc_expr.when(pl.col("_depth") == 1).then(
            pl.col("_sname").fill_null("")
        )
    else:
        desc_expr = desc_expr.when(pl.col("_depth") == 1).then(pl.lit(""))
    if has_desc:
        desc_expr = desc_expr.when(pl.col("_depth") == 2).then(
            pl.col("_desc").fill_null("")
        )
    else:
        desc_expr = desc_expr.when(pl.col("_depth") == 2).then(pl.lit(""))
    desc_expr = desc_expr.otherwise(pl.lit(""))

    parts = parts.with_columns(
        label_expr.alias("_label"),
        desc_expr.alias("_desc_rank"),
    )
    uids = parts["unique_id"].to_list()
    return (
        dict(zip(uids, parts["_label"].to_list())),
        dict(zip(uids, parts["_desc_rank"].to_list())),
    )


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


def aggregate_wmape_by_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-agregación por unique_id sobre el panel DIARIO (sin agregación
    temporal): wmape, sum_y (rotación), n_with_sales (puntos y≠0) y n_ds
    (días del spine por serie). Excluye y==0/nulos y forecast_only del
    cálculo. Se ejecuta UNA vez por (archivo, unidad) → `build_dashboard_context`.
    """
    schema = {
        "unique_id": pl.Utf8,
        "wmape": pl.Float64,
        "sum_y": pl.Float64,
        "n_with_sales": pl.UInt32,
        "n_ds": pl.UInt32,
    }
    if df.height == 0:
        return pl.DataFrame(schema=schema)

    has_ds = "ds" in df.columns
    n_ds = None
    if has_ds:
        n_ds = (
            df.select(["unique_id", "ds"])
            .unique()
            .group_by("unique_id")
            .agg(pl.len().alias("n_ds"))
        )

    base = df.filter(pl.col("y").is_not_null() & (pl.col("y") != 0))
    if "period_type" in base.columns:
        base = base.filter(pl.col("period_type") != "forecast_only")

    def _with_n_ds(frame: pl.DataFrame) -> pl.DataFrame:
        if n_ds is None:
            return frame.with_columns(pl.lit(0).cast(pl.UInt32).alias("n_ds"))
        return frame.join(n_ds, on="unique_id", how="left").with_columns(
            pl.col("n_ds").fill_null(0).cast(pl.UInt32)
        )

    if base.height == 0:
        ids = _with_n_ds(df.select("unique_id").unique())
        return ids.with_columns(
            pl.lit(0.0).alias("wmape"),
            pl.lit(0.0).alias("sum_y"),
            pl.lit(0).cast(pl.UInt32).alias("n_with_sales"),
        ).select(schema.keys())

    agg = (
        base.group_by("unique_id")
        .agg(
            pl.col("y").sum().alias("sum_y"),
            (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_error"),
            pl.len().alias("n_with_sales"),
        )
        .filter(pl.col("sum_y") != 0)
        .with_columns((pl.col("sum_abs_error") / pl.col("sum_y")).alias("wmape"))
        .select(["unique_id", "wmape", "sum_y", "n_with_sales"])
    )
    return _with_n_ds(agg).select(schema.keys())


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

    Implementado sobre `aggregate_wmape_by_id` (única fuente de la fórmula).
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

    if n_fechas_spine is None:
        n_fechas_spine = spine_n_fechas(df, ids)
    n_fechas_spine = int(n_fechas_spine or 0)

    agg = aggregate_wmape_by_id(df).filter(pl.col("unique_id").is_in(ids))
    missing = [
        i for i in ids if i not in set(agg["unique_id"].to_list())
    ]
    if missing:
        agg = pl.concat(
            [
                agg,
                pl.DataFrame(
                    {
                        "unique_id": missing,
                        "wmape": [0.0] * len(missing),
                        "sum_y": [0.0] * len(missing),
                        "n_with_sales": [0] * len(missing),
                        "n_ds": [0] * len(missing),
                    }
                ),
            ],
            how="diagonal_relaxed",
        )
    return (
        agg.with_columns(pl.lit(n_fechas_spine).cast(pl.UInt32).alias("n_points"))
        .select(["unique_id", "wmape", "sum_y", "n_points", "n_with_sales"])
        .filter(pl.col("unique_id").is_in(ids))
    )


def ranking_wmape_table(
    per_id: pl.DataFrame,
    selected_id: str,
    desc_map: dict[str, str],
    n_points_total: int,
    candidatos: list[str] | None = None,
) -> pl.DataFrame:
    """
    Tabla ÚNICA del dashboard (elimina 'Mayor rotación').

    Columnas exactas: Código | Descripción | wMAPE (%) | Rotación |
    N puntos ≠0 | % ≠0 | unique_id.

    - **Jerárquica**: si se pasa `candidatos`, solo se rankean esos (hijos
      directos del nivel seleccionado); nunca series de otro nivel/sección.
    - `N puntos ≠0` = períodos con venta (y≠0).
    - `% ≠0` = % de esos puntos respecto del spine total del nivel (etiqueta
      "Total de puntos" arriba).
    - Ordenada por wMAPE asc. Excluye el id seleccionado y las series sin
      ventas (wmape=0 → sin error que rankear).
    """
    tabla = per_id
    if candidatos:
        tabla = tabla.filter(pl.col("unique_id").is_in(candidatos))
    tabla = tabla.filter(
        (pl.col("unique_id") != selected_id) & (pl.col("wmape") != 0)
    ).sort("wmape")
    empty = pl.DataFrame(
        schema={
            "Código": pl.Utf8,
            "Descripción": pl.Utf8,
            "wMAPE (%)": pl.Utf8,
            "Rotación": pl.Utf8,
            "N puntos ≠0": pl.UInt32,
            "% ≠0": pl.Utf8,
            "unique_id": pl.Utf8,
        }
    )
    if tabla.height == 0:
        return empty
    uids = tabla["unique_id"].to_list()
    n_pts = tabla["n_with_sales"].to_list()
    pct = [
        (float(n) / n_points_total * 100.0) if n_points_total else 0.0 for n in n_pts
    ]
    return pl.DataFrame(
        {
            "Código": [settings.ranking_code(u) for u in uids],
            "Descripción": [desc_map.get(u, "") for u in uids],
            "wMAPE (%)": [f"{w * 100:,.2f}" for w in tabla["wmape"].to_list()],
            "Rotación": [f"{v:,.2f}" for v in tabla["sum_y"].to_list()],
            "N puntos ≠0": n_pts,
            "% ≠0": [f"{p:.1f}%" for p in pct],
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


# ─────────────────────────────────────────────────────────────────────────────
# Contexto de dashboard (pre-agregado UNA vez por archivo + unidad)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DashboardContext:
    """Pre-agregados que el dashboard reutiliza en cada interacción."""

    unit_df: pl.DataFrame
    label_map: dict[str, str]
    desc_map: dict[str, str]
    per_id: pl.DataFrame
    all_ids: list[str]
    has_rolling: bool
    has_value_cols: bool


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
    ranking_n_series: int
    ranking_total_points: int
    metrics: dict[str, dict[str, float | int]]
    metrics_28: dict[str, float | int]
    chart: dict[str, Any]
    detail: pl.DataFrame
    ds_min: dt.date | None
    ds_max: dt.date | None
    cutoff: dt.date
    nombre_nivel: str
    has_value_cols: bool
    has_rolling: bool
    store_id: str | None = None
    store_name: str | None = None


def has_value_columns(df: pl.DataFrame) -> bool:
    cols = set(df.columns)
    return "value" in cols and "valuehat" in cols


def build_dashboard_context(df: pl.DataFrame, unidad: str) -> DashboardContext:
    """
    Pre-agregado por (archivo, unidad). Construir esto UNA sola vez (y
    cachearlo) es lo que hace rápido al dashboard: a partir de aquí cada
    selección solo filtra subconjuntos → O(candidatos) + O(serie elegida).
    """
    has_value = has_value_columns(df)
    has_rolling = "yhat28" in df.columns or "valuehat28" in df.columns
    unit_df = prepare_unit_df(df, unidad, has_value)
    label_map, desc_map = build_label_maps(df)
    per_id = aggregate_wmape_by_id(unit_df)
    all_ids = sorted(unit_df["unique_id"].unique().to_list())
    return DashboardContext(
        unit_df=unit_df,
        label_map=label_map,
        desc_map=desc_map,
        per_id=per_id,
        all_ids=all_ids,
        has_rolling=has_rolling,
        has_value_cols=has_value,
    )


def _store_context(selected_id: str) -> tuple[str | None, str | None]:
    """contexto de tienda cuando se navega a nivel SKU (sec||store||sku)."""
    parts = selected_id.split("||")
    if len(parts) < 3:
        return None, None
    sec, store = parts[0], parts[1]
    name = settings.SECCIONES.get(sec, {}).get("local_names", {}).get(store, "")
    return (store, name) if store else (None, None)


def _resolve_n_points(
    ctx: DashboardContext, candidatos: list[str], seccion: str
) -> int:
    """Total de puntos del spine del nivel (settings alineado al panel real)."""
    hz = settings.section_horizons(seccion)
    n_spine = (hz["forecast_end"] - hz["train_start"]).days + 1
    if ctx.per_id.height and candidatos:
        sub = ctx.per_id.filter(pl.col("unique_id").is_in(candidatos))
        if sub.height:
            n_data = int(sub["n_ds"].max())
            if n_data > n_spine:
                n_spine = n_data
    return int(n_spine or 0)


def prepare_dashboard_state_from_context(
    ctx: DashboardContext,
    *,
    unidad: str,
    freq: str,
    selected_id: str,
    cutoff_date: dt.date | None,
    candidatos: list[str] | None,
    nombre_nivel: str,
) -> DashboardView:
    """
    Construye el ViewModel a partir del contexto pre-agregado.

    Por interacción solo toca: la serie seleccionada (gráfico, métricas,
    detalle) y la tabla pre-agregada `per_id` filtrada por candidatos
    (ranking). Sin re-procesar el panel completo.
    """
    unit_df = ctx.unit_df
    df_daily = filter_series(unit_df, selected_id)
    seccion = selected_id.split("||")[0]
    horizons = resolve_horizons(df_daily, seccion, set(unit_df.columns))

    train_end = horizons["train_end"]
    test_end = horizons["test_end"]
    fcst_start = horizons["forecast_start"]
    fcst_end = horizons["forecast_end"]

    ds_min, ds_max = ds_range(df_daily)
    cutoff = cutoff_date or train_end or ds_min or dt.date.today()
    if train_end and ds_min and ds_max and not (ds_min <= cutoff <= ds_max):
        cutoff = ds_min

    df_view = aggregate_temporal(df_daily, freq)

    if candidatos is None:
        candidatos = opciones_nivel(ctx.all_ids, selected_id)
    n_points_total = _resolve_n_points(ctx, candidatos, seccion)
    ranking_wmape = ranking_wmape_table(
        ctx.per_id, selected_id, ctx.desc_map, n_points_total, candidatos=candidatos
    )

    metrics = metrics_in_out_total(df_view, cutoff, test_end or cutoff)
    metrics_28 = (
        metrics_rolling28(df_view)
        if ctx.has_rolling
        else {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    )
    chart = build_chart_series(
        df_view,
        test_end or cutoff,
        fcst_start or (cutoff + dt.timedelta(days=1)),
        fcst_end or (cutoff + dt.timedelta(days=28)),
        cutoff,
        "period_type" in unit_df.columns,
    )
    detail = format_detail_display(detail_view(df_view))
    label = ctx.label_map.get(selected_id, settings.display_label(selected_id))
    store_id, store_name = _store_context(selected_id)

    return DashboardView(
        selected_id=selected_id,
        label=label,
        seccion=seccion,
        unidad=unidad,
        freq=freq,
        horizons=horizons,
        ranking_wmape=ranking_wmape,
        ranking_n_series=len(candidatos),
        ranking_total_points=n_points_total,
        metrics=metrics,
        metrics_28=metrics_28,
        chart=chart,
        detail=detail,
        ds_min=ds_min,
        ds_max=ds_max,
        cutoff=cutoff,
        nombre_nivel=nombre_nivel,
        has_value_cols=ctx.has_value_cols,
        has_rolling=ctx.has_rolling,
        store_id=store_id,
        store_name=store_name,
    )


def prepare_dashboard_state(
    res_df: pl.DataFrame,
    *,
    unidad: str,
    freq: str,
    selected_id: str,
    cutoff_date: dt.date | None = None,
    candidatos: list[str] | None = None,
    nombre_nivel: str,
    label_map: dict[str, str] | None = None,
    desc_map: dict[str, str] | None = None,
) -> DashboardView:
    """
    API de compatibilidad / tests: construye el contexto desde `res_df` y
    delega en `prepare_dashboard_state_from_context`. El dashboard productivo
    usa el contexto cacheado (`build_dashboard_context` + `..._from_context`).
    """
    ctx = build_dashboard_context(res_df, unidad)
    if label_map is not None and desc_map is not None:
        ctx.label_map = label_map
        ctx.desc_map = desc_map
    return prepare_dashboard_state_from_context(
        ctx,
        unidad=unidad,
        freq=freq,
        selected_id=selected_id,
        cutoff_date=cutoff_date,
        candidatos=candidatos,
        nombre_nivel=nombre_nivel,
    )
