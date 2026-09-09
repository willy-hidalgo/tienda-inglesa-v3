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
    "_compare_v11",
    "_compare_v12",
    "_v12_selected",
    "_model_family",
    "_cal_factor",
    "_cal_blocks",
    "_shape_applied",
    "_cal_applied",
    "_meta_probability",
    "_meta_available",
    "_meta_training_rows",
    "_meta_recent_gain",
    "_meta_weighted_gain",
    "_meta_win_rate",
    "_meta_top_driver",
    "_meta_threshold",
    "_meta_portfolio_mode",
    "_meta_bias_guard_pass",
    "_meta_policy_available",
    "_meta_policy_utility_gain",
    "_value_safety_dominance_pass",
    "_value_safety_recent_confirmations",
    "_value_safety_recent_blocks",
    "_value_safety_bias_coverage",
    "_value_safety_bias_coverage_threshold",
    "_value_safety_bias_coverage_pass",
    "_value_safety_meta_margin_pass",
    "_value_safety_best_all_mode",
    "_value_safety_reason",
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
    """Normaliza la unidad y expone aliases genéricos de diagnóstico v12.8.

    Los aliases permiten que dashboard/backend no dupliquen lógica para Qty/Valor.
    Si el parquet es anterior a v12.8 simplemente no se crean.
    """
    is_value = unidad.startswith("Valor") and has_value
    exprs: list[pl.Expr] = []
    if is_value:
        exprs.extend([pl.col("value").alias("y"), pl.col("valuehat").alias("yhat")])
        if "valuehat28" in df.columns:
            exprs.append(pl.col("valuehat28").alias("yhat28"))

    mapping = {
        # Existing official bottom-up metric components, exposed for the chart
        # diagnostic without changing metric semantics.
        "_bu_abs_error_daily": "bu_abs_error_daily_value" if is_value else "bu_abs_error_daily_y",
        "_bu_abs_y_daily": "bu_abs_y_daily_value" if is_value else "bu_abs_y_daily_y",
        "_bu_signed_error_daily": "bu_signed_error_daily_value" if is_value else "bu_signed_error_daily_y",
        "_compare_v11": "v11_valuehat_raw_before_v12" if is_value else "v11_yhat_raw_before_v12",
        "_compare_v12": "v12_candidate_valuehat_raw" if is_value else "v12_candidate_yhat_raw",
        "_v12_selected": "v12_selected_value" if is_value else "v12_selected_y",
        "_model_family": "leaf_model_family_value" if is_value else "leaf_model_family_y",
        "_cal_factor": "v12_sku_level_calibration_factor_value" if is_value else "v12_sku_level_calibration_factor_y",
        "_cal_blocks": "v12_sku_level_calibration_blocks_value" if is_value else "v12_sku_level_calibration_blocks_y",
        "_shape_applied": "v12_sku_shape_applied_value" if is_value else "v12_sku_shape_applied_y",
        "_cal_applied": "v12_sku_level_calibration_applied_value" if is_value else "v12_sku_level_calibration_applied_y",
        "_meta_probability": "v12_meta_probability_value" if is_value else "v12_meta_probability_y",
        "_meta_available": "v12_meta_model_available_value" if is_value else "v12_meta_model_available_y",
        "_meta_training_rows": "v12_meta_training_rows_value" if is_value else "v12_meta_training_rows_y",
        "_meta_recent_gain": "v12_meta_recent_gain_value" if is_value else "v12_meta_recent_gain_y",
        "_meta_weighted_gain": "v12_meta_weighted_gain_value" if is_value else "v12_meta_weighted_gain_y",
        "_meta_win_rate": "v12_meta_win_rate_value" if is_value else "v12_meta_win_rate_y",
        "_meta_top_driver": "v12_meta_top_driver_value" if is_value else "v12_meta_top_driver_y",
        "_meta_threshold": "v12_meta_threshold_value" if is_value else "v12_meta_threshold_y",
        "_meta_portfolio_mode": "v12_meta_portfolio_mode_value" if is_value else "v12_meta_portfolio_mode_y",
        "_meta_bias_guard_pass": "v12_meta_bias_guard_pass_value" if is_value else "v12_meta_bias_guard_pass_y",
        "_meta_policy_available": "v12_meta_policy_available_value" if is_value else "v12_meta_policy_available_y",
        "_meta_policy_utility_gain": "v12_meta_policy_utility_gain_value" if is_value else "v12_meta_policy_utility_gain_y",
    }
    if is_value:
        mapping.update({
            "_value_safety_dominance_pass": "v12_value_safety_dominance_pass",
            "_value_safety_recent_confirmations": "v12_value_safety_recent_confirmations",
            "_value_safety_recent_blocks": "v12_value_safety_recent_blocks",
            "_value_safety_bias_coverage": "v12_value_safety_bias_coverage",
            "_value_safety_bias_coverage_threshold": "v12_value_safety_bias_coverage_threshold",
            "_value_safety_bias_coverage_pass": "v12_value_safety_bias_coverage_pass",
            "_value_safety_meta_margin_pass": "v12_value_safety_meta_margin_pass",
            "_value_safety_best_all_mode": "v12_value_safety_best_all_mode",
            "_value_safety_reason": "v12_value_safety_reason",
            "_v129_wf_enabled": "v129_value_wf_enabled",
            "_v129_wf_available": "v129_value_wf_available",
            "_v129_wf_folds": "v129_value_wf_folds",
            "_v129_wf_win_rate": "v129_value_wf_win_rate",
            "_v129_wf_median_gain": "v129_value_wf_median_gain",
            "_v129_wf_worst_gain": "v129_value_wf_worst_gain",
            "_v129_wf_weighted_gain": "v129_value_wf_weighted_gain",
            "_v129_wf_weighted_utility_gain": "v129_value_wf_weighted_utility_gain",
            "_v129_wf_bias_worsen_max": "v129_value_wf_bias_worsen_max",
            "_v129_wf_meta_folds": "v129_value_wf_meta_folds",
            "_v129_wf_reason": "v129_value_wf_reason",
        })
    for alias, source in mapping.items():
        if source in df.columns:
            expr = pl.col(source)
            if alias in {"_compare_v11", "_compare_v12"}:
                expr = expr.round(2 if is_value else 0)
            exprs.append(expr.alias(alias))
    return df.with_columns(exprs) if exprs else df


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
    # Forecasts comparables se agregan igual que yhat. Metadatos diagnósticos
    # usan first/max para conservar contexto sin alterar totales.
    for col in ("_compare_v11", "_compare_v12", "_bu_abs_error_daily", "_bu_abs_y_daily", "_bu_signed_error_daily"):
        if col in df.columns:
            aggs.append(pl.col(col).sum())
    for col in ("_v12_selected", "_shape_applied", "_cal_applied", "_meta_available", "_meta_bias_guard_pass", "_meta_policy_available", "_value_safety_dominance_pass", "_value_safety_bias_coverage_pass", "_value_safety_meta_margin_pass"):
        if col in df.columns:
            aggs.append(pl.col(col).max())
    for col in ("_model_family", "_cal_factor", "_cal_blocks", "_meta_probability", "_meta_training_rows", "_meta_recent_gain", "_meta_weighted_gain", "_meta_win_rate", "_meta_top_driver", "_meta_threshold", "_meta_portfolio_mode", "_meta_policy_utility_gain", "_value_safety_recent_confirmations", "_value_safety_recent_blocks", "_value_safety_bias_coverage", "_value_safety_bias_coverage_threshold", "_value_safety_best_all_mode", "_value_safety_reason"):
        if col in df.columns:
            aggs.append(pl.col(col).first())
    # Nunca mezclar periodos semánticos dentro de un bucket semanal/mensual.
    # Un group_by solo por ds truncado podía sumar días in_sample + out_sample
    # (o out_sample + forecast_only) cuando el corte caía dentro de la semana/mes.
    # Eso era especialmente visible en Sec.23, cuyo OOS tiene fechas distintas.
    group_keys = ["_bucket"]
    if "period_type" in df.columns:
        group_keys.append("period_type")
    for col in ("sku_desc", "store_name", "seccion", "unique_id"):
        if col in df.columns:
            aggs.append(pl.col(col).first())

    out = (
        df.with_columns(pl.col("ds").dt.truncate(period).alias("_bucket"))
        .group_by(group_keys)
        .agg([pl.col("ds").min().alias("ds"), *aggs])
        .sort("ds")
        .with_columns((pl.col("y") - pl.col("yhat")).abs().alias("abs_error"))
        .drop("_bucket")
    )
    return out


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
    Hojas sku+tienda puntuables. Los días con y=0 permanecen para penalizar sobreforecast.

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
    )
    if leaves.height == 0:
        return leaves

    return leaves.with_columns(
        (pl.col("y") - pl.col("yhat")).abs().alias("_abs_error"),
        pl.col("y").abs().alias("_abs_y"),
        (pl.col("yhat") - pl.col("y")).alias("_signed_error"),
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
    cohort_mode: str | None = None,
) -> pl.DataFrame:
    """
    WMAPE bottom-up desde hojas sku+tienda.

    Para OOS, si ``cohort_mode`` es "active" (default configurable), los nodos
    tienda/sección/SKU puro se construyen únicamente con hojas activas:
      zero   = 0 días con venta
      sparse = 1..active_min_nonzero_days-1
      active = >= active_min_nonzero_days

    Las filas hoja se conservan todas y llevan ``metric_cohort`` para permitir
    auditar/mostrar sparse y zero-demand por separado.
    """
    schema = {
        "unique_id": pl.Utf8,
        "wmape": pl.Float64,
        "bias": pl.Float64,
        "sum_y": pl.Float64,
        "sum_yhat": pl.Float64,
        "sum_abs_y": pl.Float64,
        "sum_abs_error": pl.Float64,
        "sum_signed_error": pl.Float64,
        "n_points": pl.UInt32,
        "n_with_sales": pl.UInt32,
        "metric_cohort": pl.Utf8,
        "metric_active": pl.Boolean,
        "n_leaf_active": pl.UInt32,
        "n_leaf_sparse": pl.UInt32,
        "n_leaf_zero": pl.UInt32,
        "zero_forecast_sum": pl.Float64,
        "zero_abs_error_sum": pl.Float64,
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
    active_min = int(
        getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7)
    )
    if cohort_mode is None:
        is_oos_only = (
            period_types is not None
            and set(str(x) for x in period_types) == {"out_sample"}
        )
        cohort_mode = (
            str(getattr(settings, "OOS_METRIC_MODE", "active")).lower()
            if is_oos_only
            else "all"
        )
    cohort_mode = str(cohort_mode).lower()

    if "ds" in leaves.columns:
        n_sales_expr = (
            pl.col("ds").filter(pl.col("y") != 0).n_unique()
            .cast(pl.UInt32).alias("n_with_sales")
        )
    else:
        n_sales_expr = (
            (pl.col("y") != 0).sum().cast(pl.UInt32).alias("n_with_sales")
        )

    # Clasificación SIEMPRE a nivel hoja antes de cualquier agregación.
    leaf_base = (
        leaves.group_by("unique_id")
        .agg(
            pl.when(pl.col("y") != 0).then(pl.col("_abs_error")).otherwise(0.0)
            .sum().alias("sum_abs_error"),
            pl.when(pl.col("y") != 0).then(pl.col("_abs_y")).otherwise(0.0)
            .sum().alias("sum_abs_y"),
            pl.when(pl.col("y") != 0).then(pl.col("_signed_error")).otherwise(0.0)
            .sum().alias("sum_signed_error"),
            pl.col("y").sum().alias("sum_y"),
            pl.when(pl.col("y") != 0).then(pl.col("yhat")).otherwise(0.0)
            .sum().alias("sum_yhat"),
            pl.col("yhat").sum().alias("_sum_yhat_all"),
            pl.col("_abs_error").sum().alias("_sum_abs_error_all"),
            n_sales_expr,
            pl.col("_store_uid").first().alias("_store_uid"),
            pl.col("_seccion").first().alias("_seccion"),
            pl.col("_sku_uid").first().alias("_sku_uid"),
        )
        .with_columns(
            pl.when(pl.col("n_with_sales") == 0)
            .then(pl.lit("zero"))
            .when(pl.col("n_with_sales") < active_min)
            .then(pl.lit("sparse"))
            .otherwise(pl.lit("active"))
            .alias("metric_cohort"),
        )
        .with_columns(
            (pl.col("metric_cohort") == "active").alias("metric_active"),
            pl.when(pl.col("sum_abs_y") > 0)
            .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
            .otherwise(None)
            .alias("wmape"),
            pl.when(pl.col("sum_abs_y") > 0)
            .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
            .otherwise(None)
            .alias("bias"),
            n_spine_lit.alias("n_points"),
            pl.when(pl.col("metric_cohort") == "zero")
            .then(pl.col("_sum_yhat_all").clip(lower_bound=0.0))
            .otherwise(0.0)
            .alias("zero_forecast_sum"),
            pl.when(pl.col("metric_cohort") == "zero")
            .then(pl.col("_sum_abs_error_all"))
            .otherwise(0.0)
            .alias("zero_abs_error_sum"),
        )
    )

    # Diagnóstico de cohortes de cada alcance. Se calcula desde leaf_base para
    # no confundir nº de días con nº de hojas.
    def _cohort_summary(key: str) -> pl.DataFrame:
        return (
            leaf_base.group_by(key)
            .agg(
                (pl.col("metric_cohort") == "active").sum()
                .cast(pl.UInt32).alias("n_leaf_active"),
                (pl.col("metric_cohort") == "sparse").sum()
                .cast(pl.UInt32).alias("n_leaf_sparse"),
                (pl.col("metric_cohort") == "zero").sum()
                .cast(pl.UInt32).alias("n_leaf_zero"),
                pl.col("zero_forecast_sum").sum().alias("zero_forecast_sum"),
                pl.col("zero_abs_error_sum").sum().alias("zero_abs_error_sum"),
            )
        )

    # Hojas: métricas propias, aun si sparse/zero.
    leaf_summary = _cohort_summary("unique_id")
    out_leaf = (
        leaf_base.join(leaf_summary, on="unique_id", how="left")
        .select(
            "unique_id", "wmape", "bias", "sum_y", "sum_yhat",
            "sum_abs_y", "sum_abs_error", "sum_signed_error",
            "n_points", "n_with_sales", "metric_cohort", "metric_active",
            "n_leaf_active", "n_leaf_sparse", "n_leaf_zero",
            "zero_forecast_sum", "zero_abs_error_sum",
        )
    )

    eligible_ids = (
        leaf_base.filter(pl.col("metric_active"))
        .select("unique_id")
        if cohort_mode == "active"
        else leaf_base.select("unique_id")
    )
    parent_rows = leaves.join(eligible_ids, on="unique_id", how="semi")

    def _agg_parent(group_key: str) -> pl.DataFrame:
        # Si no hay hojas activas en el alcance, el nodo queda presente con
        # denominador 0 y wMAPE N/A, en vez de fabricar 0%.
        summary = _cohort_summary(group_key)
        if parent_rows.height:
            agg = (
                parent_rows.group_by(group_key)
                .agg(
                    pl.when(pl.col("y") != 0).then(pl.col("_abs_error")).otherwise(0.0)
                    .sum().alias("sum_abs_error"),
                    pl.when(pl.col("y") != 0).then(pl.col("_abs_y")).otherwise(0.0)
                    .sum().alias("sum_abs_y"),
                    pl.when(pl.col("y") != 0).then(pl.col("_signed_error")).otherwise(0.0)
                    .sum().alias("sum_signed_error"),
                    pl.col("y").sum().alias("sum_y"),
                    pl.when(pl.col("y") != 0).then(pl.col("yhat")).otherwise(0.0)
                    .sum().alias("sum_yhat"),
                    n_sales_expr,
                )
            )
        else:
            agg = pl.DataFrame()

        keys = leaf_base.select(group_key).unique()
        if agg.height:
            out = keys.join(agg, on=group_key, how="left")
        else:
            out = keys.with_columns(
                pl.lit(None).cast(pl.Float64).alias("sum_abs_error"),
                pl.lit(None).cast(pl.Float64).alias("sum_abs_y"),
                pl.lit(None).cast(pl.Float64).alias("sum_signed_error"),
                pl.lit(None).cast(pl.Float64).alias("sum_y"),
                pl.lit(None).cast(pl.Float64).alias("sum_yhat"),
                pl.lit(None).cast(pl.UInt32).alias("n_with_sales"),
            )
        out = (
            out.join(summary, on=group_key, how="left")
            .with_columns(
                pl.col("sum_abs_error").fill_null(0.0),
                pl.col("sum_abs_y").fill_null(0.0),
                pl.col("sum_signed_error").fill_null(0.0),
                pl.col("sum_y").fill_null(0.0),
                pl.col("sum_yhat").fill_null(0.0),
                pl.col("n_with_sales").fill_null(0).cast(pl.UInt32),
            )
            .with_columns(
                pl.when(pl.col("sum_abs_y") > 0)
                .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
                .otherwise(None)
                .alias("wmape"),
                pl.when(pl.col("sum_abs_y") > 0)
                .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
                .otherwise(None)
                .alias("bias"),
                n_spine_lit.alias("n_points"),
                pl.lit("active" if cohort_mode == "active" else "all")
                .alias("metric_cohort"),
                pl.lit(cohort_mode == "active").alias("metric_active"),
            )
        )
        if group_key != "unique_id":
            out = out.rename({group_key: "unique_id"})
        return out.select(
            "unique_id", "wmape", "bias", "sum_y", "sum_yhat",
            "sum_abs_y", "sum_abs_error", "sum_signed_error",
            "n_points", "n_with_sales", "metric_cohort", "metric_active",
            "n_leaf_active", "n_leaf_sparse", "n_leaf_zero",
            "zero_forecast_sum", "zero_abs_error_sum",
        )

    out_store = _agg_parent("_store_uid")
    out_sec = _agg_parent("_seccion")
    out_sku = _agg_parent("_sku_uid")

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
        "bias": pl.Float64,
        "sum_y": pl.Float64,
        "sum_yhat": pl.Float64,
        "sum_abs_y": pl.Float64,
        "sum_abs_error": pl.Float64,
        "sum_signed_error": pl.Float64,
        "n_points": pl.UInt32,
        "n_with_sales": pl.UInt32,
        "metric_cohort": pl.Utf8,
        "metric_active": pl.Boolean,
        "n_leaf_active": pl.UInt32,
        "n_leaf_sparse": pl.UInt32,
        "n_leaf_zero": pl.UInt32,
        "zero_forecast_sum": pl.Float64,
        "zero_abs_error_sum": pl.Float64,
    }
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    if ids is not None and len(ids) == 0:
        return pl.DataFrame(schema=schema)

    agg = wmape_bottom_up(
        df, n_fechas_spine=n_fechas_spine, period_types=period_types
    )

    # Compatibilidad con paneles/fixtures legacy que no materializan hojas
    # ``sec||T:store||S:sku``. En producción siempre se usa bottom-up desde
    # hojas; este fallback solo entra cuando no existe ninguna hoja exacta y
    # calcula la métrica directamente por unique_id, manteniendo la regla
    # vigente de excluir y=0 del cálculo de wMAPE/BIAS.
    if agg.height == 0 and "unique_id" in df.columns:
        legacy = df
        if "period_type" in legacy.columns:
            if period_types is not None:
                legacy = legacy.filter(pl.col("period_type").is_in(list(period_types)))
            else:
                legacy = legacy.filter(pl.col("period_type") != "forecast_only")
        legacy = legacy.filter(
            pl.col("y").is_not_null()
            & pl.col("yhat").is_not_null()
            & pl.col("y").is_finite()
            & pl.col("yhat").is_finite()
        )
        if legacy.height:
            n_spine = int(
                n_fechas_spine
                if n_fechas_spine is not None
                else (
                    legacy.select(pl.col("ds").n_unique()).item()
                    if "ds" in legacy.columns
                    else 0
                )
            )
            if "ds" in legacy.columns:
                n_sales_expr = (
                    pl.col("ds").filter(pl.col("y") != 0).n_unique()
                    .cast(pl.UInt32).alias("n_with_sales")
                )
            else:
                n_sales_expr = (
                    (pl.col("y") != 0).sum().cast(pl.UInt32).alias("n_with_sales")
                )
            agg = (
                legacy.group_by("unique_id")
                .agg(
                    pl.when(pl.col("y") != 0)
                    .then((pl.col("y") - pl.col("yhat")).abs())
                    .otherwise(0.0).sum().alias("sum_abs_error"),
                    pl.when(pl.col("y") != 0)
                    .then(pl.col("y").abs())
                    .otherwise(0.0).sum().alias("sum_abs_y"),
                    pl.when(pl.col("y") != 0)
                    .then(pl.col("yhat") - pl.col("y"))
                    .otherwise(0.0).sum().alias("sum_signed_error"),
                    pl.col("y").sum().alias("sum_y"),
                    pl.when(pl.col("y") != 0)
                    .then(pl.col("yhat"))
                    .otherwise(0.0).sum().alias("sum_yhat"),
                    n_sales_expr,
                )
                .with_columns(
                    pl.when(pl.col("sum_abs_y") > 0)
                    .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
                    .otherwise(None).alias("wmape"),
                    pl.when(pl.col("sum_abs_y") > 0)
                    .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
                    .otherwise(None).alias("bias"),
                    pl.lit(n_spine).cast(pl.UInt32).alias("n_points"),
                    pl.lit("legacy").alias("metric_cohort"),
                    pl.lit(True).alias("metric_active"),
                    pl.lit(0).cast(pl.UInt32).alias("n_leaf_active"),
                    pl.lit(0).cast(pl.UInt32).alias("n_leaf_sparse"),
                    pl.lit(0).cast(pl.UInt32).alias("n_leaf_zero"),
                    pl.lit(0.0).alias("zero_forecast_sum"),
                    pl.lit(0.0).alias("zero_abs_error_sum"),
                )
                .select(list(schema))
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
            pl.lit(None).cast(pl.Float64).alias("wmape"),
            pl.lit(None).cast(pl.Float64).alias("bias"),
            pl.lit(0.0).alias("sum_y"),
            pl.lit(0.0).alias("sum_yhat"),
            pl.lit(0.0).alias("sum_abs_y"),
            pl.lit(0.0).alias("sum_abs_error"),
            pl.lit(0.0).alias("sum_signed_error"),
            n_spine_lit.alias("n_points"),
            pl.lit(0).cast(pl.UInt32).alias("n_with_sales"),
            pl.lit("zero").alias("metric_cohort"),
            pl.lit(False).alias("metric_active"),
            pl.lit(0).cast(pl.UInt32).alias("n_leaf_active"),
            pl.lit(0).cast(pl.UInt32).alias("n_leaf_sparse"),
            pl.lit(0).cast(pl.UInt32).alias("n_leaf_zero"),
            pl.lit(0.0).alias("zero_forecast_sum"),
            pl.lit(0.0).alias("zero_abs_error_sum"),
        )
    missing = ids_df.join(agg.select("unique_id"), on="unique_id", how="anti")
    if missing.height == 0:
        return agg
    extra = missing.with_columns(
        pl.lit(None).cast(pl.Float64).alias("wmape"),
        pl.lit(None).cast(pl.Float64).alias("bias"),
        pl.lit(0.0).alias("sum_y"),
        pl.lit(0.0).alias("sum_yhat"),
        pl.lit(0.0).alias("sum_abs_y"),
        pl.lit(0.0).alias("sum_abs_error"),
        pl.lit(0.0).alias("sum_signed_error"),
        n_spine_lit.alias("n_points"),
        pl.lit(0).cast(pl.UInt32).alias("n_with_sales"),
        pl.lit("zero").alias("metric_cohort"),
        pl.lit(False).alias("metric_active"),
        pl.lit(0).cast(pl.UInt32).alias("n_leaf_active"),
        pl.lit(0).cast(pl.UInt32).alias("n_leaf_sparse"),
        pl.lit(0).cast(pl.UInt32).alias("n_leaf_zero"),
        pl.lit(0.0).alias("zero_forecast_sum"),
        pl.lit(0.0).alias("zero_abs_error_sum"),
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
    sum_abs_y = float(leaves["_abs_y"].sum())
    sum_err = float(leaves["_abs_error"].sum())
    wmape = (sum_err / sum_abs_y) if sum_abs_y > 0 else 0.0
    return {
        "wmape": wmape,
        "sum_y": sum_y,
        "sum_abs_y": sum_abs_y,
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
    selected_code: str | None = None,
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
    legacy_schema = "bias" not in tabla_base.columns
    empty_schema = {
        "Código": pl.Utf8,
        "Descripción": pl.Utf8,
        "wMAPE (%)": pl.Utf8,
        label_rot: pl.Utf8,
        "N puntos": pl.UInt32,
        "% ≠0": pl.Utf8,
        "unique_id": pl.Utf8,
    }
    if not legacy_schema:
        empty_schema = {
            "Código": pl.Utf8,
            "Descripción": pl.Utf8,
            "wMAPE (%)": pl.Utf8,
            "BIAS (%)": pl.Utf8,
            label_rot: pl.Utf8,
            "N puntos": pl.UInt32,
            "% ≠0": pl.Utf8,
            "Impacto error (%)": pl.Utf8,
            "Estado": pl.Utf8,
            "unique_id": pl.Utf8,
        }
    empty = pl.DataFrame(schema=empty_schema)
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
    # Compatibilidad con tablas antiguas/tests que solo traen wmape+sum_y.
    # El dashboard v11.0 usa siempre los componentes exactos materializados.
    if "sum_abs_y" not in enriched.columns and "sum_y" in enriched.columns:
        enriched = enriched.with_columns(
            pl.col("sum_y").abs().alias("sum_abs_y")
        )
    if "sum_abs_error" not in enriched.columns:
        enriched = enriched.with_columns(
            pl.col("sum_y").abs().mul(pl.col("wmape")).alias("sum_abs_error")
        )
    if "sum_signed_error" not in enriched.columns:
        enriched = enriched.with_columns(
            pl.lit(0.0).alias("sum_signed_error")
        )
    if "bias" not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(0.0).alias("bias"))
    if "sum_yhat" not in enriched.columns:
        enriched = enriched.with_columns(
            pl.col("sum_y").alias("sum_yhat")
        )

    if axis == "store":
        if fixed_peer is not None:
            # SKU seleccionado → comparar la misma hoja SKU entre tiendas.
            peer = str(fixed_peer)
            tabla = enriched.filter(
                pl.col("_store").is_not_null() & (pl.col("_sku") == peer)
            )
        else:
            # Sin SKU fijo → métricas bottom-up derivadas por tienda.
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
                if "metric_active" in leaves.columns:
                    leaves = leaves.filter(
                        pl.col("metric_active").fill_null(False)
                    )
                if exclude is not None:
                    leaves = leaves.filter(pl.col("_sku") != str(exclude))
                if leaves.height == 0:
                    return empty
                tabla = (
                    leaves.group_by("_sku")
                    .agg(
                        pl.col("sum_abs_error").sum().alias("sum_abs_error"),
                        pl.col("sum_abs_y").sum().alias("sum_abs_y"),
                        pl.col("sum_signed_error").sum().alias("sum_signed_error"),
                        pl.col("sum_y").sum().alias("sum_y"),
                        pl.col("sum_yhat").sum().alias("sum_yhat"),
                        pl.col("n_with_sales").sum().alias("n_with_sales"),
                        pl.col("n_points").first().alias("n_points"),
                    )
                    .with_columns(
                        pl.when(pl.col("sum_abs_y") > 0)
                        .then(pl.col("sum_abs_error") / pl.col("sum_abs_y"))
                        .otherwise(0.0)
                        .alias("wmape"),
                        pl.when(pl.col("sum_abs_y") > 0)
                        .then(pl.col("sum_signed_error") / pl.col("sum_abs_y"))
                        .otherwise(0.0)
                        .alias("bias"),
                        pl.concat_str(
                            [pl.lit(seccion), pl.lit("||S:"), pl.col("_sku")]
                        ).alias("unique_id"),
                    )
                )
                code_expr = pl.col("_sku")
                uid_expr = pl.col("unique_id")
                code_col = "_sku"

    scope_metric_rows = (
        tabla.filter(pl.col("metric_active").fill_null(False))
        if "metric_active" in tabla.columns else tabla
    )
    scope_abs_error = float(
        scope_metric_rows.select(pl.col("sum_abs_error").sum()).item()
        or 0.0
    )

    eligible_expr = (
        (pl.col("sum_abs_y") > 0)
        & pl.col("wmape").is_not_null()
        & (pl.col("wmape") > 0)
    )
    if "metric_active" in tabla.columns:
        eligible_expr = eligible_expr & pl.col("metric_active").fill_null(False)
    min_nonzero = int(getattr(settings, "RANKING_SKU_MIN_NONZERO_POINTS", 15))
    if axis == "sku" and not legacy_schema:
        eligible_expr = eligible_expr & (pl.col("n_with_sales") >= min_nonzero)
    tabla = tabla.with_columns(eligible_expr.alias("_eligible"))

    selected_row = pl.DataFrame()
    if selected_code is not None:
        selected_row = tabla.filter(
            pl.col(code_col).cast(pl.Utf8) == str(selected_code)
        )
    ranked = tabla.filter(pl.col("_eligible"))
    if selected_row.height:
        ranked = pl.concat(
            [ranked, selected_row], how="diagonal_relaxed"
        ).unique(subset=[code_col], keep="first", maintain_order=True)
    tabla = ranked
    if tabla.height == 0:
        return empty
    tabla = tabla.sort("wmape", descending=False, nulls_last=True).unique(
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
    impacts = [
        (
            float(err) / scope_abs_error * 100.0
            if scope_abs_error > 0
            else 0.0
        )
        for err in tabla["sum_abs_error"].to_list()
    ]

    descriptions: list[str] = []
    for u, code in zip(uids2, codes):
        if axis == "store":
            store_uid = settings.make_unique_id(seccion, store=str(code))
            d = desc_map.get(store_uid, "")
        else:
            sku_uid = settings.make_unique_id(seccion, sku=str(code))
            d = desc_map.get(sku_uid, "") or desc_map.get(u, "")
            if not d:
                d = next(
                    (
                        desc_map[k]
                        for k in desc_map
                        if f"||S:{code}" in k and desc_map[k]
                    ),
                    "",
                )
        descriptions.append(d)

    out = pl.DataFrame(
        {
            "Código": codes,
            "Descripción": descriptions,
            "wMAPE (%)": [
                f"{float(w) * 100:,.2f}" if w is not None else "N/A"
                for w in tabla["wmape"].to_list()
            ],
            "BIAS (%)": [
                f"{float(b) * 100:+,.2f}" if b is not None else "N/A"
                for b in tabla["bias"].to_list()
            ],
            label_rot: [f"{v:,.2f}" for v in tabla["sum_y"].to_list()],
            "N puntos": [int(x) for x in n_with_sales],
            "% ≠0": [f"{p:,.1f}" for p in pct],
            "Impacto error (%)": [f"{v:,.1f}" for v in impacts],
            "Estado": [
                "✓ criterio"
                if bool(v)
                else f"fuera criterio (<{min_nonzero} días ≠0)"
                for v in tabla["_eligible"].to_list()
            ],
            "unique_id": uids2,
        }
    )
    if legacy_schema:
        return out.select(
            "Código", "Descripción", "wMAPE (%)", label_rot,
            "N puntos", "% ≠0", "unique_id",
        )
    return out


def calcular_metricas(df: pl.DataFrame) -> tuple[float, float, int]:
    """
    wMAPE = Σ|y−ŷ| / Σ|y| sobre filas finitas con y!=0.
    Los días sin venta no participan en wMAPE/BIAS, conforme al contrato
    de negocio y a las métricas OOS bottom-up del pipeline.
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
    sum_abs_y = float(scored["y"].abs().sum())
    if sum_abs_y == 0:
        return 0.0, 0.0, 0
    wmape = float(scored["abs_error"].sum()) / sum_abs_y
    bias = float((scored["yhat"] - scored["y"]).sum()) / sum_abs_y
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
) -> dict[str, dict[str, float | int | bool | str | None]]:
    """
    Métricas oficiales por cohortes OOS desde hojas SKU+tienda.

    La cohorte se determina POR HOJA usando exclusivamente el OOS:
      zero   : 0 días con venta
      sparse : 1..OOS_ACTIVE_MIN_NONZERO_DAYS-1
      active : >= OOS_ACTIVE_MIN_NONZERO_DAYS

    ``out`` es la métrica oficial ACTIVE. ``out_all`` conserva el impacto de
    todas las hojas, y zero-demand se reporta aparte porque wMAPE es indefinido
    cuando Σ|y|=0.
    """
    leaves = _scored_leaves(unit_df)
    z = {
        "wmape": None, "bias": None, "n": 0, "defined": False,
        "sum_abs_y": 0.0, "sum_abs_error": 0.0,
        "sum_signed_error": 0.0,
    }
    if leaves.height == 0:
        return {
            "in": dict(z), "out": dict(z), "total": dict(z),
            "out_all": dict(z), "out_sparse": dict(z),
            "zero_demand": {
                "leaf_count": 0, "forecast_sum": 0.0,
                "abs_error_sum": 0.0, "impact_vs_active": 0.0,
            },
            "cohorts": {"active": 0, "sparse": 0, "zero": 0},
        }

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
        return {
            "in": dict(z), "out": dict(z), "total": dict(z),
            "out_all": dict(z), "out_sparse": dict(z),
            "zero_demand": {
                "leaf_count": 0, "forecast_sum": 0.0,
                "abs_error_sum": 0.0, "impact_vs_active": 0.0,
            },
            "cohorts": {"active": 0, "sparse": 0, "zero": 0},
        }

    if "period_type" in leaves.columns:
        in_rows = leaves.filter(pl.col("period_type") == "in_sample")
        oos_rows = leaves.filter(pl.col("period_type") == "out_sample")
        total_rows = leaves.filter(
            pl.col("period_type").is_in(["in_sample", "out_sample"])
        )
    else:
        d = pl.col("ds").cast(pl.Date)
        in_rows = leaves.filter(d < cutoff)
        oos_rows = leaves.filter((d >= cutoff) & (d <= test_end))
        total_rows = leaves.filter(d <= test_end)

    active_min = int(getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7))
    if oos_rows.height:
        if "ds" in oos_rows.columns:
            cohort_df = (
                oos_rows.group_by("unique_id")
                .agg(
                    pl.col("ds").filter(pl.col("y") != 0)
                    .n_unique().cast(pl.UInt32).alias("_nz"),
                    pl.col("_abs_y").sum().alias("_oos_abs_y"),
                    pl.col("yhat").clip(lower_bound=0.0)
                    .sum().alias("_oos_forecast"),
                    pl.col("_abs_error").sum().alias("_oos_abs_error"),
                )
            )
        else:
            cohort_df = (
                oos_rows.group_by("unique_id")
                .agg(
                    (pl.col("y") != 0).sum().cast(pl.UInt32).alias("_nz"),
                    pl.col("_abs_y").sum().alias("_oos_abs_y"),
                    pl.col("yhat").clip(lower_bound=0.0)
                    .sum().alias("_oos_forecast"),
                    pl.col("_abs_error").sum().alias("_oos_abs_error"),
                )
            )
        cohort_df = cohort_df.with_columns(
            pl.when(pl.col("_nz") == 0)
            .then(pl.lit("zero"))
            .when(pl.col("_nz") < active_min)
            .then(pl.lit("sparse"))
            .otherwise(pl.lit("active"))
            .alias("_cohort")
        )
    else:
        cohort_df = leaves.select("unique_id").unique().with_columns(
            pl.lit("zero").alias("_cohort"),
            pl.lit(0).cast(pl.UInt32).alias("_nz"),
            pl.lit(0.0).alias("_oos_abs_y"),
            pl.lit(0.0).alias("_oos_forecast"),
            pl.lit(0.0).alias("_oos_abs_error"),
        )

    def _ids(kind: str) -> pl.DataFrame:
        return cohort_df.filter(pl.col("_cohort") == kind).select("unique_id")

    active_ids = _ids("active")
    sparse_ids = _ids("sparse")
    zero_ids = _ids("zero")

    def _stats(
        frame: pl.DataFrame,
        label: str,
        *,
        include_zero_error: bool = False,
    ) -> dict[str, float | int | bool | str | None]:
        """Métrica de un tramo respetando la definición oficial y!=0.

        ``_scored_leaves`` conserva y=0 para clasificar cohortes y auditar
        zero-demand.  Esos días NO deben entrar al numerador del wMAPE/BIAS
        oficial.  ``include_zero_error`` se reserva para el diagnóstico
        complementario ``out_all`` que cuantifica explícitamente ese impacto.
        """
        if frame.height == 0:
            return {**z, "cohort": label}
        metric_rows = (
            frame
            if include_zero_error
            else frame.filter(pl.col("y") != 0)
        )
        den = float(frame["_abs_y"].sum())
        ae = float(metric_rows["_abs_error"].sum()) if metric_rows.height else 0.0
        se = float(metric_rows["_signed_error"].sum()) if metric_rows.height else 0.0
        n = (
            int(metric_rows.select(pl.col("ds").n_unique()).item())
            if "ds" in metric_rows.columns and metric_rows.height
            else int(metric_rows.height)
        )
        return {
            "wmape": (ae / den) if den > 0 else None,
            "bias": (se / den) if den > 0 else None,
            "n": n,
            "defined": den > 0,
            "sum_abs_y": den,
            "sum_abs_error": ae,
            "sum_signed_error": se,
            "cohort": label,
        }

    active_in = in_rows.join(active_ids, on="unique_id", how="semi")
    active_oos = oos_rows.join(active_ids, on="unique_id", how="semi")
    active_total = total_rows.join(active_ids, on="unique_id", how="semi")
    sparse_oos = oos_rows.join(sparse_ids, on="unique_id", how="semi")

    active_den = float(active_oos["_abs_y"].sum()) if active_oos.height else 0.0
    zero_meta = cohort_df.filter(pl.col("_cohort") == "zero")
    zero_forecast = (
        float(zero_meta["_oos_forecast"].sum()) if zero_meta.height else 0.0
    )
    zero_error = (
        float(zero_meta["_oos_abs_error"].sum()) if zero_meta.height else 0.0
    )

    return {
        "in": _stats(active_in, "active"),
        "out": _stats(active_oos, "active"),
        "total": _stats(active_total, "active"),
        "out_all": _stats(oos_rows, "all", include_zero_error=True),
        "out_sparse": _stats(sparse_oos, "sparse"),
        "zero_demand": {
            "leaf_count": int(zero_meta.height),
            "forecast_sum": zero_forecast,
            "abs_error_sum": zero_error,
            "impact_vs_active": (
                zero_forecast / active_den if active_den > 0 else 0.0
            ),
        },
        "cohorts": {
            "active": int(active_ids.height),
            "sparse": int(sparse_ids.height),
            "zero": int(zero_ids.height),
        },
    }


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
    # Los artefactos/fixtures pueden traer explícitamente sus horizontes y no
    # necesariamente contienen 56+ días. Priorizar esa metadata evita forzar
    # el contrato production 28x28 sobre vistas recortadas del dashboard.
    explicit = {
        key: meta_date_from_df(df_daily, key, None, available_cols)
        for key in (
            "train_end", "test_start", "test_end", "forecast_start", "forecast_end"
        )
    }
    if explicit["test_start"] is not None and explicit["test_end"] is not None:
        fcst_start = explicit["forecast_start"]
        if fcst_start is None or fcst_start <= explicit["test_end"]:
            fcst_start = explicit["test_end"] + dt.timedelta(days=1)
        fcst_end = explicit["forecast_end"]
        if fcst_end is None:
            fcst_end = fcst_start + dt.timedelta(
                days=int(getattr(settings, "RLS_BLOCK_DAYS", 28)) - 1
            )
        return {
            "train_end": explicit["train_end"],
            "test_start": explicit["test_start"],
            "test_end": explicit["test_end"],
            "forecast_start": fcst_start,
            "forecast_end": fcst_end,
        }

    first_d = df_daily["ds"].min() if df_daily.height else settings.FECHAS_TRAIN[0]
    if isinstance(first_d, dt.datetime):
        first_d = first_d.date()
    last_actual = None
    if df_daily.height and "ds" in df_daily.columns:
        actual_rows = df_daily
        if "period_type" in df_daily.columns:
            actual_rows = df_daily.filter(pl.col("period_type") != "forecast_only")
        if actual_rows.height:
            last_actual = actual_rows["ds"].max()
            if isinstance(last_actual, dt.datetime):
                last_actual = last_actual.date()
    hz = settings.section_horizons(seccion, first_d, last_actual)
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
    """Series listas para Plotly, separando in-sample, OOS y forecast-only."""
    hist, fcst = split_hist_forecast(
        df_view, test_end, forecast_start, forecast_end, has_period
    )
    d = pl.col("ds").cast(pl.Date)
    if has_period and "period_type" in df_view.columns:
        # Defensa de coherencia temporal: period_type es la etiqueta semántica,
        # pero los límites de la sección son el contrato visual. Nunca permitir
        # que una fila etiquetada in_sample se dibuje dentro del OOS de esa sección.
        ins = df_view.filter((pl.col("period_type") == "in_sample") & (d <= cutoff))
        oos = df_view.filter(
            (pl.col("period_type") == "out_sample")
            & (d > cutoff)
            & (d <= test_end)
        )
        fcst = fcst.filter(
            (pl.col("ds").cast(pl.Date) >= forecast_start)
            & (pl.col("ds").cast(pl.Date) <= forecast_end)
        )
    else:
        ins = hist.filter(d <= cutoff)
        oos = hist.filter((d > cutoff) & (d <= test_end))

    def _col(df: pl.DataFrame, name: str):
        if name in df.columns and df.height:
            return df[name].to_list()
        return []

    return {
        # Compatibilidad con dashboards/tests previos.
        "hist_ds": _col(hist, "ds"),
        "hist_y": _col(hist, "y"),
        "hist_yhat": _col(hist, "yhat"),
        "hist_yhat28": _col(hist, "yhat28"),
        # Periodos explícitos v12.7.
        "in_ds": _col(ins, "ds"),
        "in_y": _col(ins, "y"),
        "in_yhat": _col(ins, "yhat"),
        "in_yhat28": _col(ins, "yhat28"),
        "oos_ds": _col(oos, "ds"),
        "oos_y": _col(oos, "y"),
        "oos_yhat": _col(oos, "yhat"),
        "oos_yhat28": _col(oos, "yhat28"),
        "oos_v11": _col(oos, "_compare_v11"),
        "oos_v12": _col(oos, "_compare_v12"),
        "fcst_ds": _col(fcst, "ds"),
        "fcst_yhat": _col(fcst, "yhat"),
        "fcst_yhat28": _col(fcst, "yhat28"),
        # Exact daily components behind the existing bottom-up metric.
        "in_bu_abs_error": _col(ins, "_bu_abs_error_daily"),
        "in_bu_abs_y": _col(ins, "_bu_abs_y_daily"),
        "oos_bu_abs_error": _col(oos, "_bu_abs_error_daily"),
        "oos_bu_abs_y": _col(oos, "_bu_abs_y_daily"),
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
        & pl.col("yhat28").is_not_null()
        & pl.col("y").is_finite()
        & pl.col("yhat28").is_finite()
        & (pl.col("y") != 0)
    )
    if scored.height == 0:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    sum_abs_y = float(scored["y"].abs().sum())
    if sum_abs_y == 0:
        return {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    ae = float((scored["y"] - scored["yhat28"]).abs().sum())
    bias = float((scored["yhat28"] - scored["y"]).sum()) / sum_abs_y
    return {"wmape_28": ae / sum_abs_y, "bias_28": bias, "n": scored.height}
