"""
Artefactos precalculados del dashboard.
=======================================
Objetivo: el Streamlit solo lee y pinta. Cero WMAPE / agregaciones pesadas
en el hot path.

Salida (junto al forecast.parquet, o en settings.DASHBOARD_DIR si existe):

  dashboard/
    index.json              — secciones, tiendas, skus, horizontes, flags
                              (SIN label_map: va en labels.parquet)
    labels.parquet          — unique_id → label, description
    metrics.parquet         — una fila por (unique_id, unidad); wMAPE = out-of-sample
    metrics_in_sample.parquet — misma estructura para métricas in-sample
    series/seccion=<s>/...  — panel slim particionado por sección
                              (predicate pushdown + menos I/O)

Uso:
  python -m app.dashboard_artifacts
  python -m app.dashboard_artifacts --forecast path/to/forecast.parquet
  from app.dashboard_artifacts import build_artifacts, load_index, load_metrics, load_series
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

try:
    from app import backend
except ImportError:  # pragma: no cover
    import backend  # type: ignore

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 20

SERIES_COLS_PREFERRED = [
    "unique_id",
    "ds",
    "y",
    "yhat",
    "yhat28",
    "value",
    "valuehat",
    "valuehat28",
    "period_type",
    "update_block_days",
    "sku_desc",
    "store_name",
    "seccion",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "forecast_start",
    "forecast_end",
    # Dashboard v12.9.11 BU visual contract: daily leaf-level error components.
    # These do NOT redefine official wMAPE; they expose the exact bottom-up
    # numerator/denominator contributions behind the existing metric.
    "bu_abs_error_daily_y",
    "bu_abs_y_daily_y",
    "bu_signed_error_daily_y",
    "bu_abs_error_daily_value",
    "bu_abs_y_daily_value",
    "bu_signed_error_daily_value",
    "ses_level_y",
    "ses_level_value",
    "driver_effect",
    "driver_effect_value",
    "driver_factor_y",
    "driver_factor_value",
    "driver_strength_y",
    "driver_strength_value",
    "ses_recent28_y",
    "ses_recent28_value",
    "ses_recent14_y",
    "ses_recent14_value",
    "ses_recent28_coverage_y",
    "ses_recent28_coverage_value",
    "ses_regime_anchor_y",
    "ses_regime_anchor_value",
    "ses_stability_reference_y",
    "ses_stability_reference_value",
    "ses_stability_guard_y",
    "ses_stability_guard_value",
    "rls_metric_eligible",
    "rls_block",
    "rls_train_days",
    # v12.8: candidatos, meta-selector y challengers para diagnóstico visual.
    "v11_yhat_raw_before_v12",
    "v11_valuehat_raw_before_v12",
    "v12_candidate_yhat_raw",
    "v12_candidate_valuehat_raw",
    "v12_selected_y",
    "v12_selected_value",
    "leaf_model_family_y",
    "leaf_model_family_value",
    "v12_sku_forecast_uncalibrated_y",
    "v12_sku_forecast_uncalibrated_value",
    "v12_sku_forecast_calibrated_challenger_y",
    "v12_sku_forecast_calibrated_challenger_value",
    "v12_sku_level_calibration_applied_y",
    "v12_sku_level_calibration_applied_value",
    "v12_sku_forecast_base_y",
    "v12_sku_forecast_base_value",
    "v12_sku_forecast_y",
    "v12_sku_forecast_value",
    "v12_sku_level_calibration_raw_ratio_y",
    "v12_sku_level_calibration_raw_ratio_value",
    "v12_sku_level_calibration_factor_y",
    "v12_sku_level_calibration_factor_value",
    "v12_sku_level_calibration_blocks_y",
    "v12_sku_level_calibration_blocks_value",
    "v12_sku_shape_applied_y",
    "v12_sku_shape_applied_value",
    "v12_sku_shape_model_y",
    "v12_sku_shape_model_value",
    "v12_validation_improvement_y",
    "v12_validation_improvement_value",
    "v12_validation_utility_improvement_y",
    "v12_validation_utility_improvement_value",
    "v12_validation_recent_improvement_y",
    "v12_validation_recent_improvement_value",
    "v12_validation_recent_utility_improvement_y",
    "v12_validation_recent_utility_improvement_value",
    "v12_meta_probability_y",
    "v12_meta_probability_value",
    "v12_meta_model_available_y",
    "v12_meta_model_available_value",
    "v12_meta_training_rows_y",
    "v12_meta_training_rows_value",
    "v12_meta_recent_gain_y",
    "v12_meta_recent_gain_value",
    "v12_meta_weighted_gain_y",
    "v12_meta_weighted_gain_value",
    "v12_meta_win_rate_y",
    "v12_meta_win_rate_value",
    "v12_meta_top_driver_y",
    "v12_meta_top_driver_value",
    "v12_meta_bias11_y",
    "v12_meta_bias11_value",
    "v12_meta_bias12_y",
    "v12_meta_bias12_value",
    "v12_meta_bias_guard_pass_y",
    "v12_meta_bias_guard_pass_value",
    "v12_meta_threshold_y",
    "v12_meta_threshold_value",
    "v12_meta_portfolio_mode_y",
    "v12_meta_portfolio_mode_value",
    "v12_meta_policy_available_y",
    "v12_meta_policy_available_value",
    "v12_meta_policy_utility_gain_y",
    "v12_meta_policy_utility_gain_value",
    "v12_value_safety_enabled",
    "v12_value_safety_dominance_pass",
    "v12_value_safety_recent_confirmations",
    "v12_value_safety_recent_blocks",
    "v12_value_safety_bias_coverage",
    "v12_value_safety_bias_coverage_threshold",
    "v12_value_safety_bias_coverage_pass",
    "v12_value_safety_meta_margin_pass",
    "v12_value_safety_best_all_mode",
    "v12_value_safety_reason",
    "v129_value_wf_enabled",
    "v129_value_wf_available",
    "v129_value_wf_folds",
    "v129_value_wf_win_rate",
    "v129_value_wf_median_gain",
    "v129_value_wf_worst_gain",
    "v129_value_wf_weighted_gain",
    "v129_value_wf_weighted_utility_gain",
    "v129_value_wf_bias_worsen_max",
    "v129_value_wf_meta_folds",
    "v129_value_wf_fold1_gain",
    "v129_value_wf_fold2_gain",
    "v129_value_wf_fold3_gain",
    "v129_value_wf_reason",
]

# Columnas mínimas para WMAPE (reduce picos de memoria al preparar unidades)
_WMAPE_COLS = ("unique_id", "ds", "y", "yhat", "period_type", "value", "valuehat", "valuehat28")


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _progress(msg: str, t0: float | None = None) -> None:
    """Log de avance con timestamp relativo y RSS si está disponible."""
    rss = _rss_mb()
    mem = f" | RSS={rss:.0f} MB" if rss >= 0 else ""
    if t0 is None:
        logger.info("%s%s", msg, mem)
    else:
        logger.info("%s (%.1fs)%s", msg, time.perf_counter() - t0, mem)


def artifacts_dir(forecast_path: Path | None = None) -> Path:
    """Devuelve el directorio de artefactos correspondiente al forecast.

    Los escenarios multi-cadencia SIEMPRE son autocontenidos:
    ``update_blocks/block_XXd/dashboard``.  Esto evita que 1d/7d/14d/28d
    se sobrescriban entre sí aunque ``settings.DASHBOARD_DIR`` esté definido
    para la baseline legacy.
    """
    if forecast_path is not None:
        fpath = Path(forecast_path).resolve()
        try:
            rel = fpath.relative_to(Path(settings.UPDATE_BLOCKS_DIR).resolve())
        except (ValueError, AttributeError):
            rel = None
        if rel is not None and rel.parts and str(rel.parts[0]).startswith("block_"):
            return fpath.parent / "dashboard"

    custom = getattr(settings, "DASHBOARD_DIR", None)
    if custom:
        return Path(custom)
    if forecast_path is not None:
        return Path(forecast_path).resolve().parent / "dashboard"
    return Path(settings.FORECAST_PATH).resolve().parent / "dashboard"


def _index_path(adir: Path) -> Path:
    return adir / "index.json"


def _labels_path(adir: Path) -> Path:
    return adir / "labels.parquet"


def _metrics_path(adir: Path) -> Path:
    return adir / "metrics.parquet"


def _metrics_in_sample_path(adir: Path) -> Path:
    return adir / "metrics_in_sample.parquet"


def _series_dir(adir: Path) -> Path:
    return adir / "series"


def _series_legacy_path(adir: Path) -> Path:
    """Compat: un solo series.parquet (versiones anteriores)."""
    return adir / "series.parquet"


def source_fingerprint(
    forecast_path: Path,
    *,
    n_rows: int | None = None,
) -> dict[str, int | str]:
    """Fingerprint barato y estricto del forecast que originó los artefactos."""
    path = Path(forecast_path)
    stat = path.stat()
    fp: dict[str, int | str] = {
        "mtime_ns": int(stat.st_mtime_ns),
        "size_bytes": int(stat.st_size),
        "app_version": str(getattr(settings, "APP_VERSION", "")),
    }
    if n_rows is not None:
        fp["n_rows"] = int(n_rows)
    return fp


def artifacts_match_source(
    forecast_path: Path,
    index: dict[str, Any] | None = None,
) -> bool:
    """True solo si index y forecast pertenecen exactamente a la misma corrida."""
    path = Path(forecast_path)
    if not path.exists():
        return False
    if index is None:
        ipath = _index_path(artifacts_dir(path))
        if not ipath.exists():
            return False
        try:
            index = json.loads(ipath.read_text(encoding="utf-8"))
        except Exception:
            return False
    expected = index.get("forecast_fingerprint") or {}
    if not expected:
        return False
    current = source_fingerprint(path)
    stat_match = (
        int(expected.get("mtime_ns") or -1) == int(current["mtime_ns"])
        and int(expected.get("size_bytes") or -1) == int(current["size_bytes"])
        and str(expected.get("app_version") or "") == str(current["app_version"])
    )
    if not stat_match:
        return False

    # Row count is part of the identity too. Parquet metadata makes this check
    # cheap and it prevents a ranking/KPI from being served from a different
    # forecast even in the unlikely case of matching file size/timestamps.
    expected_rows = int(expected.get("n_rows") or -1)
    if expected_rows < 0:
        return False
    try:
        current_rows = int(
            pl.scan_parquet(path)
            .select(pl.len().alias("_n"))
            .collect()
            .item()
        )
    except Exception:
        return False
    return current_rows == expected_rows


def artifacts_exist(forecast_path: Path | None = None) -> bool:
    adir = artifacts_dir(forecast_path)
    has_series = _series_dir(adir).is_dir() or _series_legacy_path(adir).exists()
    ipath = _index_path(adir)
    if not (ipath.exists() and _metrics_path(adir).exists() and _metrics_in_sample_path(adir).exists() and has_series):
        return False
    try:
        payload = json.loads(ipath.read_text(encoding="utf-8"))
        return (
            int(payload.get("version") or 0) == ARTIFACT_VERSION
            and artifacts_match_source(
                Path(forecast_path or settings.FORECAST_PATH),
                payload,
            )
        )
    except Exception:
        return False


def _parse_uid_parts(uids: list[str]) -> pl.DataFrame:
    """Vectoriza seccion / store / sku / node_kind desde unique_id."""
    sec, store, sku, kind = [], [], [], []
    for uid in uids:
        p = settings.split_unique_id(uid)
        s, t, k = p["seccion"], p.get("store"), p.get("sku")
        sec.append(s)
        store.append(t)
        sku.append(k)
        if t is not None and k is not None:
            kind.append("tienda_sku")
        elif k is not None:
            kind.append("sku")
        elif t is not None:
            kind.append("tienda")
        else:
            kind.append("seccion")
    return pl.DataFrame(
        {
            "unique_id": uids,
            "seccion": sec,
            "store": store,
            "sku": sku,
            "node_kind": kind,
        }
    )


def _slim_for_wmape(df: pl.DataFrame) -> pl.DataFrame:
    """Solo columnas necesarias para WMAPE — evita duplicar el panel completo."""
    keep = [c for c in _WMAPE_COLS if c in df.columns]
    return df.select(keep)


def _wmape_table_for_unit(
    unit_df: pl.DataFrame,
    all_ids: list[str],
    unidad: str,
    *,
    period_type: str = "out_sample",
) -> pl.DataFrame:
    """
    Métricas por unique_id para una unidad (Valor o Unidades).

    Un solo group_by sobre el panel completo (no un wmape_por_id por sección).
    n_spine se asigna después por sección vía join.
    """
    empty = pl.DataFrame(
        schema={
            "unique_id": pl.Utf8,
            "wmape": pl.Float64,
            "bias": pl.Float64,
            "sum_y": pl.Float64,
            "sum_yhat": pl.Float64,
            "forecast_oos_total": pl.Float64,
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
            "unidad": pl.Utf8,
            "seccion": pl.Utf8,
            "n_spine": pl.UInt32,
        }
    )
    if unit_df.height == 0:
        return empty

    # Secciones: parseo una sola vez
    sec_from_id = (
        pl.DataFrame({"unique_id": all_ids})
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion")
        )
        .unique()
    )
    secciones = (
        sec_from_id.filter(~pl.col("unique_id").str.contains(r"\|\|", literal=False))
        ["seccion"]
        .unique()
        .to_list()
    )
    if not secciones:
        secciones = sec_from_id["seccion"].unique().to_list()

    # n_spine por sección (settings + opcional override por datos)
    spine_rows = []
    for seccion in secciones:
        hz = settings.section_horizons(seccion)
        n_spine = (hz["forecast_end"] - hz["train_start"]).days + 1
        ranking_days = (hz["test_end"] - hz["test_start"]).days + 1
        train_start = hz.get("train_start")
        train_end = hz.get("train_end")
        in_sample_days = (train_end - train_start).days + 1 if train_start is not None and train_end is not None else int(n_spine)
        spine_rows.append(
            {
                "seccion": seccion,
                "n_spine": int(n_spine),
                "ranking_days": int(ranking_days),
                "in_sample_days": int(in_sample_days),
            }
        )
    spine_df = pl.DataFrame(spine_rows).with_columns(
        pl.col("n_spine").cast(pl.UInt32),
        pl.col("ranking_days").cast(pl.UInt32),
        pl.col("in_sample_days").cast(pl.UInt32),
    )

    # Métrica del período solicitado. OOS sigue siendo la fuente oficial;
    # in-sample se precalcula para el slider de rankings sin tocar el hot path.
    tabla = backend.wmape_all_ids(
        unit_df, n_fechas_spine=0, period_types=[period_type]
    )
    if tabla.height == 0:
        return empty

    # Pronóstico TOTAL del período, incluidos días con actual=0.
    total_col = "forecast_oos_total" if period_type == "out_sample" else "forecast_in_sample_total"
    if "period_type" in unit_df.columns and "yhat" in unit_df.columns:
        period_forecast_total = (
            unit_df.filter(pl.col("period_type") == period_type)
            .group_by("unique_id")
            .agg(pl.col("yhat").sum().alias(total_col))
        )
    else:
        period_forecast_total = pl.DataFrame(
            schema={"unique_id": pl.Utf8, total_col: pl.Float64}
        )

    tabla = (
        tabla.join(period_forecast_total, on="unique_id", how="left")
        .with_columns(
            pl.col(total_col).fill_null(0.0),
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
        )
        .join(spine_df, on="seccion", how="left")
        .with_columns(
            pl.when(pl.lit(period_type) == "out_sample")
            .then(pl.col("ranking_days"))
            .otherwise(pl.col("in_sample_days"))
            .fill_null(0).alias("n_points"),
            pl.lit(unidad).alias("unidad"),
        )
        .select(
            [
                "unique_id",
                "wmape",
                "bias",
                "sum_y",
                "sum_yhat",
                total_col,
                "sum_abs_y",
                "sum_abs_error",
                "sum_signed_error",
                "n_points",
                "n_with_sales",
                "metric_cohort",
                "metric_active",
                "n_leaf_active",
                "n_leaf_sparse",
                "n_leaf_zero",
                "zero_forecast_sum",
                "zero_abs_error_sum",
                "unidad",
                "seccion",
                "n_spine",
            ]
        )
    )
    return tabla


def _build_bottom_up_aggregate_series_vectorized(res_df: pl.DataFrame) -> pl.DataFrame:
    """Materializa las series operacionales agregadas desde hojas SKU+Tienda.

    Contrato visual del dashboard:
      * SKU        = suma diaria de todas sus hojas tienda+sku.
      * Tienda     = suma diaria de todas sus hojas tienda+sku.
      * Sección    = suma diaria de todas sus hojas tienda+sku.
      * SKU+Tienda = la hoja productiva original.

    Los modelos RLS directos de sección/tienda siguen existiendo en forecast.parquet
    y siguen participando en el modelado, pero NO son la serie operacional mostrada.

    Además se materializan contribuciones diarias de error bottom-up calculadas
    hoja por hoja con y != 0. Sirven únicamente para explicar visualmente el wMAPE
    oficial existente; no cambian su definición ni metrics.parquet.
    """
    if res_df.height == 0 or "unique_id" not in res_df.columns:
        return pl.DataFrame()

    needed = [
        c for c in (
            "unique_id", "ds", "y", "yhat", "yhat28", "value", "valuehat",
            "valuehat28", "period_type", "sku_desc", "store_name", "seccion",
            "train_start", "train_end", "test_start", "test_end",
            "forecast_start", "forecast_end", "update_block_days",
            "v11_yhat_raw_before_v12", "v11_valuehat_raw_before_v12",
            "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
        ) if c in res_df.columns
    ]
    leaves = res_df.select(needed).filter(
        pl.col("unique_id").str.contains(r"\|\|T:", literal=False)
        & pl.col("unique_id").str.contains(r"\|\|S:", literal=False)
    )
    if leaves.height == 0:
        return pl.DataFrame()

    leaves = leaves.with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("_sec"),
        pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("_store"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("_sku"),
    )

    # Error components are deliberately computed BEFORE aggregation so that
    # opposite leaf errors never cancel. This mirrors the existing BU metric.
    err_exprs: list[pl.Expr] = []
    if {"y", "yhat"}.issubset(leaves.columns):
        err_exprs.extend([
            pl.when(pl.col("y") != 0).then((pl.col("y") - pl.col("yhat")).abs()).otherwise(0.0).alias("_bu_ae_y"),
            pl.when(pl.col("y") != 0).then(pl.col("y").abs()).otherwise(0.0).alias("_bu_ay_y"),
            pl.when(pl.col("y") != 0).then(pl.col("yhat") - pl.col("y")).otherwise(0.0).alias("_bu_se_y"),
        ])
    if {"value", "valuehat"}.issubset(leaves.columns):
        err_exprs.extend([
            pl.when(pl.col("value") != 0).then((pl.col("value") - pl.col("valuehat")).abs()).otherwise(0.0).alias("_bu_ae_v"),
            pl.when(pl.col("value") != 0).then(pl.col("value").abs()).otherwise(0.0).alias("_bu_ay_v"),
            pl.when(pl.col("value") != 0).then(pl.col("valuehat") - pl.col("value")).otherwise(0.0).alias("_bu_se_v"),
        ])
    if err_exprs:
        leaves = leaves.with_columns(err_exprs)

    additive = [
        c for c in (
            "y", "yhat", "yhat28", "value", "valuehat", "valuehat28",
            "v11_yhat_raw_before_v12", "v11_valuehat_raw_before_v12",
            "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
        ) if c in leaves.columns
    ]
    first_common = [
        c for c in (
            "period_type", "seccion", "train_start", "train_end", "test_start",
            "test_end", "forecast_start", "forecast_end", "update_block_days",
        ) if c in leaves.columns
    ]

    def _one(level: str) -> pl.DataFrame:
        if level == "seccion":
            keys = ["_sec", "ds"]
        elif level == "tienda":
            keys = ["_sec", "_store", "ds"]
        elif level == "sku":
            keys = ["_sec", "_sku", "ds"]
        else:
            raise ValueError(level)

        aggs: list[pl.Expr] = [pl.col(c).sum().alias(c) for c in additive]
        aggs.extend(pl.col(c).first().alias(c) for c in first_common)
        for src, out in (
            ("_bu_ae_y", "bu_abs_error_daily_y"),
            ("_bu_ay_y", "bu_abs_y_daily_y"),
            ("_bu_se_y", "bu_signed_error_daily_y"),
            ("_bu_ae_v", "bu_abs_error_daily_value"),
            ("_bu_ay_v", "bu_abs_y_daily_value"),
            ("_bu_se_v", "bu_signed_error_daily_value"),
        ):
            if src in leaves.columns:
                aggs.append(pl.col(src).sum().alias(out))

        # Description/name are meaningful only at their corresponding level.
        if level == "sku" and "sku_desc" in leaves.columns:
            aggs.append(pl.col("sku_desc").drop_nulls().first().alias("sku_desc"))
        if level == "tienda" and "store_name" in leaves.columns:
            aggs.append(pl.col("store_name").drop_nulls().first().alias("store_name"))

        out = leaves.group_by(keys).agg(aggs)
        if level == "seccion":
            out = out.with_columns(pl.col("_sec").alias("unique_id"))
        elif level == "tienda":
            out = out.with_columns(
                pl.concat_str([pl.col("_sec"), pl.lit("||T:"), pl.col("_store")]).alias("unique_id")
            )
        else:
            out = out.with_columns(
                pl.concat_str([pl.col("_sec"), pl.lit("||S:"), pl.col("_sku")]).alias("unique_id")
            )
        return out.drop([c for c in ("_sec", "_store", "_sku") if c in out.columns])

    parts = [_one("seccion"), _one("tienda"), _one("sku")]
    return pl.concat(parts, how="diagonal_relaxed")


def _build_pure_sku_series_vectorized(res_df: pl.DataFrame) -> pl.DataFrame:
    """Compat wrapper; SKU puro ahora forma parte del agregado BU completo."""
    agg = _build_bottom_up_aggregate_series_vectorized(res_df)
    if agg.height == 0:
        return agg
    return agg.filter(
        pl.col("unique_id").str.contains(r"\|\|S:", literal=False)
        & ~pl.col("unique_id").str.contains(r"\|\|T:", literal=False)
    )

def _parts_from_ids(all_ids: list[str]) -> pl.DataFrame:
    """Parseo vectorizado de unique_ids → seccion/store/sku/node_kind (una sola vez)."""
    return (
        pl.DataFrame({"unique_id": all_ids})
        .with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion"),
            pl.when(pl.col("unique_id").str.contains(r"\|\|T:", literal=False))
            .then(pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1))
            .otherwise(None)
            .alias("store"),
            pl.when(pl.col("unique_id").str.contains(r"\|\|S:", literal=False))
            .then(pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1))
            .otherwise(None)
            .alias("sku"),
        )
        .with_columns(
            pl.when(pl.col("store").is_not_null() & pl.col("sku").is_not_null())
            .then(pl.lit("tienda_sku"))
            .when(pl.col("sku").is_not_null())
            .then(pl.lit("sku"))
            .when(pl.col("store").is_not_null())
            .then(pl.lit("tienda"))
            .otherwise(pl.lit("seccion"))
            .alias("node_kind")
        )
    )


def _build_index(
    res_df: pl.DataFrame,
    all_ids: list[str],
    metrics: pl.DataFrame,
    forecast_path: Path,
    label_map: dict[str, str],
    desc_map: dict[str, str],
) -> dict[str, Any]:
    """Index liviano: sin label_map/desc_map (van a labels.parquet). Vectorizado."""
    parts = _parts_from_ids(all_ids)

    # Secciones = nodos sin store ni sku
    secciones = (
        parts.filter(pl.col("node_kind") == "seccion")["seccion"]
        .unique()
        .sort()
        .to_list()
    )
    if not secciones:
        secciones = parts["seccion"].unique().sort().to_list()

    stores_by_sec: dict[str, list[str]] = {}
    skus_by_sec: dict[str, list[str]] = {}
    stores_for_sku: dict[str, dict[str, list[str]]] = {}
    skus_for_store: dict[str, dict[str, list[str]]] = {}
    n_spine_by_sec: dict[str, int] = {}
    horizons_by_sec: dict[str, dict[str, str | None]] = {}

    # Tiendas y SKUs por sección (nodos tienda / cualquier id con sku)
    for seccion in secciones:
        sec_parts = parts.filter(pl.col("seccion") == seccion)
        stores_by_sec[seccion] = (
            sec_parts.filter(pl.col("node_kind") == "tienda")["store"]
            .drop_nulls()
            .unique()
            .sort()
            .to_list()
        )
        skus_by_sec[seccion] = (
            sec_parts.filter(pl.col("sku").is_not_null())["sku"]
            .unique()
            .sort()
            .to_list()
        )

        # Hojas: store×sku presentes
        leaves = sec_parts.filter(pl.col("node_kind") == "tienda_sku")
        if leaves.height:
            # stores_for_sku[seccion][sku] = [stores…]
            sfs: dict[str, list[str]] = {}
            for row in (
                leaves.group_by("sku")
                .agg(pl.col("store").unique().sort().alias("stores"))
                .iter_rows(named=True)
            ):
                sfs[str(row["sku"])] = [str(x) for x in row["stores"] if x is not None]
            stores_for_sku[seccion] = sfs

            # skus_for_store[seccion][store] = [skus…]
            sft: dict[str, list[str]] = {}
            for row in (
                leaves.group_by("store")
                .agg(pl.col("sku").unique().sort().alias("skus"))
                .iter_rows(named=True)
            ):
                sft[str(row["store"])] = [str(x) for x in row["skus"] if x is not None]
            skus_for_store[seccion] = sft
        else:
            stores_for_sku[seccion] = {}
            skus_for_store[seccion] = {}

        # Prefer the horizons actually used by the forecast run. Settings
        # contain only fallback/preferences; v11 resolves OOS and forecast-only
        # from the real first/last actual dates.
        hz = settings.section_horizons(seccion)
        sec_source = res_df.filter(
            pl.col("unique_id")
            .cast(pl.Utf8)
            .str.split("||")
            .list.get(0)
            == str(seccion)
        )
        for _hk in (
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "forecast_start",
            "forecast_end",
        ):
            if _hk in sec_source.columns and sec_source.height:
                _hv = sec_source.select(
                    pl.col(_hk).drop_nulls().first()
                ).item()
                if _hv is not None:
                    hz[_hk] = _hv
        horizons_by_sec[seccion] = {
            k: (v.isoformat() if hasattr(v, "isoformat") else None)
            for k, v in hz.items()
            if k
            in {
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "forecast_start",
                "forecast_end",
            }
        }
        sub = metrics.filter(pl.col("seccion") == seccion) if metrics.height else metrics
        n_val = None
        if sub.height and "n_spine" in sub.columns:
            raw = sub["n_spine"][0]
            if raw is not None:
                try:
                    n_val = int(raw)
                except (TypeError, ValueError):
                    n_val = None
        if n_val is None or n_val <= 0:
            n_val = (hz["forecast_end"] - hz["train_start"]).days + 1
        n_spine_by_sec[seccion] = int(n_val)

    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    has_rolling28 = False
    if "yhat28" in cols:
        # sample cheap: null count vs height on a projection
        has_rolling28 = res_df.select(pl.col("yhat28").null_count()).item() < res_df.height
    if not has_rolling28 and "valuehat28" in cols:
        has_rolling28 = (
            res_df.select(pl.col("valuehat28").null_count()).item() < res_df.height
        )

    # labels para SKU puro virtuales. Construir una vez el lookup O(n) y
    # reutilizarlo evita el escaneo O(n_ids × n_skus) del dashboard histórico.
    desc_lookup = _leaf_description_lookup(desc_map)
    for seccion, skus in skus_by_sec.items():
        for sku in skus:
            pure_id = settings.make_unique_id(seccion, sku=sku)
            if pure_id not in label_map:
                hit = desc_lookup.get((str(seccion), str(sku)), "")
                label_map[pure_id] = settings.display_label(pure_id, hit or None, None)
                desc_map[pure_id] = hit or settings.ranking_description(
                    pure_id, hit or None, None
                )

    return {
        "version": ARTIFACT_VERSION,
        "forecast_path": str(forecast_path.resolve()),
        "forecast_mtime": forecast_path.stat().st_mtime if forecast_path.exists() else 0.0,
        "forecast_fingerprint": source_fingerprint(
            forecast_path,
            n_rows=res_df.height,
        ),
        "app_version": str(getattr(settings, "APP_VERSION", "")),
        "update_block_days": int(
            res_df.select(pl.col("update_block_days").drop_nulls().first()).item()
            if "update_block_days" in res_df.columns and res_df.select(pl.col("update_block_days").drop_nulls()).height
            else getattr(settings, "RLS_BLOCK_DAYS", 28)
        ),
        "metric_horizon_days": int(getattr(settings, "METRIC_HORIZON_DAYS", 28)),
        "secciones": secciones,
        "stores_by_sec": stores_by_sec,
        "skus_by_sec": skus_by_sec,
        "stores_for_sku": stores_for_sku,
        "skus_for_store": skus_for_store,
        "n_spine_by_sec": n_spine_by_sec,
        "horizons_by_sec": horizons_by_sec,
        # labels viven en labels.parquet (v2); se mantienen vacíos aquí por compat
        "label_map": {},
        "desc_map": {},
        "has_value": has_value,
        "has_rolling28": has_rolling28,
        "n_unique_ids": len(all_ids),
        "n_rows_source": res_df.height,
        "series_partitioned": True,
    }


def _clear_series_dir(sdir: Path) -> None:
    if not sdir.exists():
        return
    for p in sdir.rglob("*.parquet"):
        p.unlink()
    for p in sorted(sdir.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def _write_one_section_series(part: pl.DataFrame, sdir: Path, seccion: str) -> None:
    """Escribe una partición seccion=<s>/data.parquet ordenada."""
    out = sdir / f"seccion={seccion}"
    out.mkdir(parents=True, exist_ok=True)
    (
        part.sort("unique_id", "ds")
        .write_parquet(
            out / "data.parquet",
            compression="zstd",
            compression_level=3,
            statistics=True,
            # Ordenado por unique_id: row-groups pequeños permiten que el
            # dashboard salte casi todo el parquet al cambiar una selección.
            row_group_size=4096,
        )
    )


def _write_series_partitioned(series: pl.DataFrame, adir: Path) -> None:
    """Escribe series/seccion=<s>/data.parquet ordenado por unique_id, ds."""
    sdir = _series_dir(adir)
    _clear_series_dir(sdir)
    sdir.mkdir(parents=True, exist_ok=True)

    if series.height == 0:
        return

    # Garantizar columna seccion para particionar
    if "seccion" not in series.columns:
        series = series.with_columns(
            pl.col("unique_id").str.split("||").list.get(0).alias("seccion")
        )
    else:
        series = series.with_columns(
            pl.when(pl.col("seccion").is_null() | (pl.col("seccion") == ""))
            .then(pl.col("unique_id").str.split("||").list.get(0))
            .otherwise(pl.col("seccion"))
            .alias("seccion")
        )

    n_sec = series.select(pl.col("seccion").n_unique()).item()
    _progress(f"  escribiendo series particionadas ({n_sec} secciones)…")
    for i, (seccion, part) in enumerate(
        series.partition_by("seccion", as_dict=True).items(), start=1
    ):
        sec_val = seccion[0] if isinstance(seccion, tuple) else seccion
        sec_str = str(sec_val)
        _write_one_section_series(part, sdir, sec_str)
        if i == 1 or i % 5 == 0 or i == n_sec:
            _progress(f"  serie sección {i}/{n_sec}: {sec_str} ({part.height:,} filas)")


def _labels_df(label_map: dict[str, str], desc_map: dict[str, str]) -> pl.DataFrame:
    uids = sorted(set(label_map) | set(desc_map))
    return pl.DataFrame(
        {
            "unique_id": uids,
            "label": [label_map.get(u, "") for u in uids],
            "description": [desc_map.get(u, "") for u in uids],
        }
    )



def _assert_bottom_up_metric_identities(metrics: pl.DataFrame) -> None:
    """Fail fast if parent OOS metrics cannot be reproduced from leaf components."""
    if metrics.height == 0:
        return
    needed = {
        "unique_id", "unidad", "sum_abs_y", "sum_abs_error",
        "sum_signed_error", "wmape", "bias", "metric_active", "metric_cohort",
    }
    if not needed.issubset(set(metrics.columns)):
        missing = sorted(needed - set(metrics.columns))
        raise RuntimeError(f"Dashboard metrics missing bottom-up components: {missing}")

    leaves = metrics.filter(
        (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        & pl.col("metric_active").fill_null(False)
    ).with_columns(
        pl.col("unique_id").str.split("||").list.get(0).alias("_sec"),
        pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
        pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("_sku"),
    ).with_columns(
        pl.concat_str(
            [pl.col("_sec"), pl.lit("||S:"), pl.col("_sku")]
        ).alias("_sku_uid")
    )
    if leaves.height == 0:
        return

    checks: list[tuple[str, pl.DataFrame]] = [
        ("store", leaves.group_by(["_store_uid", "unidad"]).agg(
            pl.col("sum_abs_y").sum(),
            pl.col("sum_abs_error").sum(),
            pl.col("sum_signed_error").sum(),
        ).rename({"_store_uid": "unique_id"})),
        ("section", leaves.group_by(["_sec", "unidad"]).agg(
            pl.col("sum_abs_y").sum(),
            pl.col("sum_abs_error").sum(),
            pl.col("sum_signed_error").sum(),
        ).rename({"_sec": "unique_id"})),
        ("sku", leaves.group_by(["_sku_uid", "unidad"]).agg(
            pl.col("sum_abs_y").sum(),
            pl.col("sum_abs_error").sum(),
            pl.col("sum_signed_error").sum(),
        ).rename({"_sku_uid": "unique_id"})),
    ]

    parents = metrics.select(
        "unique_id", "unidad", "sum_abs_y", "sum_abs_error",
        "sum_signed_error", "wmape", "bias",
    )
    tol = float(getattr(settings, "DASHBOARD_METRIC_IDENTITY_TOLERANCE", 1e-9))

    for level, rebuilt in checks:
        cmp = rebuilt.join(
            parents,
            on=["unique_id", "unidad"],
            how="left",
            suffix="_artifact",
        ).with_columns(
            (
                pl.col("sum_abs_error")
                / pl.col("sum_abs_y").replace(0.0, None)
            ).fill_null(0.0).alias("_wmape_rebuilt"),
            (
                pl.col("sum_signed_error")
                / pl.col("sum_abs_y").replace(0.0, None)
            ).fill_null(0.0).alias("_bias_rebuilt"),
        ).with_columns(
            (
                (pl.col("_wmape_rebuilt") - pl.col("wmape")).abs()
                / pl.max_horizontal(pl.col("_wmape_rebuilt").abs(), pl.lit(1.0))
            ).alias("_dw"),
            (
                (pl.col("_bias_rebuilt") - pl.col("bias")).abs()
                / pl.max_horizontal(pl.col("_bias_rebuilt").abs(), pl.lit(1.0))
            ).alias("_db"),
            (
                (pl.col("sum_abs_error") - pl.col("sum_abs_error_artifact")).abs()
                / pl.max_horizontal(pl.col("sum_abs_error").abs(), pl.lit(1.0))
            ).alias("_de"),
            (
                (pl.col("sum_abs_y") - pl.col("sum_abs_y_artifact")).abs()
                / pl.max_horizontal(pl.col("sum_abs_y").abs(), pl.lit(1.0))
            ).alias("_dy"),
        )
        bad = cmp.filter(
            pl.col("wmape").is_null()
            | (pl.col("_dw") > tol)
            | (pl.col("_db") > tol)
            | (pl.col("_de") > tol)
            | (pl.col("_dy") > tol)
        )
        if bad.height:
            raise RuntimeError(
                f"Bottom-up metric identity failed at {level}: "
                f"{bad.head(5).to_dicts()}"
            )


def _leaf_description_lookup(desc_map: dict[str, str]) -> dict[tuple[str, str], str]:
    """O(n_ids) lookup for pure SKU labels.

    v11.2 searched the complete desc_map for every pure SKU (roughly
    8.5k × 37k string checks in the sample).  Build the mapping once instead.
    """
    out: dict[tuple[str, str], str] = {}
    for uid, desc in desc_map.items():
        if not desc or "||S:" not in uid:
            continue
        try:
            sec = uid.split("||", 1)[0]
            sku = uid.split("||S:", 1)[1].split("||", 1)[0]
        except (IndexError, AttributeError):
            continue
        out.setdefault((sec, sku), desc)
    return out


def build_artifacts(
    forecast_path: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
) -> Path:
    """
    Lee forecast.parquet y materializa index + labels + metrics + series particionadas.
    Devuelve el directorio de artefactos.

    Diseñado para datasets grandes:
    - logs de avance por etapa + RSS
    - paneles WMAPE slim (solo columnas necesarias)
    - libera intermedios con gc
    - index vectorizado (sin O(ids × skus) en Python)
    - series escritas por sección
    """
    t_all = time.perf_counter()
    fpath = Path(forecast_path or settings.FORECAST_PATH)
    if not fpath.exists():
        raise FileNotFoundError(f"No existe forecast: {fpath}")

    adir = Path(out_dir) if out_dir else artifacts_dir(fpath)
    adir.mkdir(parents=True, exist_ok=True)
    _progress(f"Construyendo artefactos dashboard en {adir}")
    _progress(f"Fuente: {fpath} ({fpath.stat().st_size / (1024**2):.1f} MB)")

    # ── 1. Carga ──────────────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("1/6 Cargando forecast.parquet (proyección dashboard)…")
    scan = pl.scan_parquet(fpath)
    source_cols = set(scan.collect_schema().names())
    required_cols = set(SERIES_COLS_PREFERRED) | set(_WMAPE_COLS) | {
        "unique_id", "ds", "sku_desc", "store_name", "seccion",
    }
    load_cols = [c for c in source_cols if c in required_cols]
    res_df = (
        scan.select(load_cols)
        .with_columns(pl.col("ds").cast(pl.Date))
        .collect()
    )
    all_ids = res_df["unique_id"].unique().to_list()
    cols = set(res_df.columns)
    has_value = "value" in cols and "valuehat" in cols
    _progress(
        f"1/6 Cargado: {res_df.height:,} filas, {len(all_ids):,} unique_ids, "
        f"{len(cols)}/{len(source_cols)} cols",
        t,
    )

    # ── 2. Labels ─────────────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("2/6 Construyendo label_map / desc_map…")
    label_map, desc_map = backend.build_label_maps(res_df)
    _progress(f"2/6 Labels: {len(label_map):,} ids", t)

    # ── 3. Metrics (slim + liberar unit dfs) ───────────────────────────────
    t = time.perf_counter()
    _progress("3/6 Calculando metrics (WMAPE OOS bottom-up)…")
    metric_parts: list[pl.DataFrame] = []
    metric_parts_in: list[pl.DataFrame] = []

    slim = _slim_for_wmape(res_df)
    _progress(f"  panel slim WMAPE: {slim.height:,} filas, {len(slim.columns)} cols")

    unit_df_u = backend.prepare_unit_df(slim, "Unidades", has_value)
    _progress("  WMAPE Unidades OOS + in-sample…")
    metric_parts.append(_wmape_table_for_unit(unit_df_u, all_ids, "Unidades", period_type="out_sample"))
    metric_parts_in.append(_wmape_table_for_unit(unit_df_u, all_ids, "Unidades", period_type="in_sample"))
    del unit_df_u
    gc.collect()

    if has_value:
        unit_df_v = backend.prepare_unit_df(slim, "Valor ($)", has_value)
        _progress("  WMAPE Valor ($) OOS + in-sample…")
        metric_parts.append(_wmape_table_for_unit(unit_df_v, all_ids, "Valor ($)", period_type="out_sample"))
        metric_parts_in.append(_wmape_table_for_unit(unit_df_v, all_ids, "Valor ($)", period_type="in_sample"))
        del unit_df_v
        gc.collect()

    del slim
    gc.collect()

    metrics = pl.concat([p for p in metric_parts if p.height], how="diagonal_relaxed")
    metrics_in = pl.concat([p for p in metric_parts_in if p.height], how="diagonal_relaxed")
    del metric_parts, metric_parts_in
    gc.collect()

    all_metric_ids = pl.concat([metrics.select("unique_id"), metrics_in.select("unique_id")]).get_column("unique_id").unique().to_list()
    parts_df = _parse_uid_parts(all_metric_ids).unique(subset=["unique_id"])
    def _attach_parts(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.height == 0:
            return frame
        frame = frame.drop([c for c in ("store", "sku", "node_kind", "seccion") if c in frame.columns])
        frame = frame.join(parts_df, on="unique_id", how="left")
        subset = ["unique_id", "unidad"] if "unidad" in frame.columns else ["unique_id"]
        return frame.unique(subset=subset, keep="first")
    metrics = _attach_parts(metrics)
    metrics_in = _attach_parts(metrics_in)
    del parts_df

    # El cohort/active es una clasificación OOS contractual. En el ranking
    # in-sample conservamos exactamente la misma población Active para que el
    # slider cambie la métrica, no el universo de hojas.
    if metrics_in.height and metrics.height:
        oos_flags = metrics.select(["unique_id", "unidad", "metric_cohort", "metric_active"]).rename({
            "metric_cohort": "_oos_metric_cohort", "metric_active": "_oos_metric_active"
        })
        metrics_in = metrics_in.drop([c for c in ("metric_cohort", "metric_active") if c in metrics_in.columns]).join(
            oos_flags, on=["unique_id", "unidad"], how="left"
        ).rename({"_oos_metric_cohort": "metric_cohort", "_oos_metric_active": "metric_active"})

    _assert_bottom_up_metric_identities(metrics)
    _progress(f"3/6 Metrics: OOS={metrics.height:,} + in-sample={metrics_in.height:,} filas", t)

    # ── 4. Preparar series memory-safe ─────────────────────────────────────
    # No concatenar las ~7.5M filas fuente con ~2.9M filas SKU-puro en un
    # único DataFrame. Ese patrón obligaba luego a partition_by() a duplicar
    # buffers y podía abortar el proceso en Rust antes de mark_success().
    t = time.perf_counter()
    _progress("4/6 Materializando series slim + SKU puro — memory-safe por sección…")
    keep = [c for c in SERIES_COLS_PREFERRED if c in res_df.columns]
    n_rows_source = res_df.height
    _progress(
        f"4/6 Fuente retenida: {n_rows_source:,} filas; series se materializarán una sección a la vez",
        t,
    )

    # ── 5. Index + labels ─────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("5/6 Construyendo index + labels…")
    index = _build_index(res_df, all_ids, metrics, fpath, label_map, desc_map)
    index["n_rows_source"] = n_rows_source
    labels = _labels_df(label_map, desc_map)
    del label_map, desc_map, all_ids
    gc.collect()
    _progress(
        f"5/6 Index: {len(index['secciones'])} secciones, labels={labels.height:,}",
        t,
    )

    # ── 6. Write ──────────────────────────────────────────────────────────
    t = time.perf_counter()
    _progress("6/6 Escribiendo artefactos a disco…")
    _index_path(adir).write_text(
        json.dumps(index, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    labels.write_parquet(
        _labels_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    metrics.write_parquet(
        _metrics_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    metrics_in.write_parquet(
        _metrics_in_sample_path(adir), compression="zstd", compression_level=3, statistics=True
    )
    del labels, metrics, metrics_in
    gc.collect()

    # Escritura por sección: filtrar fuente, construir SKU-puro solo para esa
    # sección, escribir inmediatamente y liberar antes de pasar a la siguiente.
    # Evita mantener ~10.36M filas y sus copias de partition_by() en RAM.
    sdir = _series_dir(adir)
    _clear_series_dir(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    secciones = [str(x) for x in index.get("secciones", [])]
    _progress(f"  escribiendo series memory-safe ({len(secciones)} secciones)…")

    for i, sec_str in enumerate(secciones, start=1):
        sec_source = res_df.filter(
            pl.col("unique_id").cast(pl.Utf8).str.split("||").list.get(0) == sec_str
        )
        # Dashboard operational series: preserve productive SKU+Tienda leaves,
        # replace direct section/store parent forecasts with bottom-up sums, and
        # materialize SKU as the same leaf sum.  Metrics remain untouched.
        leaf_source = sec_source.filter(
            pl.col("unique_id").str.contains(r"\|\|T:", literal=False)
            & pl.col("unique_id").str.contains(r"\|\|S:", literal=False)
        )
        sec_series = leaf_source.select(keep)
        bottom_up = _build_bottom_up_aggregate_series_vectorized(sec_source)
        bu_rows = bottom_up.height
        bu_ids = 0
        if bu_rows:
            for c in keep:
                if c not in bottom_up.columns:
                    bottom_up = bottom_up.with_columns(pl.lit(None).alias(c))
            bottom_up = bottom_up.select(keep)
            bu_ids = int(bottom_up.select(pl.col("unique_id").n_unique()).item())
            sec_series = pl.concat([sec_series, bottom_up], how="diagonal_relaxed")

        _write_one_section_series(sec_series, sdir, sec_str)
        _progress(
            f"  serie sección {i}/{len(secciones)}: {sec_str} "
            f"({sec_series.height:,} filas; agregados BU={bu_ids:,} ids/{bu_rows:,} filas)"
        )
        del sec_series, bottom_up, leaf_source, sec_source
        gc.collect()

    del res_df
    gc.collect()

    legacy = _series_legacy_path(adir)
    if legacy.exists():
        legacy.unlink()

    _progress(
        f"✓ Artefactos listos: {len(index['secciones'])} secciones, "
        f"series particionadas en {adir}",
        t_all,
    )
    return adir


# ── loaders (usados por el dashboard) ─────────────────────────────────────────


def load_index(forecast_path: Path | None = None) -> dict[str, Any]:
    path = _index_path(artifacts_dir(forecast_path))
    if not path.exists():
        raise FileNotFoundError(f"Falta index de dashboard: {path}")
    index = json.loads(path.read_text(encoding="utf-8"))
    # v2: labels en parquet; v1: embebidos en index
    if not index.get("label_map"):
        labels_path = _labels_path(artifacts_dir(forecast_path))
        if labels_path.exists():
            lab = pl.read_parquet(labels_path)
            index["label_map"] = dict(
                zip(lab["unique_id"].to_list(), lab["label"].to_list())
            )
            index["desc_map"] = dict(
                zip(lab["unique_id"].to_list(), lab["description"].to_list())
            )
    return index


def load_metrics(forecast_path: Path | None = None) -> pl.DataFrame:
    path = _metrics_path(artifacts_dir(forecast_path))
    if not path.exists():
        raise FileNotFoundError(f"Falta metrics de dashboard: {path}")
    return pl.read_parquet(path)


def load_metrics_in_sample(forecast_path: Path | None = None) -> pl.DataFrame:
    path = _metrics_in_sample_path(artifacts_dir(forecast_path))
    if not path.exists():
        raise FileNotFoundError(f"Falta metrics in-sample de dashboard: {path}")
    return pl.read_parquet(path)


def load_metrics_scope(
    forecast_path: Path | None,
    *,
    seccion: str,
    unidad: str,
    period: str = "oos",
) -> pl.DataFrame:
    """Lee solo sección+unidad desde el parquet de métricas.

    Este es el hot path del dashboard. Predicate/projection pushdown evita cargar
    ~90k filas de todas las secciones/unidades al cambiar bloque o selección.
    """
    adir = artifacts_dir(forecast_path)
    path = _metrics_in_sample_path(adir) if period == "in_sample" else _metrics_path(adir)
    if not path.exists():
        raise FileNotFoundError(f"Falta metrics ({period}) de dashboard: {path}")
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    sec_col = "seccion" if "seccion" in schema.names() else None
    if sec_col is None:
        return lf.filter(pl.col("unidad") == unidad).collect(engine="streaming")
    return lf.filter(
        (pl.col(sec_col).cast(pl.Utf8) == str(seccion)) & (pl.col("unidad") == unidad)
    ).collect(engine="streaming")


def _series_scan_path(adir: Path, unique_id: str | None = None) -> Path | list[Path]:
    """
    Resuelve path(s) de series.
    Prefer partición por sección; fallback a series.parquet legacy.
    """
    sdir = _series_dir(adir)
    if sdir.is_dir():
        if unique_id is not None:
            seccion = unique_id.split("||", 1)[0]
            part = sdir / f"seccion={seccion}" / "data.parquet"
            if part.exists():
                return part
            # buscar cualquier partición (por si seccion no matchea el nombre)
            parts = list(sdir.glob("seccion=*/data.parquet"))
            if parts:
                return parts
        else:
            return list(sdir.glob("seccion=*/data.parquet"))
    legacy = _series_legacy_path(adir)
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Falta series de dashboard en {adir}")


def load_series(
    unique_id: str,
    forecast_path: Path | None = None,
    *,
    unidad: str = "Unidades",
    has_value: bool = False,
) -> pl.DataFrame:
    """
    Lee SOLO la serie del unique_id pedido (partición de sección + filter).
    Aplica prepare_unit_df según unidad.
    """
    adir = artifacts_dir(forecast_path)
    path = _series_scan_path(adir, unique_id)
    if isinstance(path, list):
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("unique_id") == unique_id)
            .collect(engine="streaming")
            .sort("ds")
        )
    else:
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("unique_id") == unique_id)
            .collect(engine="streaming")
            .sort("ds")
        )
    if df.height == 0:
        return df
    return backend.prepare_unit_df(df, unidad, has_value)


def load_series_many(
    unique_ids: list[str],
    forecast_path: Path | None = None,
) -> pl.DataFrame:
    if not unique_ids:
        return pl.DataFrame()
    adir = artifacts_dir(forecast_path)
    # agrupar por sección para leer solo particiones necesarias
    by_sec: dict[str, list[str]] = {}
    for uid in unique_ids:
        by_sec.setdefault(uid.split("||", 1)[0], []).append(uid)

    chunks: list[pl.DataFrame] = []
    sdir = _series_dir(adir)
    if sdir.is_dir():
        for seccion, uids in by_sec.items():
            part = sdir / f"seccion={seccion}" / "data.parquet"
            if not part.exists():
                continue
            chunks.append(
                pl.scan_parquet(part)
                .filter(pl.col("unique_id").is_in(uids))
                .collect()
            )
    else:
        legacy = _series_legacy_path(adir)
        if legacy.exists():
            chunks.append(
                pl.scan_parquet(legacy)
                .filter(pl.col("unique_id").is_in(unique_ids))
                .collect()
            )
    if not chunks:
        return pl.DataFrame()
    return pl.concat(chunks, how="diagonal_relaxed").sort("unique_id", "ds")


def load_leaves_for_scope(
    seccion: str,
    store: str | None = None,
    sku: str | None = None,
    forecast_path: Path | None = None,
    *,
    unidad: str = "Unidades",
    has_value: bool = False,
) -> pl.DataFrame:
    """
    Hojas sku+tienda del alcance desde artefactos (métricas bottom-up).
    """
    adir = artifacts_dir(forecast_path)
    sdir = _series_dir(adir)
    part = sdir / f"seccion={seccion}" / "data.parquet"
    if part.exists():
        lf = pl.scan_parquet(part)
    else:
        legacy = _series_legacy_path(adir)
        if not legacy.exists():
            return pl.DataFrame()
        lf = pl.scan_parquet(legacy).filter(
            (pl.col("unique_id") == seccion)
            | pl.col("unique_id").str.starts_with(f"{seccion}||")
        )

    lf = lf.filter(pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)

    if store is not None and sku is not None:
        uid = f"{seccion}||T:{store}||S:{sku}"
        lf = lf.filter(pl.col("unique_id") == uid)
    elif store is not None:
        prefix = f"{seccion}||T:{store}||"
        lf = lf.filter(pl.col("unique_id").str.starts_with(prefix))
    elif sku is not None:
        needle = f"||S:{sku}"
        lf = lf.filter(pl.col("unique_id").str.ends_with(needle))

    df = lf.collect()
    if df.height == 0:
        return df
    return backend.prepare_unit_df(df, unidad, has_value)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser(description="Construir artefactos del dashboard")
    parser.add_argument(
        "--forecast", type=str, default=None,
        help="Path a forecast.parquet (default: settings.FORECAST_PATH)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directorio de salida (default: <forecast_dir>/dashboard)",
    )
    parser.add_argument(
        "--update-block-days", type=int, choices=list(getattr(settings, "UPDATE_BLOCK_OPTIONS", (1,7,14,28))), default=None,
        help="Construir artefactos para un escenario precalculado.",
    )
    parser.add_argument(
        "--all-update-blocks", action="store_true",
        help="Construir artefactos para todos los escenarios 1/7/14/28 existentes.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Omite artefactos multi-bloque que ya coinciden con su forecast.",
    )
    args = parser.parse_args(argv)
    if args.all_update_blocks:
        for days in settings.UPDATE_BLOCK_OPTIONS:
            fp = settings.update_block_forecast_path(days)
            if not fp.exists():
                logger.warning("Escenario %dd no existe: %s", days, fp)
                continue
            if args.skip_existing and artifacts_exist(fp):
                print(f"Artefactos {days}d ya sincronizados: {artifacts_dir(fp)}")
                continue
            adir = build_artifacts(fp)
            print(f"Artefactos {days}d: {adir}")
        return
    if args.update_block_days is not None:
        if args.forecast:
            parser.error("No combine --forecast con --update-block-days")
        fp = settings.update_block_forecast_path(args.update_block_days)
        adir = build_artifacts(fp, out_dir=args.out_dir)
    else:
        adir = build_artifacts(args.forecast, out_dir=args.out_dir)
    print(f"Artefactos listos en: {adir}")

if __name__ == "__main__":
    main()
