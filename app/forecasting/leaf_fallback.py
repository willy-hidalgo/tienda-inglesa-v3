"""Vectorized causal fallbacks for high-error SKU+store series.

v11.6 keeps the per-leaf path vectorized, aligns fallback levels with the client metric, and adds a causal SKU-specific 52-week seasonal candidate.  Every candidate is
computed with a handful of Polars group-bys/joins across all leaves at once:

  direct_ses        : current SES level × normalized RLS factor
  deses_ses         : deseasonalize by the selected RLS factor, compute the
                      positive-sale event-time SES terminal level, reapply factor
  deses_robust_mean : deseasonalize, winsorize positive peaks robustly, compute
                      a positive-sale robust mean, reapply factor
  sku_yoy_seasonal  : multiply the existing SES+parent-shape forecast by a
                      SKU-level 13x28-day (364-day) seasonal factor estimated
                      from aggregate sales across stores

OOS model selection uses only the immediately preceding 28-day validation
block.  Forecast-only may use the already-observed OOS block.  No actual from
the target block is used to choose that block's method.
"""
from __future__ import annotations

import datetime as dt
import logging

import polars as pl

import settings

logger = logging.getLogger(__name__)


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _safe_factor_expr(col: str) -> pl.Expr:
    return (
        pl.when(
            pl.col(col).is_not_null()
            & pl.col(col).is_finite()
            & (pl.col(col) > 1e-9)
        )
        .then(pl.col(col).cast(pl.Float64))
        .otherwise(1.0)
    )


def _with_sku_yoy_factors(rows: pl.DataFrame, horizons: dict) -> pl.DataFrame:
    """Attach causal SKU-level 364-day seasonal factors to every row.

    The factor for block *b* is based only on aggregate SKU sales in blocks
    b-13 and b-14 (13 x 28 = 364 days).  Aggregating across stores makes the
    signal robust to store-level intermittency and directly targets the pattern
    seen in the acceptance report where the same SKU under-forecasts across
    several stores.  Reliability shrinkage pulls noisy factors toward 1.0.
    """
    if rows.height == 0:
        return rows
    if not bool(getattr(settings, "LEAF_SKU_SEASONAL_ENABLED", True)):
        return rows.with_columns(
            pl.lit(1.0).alias("sku_yoy_factor_y"),
            pl.lit(1.0).alias("sku_yoy_factor_value"),
            pl.lit(False).alias("sku_yoy_available_y"),
            pl.lit(False).alias("sku_yoy_available_value"),
        )
    train_start = _as_date(horizons["train_start"])
    block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
    lag_blocks = int(getattr(settings, "LEAF_SKU_SEASONAL_LAG_BLOCKS", 13))
    min_obs = int(getattr(settings, "LEAF_SKU_SEASONAL_MIN_POSITIVE_OBS", 14))
    full_weight_obs = max(1, int(getattr(settings, "LEAF_SKU_SEASONAL_FULL_WEIGHT_OBS", 56)))
    clip_lo, clip_hi = tuple(getattr(settings, "LEAF_SKU_SEASONAL_FACTOR_CLIP", (0.50, 2.50)))
    eps = 1e-9

    base = rows.with_columns(
        pl.col("unique_id").str.extract(r"\|\|S:(.+)$", 1).fill_null(pl.col("unique_id")).alias("_sku_key"),
        (((pl.col("ds").cast(pl.Date) - pl.lit(train_start)).dt.total_days()) // block_days).cast(pl.Int32).alias("_season_block"),
    )
    hist = (
        base.filter(pl.col("period_type") != "forecast_only")
        .group_by(["_sku_key", "_season_block"])
        .agg(
            pl.when(pl.col("y").is_not_null() & pl.col("y").is_finite())
            .then(pl.col("y").clip(lower_bound=0.0)).otherwise(0.0).sum().alias("_sum_y"),
            pl.when(pl.col("value").is_not_null() & pl.col("value").is_finite())
            .then(pl.col("value").clip(lower_bound=0.0)).otherwise(0.0).sum().alias("_sum_v"),
            ((pl.col("y").fill_null(0.0) > 0.0) & pl.col("y").is_finite()).sum().cast(pl.Int32).alias("_nz_y"),
            ((pl.col("value").fill_null(0.0) > 0.0) & pl.col("value").is_finite()).sum().cast(pl.Int32).alias("_nz_v"),
        )
    )
    lag13 = hist.with_columns((pl.col("_season_block") + lag_blocks).alias("_season_block")).rename({
        "_sum_y": "_lag13_y", "_sum_v": "_lag13_v",
        "_nz_y": "_lag13_nz_y", "_nz_v": "_lag13_nz_v",
    })
    lag14 = hist.with_columns((pl.col("_season_block") + lag_blocks + 1).alias("_season_block")).rename({
        "_sum_y": "_lag14_y", "_sum_v": "_lag14_v",
        "_nz_y": "_lag14_nz_y", "_nz_v": "_lag14_nz_v",
    })
    factors = (
        lag13.join(lag14, on=["_sku_key", "_season_block"], how="inner")
        .with_columns(
            pl.min_horizontal("_lag13_nz_y", "_lag14_nz_y").alias("_reliability_n_y"),
            pl.min_horizontal("_lag13_nz_v", "_lag14_nz_v").alias("_reliability_n_v"),
        )
        .with_columns(
            pl.when((pl.col("_lag14_y") > eps) & (pl.col("_reliability_n_y") >= min_obs))
            .then((pl.col("_lag13_y") / pl.col("_lag14_y")).clip(lower_bound=float(clip_lo), upper_bound=float(clip_hi)))
            .otherwise(1.0).alias("_raw_sku_yoy_y"),
            pl.when((pl.col("_lag14_v") > eps) & (pl.col("_reliability_n_v") >= min_obs))
            .then((pl.col("_lag13_v") / pl.col("_lag14_v")).clip(lower_bound=float(clip_lo), upper_bound=float(clip_hi)))
            .otherwise(1.0).alias("_raw_sku_yoy_v"),
            (pl.col("_reliability_n_y").cast(pl.Float64) / float(full_weight_obs)).clip(0.0, 1.0).alias("_rel_y"),
            (pl.col("_reliability_n_v").cast(pl.Float64) / float(full_weight_obs)).clip(0.0, 1.0).alias("_rel_v"),
        )
        .with_columns(
            (pl.col("_raw_sku_yoy_y").log() * pl.col("_rel_y")).exp().alias("sku_yoy_factor_y"),
            (pl.col("_raw_sku_yoy_v").log() * pl.col("_rel_v")).exp().alias("sku_yoy_factor_value"),
            ((pl.col("_reliability_n_y") >= min_obs) & (pl.col("_lag14_y") > eps)).alias("sku_yoy_available_y"),
            ((pl.col("_reliability_n_v") >= min_obs) & (pl.col("_lag14_v") > eps)).alias("sku_yoy_available_value"),
        )
        .select("_sku_key", "_season_block", "sku_yoy_factor_y", "sku_yoy_factor_value", "sku_yoy_available_y", "sku_yoy_available_value")
    )
    return (
        base.join(factors, on=["_sku_key", "_season_block"], how="left")
        .with_columns(
            pl.col("sku_yoy_factor_y").fill_null(1.0),
            pl.col("sku_yoy_factor_value").fill_null(1.0),
            pl.col("sku_yoy_available_y").fill_null(False),
            pl.col("sku_yoy_available_value").fill_null(False),
        )
        .drop(["_sku_key", "_season_block"])
    )


def _base_state(
    rows: pl.DataFrame,
    *,
    level_col: str,
    alpha_col: str,
    alpha_source_period: str,
    default_alpha: float,
) -> pl.DataFrame:
    """One initial post-warmup state and one causal alpha per leaf."""
    initial = (
        rows.filter(pl.col("period_type") == "in_sample")
        .group_by("unique_id")
        .agg(
            pl.col("ds").min().alias("_first_date"),
            pl.col(level_col).sort_by("ds").first().fill_null(0.0).alias("_level0"),
        )
        .with_columns(
            (pl.col("_first_date") - pl.duration(days=1)).alias("_base_date")
        )
        .select("unique_id", "_base_date", "_level0")
    )
    alpha = (
        rows.filter(pl.col("period_type") == alpha_source_period)
        .group_by("unique_id")
        .agg(
            pl.col(alpha_col)
            .drop_nulls()
            .first()
            .fill_null(default_alpha)
            .clip(lower_bound=1e-6, upper_bound=1.0)
            .alias("_alpha")
        )
    )
    return initial.join(alpha, on="unique_id", how="left").with_columns(
        pl.col("_alpha").fill_null(default_alpha)
    )


def _alpha_table(
    rows: pl.DataFrame,
    *,
    alpha_col: str,
    alpha_source_period: str,
    default_alpha: float,
) -> pl.DataFrame:
    """One causal alpha per leaf, with a deterministic fallback."""
    ids = rows.select("unique_id").unique()
    alpha = (
        rows.filter(pl.col("period_type") == alpha_source_period)
        .group_by("unique_id")
        .agg(
            pl.col(alpha_col)
            .drop_nulls()
            .first()
            .alias("_alpha")
        )
    )
    return ids.join(alpha, on="unique_id", how="left").with_columns(
        pl.col("_alpha")
        .fill_null(default_alpha)
        .clip(lower_bound=1e-6, upper_bound=1.0)
    )


def _positive_deses_history(
    rows: pl.DataFrame,
    *,
    target: str,
    factor_col: str,
    alpha_col: str,
    alpha_source_period: str,
    end_date: dt.date,
    default_alpha: float,
) -> pl.DataFrame:
    """Positive-sale deseasonalized history used by the official-metric fallbacks.

    The client's official wMAPE/BIAS excludes actual==0.  Therefore fallback
    levels must estimate the magnitude conditional on a sale, rather than the
    calendar-day expected value.  Zero/no-sale days hold the conditional level
    unchanged; they do not pull it toward zero.
    """
    actual_col = "y" if target == "y" else "value"
    alpha = _alpha_table(
        rows,
        alpha_col=alpha_col,
        alpha_source_period=alpha_source_period,
        default_alpha=default_alpha,
    )
    return (
        rows.filter(
            (pl.col("period_type") != "forecast_only")
            & (pl.col("ds") <= pl.lit(end_date))
            & pl.col(actual_col).is_not_null()
            & pl.col(actual_col).is_finite()
            & (pl.col(actual_col) > 0.0)
        )
        .select("unique_id", "ds", actual_col, factor_col)
        .join(alpha, on="unique_id", how="inner")
        .with_columns(
            (
                pl.col(actual_col).cast(pl.Float64)
                / _safe_factor_expr(factor_col)
            ).alias("_des")
        )
        .sort(["unique_id", "ds"])
        .with_columns(
            (
                pl.col("ds").rank(method="ordinal").over("unique_id") - 1
            ).cast(pl.Int32).alias("_event_idx"),
            pl.len().over("unique_id").cast(pl.Int32).alias("_n_events"),
            pl.col("ds").min().over("unique_id").alias("_first_positive_date"),
        )
        .with_columns(
            (
                (pl.lit(end_date) - pl.col("_first_positive_date"))
                .dt.total_days().cast(pl.Int32)
                + 1
            ).alias("_span_days")
        )
    )


def _deses_ses_levels(
    rows: pl.DataFrame,
    *,
    target: str,
    factor_col: str,
    level_col: str,
    alpha_col: str,
    alpha_source_period: str,
    end_date: dt.date,
    default_alpha: float,
    min_history_days: int,
) -> pl.DataFrame:
    """Terminal SES of positive-sale deseasonalized magnitudes, vectorized.

    This is standard SES in *event time*: the state updates only when there is
    a positive sale.  That is intentional because the official client metric
    also conditions on actual>0.  Between sale events the conditional level is
    held constant.  The selected parent RLS factor is reapplied afterwards.
    """
    del level_col  # kept in the signature for backward-compatible callers
    min_positive = int(
        getattr(settings, "LEAF_FALLBACK_MIN_POSITIVE_HISTORY", 14)
    )
    hist = _positive_deses_history(
        rows,
        target=target,
        factor_col=factor_col,
        alpha_col=alpha_col,
        alpha_source_period=alpha_source_period,
        end_date=end_date,
        default_alpha=default_alpha,
    )
    if hist.height == 0:
        return pl.DataFrame(schema={"unique_id": pl.Utf8, "_level": pl.Float64})

    hist = hist.with_columns(
        (1.0 - pl.col("_alpha")).alias("_decay"),
        (
            pl.col("_n_events") - 1 - pl.col("_event_idx")
        ).cast(pl.Int32).alias("_events_to_end"),
    ).with_columns(
        pl.when(pl.col("_event_idx") > 0)
        .then(
            pl.col("_alpha")
            * pl.col("_decay").pow(pl.col("_events_to_end").cast(pl.Float64))
            * pl.col("_des")
        )
        .otherwise(0.0)
        .alias("_weighted")
    )

    return (
        hist.group_by("unique_id")
        .agg(
            pl.col("_des").sort_by("ds").first().alias("_first_des"),
            pl.col("_alpha").first().alias("_alpha"),
            pl.col("_decay").first().alias("_decay"),
            pl.col("_n_events").first().alias("_n_events"),
            pl.col("_span_days").first().alias("_span_days"),
            pl.col("_weighted").sum().alias("_weighted_sum"),
        )
        .filter(
            (pl.col("_span_days") >= int(min_history_days))
            & (pl.col("_n_events") >= min_positive)
        )
        .with_columns(
            (
                pl.col("_decay").pow(
                    (pl.col("_n_events") - 1).cast(pl.Float64)
                )
                * pl.col("_first_des")
                + pl.col("_weighted_sum")
            )
            .clip(lower_bound=0.0)
            .alias("_level")
        )
        .select("unique_id", "_level")
    )


def _robust_mean_levels(
    rows: pl.DataFrame,
    *,
    target: str,
    factor_col: str,
    level_col: str,
    alpha_col: str,
    alpha_source_period: str,
    end_date: dt.date,
    default_alpha: float,
    min_history_days: int,
) -> pl.DataFrame:
    """Positive-sale deseasonalized mean after robust peak winsorization."""
    del level_col
    min_positive = int(
        getattr(settings, "LEAF_FALLBACK_MIN_POSITIVE_HISTORY", 14)
    )
    hist = _positive_deses_history(
        rows,
        target=target,
        factor_col=factor_col,
        alpha_col=alpha_col,
        alpha_source_period=alpha_source_period,
        end_date=end_date,
        default_alpha=default_alpha,
    )
    if hist.height == 0:
        return pl.DataFrame(schema={"unique_id": pl.Utf8, "_level": pl.Float64})

    positive_stats = hist.group_by("unique_id").agg(
        pl.col("_des").median().alias("_median")
    )
    mad = (
        hist.join(positive_stats, on="unique_id", how="left")
        .with_columns((pl.col("_des") - pl.col("_median")).abs().alias("_absdev"))
        .group_by("unique_id")
        .agg(pl.col("_absdev").median().fill_null(0.0).alias("_mad"))
    )
    mad_mult = float(getattr(settings, "LEAF_FALLBACK_PEAK_MAD_MULTIPLIER", 4.0))
    med_mult = float(getattr(settings, "LEAF_FALLBACK_PEAK_MEDIAN_MULTIPLIER", 3.0))
    caps = (
        positive_stats.join(mad, on="unique_id", how="left")
        .with_columns(
            (pl.col("_median") + mad_mult * 1.4826 * pl.col("_mad").fill_null(0.0)).alias("_cap_mad"),
            (pl.col("_median") * med_mult).alias("_cap_med"),
        )
        .with_columns(
            pl.max_horizontal("_median", "_cap_mad", "_cap_med").alias("_cap")
        )
        .select("unique_id", "_cap")
    )
    return (
        hist.join(caps, on="unique_id", how="left")
        .with_columns(
            pl.when(pl.col("_cap").is_not_null() & (pl.col("_des") > pl.col("_cap")))
            .then(pl.col("_cap"))
            .otherwise(pl.col("_des"))
            .alias("_clipped")
        )
        .group_by("unique_id")
        .agg(
            pl.col("_clipped").mean().alias("_level"),
            pl.col("_n_events").first().alias("_n_events"),
            pl.col("_span_days").first().alias("_span_days"),
        )
        .filter(
            (pl.col("_span_days") >= int(min_history_days))
            & (pl.col("_n_events") >= min_positive)
        )
        .select("unique_id", pl.col("_level").clip(lower_bound=0.0))
    )

def _score_level(
    rows: pl.DataFrame,
    *,
    target: str,
    factor_col: str,
    level_table: pl.DataFrame | None,
    validation_start: dt.date,
    validation_end: dt.date,
    method: str,
    bias_weight: float,
) -> pl.DataFrame:
    actual_col = "y" if target == "y" else "value"
    pred_col = "yhat_raw" if target == "y" else "valuehat_raw"
    val = rows.filter(
        (pl.col("period_type") != "forecast_only")
        & (pl.col("ds") >= pl.lit(validation_start))
        & (pl.col("ds") <= pl.lit(validation_end))
        & (pl.col(actual_col).fill_null(0.0) > 0.0)
    )
    if level_table is not None:
        val = val.join(
            level_table.rename({"_level": "_candidate_level"}),
            on="unique_id",
            how="inner",
        ).with_columns(
            (pl.col("_candidate_level") * _safe_factor_expr(factor_col))
            .clip(lower_bound=0.0)
            .alias("_candidate_pred")
        )
    else:
        val = val.with_columns(
            pl.col(pred_col).fill_nan(0.0).fill_null(0.0).clip(lower_bound=0.0).alias("_candidate_pred")
        )

    return (
        val.group_by("unique_id")
        .agg(
            (pl.col(actual_col) - pl.col("_candidate_pred")).abs().sum().alias("_ae"),
            (pl.col("_candidate_pred") - pl.col(actual_col)).sum().alias("_se"),
            pl.col(actual_col).abs().sum().alias("_den"),
            pl.len().cast(pl.UInt32).alias("_n_sales"),
        )
        .filter(pl.col("_den") > 1e-12)
        .with_columns(
            (pl.col("_ae") / pl.col("_den")).alias("_wmape"),
            (pl.col("_se") / pl.col("_den")).alias("_bias"),
        )
        .with_columns(
            (pl.col("_wmape") + bias_weight * pl.col("_bias").abs()).alias("_score"),
            pl.lit(method).alias("_method"),
        )
        .select("unique_id", "_method", "_score", "_wmape", "_bias", "_n_sales")
    )


def _score_seasonal(
    rows: pl.DataFrame,
    *,
    target: str,
    validation_start: dt.date,
    validation_end: dt.date,
    bias_weight: float,
) -> pl.DataFrame:
    actual_col = "y" if target == "y" else "value"
    pred_col = "yhat_raw" if target == "y" else "valuehat_raw"
    factor_col = "sku_yoy_factor_y" if target == "y" else "sku_yoy_factor_value"
    avail_col = "sku_yoy_available_y" if target == "y" else "sku_yoy_available_value"
    val = rows.filter(
        (pl.col("period_type") != "forecast_only")
        & (pl.col("ds") >= pl.lit(validation_start))
        & (pl.col("ds") <= pl.lit(validation_end))
        & (pl.col(actual_col).fill_null(0.0) > 0.0)
    ).with_columns(
        (pl.col(pred_col).fill_nan(0.0).fill_null(0.0).clip(lower_bound=0.0)
         * pl.col(factor_col).fill_null(1.0)).alias("_candidate_pred")
    )
    return (
        val.group_by("unique_id")
        .agg(
            (pl.col(actual_col) - pl.col("_candidate_pred")).abs().sum().alias("_ae"),
            (pl.col("_candidate_pred") - pl.col(actual_col)).sum().alias("_se"),
            pl.col(actual_col).abs().sum().alias("_den"),
            pl.len().cast(pl.UInt32).alias("_n_sales"),
            pl.col(avail_col).any().alias("_seasonal_available"),
        )
        .filter(pl.col("_den") > 1e-12)
        .with_columns(
            (pl.col("_ae") / pl.col("_den")).alias("_seasonal_wmape"),
            (pl.col("_se") / pl.col("_den")).alias("_seasonal_bias"),
        )
        .with_columns(
            (pl.col("_seasonal_wmape") + bias_weight * pl.col("_seasonal_bias").abs()).alias("_seasonal_score")
        )
        .select("unique_id", "_seasonal_score", "_seasonal_wmape", "_seasonal_bias", "_seasonal_available", "_n_sales")
    )


def _select_methods(
    direct: pl.DataFrame,
    deses: pl.DataFrame,
    robust: pl.DataFrame,
    *,
    trigger: float,
    min_improvement: float,
) -> pl.DataFrame:
    """Select one method per UID without a Python UID loop."""
    d = direct.rename({
        "_score": "_direct_score", "_wmape": "_direct_wmape", "_bias": "_direct_bias"
    }).drop("_method")
    s = deses.rename({
        "_score": "_ses_score", "_wmape": "_ses_wmape", "_bias": "_ses_bias"
    }).drop("_method")
    r = robust.rename({
        "_score": "_robust_score", "_wmape": "_robust_wmape", "_bias": "_robust_bias"
    }).drop("_method")
    inf = float("inf")
    x = d.join(s, on="unique_id", how="left").join(r, on="unique_id", how="left").with_columns(
        pl.col("_ses_score").fill_null(inf),
        pl.col("_robust_score").fill_null(inf),
    ).with_columns(
        pl.min_horizontal("_ses_score", "_robust_score").alias("_best_alt_score"),
        pl.when(pl.col("_ses_score") <= pl.col("_robust_score"))
        .then(pl.lit("deses_ses"))
        .otherwise(pl.lit("deses_robust_mean"))
        .alias("_best_alt_method"),
    ).with_columns(
        (
            (pl.col("_direct_wmape") >= trigger)
            & pl.col("_best_alt_score").is_finite()
            & ((pl.col("_direct_score") - pl.col("_best_alt_score")) >= min_improvement)
        ).alias("_use_alt")
    ).with_columns(
        pl.when(pl.col("_use_alt"))
        .then(pl.col("_best_alt_method"))
        .otherwise(pl.lit("direct_ses"))
        .alias("_selected_method"),
        pl.when(pl.col("_use_alt") & (pl.col("_best_alt_method") == "deses_ses"))
        .then(pl.col("_ses_wmape"))
        .when(pl.col("_use_alt") & (pl.col("_best_alt_method") == "deses_robust_mean"))
        .then(pl.col("_robust_wmape"))
        .otherwise(pl.col("_direct_wmape"))
        .alias("_selected_wmape"),
    )
    return x.select("unique_id", "_selected_method", "_selected_wmape", "_direct_wmape")


def _selection_for_origin(
    rows: pl.DataFrame,
    *,
    target: str,
    validation_start: dt.date,
    validation_end: dt.date,
    validation_history_end: dt.date,
    level_end: dt.date,
    alpha_source_period: str,
    trigger: float,
    min_improvement: float,
    bias_weight: float,
    default_alpha: float,
    min_history_days: int,
    min_validation_sales: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Causal sequential fallback selection.

    The business rule is evaluated literally and cheaply:
      1) direct SES;
      2) for ACTIVE leaves with reliable 364-day SKU history, try the cheap
         SKU YoY seasonal multiplier;
      3) only for remaining ACTIVE high-error leaves, try deseasonalized SES;
      4) only if that does not improve enough, try the robust mean.

    This avoids computing both expensive alternatives for every risky leaf and
    avoids tuning a fallback from sparse validation blocks that are not part of
    the official Active OOS metric.
    """
    factor_col = "driver_factor_y" if target == "y" else "driver_factor_value"
    level_col = "ses_level_y" if target == "y" else "ses_level_value"
    alpha_col = "ses_alpha_y" if target == "y" else "ses_alpha_value"

    direct_score = _score_level(
        rows, target=target, factor_col=factor_col, level_table=None,
        validation_start=validation_start, validation_end=validation_end,
        method="direct_ses", bias_weight=bias_weight,
    )
    direct_base = direct_score.select(
        "unique_id",
        pl.col("_score").alias("_direct_score"),
        pl.col("_wmape").alias("_direct_wmape"),
        pl.col("_n_sales").alias("_direct_n_sales"),
    )
    seasonal_score = _score_seasonal(
        rows, target=target, validation_start=validation_start,
        validation_end=validation_end, bias_weight=bias_weight,
    )
    direct_base = (
        direct_base.join(seasonal_score, on="unique_id", how="left")
        .with_columns(
            (
                pl.col("_seasonal_available").fill_null(False)
                & (pl.col("_direct_n_sales") >= int(min_validation_sales))
                & pl.col("_seasonal_score").is_not_null()
                & pl.col("_seasonal_score").is_finite()
                & ((pl.col("_direct_score") - pl.col("_seasonal_score")) >= min_improvement)
            ).alias("_use_seasonal")
        )
        .with_columns(
            (
                (pl.col("_direct_wmape") >= trigger)
                & (pl.col("_direct_n_sales") >= int(min_validation_sales))
                & ~pl.col("_use_seasonal")
            ).alias("_risk_eligible")
        )
    )
    risk_ids = direct_base.filter(pl.col("_risk_eligible")).select("unique_id")

    if risk_ids.height == 0:
        selected = direct_base.with_columns(
            pl.when(pl.col("_use_seasonal"))
            .then(pl.lit("sku_yoy_seasonal"))
            .otherwise(pl.lit("direct_ses"))
            .alias("_selected_method"),
            pl.when(pl.col("_use_seasonal"))
            .then(pl.col("_seasonal_wmape"))
            .otherwise(pl.col("_direct_wmape"))
            .alias("_selected_wmape"),
        ).select(
            "unique_id", "_selected_method", "_selected_wmape",
            "_direct_wmape", "_risk_eligible",
        )
        empty_s = selected.select("unique_id").head(0).with_columns(
            pl.lit(None).cast(pl.Float64).alias("_ses_final_level")
        )
        empty_r = selected.select("unique_id").head(0).with_columns(
            pl.lit(None).cast(pl.Float64).alias("_robust_final_level")
        )
        return selected, empty_s, empty_r

    risk_rows = rows.join(risk_ids, on="unique_id", how="semi")

    # Stage 2: deseasonalized SES for active/high-error leaves only.
    ses_val = _deses_ses_levels(
        risk_rows, target=target, factor_col=factor_col, level_col=level_col,
        alpha_col=alpha_col, alpha_source_period=alpha_source_period,
        end_date=validation_history_end, default_alpha=default_alpha,
        min_history_days=min_history_days,
    )
    ses_score = _score_level(
        risk_rows, target=target, factor_col=factor_col, level_table=ses_val,
        validation_start=validation_start, validation_end=validation_end,
        method="deses_ses", bias_weight=bias_weight,
    ).select(
        "unique_id",
        pl.col("_score").alias("_ses_score"),
        pl.col("_wmape").alias("_ses_wmape"),
    )

    staged = direct_base.join(ses_score, on="unique_id", how="left").with_columns(
        (
            pl.col("_risk_eligible")
            & pl.col("_ses_score").is_not_null()
            & pl.col("_ses_score").is_finite()
            & ((pl.col("_direct_score") - pl.col("_ses_score")) >= min_improvement)
        ).alias("_use_ses")
    )

    # Stage 3: robust mean ONLY where deseasonalized SES did not work.
    robust_needed_ids = staged.filter(
        pl.col("_risk_eligible") & ~pl.col("_use_ses")
    ).select("unique_id")
    robust_score = pl.DataFrame(
        schema={"unique_id": pl.Utf8, "_robust_score": pl.Float64, "_robust_wmape": pl.Float64}
    )
    if robust_needed_ids.height:
        robust_rows = rows.join(robust_needed_ids, on="unique_id", how="semi")
        robust_val = _robust_mean_levels(
            robust_rows, target=target, factor_col=factor_col, level_col=level_col,
            alpha_col=alpha_col, alpha_source_period=alpha_source_period,
            end_date=validation_history_end, default_alpha=default_alpha,
            min_history_days=min_history_days,
        )
        robust_score = _score_level(
            robust_rows, target=target, factor_col=factor_col, level_table=robust_val,
            validation_start=validation_start, validation_end=validation_end,
            method="deses_robust_mean", bias_weight=bias_weight,
        ).select(
            "unique_id",
            pl.col("_score").alias("_robust_score"),
            pl.col("_wmape").alias("_robust_wmape"),
        )

    selected = staged.join(robust_score, on="unique_id", how="left").with_columns(
        (
            pl.col("_risk_eligible")
            & ~pl.col("_use_ses")
            & pl.col("_robust_score").is_not_null()
            & pl.col("_robust_score").is_finite()
            & ((pl.col("_direct_score") - pl.col("_robust_score")) >= min_improvement)
        ).alias("_use_robust")
    ).with_columns(
        pl.when(pl.col("_use_seasonal"))
        .then(pl.lit("sku_yoy_seasonal"))
        .when(pl.col("_use_ses"))
        .then(pl.lit("deses_ses"))
        .when(pl.col("_use_robust"))
        .then(pl.lit("deses_robust_mean"))
        .otherwise(pl.lit("direct_ses"))
        .alias("_selected_method"),
        pl.when(pl.col("_use_seasonal"))
        .then(pl.col("_seasonal_wmape"))
        .when(pl.col("_use_ses"))
        .then(pl.col("_ses_wmape"))
        .when(pl.col("_use_robust"))
        .then(pl.col("_robust_wmape"))
        .otherwise(pl.col("_direct_wmape"))
        .alias("_selected_wmape"),
    ).select(
        "unique_id", "_selected_method", "_selected_wmape",
        "_direct_wmape", "_risk_eligible",
    )

    # Final refit is method-specific: do not compute both alternatives again.
    ses_ids = selected.filter(
        pl.col("_selected_method") == "deses_ses"
    ).select("unique_id")
    if ses_ids.height:
        ses_final = _deses_ses_levels(
            rows.join(ses_ids, on="unique_id", how="semi"),
            target=target, factor_col=factor_col, level_col=level_col,
            alpha_col=alpha_col, alpha_source_period=alpha_source_period,
            end_date=level_end, default_alpha=default_alpha,
            min_history_days=min_history_days,
        ).rename({"_level": "_ses_final_level"})
    else:
        ses_final = ses_ids.with_columns(
            pl.lit(None).cast(pl.Float64).alias("_ses_final_level")
        )

    robust_ids = selected.filter(
        pl.col("_selected_method") == "deses_robust_mean"
    ).select("unique_id")
    if robust_ids.height:
        robust_final = _robust_mean_levels(
            rows.join(robust_ids, on="unique_id", how="semi"),
            target=target, factor_col=factor_col, level_col=level_col,
            alpha_col=alpha_col, alpha_source_period=alpha_source_period,
            end_date=level_end, default_alpha=default_alpha,
            min_history_days=min_history_days,
        ).rename({"_level": "_robust_final_level"})
    else:
        robust_final = robust_ids.with_columns(
            pl.lit(None).cast(pl.Float64).alias("_robust_final_level")
        )
    return selected, ses_final, robust_final

def apply_robust_leaf_fallbacks(rows: pl.DataFrame, horizons: dict) -> pl.DataFrame:
    """Vectorized causal replacement of high-error OOS/forecast-only levels."""
    if not bool(getattr(settings, "LEAF_FALLBACK_ENABLED", True)) or rows.height == 0:
        return rows
    required = {
        "unique_id", "ds", "period_type", "y", "value", "yhat_raw", "valuehat_raw",
        "driver_factor_y", "driver_factor_value", "ses_level_y", "ses_level_value",
    }
    if not required.issubset(rows.columns):
        return rows

    t0 = __import__("time").perf_counter()
    rows = rows.with_columns(pl.col("ds").cast(pl.Date))

    # Performance contract: fallback selection must never drag the full
    # forecast schema (100+ columns) through repeated semi-joins/group-bys.
    # Keep one slim analytical frame and join the tiny method table back only
    # once at the end. This is especially important at production leaf counts.
    work_cols = [
        "unique_id", "ds", "period_type", "y", "value",
        "yhat_raw", "valuehat_raw",
        "driver_factor_y", "driver_factor_value",
        "ses_level_y", "ses_level_value",
        "ses_alpha_y", "ses_alpha_value",
    ]
    work = rows.select([c for c in work_cols if c in rows.columns])
    work = _with_sku_yoy_factors(work, horizons)

    test_start = _as_date(horizons["test_start"])
    test_end = _as_date(horizons["test_end"])
    train_end = _as_date(horizons["train_end"])
    block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
    validation_end = train_end
    validation_start = validation_end - dt.timedelta(days=block_days - 1)
    validation_history_end = validation_start - dt.timedelta(days=1)

    trigger = float(getattr(settings, "LEAF_FALLBACK_TRIGGER_WMAPE", 0.60))
    min_improvement = float(getattr(settings, "LEAF_FALLBACK_MIN_IMPROVEMENT", 0.02))
    bias_weight = float(getattr(settings, "LEAF_FALLBACK_BIAS_WEIGHT", 0.20))
    min_history_days = int(getattr(settings, "LEAF_FALLBACK_MIN_HISTORY_DAYS", 84))
    min_validation_sales = int(
        getattr(
            settings,
            "LEAF_FALLBACK_MIN_VALIDATION_SALES",
            getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7),
        )
    )
    default_alpha = float(getattr(settings, "LEAF_SES_ALPHA", 0.10))

    selections: dict[tuple[str, str], tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]] = {}
    for target in ("y", "value"):
        selections[(target, "out_sample")] = _selection_for_origin(
            work, target=target,
            validation_start=validation_start,
            validation_end=validation_end,
            validation_history_end=validation_history_end,
            level_end=train_end,
            alpha_source_period="out_sample",
            trigger=trigger, min_improvement=min_improvement,
            bias_weight=bias_weight, default_alpha=default_alpha,
            min_history_days=min_history_days,
            min_validation_sales=min_validation_sales,
        )
        # Forecast-only is allowed to validate on the now-observed OOS block.
        selections[(target, "forecast_only")] = _selection_for_origin(
            work, target=target,
            validation_start=test_start,
            validation_end=test_end,
            validation_history_end=train_end,
            level_end=test_end,
            alpha_source_period="forecast_only",
            trigger=trigger, min_improvement=min_improvement,
            bias_weight=bias_weight, default_alpha=default_alpha,
            min_history_days=min_history_days,
            min_validation_sales=min_validation_sales,
        )

    updates: list[pl.DataFrame] = []
    switch_counts: dict[tuple[str, str], int] = {}
    risk_counts: dict[tuple[str, str], int] = {}
    for (target, period), (sel, ses_final, robust_final) in selections.items():
        risk_counts[(target, period)] = sel.filter(pl.col("_risk_eligible")).height
        switch_counts[(target, period)] = sel.filter(pl.col("_selected_method") != "direct_ses").height
        seasonal_col = "sku_yoy_factor_y" if target == "y" else "sku_yoy_factor_value"
        seasonal_period = (
            work.filter(pl.col("period_type") == period)
            .group_by("unique_id")
            .agg(pl.col(seasonal_col).drop_nulls().first().fill_null(1.0).alias("_seasonal_target_factor"))
        )
        u = (
            sel.join(ses_final, on="unique_id", how="left")
            .join(robust_final, on="unique_id", how="left")
            .join(seasonal_period, on="unique_id", how="left")
            .with_columns(
                pl.when(pl.col("_selected_method") == "deses_ses")
                .then(pl.col("_ses_final_level"))
                .when(pl.col("_selected_method") == "deses_robust_mean")
                .then(pl.col("_robust_final_level"))
                .otherwise(None)
                .alias("_fallback_level"),
                pl.when(pl.col("_selected_method") == "sku_yoy_seasonal")
                .then(pl.col("_seasonal_target_factor").fill_null(1.0))
                .otherwise(1.0)
                .alias("_fallback_multiplier"),
                pl.lit(period).alias("period_type"),
                pl.lit(target).alias("_fallback_target"),
            )
            .select(
                "unique_id", "period_type", "_fallback_target",
                pl.col("_selected_method").alias("_fallback_method"),
                "_fallback_level", "_fallback_multiplier",
                pl.col("_selected_wmape").alias("_fallback_validation_wmape"),
            )
        )
        updates.append(u)

    upd = pl.concat(updates, how="vertical_relaxed") if updates else pl.DataFrame()
    if upd.height == 0:
        return rows.with_columns(
            pl.lit("direct_ses").alias("leaf_level_method_y"),
            pl.lit("direct_ses").alias("leaf_level_method_value"),
        )

    uy = upd.filter(pl.col("_fallback_target") == "y").drop("_fallback_target").rename({
        "_fallback_method": "_fb_method_y",
        "_fallback_level": "_fb_level_y",
        "_fallback_multiplier": "_fb_mult_y",
        "_fallback_validation_wmape": "_fb_wmape_y",
    })
    uv = upd.filter(pl.col("_fallback_target") == "value").drop("_fallback_target").rename({
        "_fallback_method": "_fb_method_v",
        "_fallback_level": "_fb_level_v",
        "_fallback_multiplier": "_fb_mult_v",
        "_fallback_validation_wmape": "_fb_wmape_v",
    })

    out = rows.join(uy, on=["unique_id", "period_type"], how="left").join(
        uv, on=["unique_id", "period_type"], how="left"
    ).with_columns(
        pl.col("_fb_method_y").fill_null("direct_ses").alias("leaf_level_method_y"),
        pl.col("_fb_method_v").fill_null("direct_ses").alias("leaf_level_method_value"),
        pl.col("_fb_wmape_y").alias("fallback_validation_wmape_y"),
        pl.col("_fb_wmape_v").alias("fallback_validation_wmape_value"),
        pl.col("_fb_mult_y").fill_null(1.0).alias("sku_seasonal_multiplier_y"),
        pl.col("_fb_mult_v").fill_null(1.0).alias("sku_seasonal_multiplier_value"),
        pl.when(pl.col("_fb_level_y").is_not_null())
        .then(pl.col("_fb_level_y"))
        .otherwise(pl.col("ses_level_y"))
        .alias("ses_level_y"),
        pl.when(pl.col("_fb_level_v").is_not_null())
        .then(pl.col("_fb_level_v"))
        .otherwise(pl.col("ses_level_value"))
        .alias("ses_level_value"),
    ).with_columns(
        (pl.col("ses_level_y") * pl.col("driver_factor_y") * pl.col("sku_seasonal_multiplier_y")).clip(lower_bound=0.0).alias("yhat_raw"),
        (pl.col("ses_level_value") * pl.col("driver_factor_value") * pl.col("sku_seasonal_multiplier_value")).clip(lower_bound=0.0).alias("valuehat_raw"),
    ).with_columns(
        pl.col("yhat_raw").round(0).alias("yhat"),
        pl.col("valuehat_raw").round(2).alias("valuehat"),
    ).drop([
        c for c in (
            "_fb_method_y", "_fb_method_v", "_fb_level_y", "_fb_level_v", "_fb_mult_y", "_fb_mult_v", "_fb_wmape_y", "_fb_wmape_v"
        ) if c in rows.columns or c in {"_fb_method_y", "_fb_method_v", "_fb_level_y", "_fb_level_v", "_fb_mult_y", "_fb_mult_v", "_fb_wmape_y", "_fb_wmape_v"}
    ])

    logger.info(
        "Leaf fallbacks VECTOR v11.6 | risk OOS y=%d v=%d | switches OOS y=%d v=%d | "
        "switches forecast y=%d v=%d | %.1fs",
        risk_counts.get(("y", "out_sample"), 0),
        risk_counts.get(("value", "out_sample"), 0),
        switch_counts.get(("y", "out_sample"), 0),
        switch_counts.get(("value", "out_sample"), 0),
        switch_counts.get(("y", "forecast_only"), 0),
        switch_counts.get(("value", "forecast_only"), 0),
        __import__("time").perf_counter() - t0,
    )
    return out
