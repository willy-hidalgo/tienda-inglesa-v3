"""v12.9.3 leaf challenger: frozen Qty + hierarchical walk-forward Value policy.

The production v11 SES+RLS leaf forecast remains the incumbent and the v12
forecast generators are unchanged from v12.8.6.  Quantity preserves the
v12.8.6 four-block policy.  Value adds a strictly causal three-fold
walk-forward portfolio selector over five already-closed historical blocks.
Portfolio scoring is bottom-up/volume-weighted and promotions are guarded by
win-rate, median gain, worst-fold degradation, |BIAS| deterioration and a
minimum utility gain.  `meta_leaf` must additionally beat the best safe all-mode
by a configured margin.  No actual from the target block is used to construct
or select its forecast.
"""
from __future__ import annotations

import datetime as dt
import gc
import logging

import numpy as np
import polars as pl

import settings

logger = logging.getLogger(__name__)


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _leaf_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("unique_id")
        .str.extract(r"\|\|S:(.+)$", 1)
        .fill_null(pl.col("unique_id"))
        .alias("_v12_sku"),
        pl.col("unique_id")
        .str.replace(r"\|\|S:.*$", "")
        .alias("_v12_store_uid"),
    )


def _finite_nonnegative(col: str) -> pl.Expr:
    return (
        pl.when(pl.col(col).is_not_null() & pl.col(col).is_finite())
        .then(pl.col(col).clip(lower_bound=0.0).cast(pl.Float64))
        .otherwise(0.0)
    )


def _history_until(
    history_all: pl.DataFrame,
    origin: dt.date,
) -> pl.DataFrame:
    """History visible at origin; target-date actuals are structurally excluded."""
    return history_all.filter(pl.col("ds") < pl.lit(origin))



_SKU_TOTAL_METHODS = (
    "v11_sum",
    "recent_weekday",
    "same_weekday_4",
    "lag28",
    "annual_scaled",
    "annual_blend",
)


def _sku_total_paths(
    prepared_history: pl.DataFrame,
    sku_daily_all: pl.DataFrame,
    sku_first_all: pl.DataFrame,
    sku_ids: pl.DataFrame,
    incumbent_rows: pl.DataFrame,
    origin: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    """Causal SKU-total candidate paths for one 28-day block.

    v12.4 separates the SKU-total problem from store allocation.  Every path
    below is available at ``origin``: the shortest actual-data lag is 28 days,
    and ``v11_sum`` is itself a causal forecast already produced by the
    incumbent.  No actual from the target block enters these candidates.
    """
    history_days = int(getattr(settings, "V12_HISTORY_DAYS", 84))
    min_history_days = int(getattr(settings, "V12_MIN_HISTORY_DAYS", 56))
    weekday_lo, weekday_hi = tuple(
        float(x) for x in getattr(settings, "V12_SKU_WEEKDAY_FACTOR_CLIP", (0.40, 2.50))
    )
    annual_lag = int(getattr(settings, "V12_SKU_ANNUAL_LAG_DAYS", 364))
    recent28_weight = float(getattr(settings, "V12_SKU_RECENT28_WEIGHT", 0.60))
    annual_weight = float(getattr(settings, "V12_SKU_ANNUAL_WEIGHT", 0.50))
    annual_scale_lo, annual_scale_hi = tuple(
        float(x) for x in getattr(settings, "V12_SKU_ANNUAL_SCALE_CLIP", (0.50, 2.00))
    )
    eps = 1e-9

    # Work on the pre-aggregated SKU×day table, not the much larger leaf panel.
    # This keeps the multi-block ensemble cheap while preserving identical
    # information for SKU-total magnitude and weekday profiles.
    recent = (
        sku_daily_all.filter(
            (pl.col("ds") < pl.lit(origin))
            & (pl.col("ds") >= pl.lit(origin - dt.timedelta(days=history_days)))
        )
        .with_columns(pl.col("ds").dt.weekday().cast(pl.Int8).alias("_v12_dow"))
    )
    sku_first = (
        sku_first_all.filter(pl.col("_sku_first_ds") < pl.lit(origin))
        .with_columns(
            pl.min_horizontal(
                pl.lit(float(history_days)),
                (pl.lit(origin) - pl.col("_sku_first_ds")).dt.total_days().cast(pl.Float64),
            )
            .clip(lower_bound=1.0)
            .alias("_sku_hist_days")
        )
    )
    recent28_for_level = (
        recent.filter(pl.col("ds") >= pl.lit(origin - dt.timedelta(days=28)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_sku_sum28_y"),
            pl.col("_sku_day_v").sum().alias("_sku_sum28_v"),
        )
    )
    sku_stats = (
        recent.group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_sku_sum_y"),
            pl.col("_sku_day_v").sum().alias("_sku_sum_v"),
        )
        .join(sku_first, on="_v12_sku", how="left")
        .join(recent28_for_level, on="_v12_sku", how="left")
        .with_columns(
            (pl.col("_sku_sum_y") / pl.col("_sku_hist_days")).alias("_sku_level84_y"),
            (pl.col("_sku_sum_v") / pl.col("_sku_hist_days")).alias("_sku_level84_v"),
            (pl.col("_sku_sum28_y").fill_null(0.0) / 28.0).alias("_sku_level28_y"),
            (pl.col("_sku_sum28_v").fill_null(0.0) / 28.0).alias("_sku_level28_v"),
            (pl.col("_sku_hist_days") >= float(min_history_days)).alias("_sku_hist_ok"),
        )
        .with_columns(
            pl.when(pl.col("_sku_hist_days") >= 28.0)
            .then(recent28_weight * pl.col("_sku_level28_y") + (1.0 - recent28_weight) * pl.col("_sku_level84_y"))
            .otherwise(pl.col("_sku_level84_y"))
            .alias("_sku_level_y"),
            pl.when(pl.col("_sku_hist_days") >= 28.0)
            .then(recent28_weight * pl.col("_sku_level28_v") + (1.0 - recent28_weight) * pl.col("_sku_level84_v"))
            .otherwise(pl.col("_sku_level84_v"))
            .alias("_sku_level_v"),
        )
    )

    sku_dow_raw = (
        recent.group_by(["_v12_sku", "_v12_dow"])
        .agg(
            pl.col("_sku_day_y").sum().alias("_sku_dow_sum_y"),
            pl.col("_sku_day_v").sum().alias("_sku_dow_sum_v"),
        )
    )
    dow_grid = sku_stats.select("_v12_sku").join(
        pl.DataFrame({"_v12_dow": list(range(1, 8))}).with_columns(pl.col("_v12_dow").cast(pl.Int8)),
        how="cross",
    )
    sku_dow = (
        dow_grid.join(sku_dow_raw, on=["_v12_sku", "_v12_dow"], how="left")
        .join(sku_stats.select("_v12_sku", "_sku_sum_y", "_sku_sum_v"), on="_v12_sku", how="left")
        .with_columns(
            pl.col("_sku_dow_sum_y").fill_null(0.0),
            pl.col("_sku_dow_sum_v").fill_null(0.0),
        )
        .with_columns(
            pl.when(pl.col("_sku_sum_y") > eps)
            .then((7.0 * pl.col("_sku_dow_sum_y") / pl.col("_sku_sum_y")).clip(weekday_lo, weekday_hi))
            .otherwise(1.0).alias("_sku_week_factor_y_raw"),
            pl.when(pl.col("_sku_sum_v") > eps)
            .then((7.0 * pl.col("_sku_dow_sum_v") / pl.col("_sku_sum_v")).clip(weekday_lo, weekday_hi))
            .otherwise(1.0).alias("_sku_week_factor_v_raw"),
        )
        .with_columns(
            pl.col("_sku_week_factor_y_raw").mean().over("_v12_sku").alias("_sku_week_mean_y"),
            pl.col("_sku_week_factor_v_raw").mean().over("_v12_sku").alias("_sku_week_mean_v"),
        )
        .with_columns(
            (pl.col("_sku_week_factor_y_raw") / pl.col("_sku_week_mean_y").clip(lower_bound=eps)).alias("_sku_week_factor_y"),
            (pl.col("_sku_week_factor_v_raw") / pl.col("_sku_week_mean_v").clip(lower_bound=eps)).alias("_sku_week_factor_v"),
        )
    )

    recent28_start = origin - dt.timedelta(days=28)
    lag28_start = recent28_start - dt.timedelta(days=annual_lag)
    lag28_end = origin - dt.timedelta(days=1 + annual_lag)
    recent28 = (
        sku_daily_all.filter((pl.col("ds") >= pl.lit(recent28_start)) & (pl.col("ds") < pl.lit(origin)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_recent28_y"),
            pl.col("_sku_day_v").sum().alias("_recent28_v"),
        )
    )
    lag28_annual = (
        sku_daily_all.filter((pl.col("ds") >= pl.lit(lag28_start)) & (pl.col("ds") <= pl.lit(lag28_end)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_lag28_annual_y"),
            pl.col("_sku_day_v").sum().alias("_lag28_annual_v"),
        )
    )
    annual_scale = (
        sku_stats.select("_v12_sku")
        .join(recent28, on="_v12_sku", how="left")
        .join(lag28_annual, on="_v12_sku", how="left")
        .with_columns(
            pl.when(pl.col("_lag28_annual_y").fill_null(0.0) > eps)
            .then((pl.col("_recent28_y").fill_null(0.0) / pl.col("_lag28_annual_y")).clip(annual_scale_lo, annual_scale_hi))
            .otherwise(None).alias("_annual_scale_y"),
            pl.when(pl.col("_lag28_annual_v").fill_null(0.0) > eps)
            .then((pl.col("_recent28_v").fill_null(0.0) / pl.col("_lag28_annual_v")).clip(annual_scale_lo, annual_scale_hi))
            .otherwise(None).alias("_annual_scale_v"),
        )
    )

    target_dates = pl.DataFrame({"ds": pl.date_range(origin, end, interval="1d", eager=True)}).with_columns(
        pl.col("ds").cast(pl.Date),
        pl.col("ds").dt.weekday().cast(pl.Int8).alias("_v12_dow"),
        (pl.col("ds") - pl.duration(days=28)).alias("_lag28_ds"),
        (pl.col("ds") - pl.duration(days=35)).alias("_lag35_ds"),
        (pl.col("ds") - pl.duration(days=42)).alias("_lag42_ds"),
        (pl.col("ds") - pl.duration(days=49)).alias("_lag49_ds"),
        (pl.col("ds") - pl.duration(days=annual_lag)).alias("_lag364_ds"),
    )
    grid = sku_ids.join(target_dates, how="cross")

    def _lag_table(days: int, date_col: str, y_col: str, v_col: str) -> pl.DataFrame:
        lo = origin - dt.timedelta(days=days)
        hi = end - dt.timedelta(days=days)
        return (
            sku_daily_all.filter((pl.col("ds") >= pl.lit(lo)) & (pl.col("ds") <= pl.lit(hi)))
            .select("_v12_sku", "ds", "_sku_day_y", "_sku_day_v")
            .rename({"ds": date_col, "_sku_day_y": y_col, "_sku_day_v": v_col})
        )

    inc_leaf = _leaf_keys(
        incumbent_rows.filter(
            (pl.col("ds") >= pl.lit(origin))
            & (pl.col("ds") <= pl.lit(end))
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        ).select("unique_id", "ds", "yhat", "valuehat")
    )
    inc_sku = (
        inc_leaf.group_by(["_v12_sku", "ds"])
        .agg(
            pl.col("yhat").sum().alias("_sku_fc_v11_y"),
            pl.col("valuehat").sum().alias("_sku_fc_v11_v"),
            pl.col("yhat").is_not_null().sum().alias("_sku_fc_v11_n_y"),
            pl.col("valuehat").is_not_null().sum().alias("_sku_fc_v11_n_v"),
        )
    )

    out = (
        grid.join(sku_stats, on="_v12_sku", how="left")
        .join(sku_dow.select("_v12_sku", "_v12_dow", "_sku_week_factor_y", "_sku_week_factor_v"), on=["_v12_sku", "_v12_dow"], how="left")
        .join(annual_scale, on="_v12_sku", how="left")
        .join(_lag_table(28, "_lag28_ds", "_lag28_y", "_lag28_v"), on=["_v12_sku", "_lag28_ds"], how="left")
        .join(_lag_table(35, "_lag35_ds", "_lag35_y", "_lag35_v"), on=["_v12_sku", "_lag35_ds"], how="left")
        .join(_lag_table(42, "_lag42_ds", "_lag42_y", "_lag42_v"), on=["_v12_sku", "_lag42_ds"], how="left")
        .join(_lag_table(49, "_lag49_ds", "_lag49_y", "_lag49_v"), on=["_v12_sku", "_lag49_ds"], how="left")
        .join(_lag_table(annual_lag, "_lag364_ds", "_annual_day_y", "_annual_day_v"), on=["_v12_sku", "_lag364_ds"], how="left")
        .join(inc_sku, on=["_v12_sku", "ds"], how="left")
        .with_columns(
            (pl.col("_sku_level_y").fill_null(0.0) * pl.col("_sku_week_factor_y").fill_null(1.0)).clip(lower_bound=0.0).alias("_sku_fc_recent_y"),
            (pl.col("_sku_level_v").fill_null(0.0) * pl.col("_sku_week_factor_v").fill_null(1.0)).clip(lower_bound=0.0).alias("_sku_fc_recent_v"),
            pl.col("_lag28_y").fill_null(0.0).clip(lower_bound=0.0).alias("_sku_fc_lag28_y"),
            pl.col("_lag28_v").fill_null(0.0).clip(lower_bound=0.0).alias("_sku_fc_lag28_v"),
        )
        .with_columns(
            (pl.sum_horizontal(
                pl.col("_lag28_y").fill_null(0.0), pl.col("_lag35_y").fill_null(0.0),
                pl.col("_lag42_y").fill_null(0.0), pl.col("_lag49_y").fill_null(0.0),
            ) / 4.0).clip(lower_bound=0.0).alias("_sku_fc_same_weekday4_y"),
            (pl.sum_horizontal(
                pl.col("_lag28_v").fill_null(0.0), pl.col("_lag35_v").fill_null(0.0),
                pl.col("_lag42_v").fill_null(0.0), pl.col("_lag49_v").fill_null(0.0),
            ) / 4.0).clip(lower_bound=0.0).alias("_sku_fc_same_weekday4_v"),
            (pl.col("_annual_day_y") * pl.col("_annual_scale_y")).clip(lower_bound=0.0).alias("_sku_fc_annual_y"),
            (pl.col("_annual_day_v") * pl.col("_annual_scale_v")).clip(lower_bound=0.0).alias("_sku_fc_annual_v"),
        )
        .with_columns(
            pl.when(pl.col("_sku_fc_annual_y").is_not_null())
            .then((1.0 - annual_weight) * pl.col("_sku_fc_recent_y") + annual_weight * pl.col("_sku_fc_annual_y"))
            .otherwise(pl.col("_sku_fc_recent_y")).clip(lower_bound=0.0).alias("_sku_fc_annual_blend_y"),
            pl.when(pl.col("_sku_fc_annual_v").is_not_null())
            .then((1.0 - annual_weight) * pl.col("_sku_fc_recent_v") + annual_weight * pl.col("_sku_fc_annual_v"))
            .otherwise(pl.col("_sku_fc_recent_v")).clip(lower_bound=0.0).alias("_sku_fc_annual_blend_v"),
            pl.when(pl.col("_sku_fc_v11_n_y").fill_null(0) > 0).then(pl.col("_sku_fc_v11_y")).otherwise(None).alias("_sku_fc_v11_y"),
            pl.when(pl.col("_sku_fc_v11_n_v").fill_null(0) > 0).then(pl.col("_sku_fc_v11_v")).otherwise(None).alias("_sku_fc_v11_v"),
        )
        .select(
            "_v12_sku", "ds", "_v12_dow",
            "_sku_fc_v11_y", "_sku_fc_v11_v",
            "_sku_fc_recent_y", "_sku_fc_recent_v",
            "_sku_fc_same_weekday4_y", "_sku_fc_same_weekday4_v",
            "_sku_fc_lag28_y", "_sku_fc_lag28_v",
            "_sku_fc_annual_y", "_sku_fc_annual_v",
            "_sku_fc_annual_blend_y", "_sku_fc_annual_blend_v",
            pl.col("_sku_hist_ok").fill_null(False).alias("_v12_sku_available_y"),
            pl.col("_sku_hist_ok").fill_null(False).alias("_v12_sku_available_v"),
        )
    )
    return out


def _sku_total_selection(
    prepared_history: pl.DataFrame,
    sku_daily_all: pl.DataFrame,
    sku_first_all: pl.DataFrame,
    sku_ids: pl.DataFrame,
    incumbent_rows: pl.DataFrame,
    origin: dt.date,
) -> pl.DataFrame:
    """Select a SKU-total submodel and causal v12.7 level calibration.

    Both the model choice and calibration use only fully closed 28-day blocks
    strictly before ``origin``.  Calibration is robust by construction: median
    actual/predicted total ratio + most recent closed ratio, shrinkage to 1.0,
    and a hard clip.  No observation from the target block can enter here.
    """
    block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
    n_blocks = int(getattr(settings, "V12_SKU_SELECTION_BLOCKS", 3))
    min_blocks = int(getattr(settings, "V12_SKU_SELECTION_MIN_BLOCKS", 2))
    min_sales = int(getattr(settings, "V12_SKU_SELECTION_MIN_POSITIVE_DAYS", 7))
    calib_enabled = bool(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_ENABLED", True))
    calib_recent_w = float(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_RECENT_WEIGHT", 0.50))
    calib_prior = float(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_PRIOR_BLOCKS", 1.50))
    calib_min_blocks = int(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_MIN_BLOCKS", min_blocks))
    calib_lo, calib_hi = (
        float(x) for x in getattr(settings, "V12_SKU_LEVEL_CALIBRATION_CLIP", (0.60, 2.00))
    )
    if not 0.0 <= calib_recent_w <= 1.0:
        raise ValueError("V12_SKU_LEVEL_CALIBRATION_RECENT_WEIGHT debe estar entre 0 y 1")
    if calib_prior < 0.0 or calib_min_blocks < 1 or calib_lo <= 0 or calib_hi < calib_lo:
        raise ValueError("Configuración inválida de calibración SKU-total v12.7")
    eps = 1e-9

    method_cols = {
        "v11_sum": ("_sku_fc_v11_y", "_sku_fc_v11_v"),
        "recent_weekday": ("_sku_fc_recent_y", "_sku_fc_recent_v"),
        "same_weekday_4": ("_sku_fc_same_weekday4_y", "_sku_fc_same_weekday4_v"),
        "lag28": ("_sku_fc_lag28_y", "_sku_fc_lag28_v"),
        "annual_scaled": ("_sku_fc_annual_y", "_sku_fc_annual_v"),
        "annual_blend": ("_sku_fc_annual_blend_y", "_sku_fc_annual_blend_v"),
    }
    scores: list[pl.DataFrame] = []
    for block_idx in range(1, n_blocks + 1):
        start = origin - dt.timedelta(days=block_days * block_idx)
        end = start + dt.timedelta(days=block_days - 1)
        paths = _sku_total_paths(
            prepared_history, sku_daily_all, sku_first_all, sku_ids,
            incumbent_rows, start, end,
        )
        actual = sku_daily_all.filter(
            (pl.col("ds") >= pl.lit(start)) & (pl.col("ds") <= pl.lit(end))
        ).select("_v12_sku", "ds", "_sku_day_y", "_sku_day_v")
        x = paths.join(actual, on=["_v12_sku", "ds"], how="left")
        aggs: list[pl.Expr] = [
            (pl.col("_sku_day_y").fill_null(0.0) > 0).sum().cast(pl.Int16).alias("_n_y"),
            (pl.col("_sku_day_v").fill_null(0.0) > 0).sum().cast(pl.Int16).alias("_n_v"),
            pl.when(pl.col("_sku_day_y").fill_null(0.0) > 0).then(pl.col("_sku_day_y")).otherwise(0.0).sum().alias("_den_y"),
            pl.when(pl.col("_sku_day_v").fill_null(0.0) > 0).then(pl.col("_sku_day_v")).otherwise(0.0).sum().alias("_den_v"),
            pl.col("_sku_day_y").fill_null(0.0).sum().alias("_actual_total_y"),
            pl.col("_sku_day_v").fill_null(0.0).sum().alias("_actual_total_v"),
        ]
        for method, (cy, cv) in method_cols.items():
            aggs.extend([
                pl.when((pl.col("_sku_day_y").fill_null(0.0) > 0) & pl.col(cy).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias(f"_pred_n_{method}_y"),
                pl.when((pl.col("_sku_day_v").fill_null(0.0) > 0) & pl.col(cv).is_not_null()).then(1).otherwise(0).sum().cast(pl.Int16).alias(f"_pred_n_{method}_v"),
                pl.when((pl.col("_sku_day_y").fill_null(0.0) > 0) & pl.col(cy).is_not_null()).then((pl.col("_sku_day_y") - pl.col(cy)).abs()).otherwise(0.0).sum().alias(f"_ae_{method}_y"),
                pl.when((pl.col("_sku_day_v").fill_null(0.0) > 0) & pl.col(cv).is_not_null()).then((pl.col("_sku_day_v") - pl.col(cv)).abs()).otherwise(0.0).sum().alias(f"_ae_{method}_v"),
                pl.col(cy).fill_null(0.0).sum().alias(f"_pred_total_{method}_y"),
                pl.col(cv).fill_null(0.0).sum().alias(f"_pred_total_{method}_v"),
            ])
        b = x.group_by("_v12_sku").agg(*aggs).with_columns(
            pl.lit(block_idx).cast(pl.Int8).alias("_sku_sel_block")
        )
        valid_exprs: list[pl.Expr] = []
        for method in method_cols:
            valid_exprs.extend([
                ((pl.col("_n_y") >= min_sales) & (pl.col(f"_pred_n_{method}_y") == pl.col("_n_y"))).alias(f"_valid_{method}_y"),
                ((pl.col("_n_v") >= min_sales) & (pl.col(f"_pred_n_{method}_v") == pl.col("_n_v"))).alias(f"_valid_{method}_v"),
            ])
        b = b.with_columns(*valid_exprs)
        ratio_exprs: list[pl.Expr] = []
        for method in method_cols:
            ratio_exprs.extend([
                pl.when(pl.col(f"_valid_{method}_y") & (pl.col(f"_pred_total_{method}_y") > eps))
                .then(pl.col("_actual_total_y") / pl.col(f"_pred_total_{method}_y"))
                .otherwise(None).alias(f"_cal_ratio_{method}_y"),
                pl.when(pl.col(f"_valid_{method}_v") & (pl.col(f"_pred_total_{method}_v") > eps))
                .then(pl.col("_actual_total_v") / pl.col(f"_pred_total_{method}_v"))
                .otherwise(None).alias(f"_cal_ratio_{method}_v"),
            ])
        scores.append(b.with_columns(*ratio_exprs))

    all_scores = pl.concat(scores, how="vertical_relaxed")
    pool_aggs: list[pl.Expr] = []
    for method in method_cols:
        pool_aggs.extend([
            pl.col(f"_valid_{method}_y").sum().cast(pl.Int16).alias(f"_blocks_{method}_y"),
            pl.col(f"_valid_{method}_v").sum().cast(pl.Int16).alias(f"_blocks_{method}_v"),
            pl.when(pl.col(f"_valid_{method}_y")).then(pl.col(f"_ae_{method}_y")).otherwise(0.0).sum().alias(f"_pool_ae_{method}_y"),
            pl.when(pl.col(f"_valid_{method}_v")).then(pl.col(f"_ae_{method}_v")).otherwise(0.0).sum().alias(f"_pool_ae_{method}_v"),
            pl.when(pl.col(f"_valid_{method}_y")).then(pl.col("_den_y")).otherwise(0.0).sum().alias(f"_pool_den_{method}_y"),
            pl.when(pl.col(f"_valid_{method}_v")).then(pl.col("_den_v")).otherwise(0.0).sum().alias(f"_pool_den_{method}_v"),
            pl.when((pl.col("_sku_sel_block") == 1) & pl.col(f"_valid_{method}_y")).then(pl.col(f"_ae_{method}_y") / pl.col("_den_y").clip(lower_bound=eps)).otherwise(None).max().alias(f"_recent_w_{method}_y"),
            pl.when((pl.col("_sku_sel_block") == 1) & pl.col(f"_valid_{method}_v")).then(pl.col(f"_ae_{method}_v") / pl.col("_den_v").clip(lower_bound=eps)).otherwise(None).max().alias(f"_recent_w_{method}_v"),
            pl.when(pl.col(f"_valid_{method}_y")).then(pl.col(f"_cal_ratio_{method}_y")).otherwise(None).median().alias(f"_cal_med_{method}_y"),
            pl.when(pl.col(f"_valid_{method}_v")).then(pl.col(f"_cal_ratio_{method}_v")).otherwise(None).median().alias(f"_cal_med_{method}_v"),
            pl.when((pl.col("_sku_sel_block") == 1) & pl.col(f"_valid_{method}_y")).then(pl.col(f"_cal_ratio_{method}_y")).otherwise(None).max().alias(f"_cal_recent_{method}_y"),
            pl.when((pl.col("_sku_sel_block") == 1) & pl.col(f"_valid_{method}_v")).then(pl.col(f"_cal_ratio_{method}_v")).otherwise(None).max().alias(f"_cal_recent_{method}_v"),
        ])
    pooled = all_scores.group_by("_v12_sku").agg(*pool_aggs)
    wmape_exprs: list[pl.Expr] = []
    for method in method_cols:
        wmape_exprs.extend([
            pl.when((pl.col(f"_blocks_{method}_y") >= min_blocks) & (pl.col(f"_pool_den_{method}_y") > eps)).then(pl.col(f"_pool_ae_{method}_y") / pl.col(f"_pool_den_{method}_y")).otherwise(None).alias(f"_w_{method}_y"),
            pl.when((pl.col(f"_blocks_{method}_v") >= min_blocks) & (pl.col(f"_pool_den_{method}_v") > eps)).then(pl.col(f"_pool_ae_{method}_v") / pl.col(f"_pool_den_{method}_v")).otherwise(None).alias(f"_w_{method}_v"),
        ])
    pooled = pooled.with_columns(*wmape_exprs)

    def _choice_expr(suffix: str) -> pl.Expr:
        ranked = [
            pl.col(f"_w_{m}_{suffix}").fill_null(999.0)
            + float(getattr(settings, "V12_SKU_RECENT_SCORE_WEIGHT", 0.25))
            * pl.coalesce([pl.col(f"_recent_w_{m}_{suffix}"), pl.col(f"_w_{m}_{suffix}"), pl.lit(999.0)])
            for m in method_cols
        ]
        best = pl.min_horizontal(ranked)
        expr = pl.lit("annual_blend")
        # reversed so v11_sum wins exact ties as the safest anchor
        for m, score in reversed(list(zip(method_cols.keys(), ranked))):
            expr = pl.when(score <= best + 1e-12).then(pl.lit(m)).otherwise(expr)
        return expr

    pooled = pooled.with_columns(
        _choice_expr("y").alias("v12_sku_model_y"),
        _choice_expr("v").alias("v12_sku_model_value"),
    ).with_columns(
        pl.when(pl.col("v12_sku_model_y") == "v11_sum").then(pl.col("_w_v11_sum_y"))
        .when(pl.col("v12_sku_model_y") == "recent_weekday").then(pl.col("_w_recent_weekday_y"))
        .when(pl.col("v12_sku_model_y") == "same_weekday_4").then(pl.col("_w_same_weekday_4_y"))
        .when(pl.col("v12_sku_model_y") == "lag28").then(pl.col("_w_lag28_y"))
        .when(pl.col("v12_sku_model_y") == "annual_scaled").then(pl.col("_w_annual_scaled_y"))
        .otherwise(pl.col("_w_annual_blend_y")).alias("v12_sku_validation_wmape_y"),
        pl.when(pl.col("v12_sku_model_value") == "v11_sum").then(pl.col("_w_v11_sum_v"))
        .when(pl.col("v12_sku_model_value") == "recent_weekday").then(pl.col("_w_recent_weekday_v"))
        .when(pl.col("v12_sku_model_value") == "same_weekday_4").then(pl.col("_w_same_weekday_4_v"))
        .when(pl.col("v12_sku_model_value") == "lag28").then(pl.col("_w_lag28_v"))
        .when(pl.col("v12_sku_model_value") == "annual_scaled").then(pl.col("_w_annual_scaled_v"))
        .otherwise(pl.col("_w_annual_blend_v")).alias("v12_sku_validation_wmape_value"),
    )

    def _selected_method_expr(model_col: str, suffix: str, prefix: str, default: float | int | None) -> pl.Expr:
        expr: pl.Expr = pl.lit(default)
        for m in reversed(list(method_cols.keys())):
            expr = pl.when(pl.col(model_col) == m).then(pl.col(f"{prefix}{m}_{suffix}")).otherwise(expr)
        return expr

    pooled = pooled.with_columns(
        _selected_method_expr("v12_sku_model_y", "y", "_cal_med_", None).alias("_cal_med_selected_y"),
        _selected_method_expr("v12_sku_model_y", "y", "_cal_recent_", None).alias("_cal_recent_selected_y"),
        _selected_method_expr("v12_sku_model_y", "y", "_blocks_", 0).cast(pl.Int16).alias("v12_sku_level_calibration_blocks_y"),
        _selected_method_expr("v12_sku_model_value", "v", "_cal_med_", None).alias("_cal_med_selected_v"),
        _selected_method_expr("v12_sku_model_value", "v", "_cal_recent_", None).alias("_cal_recent_selected_v"),
        _selected_method_expr("v12_sku_model_value", "v", "_blocks_", 0).cast(pl.Int16).alias("v12_sku_level_calibration_blocks_value"),
    ).with_columns(
        pl.coalesce([
            (1.0 - calib_recent_w) * pl.col("_cal_med_selected_y") + calib_recent_w * pl.col("_cal_recent_selected_y"),
            pl.col("_cal_recent_selected_y"),
            pl.col("_cal_med_selected_y"),
            pl.lit(1.0),
        ]).clip(lower_bound=calib_lo, upper_bound=calib_hi).alias("v12_sku_level_calibration_raw_ratio_y"),
        pl.coalesce([
            (1.0 - calib_recent_w) * pl.col("_cal_med_selected_v") + calib_recent_w * pl.col("_cal_recent_selected_v"),
            pl.col("_cal_recent_selected_v"),
            pl.col("_cal_med_selected_v"),
            pl.lit(1.0),
        ]).clip(lower_bound=calib_lo, upper_bound=calib_hi).alias("v12_sku_level_calibration_raw_ratio_value"),
    ).with_columns(
        pl.when(pl.lit(calib_enabled) & (pl.col("v12_sku_level_calibration_blocks_y") >= calib_min_blocks))
        .then(
            1.0
            + (
                pl.col("v12_sku_level_calibration_blocks_y").cast(pl.Float64)
                / (pl.col("v12_sku_level_calibration_blocks_y").cast(pl.Float64) + calib_prior)
            ) * (pl.col("v12_sku_level_calibration_raw_ratio_y") - 1.0)
        )
        .otherwise(1.0)
        .clip(lower_bound=calib_lo, upper_bound=calib_hi)
        .alias("v12_sku_level_calibration_factor_y"),
        pl.when(pl.lit(calib_enabled) & (pl.col("v12_sku_level_calibration_blocks_value") >= calib_min_blocks))
        .then(
            1.0
            + (
                pl.col("v12_sku_level_calibration_blocks_value").cast(pl.Float64)
                / (pl.col("v12_sku_level_calibration_blocks_value").cast(pl.Float64) + calib_prior)
            ) * (pl.col("v12_sku_level_calibration_raw_ratio_value") - 1.0)
        )
        .otherwise(1.0)
        .clip(lower_bound=calib_lo, upper_bound=calib_hi)
        .alias("v12_sku_level_calibration_factor_value"),
    )

    return pooled.select(
        "_v12_sku", "v12_sku_model_y", "v12_sku_model_value",
        "v12_sku_validation_wmape_y", "v12_sku_validation_wmape_value",
        "v12_sku_level_calibration_raw_ratio_y", "v12_sku_level_calibration_raw_ratio_value",
        "v12_sku_level_calibration_factor_y", "v12_sku_level_calibration_factor_value",
        "v12_sku_level_calibration_blocks_y", "v12_sku_level_calibration_blocks_value",
    )

def _selected_sku_total_forecast(
    prepared_history: pl.DataFrame,
    sku_daily_all: pl.DataFrame,
    sku_first_all: pl.DataFrame,
    sku_ids: pl.DataFrame,
    incumbent_rows: pl.DataFrame,
    origin: dt.date,
    end: dt.date,
) -> pl.DataFrame:
    paths = _sku_total_paths(
        prepared_history, sku_daily_all, sku_first_all, sku_ids,
        incumbent_rows, origin, end,
    )
    selection = _sku_total_selection(
        prepared_history, sku_daily_all, sku_first_all, sku_ids,
        incumbent_rows, origin,
    )
    return (
        paths.join(selection, on="_v12_sku", how="left")
        .with_columns(
            pl.col("v12_sku_model_y").fill_null("annual_blend"),
            pl.col("v12_sku_model_value").fill_null("annual_blend"),
            pl.col("v12_sku_level_calibration_factor_y").fill_null(1.0),
            pl.col("v12_sku_level_calibration_factor_value").fill_null(1.0),
            pl.col("v12_sku_level_calibration_raw_ratio_y").fill_null(1.0),
            pl.col("v12_sku_level_calibration_raw_ratio_value").fill_null(1.0),
            pl.col("v12_sku_level_calibration_blocks_y").fill_null(0),
            pl.col("v12_sku_level_calibration_blocks_value").fill_null(0),
        )
        .with_columns(
            pl.coalesce([
                pl.when(pl.col("v12_sku_model_y") == "v11_sum").then(pl.col("_sku_fc_v11_y"))
                .when(pl.col("v12_sku_model_y") == "recent_weekday").then(pl.col("_sku_fc_recent_y"))
                .when(pl.col("v12_sku_model_y") == "same_weekday_4").then(pl.col("_sku_fc_same_weekday4_y"))
                .when(pl.col("v12_sku_model_y") == "lag28").then(pl.col("_sku_fc_lag28_y"))
                .when(pl.col("v12_sku_model_y") == "annual_scaled").then(pl.col("_sku_fc_annual_y"))
                .otherwise(pl.col("_sku_fc_annual_blend_y")),
                pl.col("_sku_fc_annual_blend_y"),
                pl.col("_sku_fc_recent_y"),
                pl.col("_sku_fc_v11_y"),
                pl.lit(0.0),
            ]).clip(lower_bound=0.0).alias("v12_sku_forecast_uncalibrated_y"),
            pl.coalesce([
                pl.when(pl.col("v12_sku_model_value") == "v11_sum").then(pl.col("_sku_fc_v11_v"))
                .when(pl.col("v12_sku_model_value") == "recent_weekday").then(pl.col("_sku_fc_recent_v"))
                .when(pl.col("v12_sku_model_value") == "same_weekday_4").then(pl.col("_sku_fc_same_weekday4_v"))
                .when(pl.col("v12_sku_model_value") == "lag28").then(pl.col("_sku_fc_lag28_v"))
                .when(pl.col("v12_sku_model_value") == "annual_scaled").then(pl.col("_sku_fc_annual_v"))
                .otherwise(pl.col("_sku_fc_annual_blend_v")),
                pl.col("_sku_fc_annual_blend_v"),
                pl.col("_sku_fc_recent_v"),
                pl.col("_sku_fc_v11_v"),
                pl.lit(0.0),
            ]).clip(lower_bound=0.0).alias("v12_sku_forecast_uncalibrated_value"),
        )
        .with_columns(
            (pl.col("v12_sku_forecast_uncalibrated_y") * pl.col("v12_sku_level_calibration_factor_y"))
            .clip(lower_bound=0.0).alias("v12_sku_forecast_calibrated_challenger_y"),
            (pl.col("v12_sku_forecast_uncalibrated_value") * pl.col("v12_sku_level_calibration_factor_value"))
            .clip(lower_bound=0.0).alias("v12_sku_forecast_calibrated_challenger_value"),
        )
        .with_columns(
            pl.when(pl.lit(bool(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER", False))))
            .then(pl.col("v12_sku_forecast_calibrated_challenger_y"))
            .otherwise(pl.col("v12_sku_forecast_uncalibrated_y"))
            .alias("v12_sku_forecast_y"),
            pl.when(pl.lit(bool(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER", False))))
            .then(pl.col("v12_sku_forecast_calibrated_challenger_value"))
            .otherwise(pl.col("v12_sku_forecast_uncalibrated_value"))
            .alias("v12_sku_forecast_value"),
            pl.lit(bool(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER", False)))
            .alias("v12_sku_level_calibration_applied_y"),
            pl.lit(bool(getattr(settings, "V12_SKU_LEVEL_CALIBRATION_USE_CHALLENGER", False)))
            .alias("v12_sku_level_calibration_applied_value"),
        )
        .with_columns(
            # v12.8 restores the v12.6 uncalibrated SKU-total as production
            # baseline.  Calibration remains a separately materialized
            # challenger and can be enabled only by explicit configuration.
            pl.col("v12_sku_forecast_y").alias("v12_sku_forecast_base_y"),
            pl.col("v12_sku_forecast_value").alias("v12_sku_forecast_base_value"),
            pl.lit("base").alias("v12_sku_shape_model_y"),
            pl.lit("base").alias("v12_sku_shape_model_value"),
            pl.lit(0).cast(pl.Int32).alias("v12_sku_shape_training_rows_y"),
            pl.lit(0).cast(pl.Int32).alias("v12_sku_shape_training_rows_value"),
            pl.lit(False).alias("v12_sku_shape_applied_y"),
            pl.lit(False).alias("v12_sku_shape_applied_value"),
        )
        .select(
            "_v12_sku", "ds", "_v12_dow",
            "v12_sku_forecast_y", "v12_sku_forecast_value",
            "v12_sku_forecast_uncalibrated_y", "v12_sku_forecast_uncalibrated_value",
            "v12_sku_forecast_calibrated_challenger_y", "v12_sku_forecast_calibrated_challenger_value",
            "v12_sku_level_calibration_applied_y", "v12_sku_level_calibration_applied_value",
            "v12_sku_forecast_base_y", "v12_sku_forecast_base_value",
            "v12_sku_shape_model_y", "v12_sku_shape_model_value",
            "v12_sku_shape_training_rows_y", "v12_sku_shape_training_rows_value",
            "v12_sku_shape_applied_y", "v12_sku_shape_applied_value",
            "v12_sku_model_y", "v12_sku_model_value",
            "v12_sku_validation_wmape_y", "v12_sku_validation_wmape_value",
            "v12_sku_level_calibration_raw_ratio_y", "v12_sku_level_calibration_raw_ratio_value",
            "v12_sku_level_calibration_factor_y", "v12_sku_level_calibration_factor_value",
            "v12_sku_level_calibration_blocks_y", "v12_sku_level_calibration_blocks_value",
            "_v12_sku_available_y", "_v12_sku_available_v",
        )
    )

def _block_candidate(
    prepared_history: pl.DataFrame,
    sku_daily_all: pl.DataFrame,
    sku_first_all: pl.DataFrame,
    uid_first_all: pl.DataFrame,
    ids: pl.DataFrame,
    incumbent_rows: pl.DataFrame,
    origin: dt.date,
    end: dt.date,
    sku_forecast_override: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build one causal 28-day v12 candidate block for all leaves."""
    history_days = int(getattr(settings, "V12_HISTORY_DAYS", 84))
    share_recent_days = int(getattr(settings, "V12_SHARE_RECENT_DAYS", 28))
    share_stable_days = int(getattr(settings, "V12_SHARE_STABLE_DAYS", 84))
    share_recent_weight = float(getattr(settings, "V12_SHARE_RECENT_WEIGHT", 0.70))
    if not (0.0 <= share_recent_weight <= 1.0):
        raise ValueError("V12_SHARE_RECENT_WEIGHT debe estar entre 0 y 1")
    if share_recent_days <= 0 or share_stable_days <= 0:
        raise ValueError("V12_SHARE_RECENT_DAYS/V12_SHARE_STABLE_DAYS deben ser > 0")
    shrink_days = float(getattr(settings, "V12_ALLOCATION_FULL_WEIGHT_POSITIVE_DAYS", 14))
    gate = float(getattr(settings, "V12_OCCURRENCE_GATE", 0.10))
    eps = 1e-9

    hist = _history_until(prepared_history, origin)
    recent_start = origin - dt.timedelta(days=history_days)
    recent = hist.filter(pl.col("ds") >= pl.lit(recent_start))
    share_recent = hist.filter(
        pl.col("ds") >= pl.lit(origin - dt.timedelta(days=share_recent_days))
    )
    share_stable = hist.filter(
        pl.col("ds") >= pl.lit(origin - dt.timedelta(days=share_stable_days))
    )

    uid_first = (
        uid_first_all.filter(pl.col("_uid_first_ds") < pl.lit(origin))
        .with_columns(
            pl.min_horizontal(
                pl.lit(float(history_days)),
                (pl.lit(origin) - pl.col("_uid_first_ds"))
                .dt.total_days()
                .cast(pl.Float64),
            )
            .clip(lower_bound=1.0)
            .alias("_uid_hist_days")
        )
    )
    sku_stats = (
        recent.group_by("_v12_sku")
        .agg(
            pl.col("_v12_y").sum().alias("_sku_sum_y"),
            pl.col("_v12_v").sum().alias("_sku_sum_v"),
        )
    )
    sku_share_recent = share_recent.group_by("_v12_sku").agg(
        pl.col("_v12_y").sum().alias("_sku_share_recent_y"),
        pl.col("_v12_v").sum().alias("_sku_share_recent_v"),
    )
    uid_share_recent = share_recent.group_by(["unique_id", "_v12_sku"]).agg(
        pl.col("_v12_y").sum().alias("_uid_share_recent_y"),
        pl.col("_v12_v").sum().alias("_uid_share_recent_v"),
    )
    sku_share_stable = share_stable.group_by("_v12_sku").agg(
        pl.col("_v12_y").sum().alias("_sku_share_stable_y"),
        pl.col("_v12_v").sum().alias("_sku_share_stable_v"),
    )
    uid_share_stable = share_stable.group_by(["unique_id", "_v12_sku"]).agg(
        pl.col("_v12_y").sum().alias("_uid_share_stable_y"),
        pl.col("_v12_v").sum().alias("_uid_share_stable_v"),
    )
    sku_ids = ids.select("_v12_sku").unique()
    target_dates = pl.DataFrame(
        {"ds": pl.date_range(origin, end, interval="1d", eager=True)}
    ).with_columns(
        pl.col("ds").cast(pl.Date),
        pl.col("ds").dt.weekday().cast(pl.Int8).alias("_v12_dow"),
    )
    sku_forecast = (
        sku_forecast_override
        if sku_forecast_override is not None
        else _selected_sku_total_forecast(
            prepared_history=prepared_history,
            sku_daily_all=sku_daily_all,
            sku_first_all=sku_first_all,
            sku_ids=sku_ids,
            incumbent_rows=incumbent_rows,
            origin=origin,
            end=end,
        )
    )

    # Store/SKU allocation statistics. Occurrence keeps the v12.3 weekday/global
    # shrinkage. Magnitude share in v12.5 is independent of DOW and uses the
    # robust causal 28d/84d recency blend validated on rolling closed blocks.
    store_stats = (
        recent.group_by(["unique_id", "_v12_sku"])
        .agg(
            pl.col("_v12_y").sum().alias("_uid_sum_y"),
            pl.col("_v12_v").sum().alias("_uid_sum_v"),
            (pl.col("_v12_y") > 0).sum().cast(pl.Float64).alias("_uid_pos_y"),
            (pl.col("_v12_v") > 0).sum().cast(pl.Float64).alias("_uid_pos_v"),
        )
        .join(uid_first, on=["unique_id", "_v12_sku"], how="left")
        .join(sku_stats.select("_v12_sku", "_sku_sum_y", "_sku_sum_v"), on="_v12_sku", how="left")
        .with_columns(
            pl.when(pl.col("_sku_sum_y") > eps)
            .then(pl.col("_uid_sum_y") / pl.col("_sku_sum_y"))
            .otherwise(0.0)
            .alias("_share_all_y"),
            pl.when(pl.col("_sku_sum_v") > eps)
            .then(pl.col("_uid_sum_v") / pl.col("_sku_sum_v"))
            .otherwise(0.0)
            .alias("_share_all_v"),
            (pl.col("_uid_pos_y") / pl.col("_uid_hist_days")).clip(0.0, 1.0).alias("_occ_all_y"),
            (pl.col("_uid_pos_v") / pl.col("_uid_hist_days")).clip(0.0, 1.0).alias("_occ_all_v"),
            (pl.col("_uid_pos_y") / shrink_days).clip(0.0, 1.0).alias("_rel_y"),
            (pl.col("_uid_pos_v") / shrink_days).clip(0.0, 1.0).alias("_rel_v"),
        )
    )
    store_dow = (
        recent.group_by(["unique_id", "_v12_sku", "_v12_dow"])
        .agg(
            pl.col("_v12_y").sum().alias("_uid_dow_y"),
            pl.col("_v12_v").sum().alias("_uid_dow_v"),
            (pl.col("_v12_y") > 0).sum().cast(pl.Float64).alias("_uid_dow_pos_y"),
            (pl.col("_v12_v") > 0).sum().cast(pl.Float64).alias("_uid_dow_pos_v"),
        )
    )
    sku_dow_alloc = (
        recent.group_by(["_v12_sku", "_v12_dow"])
        .agg(
            pl.col("_v12_y").sum().alias("_sku_dow_alloc_y"),
            pl.col("_v12_v").sum().alias("_sku_dow_alloc_v"),
        )
    )
    nstores = ids.group_by("_v12_sku").agg(pl.len().cast(pl.Float64).alias("_n_stores"))

    leaf_grid = (
        ids.select("unique_id", "_v12_sku", "_v12_store_uid")
        .join(target_dates.select("ds", "_v12_dow"), how="cross")
        .join(store_stats, on=["unique_id", "_v12_sku"], how="left")
        .join(store_dow, on=["unique_id", "_v12_sku", "_v12_dow"], how="left")
        .join(sku_dow_alloc, on=["_v12_sku", "_v12_dow"], how="left")
        .join(uid_share_recent, on=["unique_id", "_v12_sku"], how="left")
        .join(sku_share_recent, on="_v12_sku", how="left")
        .join(uid_share_stable, on=["unique_id", "_v12_sku"], how="left")
        .join(sku_share_stable, on="_v12_sku", how="left")
        .join(nstores, on="_v12_sku", how="left")
        .with_columns(
            pl.when(pl.col("_share_all_y").fill_null(0.0) > 0)
            .then(pl.col("_share_all_y"))
            .otherwise(1.0 / pl.col("_n_stores").clip(lower_bound=1.0))
            .alias("_share_base_y"),
            pl.when(pl.col("_share_all_v").fill_null(0.0) > 0)
            .then(pl.col("_share_all_v"))
            .otherwise(1.0 / pl.col("_n_stores").clip(lower_bound=1.0))
            .alias("_share_base_v"),
            pl.when(pl.col("_sku_share_recent_y").fill_null(0.0) > eps)
            .then(pl.col("_uid_share_recent_y").fill_null(0.0) / pl.col("_sku_share_recent_y"))
            .otherwise(0.0)
            .alias("_share_recent_y"),
            pl.when(pl.col("_sku_share_recent_v").fill_null(0.0) > eps)
            .then(pl.col("_uid_share_recent_v").fill_null(0.0) / pl.col("_sku_share_recent_v"))
            .otherwise(0.0)
            .alias("_share_recent_v"),
            pl.when(pl.col("_sku_share_stable_y").fill_null(0.0) > eps)
            .then(pl.col("_uid_share_stable_y").fill_null(0.0) / pl.col("_sku_share_stable_y"))
            .otherwise(0.0)
            .alias("_share_stable_y"),
            pl.when(pl.col("_sku_share_stable_v").fill_null(0.0) > eps)
            .then(pl.col("_uid_share_stable_v").fill_null(0.0) / pl.col("_sku_share_stable_v"))
            .otherwise(0.0)
            .alias("_share_stable_v"),
        )
        .with_columns(
            pl.when(pl.col("_sku_dow_alloc_y").fill_null(0.0) > eps)
            .then(pl.col("_uid_dow_y").fill_null(0.0) / pl.col("_sku_dow_alloc_y"))
            .otherwise(pl.col("_share_base_y"))
            .alias("_share_dow_y"),
            pl.when(pl.col("_sku_dow_alloc_v").fill_null(0.0) > eps)
            .then(pl.col("_uid_dow_v").fill_null(0.0) / pl.col("_sku_dow_alloc_v"))
            .otherwise(pl.col("_share_base_v"))
            .alias("_share_dow_v"),
            (7.0 * pl.col("_uid_dow_pos_y").fill_null(0.0) / pl.col("_uid_hist_days").fill_null(float(history_days))).clip(0.0, 1.0).alias("_occ_dow_y"),
            (7.0 * pl.col("_uid_dow_pos_v").fill_null(0.0) / pl.col("_uid_hist_days").fill_null(float(history_days))).clip(0.0, 1.0).alias("_occ_dow_v"),
        )
        .with_columns(
            (
                pl.col("_rel_y").fill_null(0.0) * pl.col("_share_dow_y")
                + (1.0 - pl.col("_rel_y").fill_null(0.0)) * pl.col("_share_base_y")
            ).clip(lower_bound=0.0).alias("_share_blend_y"),
            (
                pl.col("_rel_v").fill_null(0.0) * pl.col("_share_dow_v")
                + (1.0 - pl.col("_rel_v").fill_null(0.0)) * pl.col("_share_base_v")
            ).clip(lower_bound=0.0).alias("_share_blend_v"),
            (
                pl.col("_rel_y").fill_null(0.0) * pl.col("_occ_dow_y")
                + (1.0 - pl.col("_rel_y").fill_null(0.0)) * pl.col("_occ_all_y").fill_null(0.0)
            ).clip(0.0, 1.0).alias("v12_occurrence_prob_y"),
            (
                pl.col("_rel_v").fill_null(0.0) * pl.col("_occ_dow_v")
                + (1.0 - pl.col("_rel_v").fill_null(0.0)) * pl.col("_occ_all_v").fill_null(0.0)
            ).clip(0.0, 1.0).alias("v12_occurrence_prob_value"),
        )
        .with_columns(
            # v12.3: occurrence is one retail event, not two target-specific
            # events.  Use one shared hurdle gate for quantity and value.  The
            # max probability is conservative against false closures when one
            # target is noisy/missing, while the post-gate magnitude/share is
            # still target-specific.
            pl.max_horizontal(
                pl.col("v12_occurrence_prob_y"),
                pl.col("v12_occurrence_prob_value"),
            ).alias("v12_occurrence_prob_joint"),
        )
        .with_columns(
            (pl.col("v12_occurrence_prob_joint") >= gate).alias("v12_occurrence_gate_open_joint"),
            (pl.col("v12_occurrence_prob_joint") >= gate).alias("v12_occurrence_gate_open_y"),
            (pl.col("v12_occurrence_prob_joint") >= gate).alias("v12_occurrence_gate_open_value"),
        )
        .with_columns(
            (
                share_recent_weight * pl.col("_share_recent_y").fill_null(0.0)
                + (1.0 - share_recent_weight) * pl.col("_share_stable_y").fill_null(0.0)
            ).clip(lower_bound=0.0).alias("_share_28x84_y"),
            (
                share_recent_weight * pl.col("_share_recent_v").fill_null(0.0)
                + (1.0 - share_recent_weight) * pl.col("_share_stable_v").fill_null(0.0)
            ).clip(lower_bound=0.0).alias("_share_28x84_v"),
        )
        .with_columns(
            pl.when(pl.col("v12_occurrence_gate_open_y"))
            .then(pl.col("_share_28x84_y"))
            .otherwise(0.0)
            .alias("_alloc_weight_y"),
            pl.when(pl.col("v12_occurrence_gate_open_value"))
            .then(pl.col("_share_28x84_v"))
            .otherwise(0.0)
            .alias("_alloc_weight_v"),
        )
        .with_columns(
            pl.col("_alloc_weight_y").sum().over(["_v12_sku", "ds"]).alias("_weight_sum_y"),
            pl.col("_alloc_weight_v").sum().over(["_v12_sku", "ds"]).alias("_weight_sum_v"),
            pl.col("v12_occurrence_gate_open_y").cast(pl.Int32).sum().over(["_v12_sku", "ds"]).alias("_open_count_y"),
            pl.col("v12_occurrence_gate_open_value").cast(pl.Int32).sum().over(["_v12_sku", "ds"]).alias("_open_count_v"),
            pl.col("_share_base_y").sum().over(["_v12_sku", "ds"]).alias("_base_sum_y"),
            pl.col("_share_base_v").sum().over(["_v12_sku", "ds"]).alias("_base_sum_v"),
        )
        .with_columns(
            pl.when(pl.col("_weight_sum_y") > eps)
            .then(pl.col("_alloc_weight_y") / pl.col("_weight_sum_y"))
            .when(pl.col("v12_occurrence_gate_open_y") & (pl.col("_open_count_y") > 0))
            .then(1.0 / pl.col("_open_count_y").cast(pl.Float64))
            .otherwise(pl.col("_share_base_y") / pl.col("_base_sum_y").clip(lower_bound=eps))
            .alias("v12_store_share_y"),
            pl.when(pl.col("_weight_sum_v") > eps)
            .then(pl.col("_alloc_weight_v") / pl.col("_weight_sum_v"))
            .when(pl.col("v12_occurrence_gate_open_value") & (pl.col("_open_count_v") > 0))
            .then(1.0 / pl.col("_open_count_v").cast(pl.Float64))
            .otherwise(pl.col("_share_base_v") / pl.col("_base_sum_v").clip(lower_bound=eps))
            .alias("v12_store_share_value"),
        )
        .select(
            "unique_id", "_v12_sku", "ds",
            "v12_occurrence_prob_y", "v12_occurrence_prob_value", "v12_occurrence_prob_joint",
            "v12_occurrence_gate_open_y", "v12_occurrence_gate_open_value", "v12_occurrence_gate_open_joint",
            "v12_store_share_y", "v12_store_share_value",
        )
    )

    out = (
        leaf_grid.join(sku_forecast, on=["_v12_sku", "ds"], how="left")
        .with_columns(
            (pl.col("v12_sku_forecast_y") * pl.col("v12_store_share_y"))
            .clip(lower_bound=0.0)
            .alias("v12_candidate_yhat_raw"),
            (pl.col("v12_sku_forecast_value") * pl.col("v12_store_share_value"))
            .clip(lower_bound=0.0)
            .alias("v12_candidate_valuehat_raw"),
            pl.col("_v12_sku_available_y").fill_null(False).alias("v12_candidate_available_y"),
            pl.col("_v12_sku_available_v").fill_null(False).alias("v12_candidate_available_value"),
        )
        .select(
            "unique_id", "_v12_sku", "ds",
            "v12_candidate_yhat_raw", "v12_candidate_valuehat_raw",
            "v12_sku_forecast_y", "v12_sku_forecast_value",
            "v12_sku_forecast_uncalibrated_y", "v12_sku_forecast_uncalibrated_value",
            "v12_sku_forecast_calibrated_challenger_y", "v12_sku_forecast_calibrated_challenger_value",
            "v12_sku_level_calibration_applied_y", "v12_sku_level_calibration_applied_value",
            "v12_sku_forecast_base_y", "v12_sku_forecast_base_value",
            "v12_sku_level_calibration_raw_ratio_y", "v12_sku_level_calibration_raw_ratio_value",
            "v12_sku_level_calibration_factor_y", "v12_sku_level_calibration_factor_value",
            "v12_sku_level_calibration_blocks_y", "v12_sku_level_calibration_blocks_value",
            "v12_sku_shape_model_y", "v12_sku_shape_model_value",
            "v12_sku_shape_training_rows_y", "v12_sku_shape_training_rows_value",
            "v12_sku_shape_applied_y", "v12_sku_shape_applied_value",
            "v12_sku_model_y", "v12_sku_model_value",
            "v12_sku_validation_wmape_y", "v12_sku_validation_wmape_value",
            "v12_store_share_y", "v12_store_share_value",
            "v12_occurrence_prob_y", "v12_occurrence_prob_value", "v12_occurrence_prob_joint",
            "v12_occurrence_gate_open_y", "v12_occurrence_gate_open_value", "v12_occurrence_gate_open_joint",
            "v12_candidate_available_y", "v12_candidate_available_value",
        )
    )
    return out


def _score_closed_block(
    incumbent_rows: pl.DataFrame,
    candidate_rows: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    block_idx: int,
) -> pl.DataFrame:
    """Score one already-closed 28-day block without target leakage.

    v12.7 keeps wMAPE as the primary metric but also exposes a causal utility
    score = wMAPE + lambda*|BIAS|.  The selector can therefore reject a noisy
    wMAPE win that materially worsens systematic under/over-forecasting.
    """
    min_sales = int(getattr(settings, "V12_MIN_VALIDATION_SALES", 7))
    bias_weight = float(getattr(settings, "V12_SELECTOR_BIAS_WEIGHT", 0.25))
    eps = 1e-9
    # v12.9.9: exact value-rescue scoring grid on each closed block.
    # These columns are diagnostic only and let the segmented selector compare
    # multiplicative uplifts without approximating wMAPE from block totals.
    v1299_grid = tuple(float(x) for x in getattr(
        settings, "V1299_VALUE_SPARSE_FACTOR_GRID",
        (1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15),
    ))

    _hist_select = [
        "unique_id", "ds", "y", "value",
        pl.col("yhat").cast(pl.Float64).alias("_v11_yhat"),
        pl.col("valuehat").cast(pl.Float64).alias("_v11_vhat"),
    ]
    # v12.9.6 stress diagnostics need the segment state as it existed in each
    # historical block, rather than today's segment label. These fields are
    # diagnostic only and never participate in the incumbent forecast itself.
    if "leaf_level_method_value" in incumbent_rows.columns:
        _hist_select.append(
            pl.col("leaf_level_method_value").cast(pl.Utf8).alias("_hist_level_method_value")
        )
    else:
        _hist_select.append(pl.lit("unknown").alias("_hist_level_method_value"))
    if "sku_seasonal_multiplier_value" in incumbent_rows.columns:
        _hist_select.append(
            pl.col("sku_seasonal_multiplier_value").cast(pl.Float64).alias("_hist_seasonal_multiplier_value")
        )
    else:
        _hist_select.append(pl.lit(1.0).alias("_hist_seasonal_multiplier_value"))

    base = (
        incumbent_rows.filter(
            (pl.col("ds") >= pl.lit(start)) & (pl.col("ds") <= pl.lit(end))
        )
        .select(*_hist_select)
        .join(
            candidate_rows.select(
                "unique_id", "ds",
                pl.col("v12_candidate_yhat_raw").round(0).alias("v12_candidate_yhat_raw"),
                pl.col("v12_candidate_valuehat_raw").round(2).alias("v12_candidate_valuehat_raw"),
                "v12_candidate_available_y", "v12_candidate_available_value",
            ),
            on=["unique_id", "ds"],
            how="left",
        )
    )
    # Exact positive-day errors for every rescue factor and both baseline families.
    # Keep factor 1.00 on the legacy columns to avoid duplicate storage.
    rescue_exprs: list[pl.Expr] = []
    for factor in v1299_grid:
        if abs(factor - 1.0) < 1e-12:
            continue
        tag = str(int(round(factor * 1000)))
        rescue_exprs.extend([
            pl.when(pl.col("value") > 0)
            .then((pl.col("value") - pl.col("_v11_vhat") * factor).abs())
            .otherwise(0.0).alias(f"_v1299_ae11_v_f{tag}"),
            pl.when(pl.col("value") > 0)
            .then(pl.col("_v11_vhat") * factor - pl.col("value"))
            .otherwise(0.0).alias(f"_v1299_se11_v_f{tag}"),
            pl.when(pl.col("value") > 0)
            .then((pl.col("value") - pl.col("v12_candidate_valuehat_raw") * factor).abs())
            .otherwise(0.0).alias(f"_v1299_ae12_v_f{tag}"),
            pl.when(pl.col("value") > 0)
            .then(pl.col("v12_candidate_valuehat_raw") * factor - pl.col("value"))
            .otherwise(0.0).alias(f"_v1299_se12_v_f{tag}"),
        ])
    if rescue_exprs:
        base = base.with_columns(*rescue_exprs)

    return (
        base.group_by("unique_id")
        .agg(
            pl.when(pl.col("y") > 0).then(pl.col("y")).otherwise(0.0).sum().alias("_den_y"),
            pl.when(pl.col("value") > 0).then(pl.col("value")).otherwise(0.0).sum().alias("_den_v"),
            (pl.col("y") > 0).sum().cast(pl.Int32).alias("_n_y"),
            (pl.col("value") > 0).sum().cast(pl.Int32).alias("_n_v"),
            pl.when(pl.col("y") > 0).then((pl.col("y") - pl.col("_v11_yhat")).abs()).otherwise(0.0).sum().alias("_ae11_y"),
            pl.when(pl.col("y") > 0).then(pl.col("_v11_yhat") - pl.col("y")).otherwise(0.0).sum().alias("_se11_y"),
            pl.when(pl.col("y") > 0).then((pl.col("y") - pl.col("v12_candidate_yhat_raw")).abs()).otherwise(0.0).sum().alias("_ae12_y"),
            pl.when(pl.col("y") > 0).then(pl.col("v12_candidate_yhat_raw") - pl.col("y")).otherwise(0.0).sum().alias("_se12_y"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("_v11_vhat")).abs()).otherwise(0.0).sum().alias("_ae11_v"),
            pl.when(pl.col("value") > 0).then(pl.col("_v11_vhat") - pl.col("value")).otherwise(0.0).sum().alias("_se11_v"),
            pl.when(pl.col("value") > 0).then((pl.col("value") - pl.col("v12_candidate_valuehat_raw")).abs()).otherwise(0.0).sum().alias("_ae12_v"),
            pl.when(pl.col("value") > 0).then(pl.col("v12_candidate_valuehat_raw") - pl.col("value")).otherwise(0.0).sum().alias("_se12_v"),
            *[
                pl.col(f"_v1299_{kind}{family}_v_f{str(int(round(factor * 1000)))}").sum()
                .alias(f"_v1299_{kind}{family}_v_f{str(int(round(factor * 1000)))}")
                for factor in v1299_grid if abs(factor - 1.0) >= 1e-12
                for family in ("11", "12")
                for kind in ("ae", "se")
            ],
            pl.col("v12_candidate_available_y").fill_null(False).all().alias("_available_y"),
            pl.col("v12_candidate_available_value").fill_null(False).all().alias("_available_v"),
            pl.col("_hist_level_method_value").drop_nulls().first().fill_null("unknown").alias("_hist_level_method_value"),
            pl.col("_hist_seasonal_multiplier_value").drop_nulls().median().fill_null(1.0).alias("_hist_seasonal_multiplier_value"),
        )
        .with_columns(
            pl.when(pl.col("_den_y") > eps).then(pl.col("_ae11_y") / pl.col("_den_y")).otherwise(None).alias("_wmape11_y"),
            pl.when(pl.col("_den_y") > eps).then(pl.col("_ae12_y") / pl.col("_den_y")).otherwise(None).alias("_wmape12_y"),
            pl.when(pl.col("_den_y") > eps).then(pl.col("_se11_y") / pl.col("_den_y")).otherwise(None).alias("_bias11_y"),
            pl.when(pl.col("_den_y") > eps).then(pl.col("_se12_y") / pl.col("_den_y")).otherwise(None).alias("_bias12_y"),
            pl.when(pl.col("_den_v") > eps).then(pl.col("_ae11_v") / pl.col("_den_v")).otherwise(None).alias("_wmape11_v"),
            pl.when(pl.col("_den_v") > eps).then(pl.col("_ae12_v") / pl.col("_den_v")).otherwise(None).alias("_wmape12_v"),
            pl.when(pl.col("_den_v") > eps).then(pl.col("_se11_v") / pl.col("_den_v")).otherwise(None).alias("_bias11_v"),
            pl.when(pl.col("_den_v") > eps).then(pl.col("_se12_v") / pl.col("_den_v")).otherwise(None).alias("_bias12_v"),
        )
        .with_columns(
            (pl.col("_wmape11_y") + bias_weight * pl.col("_bias11_y").abs()).alias("_score11_y"),
            (pl.col("_wmape12_y") + bias_weight * pl.col("_bias12_y").abs()).alias("_score12_y"),
            (pl.col("_wmape11_v") + bias_weight * pl.col("_bias11_v").abs()).alias("_score11_v"),
            (pl.col("_wmape12_v") + bias_weight * pl.col("_bias12_v").abs()).alias("_score12_v"),
        )
        .with_columns(
            (pl.col("_available_y") & (pl.col("_n_y") >= min_sales) & pl.col("_wmape12_y").is_not_null()).alias("_eligible_y"),
            (pl.col("_available_v") & (pl.col("_n_v") >= min_sales) & pl.col("_wmape12_v").is_not_null()).alias("_eligible_v"),
            (pl.col("_wmape11_y") - pl.col("_wmape12_y")).alias("_improvement_y"),
            (pl.col("_wmape11_v") - pl.col("_wmape12_v")).alias("_improvement_v"),
            (pl.col("_score11_y") - pl.col("_score12_y")).alias("_utility_improvement_y"),
            (pl.col("_score11_v") - pl.col("_score12_v")).alias("_utility_improvement_v"),
            pl.lit(int(block_idx)).cast(pl.Int8).alias("_validation_block"),
        )
    )



_META_FEATURE_NAMES = (
    "meta_recent_gain",
    "meta_previous_gain",
    "meta_weighted_gain",
    "meta_weighted_utility",
    "meta_wmape11",
    "meta_wmape12",
    "meta_bias11",
    "meta_bias12",
    "meta_gain_range",
    "meta_win_rate",
    "meta_sales_coverage",
    "meta_log_volume",
)


def _meta_feature_frame(
    block_scores: pl.DataFrame,
    suffix: str,
    feature_blocks: tuple[int, ...],
) -> pl.DataFrame:
    """Build causal recency features from already-closed blocks only."""
    configured = tuple(float(x) for x in getattr(
        settings, "V12_META_SELECTOR_RECENCY_WEIGHTS", (0.55, 0.30, 0.15)
    ))
    if not configured or any(w < 0.0 for w in configured):
        raise ValueError("V12_META_SELECTOR_RECENCY_WEIGHTS inválido")
    weights = tuple(configured[i] if i < len(configured) else 0.0 for i in range(len(feature_blocks)))
    if sum(weights) <= 0.0:
        weights = tuple(1.0 for _ in feature_blocks)

    elig = f"_eligible_{suffix}"
    ncol = "_n_y" if suffix == "y" else "_n_v"
    dencol = "_den_y" if suffix == "y" else "_den_v"
    cols = {
        "imp": f"_improvement_{suffix}",
        "utility": f"_utility_improvement_{suffix}",
        "w11": f"_wmape11_{suffix}",
        "w12": f"_wmape12_{suffix}",
        "b11": f"_bias11_{suffix}",
        "b12": f"_bias12_{suffix}",
        "n": ncol,
        "den": dencol,
    }
    aggs: list[pl.Expr] = []
    for pos, block in enumerate(feature_blocks):
        for short, col in cols.items():
            aggs.append(
                pl.when((pl.col("_validation_block") == block) & pl.col(elig))
                .then(pl.col(col))
                .otherwise(None)
                .max()
                .alias(f"_mb{pos}_{short}")
            )
    out = block_scores.group_by("unique_id").agg(*aggs)

    present = [pl.col(f"_mb{i}_imp").is_not_null().cast(pl.Float64) for i in range(len(feature_blocks))]
    weight_den = sum(
        (pl.lit(weights[i]) * present[i] for i in range(len(feature_blocks))),
        pl.lit(0.0),
    )

    def weighted(short: str) -> pl.Expr:
        num = sum(
            (
                pl.lit(weights[i])
                * pl.col(f"_mb{i}_{short}").fill_null(0.0).cast(pl.Float64)
                * present[i]
                for i in range(len(feature_blocks))
            ),
            pl.lit(0.0),
        )
        return pl.when(weight_den > 0).then(num / weight_den).otherwise(None)

    imp_filled = [pl.col(f"_mb{i}_imp").fill_null(0.0).cast(pl.Float64) for i in range(len(feature_blocks))]
    wins = sum(
        (pl.col(f"_mb{i}_imp").fill_null(-999.0).gt(0.0).cast(pl.Float64) for i in range(len(feature_blocks))),
        pl.lit(0.0),
    )
    count = sum(present, pl.lit(0.0))
    out = out.with_columns(
        count.cast(pl.Int8).alias("meta_feature_blocks"),
        pl.col("_mb0_imp").fill_null(0.0).alias("meta_recent_gain"),
        (pl.col("_mb1_imp").fill_null(0.0) if len(feature_blocks) > 1 else pl.lit(0.0)).alias("meta_previous_gain"),
        weighted("imp").fill_null(0.0).alias("meta_weighted_gain"),
        weighted("utility").fill_null(0.0).alias("meta_weighted_utility"),
        weighted("w11").fill_null(0.0).alias("meta_wmape11"),
        weighted("w12").fill_null(0.0).alias("meta_wmape12"),
        weighted("b11").fill_null(0.0).alias("meta_bias11"),
        weighted("b12").fill_null(0.0).alias("meta_bias12"),
        (pl.max_horizontal(*imp_filled) - pl.min_horizontal(*imp_filled)).alias("meta_gain_range"),
        pl.when(count > 0).then(wins / count).otherwise(0.0).alias("meta_win_rate"),
        (weighted("n").fill_null(0.0) / float(getattr(settings, "RLS_BLOCK_DAYS", 28))).clip(0.0, 1.0).alias("meta_sales_coverage"),
        (weighted("den").fill_null(0.0).clip(lower_bound=0.0) + 1.0).log().alias("meta_log_volume"),
    )
    return out.select("unique_id", "meta_feature_blocks", *_META_FEATURE_NAMES)


def _fit_meta_model(
    block_scores: pl.DataFrame,
    suffix: str,
    *,
    label_block: int,
    train_feature_blocks: tuple[int, ...],
    score_feature_blocks: tuple[int, ...],
) -> pl.DataFrame:
    """Fit one temporally shifted pooled L2 logistic meta-model.

    ``label_block`` is predicted only from strictly older ``train_feature_blocks``.
    The fitted model is then applied to ``score_feature_blocks``.  This helper is
    used twice in v12.8.2: an older fit calibrates threshold/portfolio mode on
    the latest CLOSED block, and the production fit is shifted one block forward
    to score the actual target horizon.
    """
    min_feature_blocks = int(getattr(settings, "V12_META_SELECTOR_MIN_FEATURE_BLOCKS", 2))
    min_train_rows = int(getattr(settings, "V12_META_SELECTOR_MIN_TRAIN_ROWS", 500))
    l2 = float(getattr(settings, "V12_META_SELECTOR_L2", 1.0))
    max_iter = int(getattr(settings, "V12_META_SELECTOR_MAX_ITER", 30))
    eps = 1e-9

    required_train_blocks = min(min_feature_blocks, max(1, len(train_feature_blocks)))
    required_score_blocks = min(min_feature_blocks, max(1, len(score_feature_blocks)))
    train_features = _meta_feature_frame(block_scores, suffix, train_feature_blocks)
    score_features = _meta_feature_frame(block_scores, suffix, score_feature_blocks)
    elig = f"_eligible_{suffix}"
    label = (
        block_scores.filter((pl.col("_validation_block") == int(label_block)) & pl.col(elig))
        .select(
            "unique_id",
            (pl.col(f"_wmape12_{suffix}") < pl.col(f"_wmape11_{suffix}"))
            .cast(pl.Int8)
            .alias("_meta_label"),
        )
        .unique(subset=["unique_id"])
    )
    train = (
        train_features.join(label, on="unique_id", how="inner")
        .filter(pl.col("meta_feature_blocks") >= required_train_blocks)
    )
    score = score_features.filter(pl.col("meta_feature_blocks") >= required_score_blocks)

    def _empty_result(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(
            "unique_id",
            "meta_recent_gain",
            "meta_weighted_gain",
            "meta_win_rate",
            "meta_bias11",
            "meta_bias12",
            "meta_weighted_utility",
        ).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_meta_probability"),
            pl.lit(False).alias("_meta_model_available"),
            pl.lit(int(train.height)).cast(pl.Int32).alias("_meta_training_rows"),
            pl.lit(None, dtype=pl.Utf8).alias("_meta_top_driver"),
        )

    if train.height < min_train_rows or score.height == 0:
        return _empty_result(score)

    X = train.select(*_META_FEATURE_NAMES).to_numpy().astype(np.float64, copy=False)
    y = train.get_column("_meta_label").to_numpy().astype(np.float64, copy=False)
    Xp = score.select(*_META_FEATURE_NAMES).to_numpy().astype(np.float64, copy=False)
    if np.unique(y).size < 2:
        return _empty_result(score)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    Xs = (X - mean) / std
    Xps = (Xp - mean) / std
    Xd = np.column_stack([np.ones(Xs.shape[0]), Xs])
    Xpd = np.column_stack([np.ones(Xps.shape[0]), Xps])

    pos = float(y.mean())
    sample_w = np.where(y > 0.5, 0.5 / max(pos, eps), 0.5 / max(1.0 - pos, eps))
    beta = np.zeros(Xd.shape[1], dtype=np.float64)
    reg = np.eye(Xd.shape[1], dtype=np.float64) * max(l2, 0.0)
    reg[0, 0] = 0.0
    for _ in range(max(1, max_iter)):
        z = np.clip(Xd @ beta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        curvature = np.maximum(prob * (1.0 - prob) * sample_w, 1e-8)
        grad = Xd.T @ ((prob - y) * sample_w) + reg @ beta
        hess = Xd.T @ (Xd * curvature[:, None]) + reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta -= step
        if float(np.max(np.abs(step))) < 1e-6:
            break

    pcur = 1.0 / (1.0 + np.exp(-np.clip(Xpd @ beta, -30.0, 30.0)))
    contrib = Xps * beta[1:]
    top_idx = np.argmax(np.abs(contrib), axis=1)
    top_driver = [str(_META_FEATURE_NAMES[int(i)]) for i in top_idx]
    return score.select(
        "unique_id",
        "meta_recent_gain",
        "meta_weighted_gain",
        "meta_win_rate",
        "meta_bias11",
        "meta_bias12",
        "meta_weighted_utility",
    ).with_columns(
        pl.Series("_meta_probability", pcur.tolist(), dtype=pl.Float64),
        pl.lit(True).alias("_meta_model_available"),
        pl.lit(int(train.height)).cast(pl.Int32).alias("_meta_training_rows"),
        pl.Series("_meta_top_driver", top_driver, dtype=pl.Utf8),
    )


def _fit_meta_selector(block_scores: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Production v12.8.2 meta-selector, shifted one block ahead of target."""
    out_suffix = "y" if suffix == "y" else "value"
    current = _fit_meta_model(
        block_scores,
        suffix,
        label_block=1,
        train_feature_blocks=(2, 3, 4),
        score_feature_blocks=(1, 2, 3),
    )
    return current.select(
        "unique_id",
        pl.col("meta_recent_gain").alias(f"v12_meta_recent_gain_{out_suffix}"),
        pl.col("meta_weighted_gain").alias(f"v12_meta_weighted_gain_{out_suffix}"),
        pl.col("meta_win_rate").alias(f"v12_meta_win_rate_{out_suffix}"),
        pl.col("meta_bias11").alias(f"v12_meta_bias11_{out_suffix}"),
        pl.col("meta_bias12").alias(f"v12_meta_bias12_{out_suffix}"),
        pl.col("_meta_probability").alias(f"v12_meta_probability_{out_suffix}"),
        pl.col("_meta_model_available").alias(f"v12_meta_model_available_{out_suffix}"),
        pl.col("_meta_training_rows").alias(f"v12_meta_training_rows_{out_suffix}"),
        pl.col("_meta_top_driver").alias(f"v12_meta_top_driver_{out_suffix}"),
    )



def _fit_value_expected_gain_selector(block_scores: pl.DataFrame) -> pl.DataFrame:
    """v12.9.2 calibrated expected-impact selector for Value ($).

    The production ridge score is trained on closed pseudo-OOS leaf examples,
    weighted by the official bottom-up denominator.  Its magnitude is *not*
    trusted directly: a strictly temporal out-of-fold layer maps raw scores to
    conservative realized utility-gain buckets.  Metadata also exposes leaf and
    bucket downside risk so the portfolio stage can apply asymmetric guards.
    """
    enabled = bool(getattr(settings, "V1292_VALUE_EXPECTED_IMPACT_ENABLED", True))
    out_cols = [
        "unique_id", "v1291_expected_gain_value", "v1291_expected_gain_available_value",
        "v1291_expected_gain_training_rows_value", "v1291_expected_gain_weight_value",
        "v1291_expected_gain_resid_rmse_value", "v1291_expected_gain_confidence_value",
        "v1291_expected_gain_recent_value", "v1291_expected_gain_range_value",
        "v1291_expected_gain_top_driver_value",
        "v1292_raw_expected_gain_value", "v1292_calibrated_gain_value",
        "v1292_calibration_bucket_value", "v1292_calibration_rows_value",
        "v1292_bucket_loss_rate_value", "v1292_bucket_p10_gain_value",
        "v1292_leaf_loss_rate_value", "v1292_leaf_p10_gain_value",
        "v1292_impact_percentile_value", "v1292_risk_score_value",
    ]
    feature_n = max(2, int(getattr(settings, "V1291_VALUE_EXPECTED_GAIN_FEATURE_BLOCKS", 3)))
    max_block_raw = block_scores.select(pl.col("_validation_block").max()).item()
    max_block = int(max_block_raw or 0)
    score_blocks = tuple(range(1, min(max_block, feature_n) + 1))
    score = _meta_feature_frame(block_scores, "v", score_blocks)

    def empty(frame: pl.DataFrame, n_train: int = 0) -> pl.DataFrame:
        return frame.select("unique_id").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("v1291_expected_gain_value"),
            pl.lit(False).alias("v1291_expected_gain_available_value"),
            pl.lit(int(n_train), dtype=pl.Int32).alias("v1291_expected_gain_training_rows_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1291_expected_gain_weight_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1291_expected_gain_resid_rmse_value"),
            pl.lit(0.0, dtype=pl.Float64).alias("v1291_expected_gain_confidence_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1291_expected_gain_recent_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1291_expected_gain_range_value"),
            pl.lit(None, dtype=pl.Utf8).alias("v1291_expected_gain_top_driver_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_raw_expected_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_calibrated_gain_value"),
            pl.lit(None, dtype=pl.Int8).alias("v1292_calibration_bucket_value"),
            pl.lit(0, dtype=pl.Int32).alias("v1292_calibration_rows_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_bucket_loss_rate_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_bucket_p10_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_leaf_loss_rate_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_leaf_p10_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_impact_percentile_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_risk_score_value"),
        ).select(*out_cols)

    if not enabled or len(score_blocks) < 2:
        return empty(score)

    label_blocks = tuple(int(x) for x in getattr(
        settings, "V1291_VALUE_EXPECTED_GAIN_LABEL_BLOCKS", (1, 2, 3)
    ))
    train_parts: list[pl.DataFrame] = []
    for label_block in label_blocks:
        feature_blocks = tuple(
            b for b in range(label_block + 1, label_block + feature_n + 1)
            if b <= max_block
        )
        if len(feature_blocks) < 2:
            continue
        feat = _meta_feature_frame(block_scores, "v", feature_blocks)
        lab = (
            block_scores.filter(
                (pl.col("_validation_block") == label_block) & pl.col("_eligible_v")
            )
            .select(
                "unique_id",
                pl.col("_utility_improvement_v").cast(pl.Float64).alias("_eg_target"),
                pl.col("_den_v").cast(pl.Float64).alias("_eg_weight"),
            )
            .filter(pl.col("_eg_target").is_finite() & (pl.col("_eg_weight") > 0))
        )
        part = (
            feat.join(lab, on="unique_id", how="inner")
            .filter(pl.col("meta_feature_blocks") >= min(2, len(feature_blocks)))
            .with_columns(pl.lit(label_block, dtype=pl.Int8).alias("_eg_label_block"))
        )
        if part.height:
            train_parts.append(part)
    if not train_parts:
        return empty(score)
    train = pl.concat(train_parts, how="vertical_relaxed")
    min_rows = int(getattr(settings, "V1291_VALUE_EXPECTED_GAIN_MIN_TRAIN_ROWS", 1000))
    if train.height < min_rows or score.height == 0:
        return empty(score, train.height)

    l2 = max(0.0, float(getattr(settings, "V1291_VALUE_EXPECTED_GAIN_L2", 2.0)))

    def fit_predict(fit_df: pl.DataFrame, pred_df: pl.DataFrame):
        X = fit_df.select(*_META_FEATURE_NAMES).to_numpy().astype(np.float64, copy=False)
        y = fit_df.get_column("_eg_target").to_numpy().astype(np.float64, copy=False)
        w = fit_df.get_column("_eg_weight").to_numpy().astype(np.float64, copy=False)
        Xp = pred_df.select(*_META_FEATURE_NAMES).to_numpy().astype(np.float64, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        pos = w > 0
        if int(pos.sum()) < max(50, min_rows // 4):
            return None
        X, y, w = X[pos], y[pos], w[pos]
        w = w / max(float(w.mean()), 1e-12)
        mean = np.average(X, axis=0, weights=w)
        var = np.average((X - mean) ** 2, axis=0, weights=w)
        std = np.sqrt(np.maximum(var, 1e-12))
        std = np.where(std > 1e-8, std, 1.0)
        Xs = (X - mean) / std
        Xps = (Xp - mean) / std
        Xd = np.column_stack([np.ones(Xs.shape[0]), Xs])
        Xpd = np.column_stack([np.ones(Xps.shape[0]), Xps])
        sw = np.sqrt(w)
        Xw = Xd * sw[:, None]
        yw = y * sw
        reg = np.eye(Xd.shape[1], dtype=np.float64) * l2
        reg[0, 0] = 0.0
        try:
            beta = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Xw.T @ Xw + reg, Xw.T @ yw, rcond=None)[0]
        pred_fit = Xd @ beta
        rmse = float(np.sqrt(np.average((y - pred_fit) ** 2, weights=w)))
        pred = Xpd @ beta
        contrib = Xps * beta[1:]
        top_idx = np.argmax(np.abs(contrib), axis=1)
        top = [str(_META_FEATURE_NAMES[int(i)]) for i in top_idx]
        return pred, rmse, top

    production = fit_predict(train, score)
    if production is None:
        return empty(score, train.height)
    raw_pred, raw_rmse, top_driver = production

    # Strictly temporal OOF calibration: block 1 may learn from labels 2/3;
    # block 2 may learn from label 3. No newer label is ever used to score an
    # older pseudo-OOS fold, and the actual target horizon is never referenced.
    oof_raw: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    oof_w: list[np.ndarray] = []
    for label_block in sorted(set(label_blocks)):
        fit_df = train.filter(pl.col("_eg_label_block") > label_block)
        valid_df = train.filter(pl.col("_eg_label_block") == label_block)
        if fit_df.height < max(250, min_rows // 2) or valid_df.height == 0:
            continue
        fit = fit_predict(fit_df, valid_df)
        if fit is None:
            continue
        oof_raw.append(np.asarray(fit[0], dtype=np.float64))
        oof_y.append(valid_df.get_column("_eg_target").to_numpy().astype(np.float64, copy=False))
        oof_w.append(valid_df.get_column("_eg_weight").to_numpy().astype(np.float64, copy=False))

    if not oof_raw:
        return empty(score, train.height)
    raw_oof = np.concatenate(oof_raw)
    y_oof = np.concatenate(oof_y)
    w_oof = np.maximum(np.concatenate(oof_w), 0.0)
    ok = np.isfinite(raw_oof) & np.isfinite(y_oof) & np.isfinite(w_oof) & (w_oof > 0)
    raw_oof, y_oof, w_oof = raw_oof[ok], y_oof[ok], w_oof[ok]
    min_cal_rows = int(getattr(settings, "V1292_VALUE_MIN_CALIBRATION_ROWS", 1000))
    if raw_oof.size < min_cal_rows:
        return empty(score, train.height)

    quantiles = np.quantile(raw_oof, [0.2, 0.4, 0.6, 0.8])
    # searchsorted handles duplicate edges safely; buckets remain 0..4.
    oof_bucket = np.searchsorted(quantiles, raw_oof, side="right")
    cur_bucket = np.searchsorted(quantiles, raw_pred, side="right")
    bucket_gain = np.zeros(5, dtype=np.float64)
    bucket_rmse = np.full(5, max(raw_rmse, 1e-6), dtype=np.float64)
    bucket_loss = np.ones(5, dtype=np.float64)
    bucket_p10 = np.full(5, -1.0, dtype=np.float64)
    bucket_rows = np.zeros(5, dtype=np.int64)
    for b in range(5):
        mask = oof_bucket == b
        if not np.any(mask):
            continue
        yy, ww = y_oof[mask], w_oof[mask]
        bucket_rows[b] = int(mask.sum())
        mean_gain = float(np.average(yy, weights=ww))
        median_gain = float(np.median(yy))
        # Conservative magnitude: a positive bucket must be positive in both
        # weighted mean and median; otherwise it is not granted positive gain.
        if mean_gain > 0.0 and median_gain > 0.0:
            cal = min(mean_gain, median_gain)
        elif mean_gain < 0.0 and median_gain < 0.0:
            cal = max(mean_gain, median_gain)
        else:
            cal = 0.0
        bucket_gain[b] = cal
        bucket_rmse[b] = float(np.sqrt(np.average((yy - cal) ** 2, weights=ww)))
        bucket_loss[b] = float(np.average((yy <= 0.0).astype(np.float64), weights=ww))
        bucket_p10[b] = float(np.quantile(yy, 0.10))

    calibrated = bucket_gain[cur_bucket]
    cal_rmse = bucket_rmse[cur_bucket]
    cal_loss = bucket_loss[cur_bucket]
    cal_p10 = bucket_p10[cur_bucket]
    cal_rows = bucket_rows[cur_bucket]

    # Leaf-specific downside from the latest closed blocks complements pooled
    # bucket risk. It is descriptive of already-known history only.
    risk = (
        block_scores.filter(
            pl.col("_validation_block").is_in(list(score_blocks)) & pl.col("_eligible_v")
        )
        .group_by("unique_id")
        .agg(
            (pl.col("_improvement_v") <= 0.0).cast(pl.Float64).mean().alias("_leaf_loss_rate"),
            pl.col("_improvement_v").quantile(0.10, interpolation="nearest").alias("_leaf_p10"),
        )
    )
    score_risk = score.select("unique_id").join(risk, on="unique_id", how="left")
    leaf_loss = score_risk.get_column("_leaf_loss_rate").fill_null(1.0).to_numpy().astype(np.float64)
    leaf_p10 = score_risk.get_column("_leaf_p10").fill_null(-1.0).to_numpy().astype(np.float64)

    gain_range = score.get_column("meta_gain_range").to_numpy().astype(np.float64, copy=False)
    recent = score.get_column("meta_recent_gain").to_numpy().astype(np.float64, copy=False)
    volume_weight = np.exp(np.clip(score.get_column("meta_log_volume").to_numpy(), -20.0, 30.0)) - 1.0
    # True probability-like confidence: bounded [0,1], reduced by both pooled
    # calibration error and historical loss frequency.
    positive_gain = np.maximum(calibrated, 0.0)
    confidence = positive_gain / (positive_gain + np.maximum(cal_rmse, 1e-6))
    confidence *= np.clip(1.0 - 0.5 * cal_loss - 0.5 * leaf_loss, 0.0, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)

    order = np.argsort(-volume_weight, kind="stable")
    pct = np.empty(score.height, dtype=np.float64)
    # v12.9.3 semantic contract: 1.0 = highest-impact leaf, values near 0
    # = lowest impact.  v12.9.2 exposed the inverse rank, which made risk
    # guards difficult to reason about.  The ridge layer is diagnostic only
    # from v12.9.3 onward, but its metadata remains unambiguous.
    n_pct = max(float(score.height), 1.0)
    pct[order] = (n_pct - np.arange(score.height, dtype=np.float64)) / n_pct
    downside = np.maximum(-np.minimum(cal_p10, leaf_p10), 0.0)
    risk_score = calibrated * confidence / (1.0 + cal_loss + leaf_loss + 5.0 * downside)

    global_cal_rmse = float(np.sqrt(np.average((y_oof - bucket_gain[oof_bucket]) ** 2, weights=w_oof)))
    return score.select("unique_id").with_columns(
        pl.Series("v1291_expected_gain_value", calibrated.tolist(), dtype=pl.Float64),
        pl.lit(True).alias("v1291_expected_gain_available_value"),
        pl.lit(int(train.height), dtype=pl.Int32).alias("v1291_expected_gain_training_rows_value"),
        pl.Series("v1291_expected_gain_weight_value", volume_weight.tolist(), dtype=pl.Float64),
        pl.lit(global_cal_rmse, dtype=pl.Float64).alias("v1291_expected_gain_resid_rmse_value"),
        pl.Series("v1291_expected_gain_confidence_value", confidence.tolist(), dtype=pl.Float64),
        pl.Series("v1291_expected_gain_recent_value", recent.tolist(), dtype=pl.Float64),
        pl.Series("v1291_expected_gain_range_value", gain_range.tolist(), dtype=pl.Float64),
        pl.Series("v1291_expected_gain_top_driver_value", top_driver, dtype=pl.Utf8),
        pl.Series("v1292_raw_expected_gain_value", raw_pred.tolist(), dtype=pl.Float64),
        pl.Series("v1292_calibrated_gain_value", calibrated.tolist(), dtype=pl.Float64),
        pl.Series("v1292_calibration_bucket_value", (cur_bucket + 1).tolist(), dtype=pl.Int8),
        pl.Series("v1292_calibration_rows_value", cal_rows.tolist(), dtype=pl.Int32),
        pl.Series("v1292_bucket_loss_rate_value", cal_loss.tolist(), dtype=pl.Float64),
        pl.Series("v1292_bucket_p10_gain_value", cal_p10.tolist(), dtype=pl.Float64),
        pl.Series("v1292_leaf_loss_rate_value", leaf_loss.tolist(), dtype=pl.Float64),
        pl.Series("v1292_leaf_p10_gain_value", leaf_p10.tolist(), dtype=pl.Float64),
        pl.Series("v1292_impact_percentile_value", pct.tolist(), dtype=pl.Float64),
        pl.Series("v1292_risk_score_value", risk_score.tolist(), dtype=pl.Float64),
    ).select(*out_cols)

def _calibrate_meta_policy(block_scores: pl.DataFrame, suffix: str) -> dict[str, object]:
    """Choose threshold and section/target portfolio mode with no target leakage.

    An *older* meta-model predicts block 1 from blocks 2/3/4 after learning the
    block-2 winner only from blocks 3/4.  Block 1 is therefore a genuinely
    forward validation block for policy tuning.  Its actuals may choose among
    ``v11_all``, ``v12_all`` and ``meta_leaf`` because block 1 is already closed
    before the target origin.
    """
    default_threshold = float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))
    adaptive = bool(getattr(settings, "V12_META_SELECTOR_ADAPTIVE_POLICY", True))
    if not adaptive:
        return {
            "mode": "meta_leaf", "threshold": default_threshold,
            "available": False, "utility_gain": None,
            "wmape_v11": None, "wmape_selected": None,
            "bias_v11": None, "bias_selected": None,
        }

    thresholds = tuple(float(x) for x in getattr(
        settings, "V12_META_SELECTOR_THRESHOLDS", (0.45, 0.50, 0.55, 0.60, 0.65)
    ))
    thresholds = tuple(sorted({min(max(x, 0.01), 0.99) for x in thresholds})) or (default_threshold,)
    bias_weight = float(getattr(settings, "V12_SELECTOR_BIAS_WEIGHT", 0.25))
    leaf_bias_tol = float(getattr(settings, "V12_META_SELECTOR_BIAS_GUARD_TOLERANCE", 0.03))
    portfolio_bias_tol = float(getattr(settings, "V12_META_PORTFOLIO_BIAS_GUARD_TOLERANCE", 0.03))
    min_gain = float(getattr(settings, "V12_META_PORTFOLIO_MIN_UTILITY_GAIN", 0.0025))
    max_recent_degradation = float(getattr(settings, "V12_META_SELECTOR_MAX_RECENT_DEGRADATION", 0.08))
    elig = f"_eligible_{suffix}"
    den_col = f"_den_{suffix}"
    ae11_col = f"_ae11_{suffix}"
    ae12_col = f"_ae12_{suffix}"
    se11_col = f"_se11_{suffix}"
    se12_col = f"_se12_{suffix}"

    older = _fit_meta_model(
        block_scores,
        suffix,
        label_block=2,
        train_feature_blocks=(3, 4),
        score_feature_blocks=(2, 3, 4),
    )
    closed = (
        block_scores.filter((pl.col("_validation_block") == 1) & pl.col(elig))
        .select("unique_id", den_col, ae11_col, ae12_col, se11_col, se12_col)
        .join(older, on="unique_id", how="left")
    )
    if closed.height == 0 or not bool(closed.select(pl.col("_meta_model_available").fill_null(False).any()).item()):
        return {
            "mode": "meta_leaf", "threshold": default_threshold,
            "available": False, "utility_gain": None,
            "wmape_v11": None, "wmape_selected": None,
            "bias_v11": None, "bias_selected": None,
        }

    def evaluate(select_expr: pl.Expr) -> dict[str, float | None]:
        row = closed.select(
            pl.col(den_col).sum().alias("den"),
            pl.when(select_expr).then(pl.col(ae12_col)).otherwise(pl.col(ae11_col)).sum().alias("ae"),
            pl.when(select_expr).then(pl.col(se12_col)).otherwise(pl.col(se11_col)).sum().alias("se"),
        ).row(0, named=True)
        den = float(row["den"] or 0.0)
        if den <= 1e-9:
            return {"wmape": None, "bias": None, "utility": None}
        wmape = float(row["ae"] or 0.0) / den
        bias = float(row["se"] or 0.0) / den
        return {"wmape": wmape, "bias": bias, "utility": wmape + bias_weight * abs(bias)}

    v11 = evaluate(pl.lit(False))
    if v11["utility"] is None:
        return {
            "mode": "meta_leaf", "threshold": default_threshold,
            "available": False, "utility_gain": None,
            "wmape_v11": None, "wmape_selected": None,
            "bias_v11": None, "bias_selected": None,
        }

    candidates: list[tuple[str, float, dict[str, float | None]]] = [
        ("v11_all", default_threshold, v11),
        ("v12_all", default_threshold, evaluate(pl.lit(True))),
    ]
    bias_guard = (
        pl.col("meta_bias12").abs()
        <= pl.col("meta_bias11").abs() + leaf_bias_tol
    ).fill_null(False)
    recent_guard = (
        pl.col("meta_recent_gain").fill_null(-999.0) >= -max_recent_degradation
    )
    available = pl.col("_meta_model_available").fill_null(False)
    for threshold in thresholds:
        selected = (
            available
            & (pl.col("_meta_probability").fill_null(0.0) >= float(threshold))
            & recent_guard
            & bias_guard
        )
        candidates.append(("meta_leaf", float(threshold), evaluate(selected)))

    base_bias = abs(float(v11["bias"] or 0.0))
    valid: list[tuple[str, float, dict[str, float | None]]] = []
    for mode, threshold, stats in candidates:
        if stats["utility"] is None or stats["bias"] is None:
            continue
        if mode != "v11_all" and abs(float(stats["bias"])) > base_bias + portfolio_bias_tol:
            continue
        valid.append((mode, threshold, stats))
    if not valid:
        valid = [("v11_all", default_threshold, v11)]

    mode, threshold, best = min(valid, key=lambda item: float(item[2]["utility"] or 1e9))
    utility_gain = float(v11["utility"] or 0.0) - float(best["utility"] or 0.0)
    if mode != "v11_all" and utility_gain < min_gain:
        mode, threshold, best, utility_gain = "v11_all", default_threshold, v11, 0.0
    return {
        "mode": mode,
        "threshold": float(threshold),
        "available": True,
        "utility_gain": float(utility_gain),
        "wmape_v11": float(v11["wmape"]) if v11["wmape"] is not None else None,
        "wmape_selected": float(best["wmape"]) if best["wmape"] is not None else None,
        "bias_v11": float(v11["bias"]) if v11["bias"] is not None else None,
        "bias_selected": float(best["bias"]) if best["bias"] is not None else None,
    }


def _calibrate_value_safety_policy(block_scores: pl.DataFrame) -> dict[str, object]:
    """v12.8.3 causal safety layer for the Value ($) portfolio only.

    Quantity intentionally stays on the v12.8.2 policy.  For Value, an all-
    portfolio switch to v12 is allowed only when the latest CLOSED portfolio
    dominates v11 on wMAPE without materially worsening |BIAS|, the same signal
    is confirmed in both of the two most recent closed blocks, and enough leaves
    pass the historical leaf-level BIAS guard.  The coverage requirement is
    chosen from a configured grid using only the latest closed validation block.

    ``meta_leaf`` is evaluated on that same genuinely forward closed block and
    may replace the safest all-mode only if its utility improvement clears a
    minimum margin.  Missing/insufficient evidence always falls back to v11_all.
    """
    suffix = "v"
    default_threshold = float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))
    enabled = bool(getattr(settings, "V12_VALUE_PORTFOLIO_SAFETY_ENABLED", True))
    if not enabled:
        base = _calibrate_meta_policy(block_scores, suffix)
        return {
            **base,
            "value_safety_enabled": False,
            "value_safety_dominance_pass": None,
            "value_safety_recent_confirmations": 0,
            "value_safety_recent_blocks": 0,
            "value_safety_bias_coverage": None,
            "value_safety_bias_coverage_threshold": None,
            "value_safety_bias_coverage_pass": None,
            "value_safety_meta_margin_pass": None,
            "value_safety_best_all_mode": None,
            "value_safety_reason": "disabled",
        }

    thresholds = tuple(float(x) for x in getattr(
        settings, "V12_META_SELECTOR_THRESHOLDS", (0.45, 0.50, 0.55, 0.60, 0.65)
    ))
    thresholds = tuple(sorted({min(max(x, 0.01), 0.99) for x in thresholds})) or (default_threshold,)
    coverage_grid = tuple(float(x) for x in getattr(
        settings, "V12_VALUE_ALL_BIAS_COVERAGE_GRID", (0.50, 0.55, 0.60, 0.65)
    ))
    coverage_grid = tuple(sorted({min(max(x, 0.0), 1.0) for x in coverage_grid})) or (0.50,)
    min_confirm = max(1, int(getattr(settings, "V12_VALUE_ALL_MIN_RECENT_CONFIRMATIONS", 2)))
    min_closed_leaves = max(1, int(getattr(settings, "V12_VALUE_SAFETY_MIN_CLOSED_LEAVES", 100)))
    meta_margin = float(getattr(settings, "V12_VALUE_META_LEAF_MIN_UTILITY_GAIN", 0.0025))
    bias_weight = float(getattr(settings, "V12_SELECTOR_BIAS_WEIGHT", 0.25))
    leaf_bias_tol = float(getattr(settings, "V12_META_SELECTOR_BIAS_GUARD_TOLERANCE", 0.03))
    portfolio_bias_tol = float(getattr(settings, "V12_META_PORTFOLIO_BIAS_GUARD_TOLERANCE", 0.03))
    min_gain = float(getattr(settings, "V12_META_PORTFOLIO_MIN_UTILITY_GAIN", 0.0025))
    max_recent_degradation = float(getattr(settings, "V12_META_SELECTOR_MAX_RECENT_DEGRADATION", 0.08))

    elig = "_eligible_v"
    den_col, ae11_col, ae12_col = "_den_v", "_ae11_v", "_ae12_v"
    se11_col, se12_col = "_se11_v", "_se12_v"

    older = _fit_meta_model(
        block_scores,
        suffix,
        label_block=2,
        train_feature_blocks=(3, 4),
        score_feature_blocks=(2, 3, 4),
    )
    closed = (
        block_scores.filter((pl.col("_validation_block") == 1) & pl.col(elig))
        .select("unique_id", den_col, ae11_col, ae12_col, se11_col, se12_col)
        .join(older, on="unique_id", how="left")
    )

    fallback = {
        "mode": "v11_all", "threshold": default_threshold,
        "available": False, "utility_gain": 0.0,
        "wmape_v11": None, "wmape_selected": None,
        "bias_v11": None, "bias_selected": None,
        "value_safety_enabled": True,
        "value_safety_dominance_pass": False,
        "value_safety_recent_confirmations": 0,
        "value_safety_recent_blocks": 0,
        "value_safety_bias_coverage": None,
        "value_safety_bias_coverage_threshold": float(coverage_grid[0]),
        "value_safety_bias_coverage_pass": False,
        "value_safety_meta_margin_pass": False,
        "value_safety_best_all_mode": "v11_all",
        "value_safety_reason": "insufficient_closed_evidence",
    }
    if closed.height < min_closed_leaves:
        return fallback
    if not bool(closed.select(pl.col("_meta_model_available").fill_null(False).any()).item()):
        return fallback

    def evaluate(frame: pl.DataFrame, select_expr: pl.Expr) -> dict[str, float | None]:
        row = frame.select(
            pl.col(den_col).sum().alias("den"),
            pl.when(select_expr).then(pl.col(ae12_col)).otherwise(pl.col(ae11_col)).sum().alias("ae"),
            pl.when(select_expr).then(pl.col(se12_col)).otherwise(pl.col(se11_col)).sum().alias("se"),
        ).row(0, named=True)
        den = float(row["den"] or 0.0)
        if den <= 1e-9:
            return {"wmape": None, "bias": None, "utility": None}
        wmape = float(row["ae"] or 0.0) / den
        bias = float(row["se"] or 0.0) / den
        return {"wmape": wmape, "bias": bias, "utility": wmape + bias_weight * abs(bias)}

    v11 = evaluate(closed, pl.lit(False))
    v12 = evaluate(closed, pl.lit(True))
    if v11["utility"] is None or v12["utility"] is None:
        return fallback

    # Portfolio dominance/confirmation uses only fully CLOSED blocks 1 and 2.
    recent = (
        block_scores.filter(pl.col("_validation_block").is_in([1, 2]) & pl.col(elig))
        .group_by("_validation_block")
        .agg(
            pl.col(den_col).sum().alias("den"),
            pl.col(ae11_col).sum().alias("ae11"),
            pl.col(ae12_col).sum().alias("ae12"),
            pl.col(se11_col).sum().alias("se11"),
            pl.col(se12_col).sum().alias("se12"),
        )
        .filter(pl.col("den") > 1e-9)
        .with_columns(
            (pl.col("ae11") / pl.col("den")).alias("w11"),
            (pl.col("ae12") / pl.col("den")).alias("w12"),
            (pl.col("se11") / pl.col("den")).alias("b11"),
            (pl.col("se12") / pl.col("den")).alias("b12"),
        )
        .with_columns(
            (
                (pl.col("w12") < pl.col("w11"))
                & (pl.col("b12").abs() <= pl.col("b11").abs() + portfolio_bias_tol)
            ).alias("dominates")
        )
    )
    recent_blocks = int(recent.height)
    recent_confirmations = int(recent.select(pl.col("dominates").sum()).item() or 0) if recent.height else 0
    latest = recent.filter(pl.col("_validation_block") == 1)
    dominance_pass = bool(latest["dominates"][0]) if latest.height else False

    bias_guard_expr = (
        pl.col("meta_bias12").abs() <= pl.col("meta_bias11").abs() + leaf_bias_tol
    ).fill_null(False)
    available_expr = pl.col("_meta_model_available").fill_null(False)
    coverage = closed.select(
        pl.when(available_expr).then(bias_guard_expr.cast(pl.Float64)).otherwise(None).mean().alias("coverage")
    )["coverage"][0]
    coverage = None if coverage is None else float(coverage)
    passing_cov = [x for x in coverage_grid if coverage is not None and coverage + 1e-12 >= x]
    coverage_threshold = float(max(passing_cov)) if passing_cov else float(coverage_grid[0])
    coverage_pass = bool(passing_cov)

    v12_all_safe = (
        dominance_pass
        and recent_blocks >= min_confirm
        and recent_confirmations >= min_confirm
        and coverage_pass
    )

    all_candidates: list[tuple[str, float, dict[str, float | None]]] = [
        ("v11_all", default_threshold, v11)
    ]
    if v12_all_safe:
        v12_gain = float(v11["utility"] or 0.0) - float(v12["utility"] or 0.0)
        if v12_gain >= min_gain:
            all_candidates.append(("v12_all", default_threshold, v12))
    best_all_mode, best_all_threshold, best_all = min(
        all_candidates, key=lambda item: float(item[2]["utility"] or 1e9)
    )

    recent_guard = pl.col("meta_recent_gain").fill_null(-999.0) >= -max_recent_degradation
    meta_candidates: list[tuple[str, float, dict[str, float | None]]] = []
    for threshold in thresholds:
        selected = (
            available_expr
            & (pl.col("_meta_probability").fill_null(0.0) >= float(threshold))
            & recent_guard
            & bias_guard_expr
        )
        stats = evaluate(closed, selected)
        if stats["utility"] is None or stats["bias"] is None:
            continue
        if abs(float(stats["bias"])) > abs(float(v11["bias"] or 0.0)) + portfolio_bias_tol:
            continue
        meta_candidates.append(("meta_leaf", float(threshold), stats))

    best_meta = min(meta_candidates, key=lambda item: float(item[2]["utility"] or 1e9)) if meta_candidates else None
    meta_margin_pass = False
    if best_meta is not None:
        meta_margin_pass = (
            float(best_all["utility"] or 1e9) - float(best_meta[2]["utility"] or 1e9)
            >= meta_margin
        )

    if best_meta is not None and meta_margin_pass:
        mode, threshold, best = best_meta
        reason = "meta_leaf_beats_safe_all"
    else:
        mode, threshold, best = best_all_mode, best_all_threshold, best_all
        if best_all_mode == "v12_all":
            reason = "v12_all_safe_dominance"
        elif not dominance_pass:
            reason = "v12_all_blocked_dominance"
        elif recent_confirmations < min_confirm:
            reason = "v12_all_blocked_recent_confirmations"
        elif not coverage_pass:
            reason = "v12_all_blocked_bias_coverage"
        else:
            reason = "v11_all_best_safe_mode"

    utility_gain = float(v11["utility"] or 0.0) - float(best["utility"] or 0.0)
    if mode != "v11_all" and utility_gain < min_gain:
        mode, threshold, best, utility_gain = "v11_all", default_threshold, v11, 0.0
        reason = "fallback_min_utility_gain"

    return {
        "mode": mode,
        "threshold": float(threshold),
        "available": True,
        "utility_gain": float(utility_gain),
        "wmape_v11": float(v11["wmape"]) if v11["wmape"] is not None else None,
        "wmape_selected": float(best["wmape"]) if best["wmape"] is not None else None,
        "bias_v11": float(v11["bias"]) if v11["bias"] is not None else None,
        "bias_selected": float(best["bias"]) if best["bias"] is not None else None,
        "value_safety_enabled": True,
        "value_safety_dominance_pass": bool(dominance_pass),
        "value_safety_recent_confirmations": int(recent_confirmations),
        "value_safety_recent_blocks": int(recent_blocks),
        "value_safety_bias_coverage": coverage,
        "value_safety_bias_coverage_threshold": float(coverage_threshold),
        "value_safety_bias_coverage_pass": bool(coverage_pass),
        "value_safety_meta_margin_pass": bool(meta_margin_pass),
        "value_safety_best_all_mode": str(best_all_mode),
        "value_safety_reason": reason,
    }


def _calibrate_value_walkforward_policy(block_scores: pl.DataFrame) -> dict[str, object]:
    """v12.9.2 walk-forward portfolio policy for Value ($).

    All evidence comes from already-closed validation blocks.  Three pseudo-OOS
    folds are evaluated with bottom-up (denominator/volume weighted) portfolio
    metrics.  ``v12_all`` or ``meta_leaf`` is promoted only when its historical
    gains are stable, its worst fold is bounded, and |BIAS| does not deteriorate
    beyond the configured tolerance.  Otherwise the policy falls back to
    ``v11_all``.  Quantity does not use this function.
    """
    legacy = _calibrate_value_safety_policy(block_scores)
    enabled = bool(getattr(settings, "V129_VALUE_WALK_FORWARD_ENABLED", True))
    if not enabled:
        return {
            **legacy,
            "v129_value_wf_enabled": False,
            "v129_value_wf_available": False,
            "v129_value_wf_folds": 0,
            "v129_value_wf_win_rate": None,
            "v129_value_wf_median_gain": None,
            "v129_value_wf_worst_gain": None,
            "v129_value_wf_weighted_gain": None,
            "v129_value_wf_weighted_utility_gain": None,
            "v129_value_wf_bias_worsen_max": None,
            "v129_value_wf_meta_folds": 0,
            "v129_value_wf_fold1_gain": None, "v129_value_wf_fold2_gain": None, "v129_value_wf_fold3_gain": None,
            "v129_value_wf_reason": "disabled",
        }

    suffix = "v"
    elig = "_eligible_v"
    den_col, ae11_col, ae12_col = "_den_v", "_ae11_v", "_ae12_v"
    se11_col, se12_col = "_se11_v", "_se12_v"
    default_threshold = float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))
    thresholds = tuple(float(x) for x in getattr(
        settings, "V12_META_SELECTOR_THRESHOLDS", (0.45, 0.50, 0.55, 0.60, 0.65)
    ))
    thresholds = tuple(sorted({min(max(x, 0.01), 0.99) for x in thresholds})) or (default_threshold,)
    bias_weight = float(getattr(settings, "V12_SELECTOR_BIAS_WEIGHT", 0.25))
    leaf_bias_tol = float(getattr(settings, "V12_META_SELECTOR_BIAS_GUARD_TOLERANCE", 0.03))
    max_recent_degradation = float(getattr(settings, "V12_META_SELECTOR_MAX_RECENT_DEGRADATION", 0.08))
    requested_folds = max(1, int(getattr(settings, "V129_VALUE_WF_FOLDS", 3)))
    min_folds = max(1, int(getattr(settings, "V129_VALUE_WF_MIN_FOLDS", 3)))
    min_win_rate = float(getattr(settings, "V129_VALUE_WF_MIN_WIN_RATE", 2.0 / 3.0))
    min_median_gain = float(getattr(settings, "V129_VALUE_WF_MIN_MEDIAN_WMAPE_GAIN", 0.0025))
    max_worst_degradation = float(getattr(settings, "V129_VALUE_WF_MAX_WORST_WMAPE_DEGRADATION", 0.04))
    max_bias_worsen = float(getattr(settings, "V129_VALUE_WF_MAX_ABS_BIAS_WORSEN", 0.02))
    min_promotion_gain = float(getattr(settings, "V129_VALUE_WF_MIN_PROMOTION_UTILITY_GAIN", 0.0030))
    meta_min_folds = max(1, int(getattr(settings, "V129_VALUE_WF_META_MIN_FOLDS", 2)))
    meta_margin = float(getattr(settings, "V129_VALUE_WF_META_MARGIN_OVER_ALL", 0.0025))
    min_closed_leaves = max(1, int(getattr(settings, "V12_VALUE_SAFETY_MIN_CLOSED_LEAVES", 100)))

    max_block_raw = block_scores.select(pl.col("_validation_block").max()).item()
    max_block = int(max_block_raw or 0)
    fold_ids = [b for b in range(1, min(requested_folds, max_block) + 1)]

    def _evaluate(frame: pl.DataFrame, select_expr: pl.Expr) -> dict[str, float | None]:
        row = frame.select(
            pl.col(den_col).sum().alias("den"),
            pl.when(select_expr).then(pl.col(ae12_col)).otherwise(pl.col(ae11_col)).sum().alias("ae"),
            pl.when(select_expr).then(pl.col(se12_col)).otherwise(pl.col(se11_col)).sum().alias("se"),
        ).row(0, named=True)
        den = float(row["den"] or 0.0)
        if den <= 1e-9:
            return {"den": 0.0, "ae": None, "se": None, "wmape": None, "bias": None, "utility": None}
        ae = float(row["ae"] or 0.0)
        se = float(row["se"] or 0.0)
        wmape = ae / den
        bias = se / den
        return {"den": den, "ae": ae, "se": se, "wmape": wmape, "bias": bias,
                "utility": wmape + bias_weight * abs(bias)}

    folds: list[dict[str, object]] = []
    meta_by_threshold: dict[float, list[dict[str, object]]] = {t: [] for t in thresholds}
    for fold in fold_ids:
        frame = block_scores.filter((pl.col("_validation_block") == fold) & pl.col(elig))
        if frame.height < min_closed_leaves:
            continue
        v11 = _evaluate(frame, pl.lit(False))
        v12 = _evaluate(frame, pl.lit(True))
        if v11["wmape"] is None or v12["wmape"] is None:
            continue
        folds.append({"fold": fold, "v11": v11, "v12": v12})

        # Temporally shifted meta-model for this pseudo-OOS fold.  The winner
        # label comes from fold+1 and features only from still older blocks.
        label_block = fold + 1
        train_blocks = tuple(range(fold + 2, min(max_block, fold + 4) + 1))
        score_blocks = tuple(range(fold + 1, min(max_block, fold + 3) + 1))
        if label_block > max_block or not train_blocks or not score_blocks:
            continue
        model = _fit_meta_model(
            block_scores, suffix,
            label_block=label_block,
            train_feature_blocks=train_blocks,
            score_feature_blocks=score_blocks,
        )
        fold_meta = frame.select(
            "unique_id", den_col, ae11_col, ae12_col, se11_col, se12_col
        ).join(model, on="unique_id", how="left")
        if fold_meta.height == 0 or not bool(
            fold_meta.select(pl.col("_meta_model_available").fill_null(False).any()).item()
        ):
            continue
        available = pl.col("_meta_model_available").fill_null(False)
        bias_guard = (
            pl.col("meta_bias12").abs() <= pl.col("meta_bias11").abs() + leaf_bias_tol
        ).fill_null(False)
        recent_guard = pl.col("meta_recent_gain").fill_null(-999.0) >= -max_recent_degradation
        for threshold in thresholds:
            selected = (
                available
                & (pl.col("_meta_probability").fill_null(0.0) >= float(threshold))
                & bias_guard
                & recent_guard
            )
            stats = _evaluate(fold_meta, selected)
            if stats["wmape"] is not None:
                meta_by_threshold[threshold].append({"fold": fold, "stats": stats, "v11": v11})

    def _summarize(candidate_folds: list[dict[str, object]], *, candidate_key: str | None = None) -> dict[str, object]:
        rows = []
        for item in candidate_folds:
            v11 = item["v11"]
            cand = item[candidate_key] if candidate_key else item["stats"]
            assert isinstance(v11, dict) and isinstance(cand, dict)
            if v11.get("wmape") is None or cand.get("wmape") is None:
                continue
            gain = float(v11["wmape"]) - float(cand["wmape"])
            utility_gain = float(v11["utility"]) - float(cand["utility"])
            bias_worsen = abs(float(cand["bias"])) - abs(float(v11["bias"]))
            rows.append((gain, utility_gain, bias_worsen, v11, cand))
        if not rows:
            return {"folds": 0, "safe": False}
        gains = np.array([r[0] for r in rows], dtype=np.float64)
        bias_worsen = np.array([r[2] for r in rows], dtype=np.float64)
        wins = int(np.sum(gains > 0.0))
        den = float(sum(float(r[3]["den"]) for r in rows))
        ae11 = float(sum(float(r[3]["ae"]) for r in rows))
        se11 = float(sum(float(r[3]["se"]) for r in rows))
        aec = float(sum(float(r[4]["ae"]) for r in rows))
        sec = float(sum(float(r[4]["se"]) for r in rows))
        w11 = ae11 / den if den > 1e-9 else float("nan")
        wc = aec / den if den > 1e-9 else float("nan")
        b11 = se11 / den if den > 1e-9 else float("nan")
        bc = sec / den if den > 1e-9 else float("nan")
        u11 = w11 + bias_weight * abs(b11)
        uc = wc + bias_weight * abs(bc)
        n = len(rows)
        win_rate = wins / n
        median_gain = float(np.median(gains))
        worst_gain = float(np.min(gains))
        weighted_gain = float(w11 - wc)
        weighted_utility_gain = float(u11 - uc)
        bias_worsen_max = float(np.max(bias_worsen))
        safe = (
            n >= min_folds
            and win_rate + 1e-12 >= min_win_rate
            and median_gain + 1e-12 >= min_median_gain
            and worst_gain + 1e-12 >= -max_worst_degradation
            and bias_worsen_max <= max_bias_worsen + 1e-12
            and weighted_utility_gain + 1e-12 >= min_promotion_gain
        )
        return {
            "folds": n, "wins": wins, "win_rate": win_rate,
            "fold_gains": [float(x) for x in gains.tolist()],
            "median_gain": median_gain, "worst_gain": worst_gain,
            "weighted_gain": weighted_gain,
            "weighted_utility_gain": weighted_utility_gain,
            "bias_worsen_max": bias_worsen_max,
            "wmape_v11": w11, "wmape_candidate": wc,
            "bias_v11": b11, "bias_candidate": bc,
            "utility_v11": u11, "utility_candidate": uc,
            "safe": bool(safe),
        }

    v12_summary = _summarize(folds, candidate_key="v12")
    safe_all: list[tuple[str, float, dict[str, object]]] = []
    # v11 is always the conservative reference; represent zero gain explicitly.
    if folds:
        base = _summarize([
            {"v11": f["v11"], "stats": f["v11"]} for f in folds
        ])
        base["safe"] = True
        safe_all.append(("v11_all", default_threshold, base))
    if bool(v12_summary.get("safe", False)):
        safe_all.append(("v12_all", default_threshold, v12_summary))

    best_all = max(
        safe_all,
        key=lambda x: float(x[2].get("weighted_utility_gain") or 0.0),
        default=("v11_all", default_threshold, {"weighted_utility_gain": 0.0, "safe": True}),
    )

    best_meta: tuple[str, float, dict[str, object]] | None = None
    for threshold, entries in meta_by_threshold.items():
        summary = _summarize(entries)
        if int(summary.get("folds", 0) or 0) < meta_min_folds:
            continue
        # Meta may use fewer folds, so relax only the fold-count requirement;
        # all other stability/bias/promotion guards remain identical.
        meta_safe = (
            float(summary.get("win_rate") or 0.0) + 1e-12 >= min_win_rate
            and float(summary.get("median_gain") or -999.0) + 1e-12 >= min_median_gain
            and float(summary.get("worst_gain") or -999.0) + 1e-12 >= -max_worst_degradation
            and float(summary.get("bias_worsen_max") or 999.0) <= max_bias_worsen + 1e-12
            and float(summary.get("weighted_utility_gain") or -999.0) + 1e-12 >= min_promotion_gain
        )
        summary["safe"] = bool(meta_safe)
        if not meta_safe:
            continue
        cand = ("meta_leaf", float(threshold), summary)
        if best_meta is None or float(summary["weighted_utility_gain"]) > float(best_meta[2]["weighted_utility_gain"]):
            best_meta = cand

    mode, threshold, chosen = best_all
    reason = "v11_all_walkforward_fallback" if mode == "v11_all" else "v12_all_walkforward_stable"
    if best_meta is not None:
        all_gain = float(chosen.get("weighted_utility_gain") or 0.0)
        meta_gain = float(best_meta[2].get("weighted_utility_gain") or 0.0)
        if meta_gain >= all_gain + meta_margin:
            mode, threshold, chosen = best_meta
            reason = "meta_leaf_walkforward_margin"

    available = int(v12_summary.get("folds", 0) or 0) >= min_folds
    if not folds:
        # Preserve legacy diagnostics while making the production decision safe.
        return {
            **legacy,
            "mode": "v11_all", "threshold": default_threshold,
            "available": False, "utility_gain": 0.0,
            "v129_value_wf_enabled": True,
            "v129_value_wf_available": False,
            "v129_value_wf_folds": 0,
            "v129_value_wf_win_rate": None,
            "v129_value_wf_median_gain": None,
            "v129_value_wf_worst_gain": None,
            "v129_value_wf_weighted_gain": None,
            "v129_value_wf_weighted_utility_gain": None,
            "v129_value_wf_bias_worsen_max": None,
            "v129_value_wf_meta_folds": 0,
            "v129_value_wf_fold1_gain": None, "v129_value_wf_fold2_gain": None, "v129_value_wf_fold3_gain": None,
            "v129_value_wf_reason": "insufficient_walkforward_evidence",
        }

    return {
        **legacy,
        "mode": str(mode),
        "threshold": float(threshold),
        "available": bool(available),
        "utility_gain": float(chosen.get("weighted_utility_gain") or 0.0),
        "wmape_v11": chosen.get("wmape_v11"),
        "wmape_selected": chosen.get("wmape_candidate"),
        "bias_v11": chosen.get("bias_v11"),
        "bias_selected": chosen.get("bias_candidate"),
        "v129_value_wf_enabled": True,
        "v129_value_wf_available": bool(available),
        "v129_value_wf_folds": int(v12_summary.get("folds", 0) or 0),
        "v129_value_wf_win_rate": v12_summary.get("win_rate"),
        "v129_value_wf_median_gain": v12_summary.get("median_gain"),
        "v129_value_wf_worst_gain": v12_summary.get("worst_gain"),
        "v129_value_wf_weighted_gain": v12_summary.get("weighted_gain"),
        "v129_value_wf_weighted_utility_gain": v12_summary.get("weighted_utility_gain"),
        "v129_value_wf_bias_worsen_max": v12_summary.get("bias_worsen_max"),
        "v129_value_wf_meta_folds": int(best_meta[2].get("folds", 0) or 0) if best_meta else 0,
        "v129_value_wf_fold1_gain": (v12_summary.get("fold_gains") or [None, None, None])[0] if len(v12_summary.get("fold_gains") or []) > 0 else None,
        "v129_value_wf_fold2_gain": (v12_summary.get("fold_gains") or [None, None, None])[1] if len(v12_summary.get("fold_gains") or []) > 1 else None,
        "v129_value_wf_fold3_gain": (v12_summary.get("fold_gains") or [None, None, None])[2] if len(v12_summary.get("fold_gains") or []) > 2 else None,
        "v129_value_wf_reason": reason,
    }


def _fit_value_hierarchical_segment_selector(
    block_scores: pl.DataFrame,
    target_profile: pl.DataFrame,
) -> pl.DataFrame:
    """v12.9.6 causal hierarchical segment selector for Value ($).

    Decisions are learned only from already-closed validation blocks.  The
    hierarchy backs off from a specific, interpretable segment to broader
    evidence and finally to the conservative v11 incumbent.  The old ridge /
    expected-impact model is retained only as a diagnostic challenger.
    """
    enabled = bool(getattr(settings, "V1293_VALUE_SEGMENT_SELECTOR_ENABLED", True)) and int(
        getattr(settings, "RLS_BLOCK_DAYS", 28)
    ) == 28
    folds_cfg = max(3, int(getattr(settings, "V1293_VALUE_SEGMENT_FOLDS", 3)))
    bias_weight = float(getattr(settings, "V12_SELECTOR_BIAS_WEIGHT", 0.25))
    budget_share = float(getattr(settings, "V1293_VALUE_SEGMENT_MAX_VOLUME_SHARE", 0.45))
    strong_section_gain = float(getattr(settings, "V1293_VALUE_SECTION_STRONG_GAIN", 0.030))

    profile = target_profile.select(
        "unique_id",
        pl.col("_seg_level_method_value").fill_null("unknown"),
        pl.col("_seg_seasonal_multiplier_value").fill_null(1.0),
    ).unique("unique_id")
    latest = (
        block_scores.filter(pl.col("_validation_block") == 1)
        .select("unique_id", pl.col("_n_v").alias("_seg_sales_days"), pl.col("_den_v").alias("_seg_recent_volume"))
        .unique("unique_id")
    )
    profile = (
        profile.join(latest, on="unique_id", how="left")
        .with_columns(
            pl.col("unique_id").str.split("||").list.first().fill_null("?").alias("_seg_section"),
            (
                (pl.col("_seg_level_method_value") == "sku_yoy_seasonal")
                | ((pl.col("_seg_seasonal_multiplier_value") - 1.0).abs() > 0.02)
            ).alias("_seg_seasonal"),
            pl.when(pl.col("_seg_sales_days").fill_null(0) <= 9).then(pl.lit("07-09"))
            .when(pl.col("_seg_sales_days") <= 13).then(pl.lit("10-13"))
            .when(pl.col("_seg_sales_days") <= 20).then(pl.lit("14-20"))
            .otherwise(pl.lit("21-28")).alias("_seg_sales_bucket"),
            pl.col("_seg_recent_volume").fill_null(0.0),
        )
    )
    section_values = profile.get_column("_seg_section").drop_nulls().unique().to_list() if profile.height else []
    section_id = str(section_values[0]) if len(section_values) == 1 else "?"
    if section_id == "23":
        budget_share = float(getattr(settings, "V1294_VALUE_SEC23_MAX_VOLUME_SHARE", 0.25))
    # v12.9.6 impact semantics: 1.0 = highest-volume leaf, near 0 = lowest.
    if profile.height:
        _vol = profile.get_column("_seg_recent_volume").fill_null(0.0).to_numpy().astype(np.float64, copy=False)
        _order = np.argsort(-_vol, kind="stable")
        _pct = np.empty(profile.height, dtype=np.float64)
        _pct[_order] = (float(profile.height) - np.arange(profile.height, dtype=np.float64)) / max(float(profile.height), 1.0)
        profile = profile.with_columns(pl.Series("_seg_impact_percentile", _pct.tolist(), dtype=pl.Float64))
    else:
        profile = profile.with_columns(pl.lit(None, dtype=pl.Float64).alias("_seg_impact_percentile"))

    base = profile.select("unique_id").with_columns(
        pl.lit(False).alias("v1293_segment_prebudget_value"),
        pl.lit(False).alias("v1293_segment_budget_selected_value"),
        pl.lit("fallback").alias("v1293_segment_level_value"),
        pl.lit("v11_insufficient_or_unstable_segment").alias("v1293_segment_reason_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v1293_segment_key_value"),
        pl.lit(0, dtype=pl.Int8).alias("v1293_segment_folds_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1293_segment_win_rate_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1293_segment_median_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1293_segment_worst_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1293_segment_weighted_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1293_segment_utility_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1293_segment_bias_worsen_value"),
        pl.lit(0, dtype=pl.Int32).alias("v1293_segment_min_leaves_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1294_segment_risk_score_value"),
        pl.lit(0.0, dtype=pl.Float64).alias("v1294_segment_exposure_budget_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1294_impact_percentile_value"),
        pl.lit(False).alias("v1294_high_impact_guard_pass_value"),
    )
    if (not enabled) or profile.height == 0:
        return base.with_columns(
            pl.lit(False).alias("v1293_segment_selector_enabled_value"),
            pl.lit(0.0).alias("v1293_segment_selected_volume_share_value"),
            pl.lit(False).alias("v1293_segment_section_strong_value"),
            pl.lit(budget_share).alias("v1293_segment_budget_value"),
        )

    use_blocks = list(range(1, folds_cfg + 1))
    enriched = (
        block_scores.filter(pl.col("_validation_block").is_in(use_blocks) & pl.col("_eligible_v"))
        .join(profile, on="unique_id", how="inner")
    )

    hierarchy = [
        ("L4", ["_seg_section", "_seg_level_method_value", "_seg_seasonal", "_seg_sales_bucket"],
         int(getattr(settings, "V1293_VALUE_L4_MIN_LEAVES", 20)), 0.005, 0.0025, -0.030, 0.030),
        ("L3", ["_seg_section", "_seg_level_method_value", "_seg_seasonal"],
         int(getattr(settings, "V1293_VALUE_L3_MIN_LEAVES", 40)), 0.006, 0.0030, -0.025, 0.025),
        ("L2", ["_seg_section", "_seg_level_method_value"],
         int(getattr(settings, "V1293_VALUE_L2_MIN_LEAVES", 80)), 0.008, 0.0040, -0.020, 0.025),
        ("L1", ["_seg_section"],
         int(getattr(settings, "V1293_VALUE_L1_MIN_LEAVES", 500)), strong_section_gain, 0.0150, 0.0, 0.020),
    ]

    result = base
    assigned = set()
    section_strong = False
    for level_name, keys, min_leaves, min_weighted_gain, min_median_gain, min_worst_gain, max_bias_worsen in hierarchy:
        if enriched.height == 0:
            continue
        fold = (
            enriched.group_by(keys + ["_validation_block"])
            .agg(
                pl.col("_den_v").sum().alias("_den"),
                pl.col("_ae11_v").sum().alias("_ae11"),
                pl.col("_ae12_v").sum().alias("_ae12"),
                pl.col("_se11_v").sum().alias("_se11"),
                pl.col("_se12_v").sum().alias("_se12"),
                pl.col("unique_id").n_unique().cast(pl.Int32).alias("_n_leaves"),
            )
            .filter(pl.col("_den") > 1e-9)
            .with_columns(
                ((pl.col("_ae11") - pl.col("_ae12")) / pl.col("_den")).alias("_gain"),
                (pl.col("_se11") / pl.col("_den")).alias("_b11"),
                (pl.col("_se12") / pl.col("_den")).alias("_b12"),
            )
            .with_columns((pl.col("_b12").abs() - pl.col("_b11").abs()).alias("_bias_worsen"))
        )
        stats = (
            fold.group_by(keys)
            .agg(
                pl.col("_validation_block").n_unique().cast(pl.Int8).alias("_folds"),
                (pl.col("_gain") > 0.0).cast(pl.Float64).mean().alias("_win_rate"),
                pl.col("_gain").median().alias("_median_gain"),
                pl.col("_gain").min().alias("_worst_gain"),
                pl.col("_bias_worsen").max().alias("_bias_worsen_max"),
                pl.col("_n_leaves").min().cast(pl.Int32).alias("_min_leaves"),
                pl.col("_den").sum().alias("_den_all"),
                pl.col("_ae11").sum().alias("_ae11_all"),
                pl.col("_ae12").sum().alias("_ae12_all"),
                pl.col("_se11").sum().alias("_se11_all"),
                pl.col("_se12").sum().alias("_se12_all"),
            )
            .with_columns(
                ((pl.col("_ae11_all") - pl.col("_ae12_all")) / pl.col("_den_all")).alias("_weighted_gain"),
                (
                    (pl.col("_ae11_all") - pl.col("_ae12_all")) / pl.col("_den_all")
                    + bias_weight * (
                        (pl.col("_se11_all") / pl.col("_den_all")).abs()
                        - (pl.col("_se12_all") / pl.col("_den_all")).abs()
                    )
                ).alias("_utility_gain"),
            )
            .with_columns(
                (
                    (pl.col("_folds") >= folds_cfg)
                    & (pl.col("_win_rate") >= (2.0 / 3.0))
                    & (pl.col("_median_gain") >= min_median_gain)
                    & (pl.col("_weighted_gain") >= min_weighted_gain)
                    & (pl.col("_worst_gain") >= min_worst_gain)
                    & (pl.col("_bias_worsen_max") <= max_bias_worsen)
                    & (pl.col("_min_leaves") >= min_leaves)
                ).alias("_segment_pass")
            )
        )
        candidates = (
            profile.join(stats, on=keys, how="left")
            .filter(pl.col("_segment_pass").fill_null(False))
        )
        if assigned:
            candidates = candidates.filter(~pl.col("unique_id").is_in(list(assigned)))
        if candidates.height == 0:
            continue
        if level_name == "L1":
            section_strong = True
        chosen_ids = candidates.get_column("unique_id").to_list()
        assigned.update(chosen_ids)
        key_expr = pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|")
        chosen = candidates.select(
            "unique_id",
            pl.lit(True).alias("_new_pre"),
            pl.lit(level_name).alias("_new_level"),
            pl.lit(f"v12_{level_name.lower()}_walkforward_stable").alias("_new_reason"),
            key_expr.alias("_new_key"),
            pl.col("_folds").alias("_new_folds"),
            pl.col("_win_rate").alias("_new_win"),
            pl.col("_median_gain").alias("_new_med"),
            pl.col("_worst_gain").alias("_new_worst"),
            pl.col("_weighted_gain").alias("_new_wg"),
            pl.col("_utility_gain").alias("_new_ug"),
            pl.col("_bias_worsen_max").alias("_new_bw"),
            pl.col("_min_leaves").alias("_new_n"),
        )
        result = (
            result.join(chosen, on="unique_id", how="left")
            .with_columns(
                pl.coalesce([pl.col("_new_pre"), pl.col("v1293_segment_prebudget_value")]).alias("v1293_segment_prebudget_value"),
                pl.coalesce([pl.col("_new_level"), pl.col("v1293_segment_level_value")]).alias("v1293_segment_level_value"),
                pl.coalesce([pl.col("_new_reason"), pl.col("v1293_segment_reason_value")]).alias("v1293_segment_reason_value"),
                pl.coalesce([pl.col("_new_key"), pl.col("v1293_segment_key_value")]).alias("v1293_segment_key_value"),
                pl.coalesce([pl.col("_new_folds"), pl.col("v1293_segment_folds_value")]).cast(pl.Int8).alias("v1293_segment_folds_value"),
                pl.coalesce([pl.col("_new_win"), pl.col("v1293_segment_win_rate_value")]).alias("v1293_segment_win_rate_value"),
                pl.coalesce([pl.col("_new_med"), pl.col("v1293_segment_median_gain_value")]).alias("v1293_segment_median_gain_value"),
                pl.coalesce([pl.col("_new_worst"), pl.col("v1293_segment_worst_gain_value")]).alias("v1293_segment_worst_gain_value"),
                pl.coalesce([pl.col("_new_wg"), pl.col("v1293_segment_weighted_gain_value")]).alias("v1293_segment_weighted_gain_value"),
                pl.coalesce([pl.col("_new_ug"), pl.col("v1293_segment_utility_gain_value")]).alias("v1293_segment_utility_gain_value"),
                pl.coalesce([pl.col("_new_bw"), pl.col("v1293_segment_bias_worsen_value")]).alias("v1293_segment_bias_worsen_value"),
                pl.coalesce([pl.col("_new_n"), pl.col("v1293_segment_min_leaves_value")]).cast(pl.Int32).alias("v1293_segment_min_leaves_value"),
            )
            .drop([c for c in ["_new_pre","_new_level","_new_reason","_new_key","_new_folds","_new_win","_new_med","_new_worst","_new_wg","_new_ug","_new_bw","_new_n"] if c in result.columns or c in chosen.columns], strict=False)
        )

    # v12.9.6 volume-risk control. Sec.1 is frozen at v12.9.3 semantics.
    # Sec.23 receives a hard 25% section-volume budget, an 8% per-segment cap,
    # and asymmetric guards for the highest-impact leaves. No leaf may push a
    # segment or the global portfolio above its budget.
    ranked = (
        result.join(
            profile.select("unique_id", "_seg_recent_volume", "_seg_impact_percentile"),
            on="unique_id", how="left"
        )
        .filter(pl.col("v1293_segment_prebudget_value"))
        .with_columns(
            pl.col("_seg_recent_volume").fill_null(0.0).alias("_seg_w"),
            pl.col("_seg_impact_percentile").fill_null(0.0).alias("_impact_pct"),
            (
                pl.col("v1293_segment_utility_gain_value").fill_null(0.0)
                * pl.col("_seg_recent_volume").fill_null(0.0)
            ).alias("_seg_score"),
        )
    )
    total_w = float(profile.select(pl.col("_seg_recent_volume").sum()).item() or 0.0)
    budget_ids = pl.DataFrame({"unique_id": []}, schema={"unique_id": pl.Utf8})
    risk_meta = pl.DataFrame(
        schema={
            "unique_id": pl.Utf8,
            "v1294_segment_risk_score_value": pl.Float64,
            "v1294_segment_exposure_budget_value": pl.Float64,
            "v1294_impact_percentile_value": pl.Float64,
            "v1294_high_impact_guard_pass_value": pl.Boolean,
        }
    )
    if ranked.height:
        if section_id != "23" and section_strong:
            # Frozen Sec.1: retain the successful v12.9.3 full-section promotion.
            budget_ids = ranked.select("unique_id")
            risk_meta = ranked.select(
                "unique_id",
                pl.col("v1293_segment_utility_gain_value").fill_null(0.0).alias("v1294_segment_risk_score_value"),
                pl.lit(1.0).alias("v1294_segment_exposure_budget_value"),
                pl.col("_impact_pct").alias("v1294_impact_percentile_value"),
                pl.lit(True).alias("v1294_high_impact_guard_pass_value"),
            )
        elif section_id != "23":
            # Preserve legacy mixed-budget behavior outside Sec.23, but make the
            # boundary strict: cumulative volume INCLUDING the candidate must fit.
            budget_w = max(0.0, budget_share) * total_w
            rr = ranked.sort(["_seg_score", "_seg_w"], descending=[True, True]).with_columns(
                pl.col("_seg_w").cum_sum().alias("_cum")
            )
            budget_ids = rr.filter(pl.col("_cum") <= budget_w + 1e-9).select("unique_id")
            risk_meta = rr.select(
                "unique_id",
                pl.col("v1293_segment_utility_gain_value").fill_null(0.0).alias("v1294_segment_risk_score_value"),
                pl.lit(float(budget_share)).alias("v1294_segment_exposure_budget_value"),
                pl.col("_impact_pct").alias("v1294_impact_percentile_value"),
                pl.lit(True).alias("v1294_high_impact_guard_pass_value"),
            )
        else:
            top01 = float(getattr(settings, "V1294_VALUE_TOP01_PERCENTILE", 0.99))
            top05 = float(getattr(settings, "V1294_VALUE_TOP05_PERCENTILE", 0.95))
            top01_gain = float(getattr(settings, "V1294_VALUE_TOP01_MIN_WEIGHTED_GAIN", 0.040))
            top01_worst = float(getattr(settings, "V1294_VALUE_TOP01_MIN_WORST_GAIN", 0.020))
            top05_gain = float(getattr(settings, "V1294_VALUE_TOP05_MIN_WEIGHTED_GAIN", 0.025))
            top05_worst = float(getattr(settings, "V1294_VALUE_TOP05_MIN_WORST_GAIN", 0.010))
            top01_bias = float(getattr(settings, "V1294_VALUE_TOP01_MAX_BIAS_WORSEN", 0.0))
            top05_bias = float(getattr(settings, "V1294_VALUE_TOP05_MAX_BIAS_WORSEN", 0.010))
            downside_weight = float(getattr(settings, "V1294_VALUE_SEGMENT_DOWNSIDE_WEIGHT", 5.0))
            max_seg_share = float(getattr(settings, "V1294_VALUE_SEC23_MAX_SEGMENT_VOLUME_SHARE", 0.08))
            guard_expr = (
                pl.when(pl.col("_impact_pct") >= top01)
                .then(
                    (pl.col("v1293_segment_weighted_gain_value").fill_null(-1.0) >= top01_gain)
                    & (pl.col("v1293_segment_worst_gain_value").fill_null(-1.0) >= top01_worst)
                    & (pl.col("v1293_segment_win_rate_value").fill_null(0.0) >= 1.0)
                    & (pl.col("v1293_segment_bias_worsen_value").fill_null(1.0) <= top01_bias)
                )
                .when(pl.col("_impact_pct") >= top05)
                .then(
                    (pl.col("v1293_segment_weighted_gain_value").fill_null(-1.0) >= top05_gain)
                    & (pl.col("v1293_segment_worst_gain_value").fill_null(-1.0) >= top05_worst)
                    & (pl.col("v1293_segment_win_rate_value").fill_null(0.0) >= 1.0)
                    & (pl.col("v1293_segment_bias_worsen_value").fill_null(1.0) <= top05_bias)
                )
                .otherwise(True)
            )
            rr = ranked.with_columns(guard_expr.alias("_high_guard"))
            rr = rr.with_columns(
                (
                    pl.col("v1293_segment_utility_gain_value").fill_null(0.0).clip(lower_bound=0.0)
                    * pl.col("v1293_segment_win_rate_value").fill_null(0.0)
                    / (
                        1.0
                        + downside_weight * (
                            (-pl.col("v1293_segment_worst_gain_value").fill_null(-1.0)).clip(lower_bound=0.0)
                            + pl.col("v1293_segment_bias_worsen_value").fill_null(0.0).clip(lower_bound=0.0)
                        )
                    )
                ).alias("_risk_score")
            )
            eligible = rr.filter(pl.col("_high_guard") & (pl.col("_risk_score") > 0.0))
            if eligible.height:
                segs = (
                    eligible.group_by(["v1293_segment_level_value", "v1293_segment_key_value"])
                    .agg(
                        pl.col("_seg_w").sum().alias("_candidate_w"),
                        pl.col("_risk_score").first().alias("_risk_score"),
                    )
                    .with_columns(
                        (
                            pl.col("_risk_score")
                            * (pl.col("_candidate_w") / max(total_w, 1e-9)).sqrt()
                        ).alias("_allocation_weight")
                    )
                )
                alloc_total = float(segs.select(pl.col("_allocation_weight").sum()).item() or 0.0)
                alloc_map = {}
                if alloc_total > 1e-12:
                    for sr in segs.iter_rows(named=True):
                        key = (str(sr["v1293_segment_level_value"]), str(sr["v1293_segment_key_value"]))
                        proportional = max(0.0, budget_share) * total_w * float(sr["_allocation_weight"] or 0.0) / alloc_total
                        cap = min(float(sr["_candidate_w"] or 0.0), max_seg_share * total_w, proportional)
                        alloc_map[key] = max(0.0, cap)
                selected_ids = []
                meta_rows = []
                for key, cap in alloc_map.items():
                    part = eligible.filter(
                        (pl.col("v1293_segment_level_value") == key[0])
                        & (pl.col("v1293_segment_key_value") == key[1])
                    ).sort(["_impact_pct", "_seg_w", "unique_id"], descending=[False, False, False])
                    used = 0.0
                    for row in part.iter_rows(named=True):
                        w = float(row["_seg_w"] or 0.0)
                        # Strict boundary: the candidate itself must fit the segment cap.
                        if used + w <= cap + 1e-9:
                            selected_ids.append(str(row["unique_id"]))
                            used += w
                    exposure_share = cap / total_w if total_w > 1e-9 else 0.0
                    for row in part.iter_rows(named=True):
                        meta_rows.append({
                            "unique_id": str(row["unique_id"]),
                            "v1294_segment_risk_score_value": float(row["_risk_score"] or 0.0),
                            "v1294_segment_exposure_budget_value": float(exposure_share),
                            "v1294_impact_percentile_value": float(row["_impact_pct"] or 0.0),
                            "v1294_high_impact_guard_pass_value": bool(row["_high_guard"]),
                        })
                # Global hard budget invariant; per-segment allocations sum to <= global budget.
                budget_ids = pl.DataFrame({"unique_id": selected_ids}, schema={"unique_id": pl.Utf8}) if selected_ids else budget_ids
                risk_meta = pl.DataFrame(meta_rows) if meta_rows else risk_meta
            # Attach metadata also for prebudget candidates rejected by high-impact guard.
            if rr.height:
                fallback_meta = rr.select(
                    "unique_id",
                    pl.col("_risk_score").alias("_risk_fb"),
                    pl.col("_impact_pct").alias("_pct_fb"),
                    pl.col("_high_guard").alias("_guard_fb"),
                )
                risk_meta = (
                    fallback_meta.join(risk_meta, on="unique_id", how="left")
                    .select(
                        "unique_id",
                        pl.coalesce([pl.col("v1294_segment_risk_score_value"), pl.col("_risk_fb")]).alias("v1294_segment_risk_score_value"),
                        pl.col("v1294_segment_exposure_budget_value").fill_null(0.0),
                        pl.coalesce([pl.col("v1294_impact_percentile_value"), pl.col("_pct_fb")]).alias("v1294_impact_percentile_value"),
                        pl.coalesce([pl.col("v1294_high_impact_guard_pass_value"), pl.col("_guard_fb")]).fill_null(False).alias("v1294_high_impact_guard_pass_value"),
                    )
                )

        if budget_ids.height:
            budget_ids = budget_ids.unique("unique_id").with_columns(pl.lit(True).alias("_budget_ok"))
            result = result.join(budget_ids, on="unique_id", how="left").with_columns(
                pl.col("_budget_ok").fill_null(False).alias("v1293_segment_budget_selected_value")
            ).drop("_budget_ok")
        if risk_meta.height:
            result = (
                result.drop([
                    "v1294_segment_risk_score_value", "v1294_segment_exposure_budget_value",
                    "v1294_impact_percentile_value", "v1294_high_impact_guard_pass_value"
                ])
                .join(risk_meta.unique("unique_id"), on="unique_id", how="left")
                .with_columns(
                    pl.col("v1294_segment_exposure_budget_value").fill_null(0.0),
                    pl.col("v1294_high_impact_guard_pass_value").fill_null(False),
                )
            )
    selected_w = float(
        result.join(profile.select("unique_id", "_seg_recent_volume"), on="unique_id", how="left")
        .select(
            pl.when(pl.col("v1293_segment_budget_selected_value"))
            .then(pl.col("_seg_recent_volume").fill_null(0.0)).otherwise(0.0).sum()
        ).item() or 0.0
    )
    share = selected_w / total_w if total_w > 1e-9 else 0.0
    return result.with_columns(
        pl.lit(True).alias("v1293_segment_selector_enabled_value"),
        pl.lit(float(share)).alias("v1293_segment_selected_volume_share_value"),
        pl.lit(bool(section_strong)).alias("v1293_segment_section_strong_value"),
        pl.lit(float(budget_share)).alias("v1293_segment_budget_value"),
    )



def _value_long_horizon_segment_stress(
    block_scores: pl.DataFrame,
    target_profile: pl.DataFrame,
) -> pl.DataFrame:
    """v12.9.6 long-horizon, strictly historical segment stress diagnostic.

    This layer is intentionally NON-PRODUCTIVE.  It never changes
    ``v12_selected_value``.  Historical segment membership is reconstructed
    from the state stored on each already-closed block, and the current leaf is
    mapped to the most-specific hierarchy level with enough historical folds.
    The current OOS block is never part of ``block_scores`` passed here.
    """
    enabled = bool(getattr(settings, "V1295_STRESS_TEST_ENABLED", True)) and int(
        getattr(settings, "RLS_BLOCK_DAYS", 28)
    ) == 28
    requested = max(3, int(getattr(settings, "V1295_STRESS_TEST_BLOCKS", 12)))
    min_folds = max(3, int(getattr(settings, "V1295_STRESS_TEST_MIN_FOLDS", 8)))
    min_win = float(getattr(settings, "V1295_STRESS_MIN_WIN_RATE", 0.75))
    min_med = float(getattr(settings, "V1295_STRESS_MIN_MEDIAN_GAIN", 0.0))
    min_p25 = float(getattr(settings, "V1295_STRESS_MIN_P25_GAIN", 0.0))
    min_p10 = float(getattr(settings, "V1295_STRESS_MIN_P10_GAIN", -0.03))
    max_bias_p90 = float(getattr(settings, "V1295_STRESS_MAX_BIAS_P90_WORSEN", 0.02))
    max_loss_run = max(0, int(getattr(settings, "V1295_STRESS_MAX_CONSECUTIVE_LOSSES", 2)))

    base = target_profile.select(
        "unique_id",
        pl.col("_seg_level_method_value").fill_null("unknown"),
        pl.col("_seg_seasonal_multiplier_value").fill_null(1.0),
    ).unique("unique_id")
    if block_scores.height:
        latest = (
            block_scores.filter(pl.col("_validation_block") == 1)
            .select("unique_id", pl.col("_n_v").alias("_stress_current_sales_days"))
            .unique("unique_id")
        )
        base = base.join(latest, on="unique_id", how="left")
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Int32).alias("_stress_current_sales_days"))
    base = base.with_columns(
        pl.col("unique_id").str.split("||").list.first().fill_null("?").alias("_stress_section"),
        (
            (pl.col("_seg_level_method_value") == "sku_yoy_seasonal")
            | ((pl.col("_seg_seasonal_multiplier_value") - 1.0).abs() > 0.02)
        ).alias("_stress_seasonal"),
        pl.when(pl.col("_stress_current_sales_days").fill_null(0) <= 9).then(pl.lit("07-09"))
        .when(pl.col("_stress_current_sales_days") <= 13).then(pl.lit("10-13"))
        .when(pl.col("_stress_current_sales_days") <= 20).then(pl.lit("14-20"))
        .otherwise(pl.lit("21-28")).alias("_stress_sales_bucket"),
    )

    _materialized_raw = block_scores.select(pl.col("_validation_block").max()).item() if block_scores.height else 0
    materialized = int(_materialized_raw or 0)
    default = base.select("unique_id").with_columns(
        pl.lit(bool(enabled)).alias("v1295_stress_enabled_value"),
        pl.lit(int(requested), dtype=pl.Int8).alias("v1295_stress_requested_blocks_value"),
        pl.lit(int(materialized), dtype=pl.Int8).alias("v1295_stress_materialized_blocks_value"),
        pl.lit("fallback").alias("v1295_stress_level_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v1295_stress_key_value"),
        pl.lit(0, dtype=pl.Int8).alias("v1295_stress_folds_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_win_rate_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_median_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_p25_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_p10_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_worst_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_std_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_weighted_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_bias_median_worsen_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_bias_p90_worsen_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1295_stress_bias_worst_worsen_value"),
        pl.lit(0, dtype=pl.Int8).alias("v1295_stress_max_consecutive_losses_value"),
        pl.lit(0, dtype=pl.Int32).alias("v1295_stress_min_leaves_value"),
        pl.lit(False).alias("v1295_stress_pass_value"),
        pl.lit("insufficient_long_history").alias("v1295_stress_reason_value"),
    )
    if (not enabled) or block_scores.height == 0 or base.height == 0:
        return default

    hist = (
        block_scores.filter(
            (pl.col("_validation_block") <= requested) & pl.col("_eligible_v")
        )
        .with_columns(
            pl.col("unique_id").str.split("||").list.first().fill_null("?").alias("_stress_section"),
            pl.col("_hist_level_method_value").fill_null("unknown").alias("_stress_level_method"),
            (
                (pl.col("_hist_level_method_value").fill_null("unknown") == "sku_yoy_seasonal")
                | ((pl.col("_hist_seasonal_multiplier_value").fill_null(1.0) - 1.0).abs() > 0.02)
            ).alias("_stress_seasonal"),
            pl.when(pl.col("_n_v") <= 9).then(pl.lit("07-09"))
            .when(pl.col("_n_v") <= 13).then(pl.lit("10-13"))
            .when(pl.col("_n_v") <= 20).then(pl.lit("14-20"))
            .otherwise(pl.lit("21-28")).alias("_stress_sales_bucket"),
        )
    )
    if hist.height == 0:
        return default

    hierarchy = [
        ("L4", ["_stress_section", "_stress_level_method", "_stress_seasonal", "_stress_sales_bucket"],
         int(getattr(settings, "V1293_VALUE_L4_MIN_LEAVES", 20))),
        ("L3", ["_stress_section", "_stress_level_method", "_stress_seasonal"],
         int(getattr(settings, "V1293_VALUE_L3_MIN_LEAVES", 40))),
        ("L2", ["_stress_section", "_stress_level_method"],
         int(getattr(settings, "V1293_VALUE_L2_MIN_LEAVES", 80))),
        ("L1", ["_stress_section"], int(getattr(settings, "V1293_VALUE_L1_MIN_LEAVES", 500))),
    ]

    result = default
    assigned: set[str] = set()
    current = base.rename({"_seg_level_method_value": "_stress_level_method"})
    for level_name, keys, min_leaves in hierarchy:
        fold = (
            hist.group_by(keys + ["_validation_block"])
            .agg(
                pl.col("_den_v").sum().alias("_den"),
                pl.col("_ae11_v").sum().alias("_ae11"),
                pl.col("_ae12_v").sum().alias("_ae12"),
                pl.col("_se11_v").sum().alias("_se11"),
                pl.col("_se12_v").sum().alias("_se12"),
                pl.col("unique_id").n_unique().cast(pl.Int32).alias("_n_leaves"),
            )
            .filter(pl.col("_den") > 1e-9)
            .with_columns(
                ((pl.col("_ae11") - pl.col("_ae12")) / pl.col("_den")).alias("_gain"),
                ((pl.col("_se12") / pl.col("_den")).abs() - (pl.col("_se11") / pl.col("_den")).abs()).alias("_bias_worsen"),
            )
        )
        if fold.height == 0:
            continue

        # Consecutive losses are computed on validation_block order, where 1 is
        # the most recent closed block and larger numbers move backward in time.
        loss_run_map: dict[tuple, int] = {}
        for row in fold.sort(keys + ["_validation_block"]).select(*(keys + ["_gain"])).iter_rows(named=True):
            k = tuple(row[x] for x in keys)
            prev_max, run = loss_run_map.get(k, (0, 0)) if k in loss_run_map else (0, 0)
            if float(row["_gain"] or 0.0) <= 0.0:
                run += 1
            else:
                run = 0
            loss_run_map[k] = (max(prev_max, run), run)
        loss_rows = []
        for k, (mx, _run) in loss_run_map.items():
            rec = {keys[i]: k[i] for i in range(len(keys))}
            rec["_max_loss_run"] = int(mx)
            loss_rows.append(rec)
        loss_df = pl.DataFrame(loss_rows) if loss_rows else pl.DataFrame()

        stats = (
            fold.group_by(keys)
            .agg(
                pl.col("_validation_block").n_unique().cast(pl.Int8).alias("_folds"),
                (pl.col("_gain") > 0.0).cast(pl.Float64).mean().alias("_win_rate"),
                pl.col("_gain").median().alias("_median_gain"),
                pl.col("_gain").quantile(0.25, interpolation="linear").alias("_p25_gain"),
                pl.col("_gain").quantile(0.10, interpolation="linear").alias("_p10_gain"),
                pl.col("_gain").min().alias("_worst_gain"),
                pl.col("_gain").std(ddof=0).alias("_std_gain"),
                pl.col("_bias_worsen").median().alias("_bias_median"),
                pl.col("_bias_worsen").quantile(0.90, interpolation="linear").alias("_bias_p90"),
                pl.col("_bias_worsen").max().alias("_bias_worst"),
                pl.col("_n_leaves").min().cast(pl.Int32).alias("_min_leaves"),
                pl.col("_den").sum().alias("_den_all"),
                pl.col("_ae11").sum().alias("_ae11_all"),
                pl.col("_ae12").sum().alias("_ae12_all"),
            )
            .with_columns(((pl.col("_ae11_all") - pl.col("_ae12_all")) / pl.col("_den_all")).alias("_weighted_gain"))
        )
        if loss_df.height:
            stats = stats.join(loss_df, on=keys, how="left")
        else:
            stats = stats.with_columns(pl.lit(0, dtype=pl.Int8).alias("_max_loss_run"))
        stats = stats.with_columns(
            pl.col("_max_loss_run").fill_null(0).cast(pl.Int8),
            (
                (pl.col("_folds") >= min_folds)
                & (pl.col("_win_rate") >= min_win)
                & (pl.col("_median_gain") > min_med)
                & (pl.col("_p25_gain") > min_p25)
                & (pl.col("_p10_gain") >= min_p10)
                & (pl.col("_bias_p90") <= max_bias_p90)
                & (pl.col("_max_loss_run") <= max_loss_run)
                & (pl.col("_min_leaves") >= min_leaves)
            ).alias("_stress_pass")
        )
        cand = current.join(stats, on=keys, how="left").filter(
            (pl.col("_folds").fill_null(0) >= min_folds)
            & (pl.col("_min_leaves").fill_null(0) >= min_leaves)
        )
        if assigned:
            cand = cand.filter(~pl.col("unique_id").is_in(list(assigned)))
        if cand.height == 0:
            continue
        ids = cand.get_column("unique_id").to_list()
        assigned.update(ids)
        key_expr = pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|")
        meta = cand.select(
            "unique_id",
            pl.lit(level_name).alias("_stress_new_level"),
            key_expr.alias("_stress_new_key"),
            pl.col("_folds").alias("_stress_new_folds"),
            pl.col("_win_rate").alias("_stress_new_win"),
            pl.col("_median_gain").alias("_stress_new_med"),
            pl.col("_p25_gain").alias("_stress_new_p25"),
            pl.col("_p10_gain").alias("_stress_new_p10"),
            pl.col("_worst_gain").alias("_stress_new_worst"),
            pl.col("_std_gain").alias("_stress_new_std"),
            pl.col("_weighted_gain").alias("_stress_new_wg"),
            pl.col("_bias_median").alias("_stress_new_bmed"),
            pl.col("_bias_p90").alias("_stress_new_bp90"),
            pl.col("_bias_worst").alias("_stress_new_bworst"),
            pl.col("_max_loss_run").alias("_stress_new_lossrun"),
            pl.col("_min_leaves").alias("_stress_new_n"),
            pl.col("_stress_pass").alias("_stress_new_pass"),
        )
        result = result.join(meta, on="unique_id", how="left").with_columns(
            pl.coalesce([pl.col("_stress_new_level"), pl.col("v1295_stress_level_value")]).alias("v1295_stress_level_value"),
            pl.coalesce([pl.col("_stress_new_key"), pl.col("v1295_stress_key_value")]).alias("v1295_stress_key_value"),
            pl.coalesce([pl.col("_stress_new_folds"), pl.col("v1295_stress_folds_value")]).cast(pl.Int8).alias("v1295_stress_folds_value"),
            pl.coalesce([pl.col("_stress_new_win"), pl.col("v1295_stress_win_rate_value")]).alias("v1295_stress_win_rate_value"),
            pl.coalesce([pl.col("_stress_new_med"), pl.col("v1295_stress_median_gain_value")]).alias("v1295_stress_median_gain_value"),
            pl.coalesce([pl.col("_stress_new_p25"), pl.col("v1295_stress_p25_gain_value")]).alias("v1295_stress_p25_gain_value"),
            pl.coalesce([pl.col("_stress_new_p10"), pl.col("v1295_stress_p10_gain_value")]).alias("v1295_stress_p10_gain_value"),
            pl.coalesce([pl.col("_stress_new_worst"), pl.col("v1295_stress_worst_gain_value")]).alias("v1295_stress_worst_gain_value"),
            pl.coalesce([pl.col("_stress_new_std"), pl.col("v1295_stress_std_gain_value")]).alias("v1295_stress_std_gain_value"),
            pl.coalesce([pl.col("_stress_new_wg"), pl.col("v1295_stress_weighted_gain_value")]).alias("v1295_stress_weighted_gain_value"),
            pl.coalesce([pl.col("_stress_new_bmed"), pl.col("v1295_stress_bias_median_worsen_value")]).alias("v1295_stress_bias_median_worsen_value"),
            pl.coalesce([pl.col("_stress_new_bp90"), pl.col("v1295_stress_bias_p90_worsen_value")]).alias("v1295_stress_bias_p90_worsen_value"),
            pl.coalesce([pl.col("_stress_new_bworst"), pl.col("v1295_stress_bias_worst_worsen_value")]).alias("v1295_stress_bias_worst_worsen_value"),
            pl.coalesce([pl.col("_stress_new_lossrun"), pl.col("v1295_stress_max_consecutive_losses_value")]).cast(pl.Int8).alias("v1295_stress_max_consecutive_losses_value"),
            pl.coalesce([pl.col("_stress_new_n"), pl.col("v1295_stress_min_leaves_value")]).cast(pl.Int32).alias("v1295_stress_min_leaves_value"),
            pl.coalesce([pl.col("_stress_new_pass"), pl.col("v1295_stress_pass_value")]).fill_null(False).alias("v1295_stress_pass_value"),
            pl.when(pl.col("_stress_new_level").is_not_null() & pl.col("_stress_new_pass").fill_null(False))
            .then(pl.lit("long_horizon_stable"))
            .when(pl.col("_stress_new_level").is_not_null())
            .then(pl.lit("long_horizon_tail_risk"))
            .otherwise(pl.col("v1295_stress_reason_value")).alias("v1295_stress_reason_value"),
        ).drop([
            "_stress_new_level", "_stress_new_key", "_stress_new_folds", "_stress_new_win",
            "_stress_new_med", "_stress_new_p25", "_stress_new_p10", "_stress_new_worst",
            "_stress_new_std", "_stress_new_wg", "_stress_new_bmed", "_stress_new_bp90",
            "_stress_new_bworst", "_stress_new_lossrun", "_stress_new_n", "_stress_new_pass",
        ], strict=False)

    return result


def _apply_v1296_sec23_long_horizon_gate(
    selection: pl.DataFrame,
    scoring_blocks: pl.DataFrame,
) -> pl.DataFrame:
    """v12.9.6 production gate for Value Sec.23 using long-horizon stress evidence.

    Sec.23 starts from v11. A leaf may move to v12 only when its most-specific
    available stress level is L4, or L3 solely because L4 lacked sufficient
    evidence. L2/L1 never authorize Sec.23 production. A sufficiently evidenced
    FAIL is terminal: broader evidence cannot overwrite it.
    """
    enabled = bool(getattr(settings, "V1296_SEC23_STRESS_GATE_ENABLED", True)) and int(
        getattr(settings, "RLS_BLOCK_DAYS", 28)
    ) == 28
    min_folds = int(getattr(settings, "V1296_SEC23_MIN_FOLDS", 8))
    min_win = float(getattr(settings, "V1296_SEC23_MIN_WIN_RATE", 0.75))
    min_med = float(getattr(settings, "V1296_SEC23_MIN_MEDIAN_GAIN", 0.0))
    min_p25 = float(getattr(settings, "V1296_SEC23_MIN_P25_GAIN", 0.0))
    min_p10 = float(getattr(settings, "V1296_SEC23_MIN_P10_GAIN", -0.01))
    max_bias_p90 = float(getattr(settings, "V1296_SEC23_MAX_BIAS_P90_WORSEN", 0.015))
    max_loss_run = int(getattr(settings, "V1296_SEC23_MAX_CONSECUTIVE_LOSSES", 2))
    allow_l3 = bool(getattr(settings, "V1296_SEC23_ALLOW_L3_BACKOFF", True))
    budget_share = float(getattr(settings, "V1296_SEC23_MAX_VOLUME_SHARE", 0.15))
    top05 = float(getattr(settings, "V1296_SEC23_TOP05_PERCENTILE", 0.95))
    top01 = float(getattr(settings, "V1296_SEC23_TOP01_PERCENTILE", 0.99))
    high_p10 = float(getattr(settings, "V1296_SEC23_HIGH_IMPACT_MIN_P10_GAIN", 0.0))

    out = selection.with_columns(
        pl.lit(bool(enabled)).alias("v1296_stress_gate_enabled_value"),
        pl.lit(False).alias("v1296_stress_eligible_value"),
        pl.lit(False).alias("v1296_stress_gate_pass_value"),
        pl.lit(False).alias("v1296_budget_selected_value"),
        pl.lit(False).alias("v1296_high_impact_guard_pass_value"),
        pl.lit("not_sec23_or_disabled").alias("v1296_reason_value"),
        pl.lit(float(budget_share)).alias("v1296_budget_value"),
        pl.lit(0.0).alias("v1296_selected_volume_share_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1296_segment_score_value"),
    )
    if (not enabled) or selection.height == 0:
        return out

    sec23 = out.filter(pl.col("unique_id").str.starts_with("23||"))
    if sec23.height == 0:
        return out

    latest = (
        scoring_blocks.filter(pl.col("_validation_block") == 1)
        .select("unique_id", pl.col("_den_v").fill_null(0.0).alias("_v1296_volume"))
        .unique("unique_id")
    )
    q = sec23.join(latest, on="unique_id", how="left").with_columns(
        pl.col("_v1296_volume").fill_null(0.0),
        pl.col("v1294_impact_percentile_value").fill_null(0.0).alias("_impact_pct"),
    )
    allowed_level = (
        (pl.col("v1295_stress_level_value") == "L4")
        | (pl.lit(allow_l3) & (pl.col("v1295_stress_level_value") == "L3"))
    )
    sufficient = (
        (pl.col("v1295_stress_folds_value").fill_null(0) >= min_folds)
        & (pl.col("v1295_stress_min_leaves_value").fill_null(0) > 0)
    )
    normal_gate = (
        allowed_level
        & sufficient
        & (pl.col("v1295_stress_win_rate_value").fill_null(0.0) >= min_win)
        & (pl.col("v1295_stress_median_gain_value").fill_null(-999.0) > min_med)
        & (pl.col("v1295_stress_p25_gain_value").fill_null(-999.0) > min_p25)
        & (pl.col("v1295_stress_p10_gain_value").fill_null(-999.0) >= min_p10)
        & (pl.col("v1295_stress_bias_p90_worsen_value").fill_null(999.0) <= max_bias_p90)
        & (pl.col("v1295_stress_max_consecutive_losses_value").fill_null(99) <= max_loss_run)
    )
    high_guard = (
        pl.when(pl.col("_impact_pct") >= top01)
        .then(
            (pl.col("v1295_stress_p25_gain_value").fill_null(-999.0) > 0.0)
            & (pl.col("v1295_stress_p10_gain_value").fill_null(-999.0) >= high_p10)
        )
        .when(pl.col("_impact_pct") >= top05)
        .then(pl.col("v1295_stress_p10_gain_value").fill_null(-999.0) >= high_p10)
        .otherwise(True)
    )
    q = q.with_columns(
        allowed_level.alias("_allowed_level"),
        sufficient.alias("_sufficient"),
        normal_gate.alias("_normal_gate"),
        high_guard.alias("_high_guard"),
    ).with_columns(
        (pl.col("_normal_gate") & pl.col("_high_guard")).alias("_eligible"),
        (
            pl.col("v1295_stress_p25_gain_value").fill_null(-1.0)
            + 0.50 * pl.col("v1295_stress_p10_gain_value").fill_null(-1.0)
            + 0.25 * pl.col("v1295_stress_median_gain_value").fill_null(-1.0)
            + 0.25 * pl.col("v1295_stress_weighted_gain_value").fill_null(-1.0)
        ).alias("_segment_score"),
        pl.when(~pl.col("_allowed_level"))
        .then(pl.lit("v11_broad_level_not_productive"))
        .when(~pl.col("_sufficient"))
        .then(pl.lit("v11_insufficient_specific_history"))
        .when(~pl.col("_normal_gate"))
        .then(pl.lit("v11_long_horizon_tail_risk"))
        .when(~pl.col("_high_guard"))
        .then(pl.lit("v11_high_impact_tail_guard"))
        .otherwise(pl.lit("v12_long_horizon_specific_pass"))
        .alias("_reason"),
    )

    total_w = float(q.select(pl.col("_v1296_volume").sum()).item() or 0.0)
    cap_w = max(0.0, budget_share) * total_w
    eligible = q.filter(pl.col("_eligible"))
    selected_ids: list[str] = []
    if eligible.height and cap_w > 0.0:
        segs = (
            eligible.group_by(["v1295_stress_level_value", "v1295_stress_key_value"])
            .agg(
                pl.col("_v1296_volume").sum().alias("_seg_w"),
                pl.col("_segment_score").first().alias("_score"),
            )
            .sort(["_score", "_seg_w"], descending=[True, False])
        )
        used = 0.0
        for sr in segs.iter_rows(named=True):
            seg_w = float(sr["_seg_w"] or 0.0)
            if used + seg_w > cap_w + 1e-9:
                continue
            part = eligible.filter(
                (pl.col("v1295_stress_level_value") == sr["v1295_stress_level_value"])
                & (pl.col("v1295_stress_key_value") == sr["v1295_stress_key_value"])
            )
            selected_ids.extend(str(x) for x in part.get_column("unique_id").to_list())
            used += seg_w

    selected_df = (
        pl.DataFrame({"unique_id": selected_ids}, schema={"unique_id": pl.Utf8})
        .unique("unique_id")
        .with_columns(pl.lit(True).alias("_v1296_selected"))
        if selected_ids else pl.DataFrame(schema={"unique_id": pl.Utf8, "_v1296_selected": pl.Boolean})
    )
    q = q.join(selected_df, on="unique_id", how="left").with_columns(
        pl.col("_v1296_selected").fill_null(False)
    ).with_columns(
        pl.when(pl.col("_eligible") & ~pl.col("_v1296_selected"))
        .then(pl.lit("v11_stress_pass_budget_safety"))
        .otherwise(pl.col("_reason"))
        .alias("_reason")
    )
    selected_w = float(
        q.select(pl.when(pl.col("_v1296_selected")).then(pl.col("_v1296_volume")).otherwise(0.0).sum()).item() or 0.0
    )
    selected_share = selected_w / total_w if total_w > 1e-9 else 0.0
    meta = q.select(
        "unique_id",
        pl.col("_eligible").alias("_v1296_eligible"),
        pl.col("_normal_gate").alias("_v1296_gate"),
        pl.col("_v1296_selected"),
        pl.col("_high_guard").alias("_v1296_high_guard"),
        pl.col("_reason").alias("_v1296_reason"),
        pl.col("_segment_score").alias("_v1296_score"),
    )
    out = out.join(meta, on="unique_id", how="left").with_columns(
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_eligible").fill_null(False))
        .otherwise(pl.col("v1296_stress_eligible_value"))
        .alias("v1296_stress_eligible_value"),
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_gate").fill_null(False))
        .otherwise(pl.col("v1296_stress_gate_pass_value"))
        .alias("v1296_stress_gate_pass_value"),
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_selected").fill_null(False))
        .otherwise(pl.col("v1296_budget_selected_value"))
        .alias("v1296_budget_selected_value"),
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_high_guard").fill_null(False))
        .otherwise(pl.col("v1296_high_impact_guard_pass_value"))
        .alias("v1296_high_impact_guard_pass_value"),
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_reason").fill_null("v11_no_long_horizon_evidence"))
        .otherwise(pl.col("v1296_reason_value"))
        .alias("v1296_reason_value"),
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.lit(float(selected_share)))
        .otherwise(pl.col("v1296_selected_volume_share_value"))
        .alias("v1296_selected_volume_share_value"),
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_score"))
        .otherwise(pl.col("v1296_segment_score_value"))
        .alias("v1296_segment_score_value"),
        # The production family is overridden only in Sec.23. Sec.1 remains
        # byte-for-byte on the v12.9.3/5 decision path.
        pl.when(pl.col("unique_id").str.starts_with("23||"))
        .then(pl.col("_v1296_selected").fill_null(False))
        .otherwise(pl.col("v12_selected_value"))
        .alias("v12_selected_value"),
    ).drop([
        "_v1296_eligible", "_v1296_gate", "_v1296_selected", "_v1296_high_guard",
        "_v1296_reason", "_v1296_score"
    ], strict=False)
    return out



def _v1297_sec23_temporal_replay(
    block_scores: pl.DataFrame,
    target_profile: pl.DataFrame,
) -> pl.DataFrame:
    """Causal multi-cutoff replay of the Sec23 Value v12.9.6 production rule.

    Diagnostic only: it never changes ``v12_selected_value``.  Validation block
    1 is the most recent closed historical block; larger block numbers are
    older.  For pseudo-OOS cutoff ``b`` every decision input comes strictly from
    blocks ``b+1..N``.  In particular, L4 membership, impact and budget weights
    are taken from block ``b+1`` (the most recent *already closed* block), never
    from the pseudo-OOS block ``b``.  The actuals in block ``b`` are used only
    after selection to score generalization.
    """
    enabled = bool(getattr(settings, "V1297_TEMPORAL_REPLAY_ENABLED", True)) and int(
        getattr(settings, "RLS_BLOCK_DAYS", 28)
    ) == 28
    requested = max(1, int(getattr(settings, "V1297_TEMPORAL_REPLAY_MAX_CUTOFFS", 4)))
    min_train = max(3, int(getattr(settings, "V1297_TEMPORAL_REPLAY_MIN_TRAIN_FOLDS", 8)))
    replay_min_win = float(getattr(settings, "V1297_TEMPORAL_REPLAY_MIN_WIN_RATE", 0.75))
    replay_min_med = float(getattr(settings, "V1297_TEMPORAL_REPLAY_MIN_MEDIAN_GAIN", 0.0))
    replay_min_worst = float(getattr(settings, "V1297_TEMPORAL_REPLAY_MIN_WORST_GAIN", -0.005))
    replay_max_bias = float(getattr(settings, "V1297_TEMPORAL_REPLAY_MAX_BIAS_WORSEN", 0.01))

    # Replay exactly the productive v12.9.6 Sec23 L4 gate.
    min_win = float(getattr(settings, "V1296_SEC23_MIN_WIN_RATE", 0.75))
    min_med = float(getattr(settings, "V1296_SEC23_MIN_MEDIAN_GAIN", 0.0))
    min_p25 = float(getattr(settings, "V1296_SEC23_MIN_P25_GAIN", 0.0))
    min_p10 = float(getattr(settings, "V1296_SEC23_MIN_P10_GAIN", -0.01))
    max_bias_p90 = float(getattr(settings, "V1296_SEC23_MAX_BIAS_P90_WORSEN", 0.015))
    max_loss_run = int(getattr(settings, "V1296_SEC23_MAX_CONSECUTIVE_LOSSES", 2))
    budget_share = float(getattr(settings, "V1296_SEC23_MAX_VOLUME_SHARE", 0.15))
    top05 = float(getattr(settings, "V1296_SEC23_TOP05_PERCENTILE", 0.95))
    top01 = float(getattr(settings, "V1296_SEC23_TOP01_PERCENTILE", 0.99))
    high_p10 = float(getattr(settings, "V1296_SEC23_HIGH_IMPACT_MIN_P10_GAIN", 0.0))
    min_leaves = int(getattr(settings, "V1293_VALUE_L4_MIN_LEAVES", 20))

    base = target_profile.select("unique_id").unique("unique_id")
    default = base.with_columns(
        pl.lit(bool(enabled)).alias("v1297_temporal_replay_enabled_value"),
        pl.lit(0, dtype=pl.Int8).alias("v1297_temporal_replay_cutoffs_value"),
        pl.lit(0.0).alias("v1297_temporal_replay_win_rate_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1297_temporal_replay_median_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1297_temporal_replay_worst_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1297_temporal_replay_weighted_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1297_temporal_replay_bias_worst_value"),
        # 0.0 is a valid neutral value when the diagnostic cannot materialize;
        # validators distinguish that case with cutoffs==0.
        pl.lit(0.0).alias("v1297_temporal_replay_selected_share_median_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v1297_temporal_replay_fold_gains_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v1297_temporal_replay_fold_shares_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v1297_temporal_replay_fold_bias_value"),
        pl.lit(False).alias("v1297_temporal_replay_pass_value"),
    )
    if (not enabled) or block_scores.height == 0 or base.height == 0:
        return default

    hist = (
        block_scores.filter(
            pl.col("_eligible_v")
            & pl.col("unique_id").str.starts_with("23||")
            & pl.col("unique_id").str.contains("||T:", literal=True)
        )
        .with_columns(
            pl.col("_hist_level_method_value").fill_null("unknown").alias("_rp_method"),
            (
                (pl.col("_hist_level_method_value").fill_null("unknown") == "sku_yoy_seasonal")
                | ((pl.col("_hist_seasonal_multiplier_value").fill_null(1.0) - 1.0).abs() > 0.02)
            ).alias("_rp_seasonal"),
            pl.when(pl.col("_n_v") <= 9).then(pl.lit("07-09"))
            .when(pl.col("_n_v") <= 13).then(pl.lit("10-13"))
            .when(pl.col("_n_v") <= 20).then(pl.lit("14-20"))
            .otherwise(pl.lit("21-28")).alias("_rp_bucket"),
            pl.lit("23").alias("_rp_section"),
        )
    )
    if hist.height == 0:
        return default

    max_block = int(hist.select(pl.col("_validation_block").max()).item() or 0)
    n_cutoffs = min(requested, max(0, max_block - min_train))
    if n_cutoffs <= 0:
        return default

    keys = ["_rp_section", "_rp_method", "_rp_seasonal", "_rp_bucket"]
    fold_results: list[dict[str, float]] = []

    for cutoff in range(1, n_cutoffs + 1):
        # Strict causality: train and current profile are both older than the
        # pseudo-OOS validation block.  Never use block `cutoff` to construct a
        # segment, impact percentile or budget weight.
        train = hist.filter(pl.col("_validation_block") > cutoff)
        prior = hist.filter(pl.col("_validation_block") == cutoff + 1)
        valid = hist.filter(pl.col("_validation_block") == cutoff)
        if train.height == 0 or prior.height == 0 or valid.height == 0:
            continue

        fold = (
            train.group_by(keys + ["_validation_block"])
            .agg(
                pl.col("_den_v").sum().alias("_den"),
                pl.col("_ae11_v").sum().alias("_ae11"),
                pl.col("_ae12_v").sum().alias("_ae12"),
                pl.col("_se11_v").sum().alias("_se11"),
                pl.col("_se12_v").sum().alias("_se12"),
                pl.col("unique_id").n_unique().cast(pl.Int32).alias("_n_leaves"),
            )
            .filter(pl.col("_den") > 1e-9)
            .with_columns(
                ((pl.col("_ae11") - pl.col("_ae12")) / pl.col("_den")).alias("_gain"),
                ((pl.col("_se12") / pl.col("_den")).abs() - (pl.col("_se11") / pl.col("_den")).abs()).alias("_bias_worsen"),
            )
        )
        if fold.height == 0:
            continue

        loss_state: dict[tuple, tuple[int, int]] = {}
        for row in fold.sort(keys + ["_validation_block"]).select(*(keys + ["_gain"])).iter_rows(named=True):
            k = tuple(row[x] for x in keys)
            mx, run = loss_state.get(k, (0, 0))
            if float(row["_gain"] or 0.0) <= 0.0:
                run += 1
            else:
                run = 0
            loss_state[k] = (max(mx, run), run)
        loss_rows: list[dict] = []
        for k, (mx, _run) in loss_state.items():
            rec = {keys[i]: k[i] for i in range(len(keys))}
            rec["_loss_run"] = int(mx)
            loss_rows.append(rec)
        loss_df = pl.DataFrame(loss_rows) if loss_rows else pl.DataFrame()

        stats = (
            fold.group_by(keys)
            .agg(
                pl.col("_validation_block").n_unique().cast(pl.Int8).alias("_folds"),
                (pl.col("_gain") > 0.0).cast(pl.Float64).mean().alias("_win"),
                pl.col("_gain").median().alias("_med"),
                pl.col("_gain").quantile(0.25, interpolation="linear").alias("_p25"),
                pl.col("_gain").quantile(0.10, interpolation="linear").alias("_p10"),
                pl.col("_bias_worsen").quantile(0.90, interpolation="linear").alias("_bp90"),
                pl.col("_n_leaves").min().cast(pl.Int32).alias("_min_n"),
                pl.col("_den").sum().alias("_den_all"),
                pl.col("_ae11").sum().alias("_ae11_all"),
                pl.col("_ae12").sum().alias("_ae12_all"),
            )
            .with_columns(((pl.col("_ae11_all") - pl.col("_ae12_all")) / pl.col("_den_all")).alias("_wg"))
        )
        if loss_df.height:
            stats = stats.join(loss_df, on=keys, how="left")
        else:
            stats = stats.with_columns(pl.lit(0).alias("_loss_run"))
        stats = stats.with_columns(pl.col("_loss_run").fill_null(0))

        # Pseudo-current leaf state/volume comes from the immediately preceding
        # closed block (cutoff+1), not the validation block itself.
        profile = (
            prior.select(
                "unique_id", *keys,
                pl.col("_den_v").fill_null(0.0).alias("_rp_weight"),
            )
            .unique("unique_id")
            .join(stats, on=keys, how="left")
        )
        if profile.height == 0:
            continue
        n_profile = max(1, profile.height)
        profile = profile.with_columns(
            (pl.col("_rp_weight").rank(method="ordinal", descending=False).cast(pl.Float64) / float(n_profile)).alias("_impact_pct")
        )
        normal_gate = (
            (pl.col("_folds").fill_null(0) >= min_train)
            & (pl.col("_min_n").fill_null(0) >= min_leaves)
            & (pl.col("_win").fill_null(0.0) >= min_win)
            & (pl.col("_med").fill_null(-999.0) > min_med)
            & (pl.col("_p25").fill_null(-999.0) > min_p25)
            & (pl.col("_p10").fill_null(-999.0) >= min_p10)
            & (pl.col("_bp90").fill_null(999.0) <= max_bias_p90)
            & (pl.col("_loss_run").fill_null(99) <= max_loss_run)
        )
        high_guard = (
            pl.when(pl.col("_impact_pct") >= top01)
            .then((pl.col("_p25").fill_null(-999.0) > 0.0) & (pl.col("_p10").fill_null(-999.0) >= high_p10))
            .when(pl.col("_impact_pct") >= top05)
            .then(pl.col("_p10").fill_null(-999.0) >= high_p10)
            .otherwise(True)
        )
        profile = profile.with_columns(
            (normal_gate & high_guard).alias("_eligible"),
            (
                pl.col("_p25").fill_null(-1.0)
                + 0.50 * pl.col("_p10").fill_null(-1.0)
                + 0.25 * pl.col("_med").fill_null(-1.0)
                + 0.25 * pl.col("_wg").fill_null(-1.0)
            ).alias("_score"),
        )

        total_prior_w = float(profile.select(pl.col("_rp_weight").sum()).item() or 0.0)
        cap = max(0.0, budget_share) * total_prior_w
        eligible = profile.filter(pl.col("_eligible"))
        selected_ids: list[str] = []
        used = 0.0
        if eligible.height and cap > 0.0:
            segs = (
                eligible.group_by(keys)
                .agg(pl.col("_rp_weight").sum().alias("_seg_w"), pl.col("_score").first().alias("_score"))
                .sort(["_score", "_seg_w"], descending=[True, False])
            )
            for sr in segs.iter_rows(named=True):
                seg_w = float(sr["_seg_w"] or 0.0)
                if used + seg_w > cap + 1e-9:
                    continue
                mask = None
                for k in keys:
                    cond = pl.col(k) == sr[k]
                    mask = cond if mask is None else (mask & cond)
                part = eligible.filter(mask)
                selected_ids.extend(str(x) for x in part.get_column("unique_id").to_list())
                used += seg_w

        # Evaluation only starts here: pseudo-OOS actuals never influenced the
        # decision above.
        q = valid
        total_den = float(q.select(pl.col("_den_v").sum()).item() or 0.0)
        if total_den <= 1e-9:
            continue
        if selected_ids:
            selected = q.filter(pl.col("unique_id").is_in(selected_ids))
            improvement_num = float(selected.select((pl.col("_ae11_v") - pl.col("_ae12_v")).sum()).item() or 0.0)
            selected_se11 = float(selected.select(pl.col("_se11_v").sum()).item() or 0.0)
            selected_se12 = float(selected.select(pl.col("_se12_v").sum()).item() or 0.0)
        else:
            improvement_num = selected_se11 = selected_se12 = 0.0
        all_se11 = float(q.select(pl.col("_se11_v").sum()).item() or 0.0)
        final_se = all_se11 - selected_se11 + selected_se12
        gain = improvement_num / total_den
        bias_worsen = abs(final_se / total_den) - abs(all_se11 / total_den)
        fold_results.append({
            "cutoff": float(cutoff),
            "gain": float(gain),
            "share": float(used / total_prior_w) if total_prior_w > 1e-9 else 0.0,
            "bias": float(bias_worsen),
            "den": float(total_den),
            "improvement_num": float(improvement_num),
        })

    if not fold_results:
        return default
    gains = [x["gain"] for x in fold_results]
    shares = [x["share"] for x in fold_results]
    biases = [x["bias"] for x in fold_results]
    wins = sum(1 for x in gains if x > 0.0)
    cutoffs = len(gains)
    gains_sorted = sorted(gains)
    median_gain = gains_sorted[len(gains_sorted)//2] if len(gains_sorted) % 2 else 0.5 * (gains_sorted[len(gains_sorted)//2-1] + gains_sorted[len(gains_sorted)//2])
    shares_sorted = sorted(shares)
    med_share = shares_sorted[len(shares_sorted)//2] if len(shares_sorted) % 2 else 0.5 * (shares_sorted[len(shares_sorted)//2-1] + shares_sorted[len(shares_sorted)//2])
    total_den = sum(x["den"] for x in fold_results)
    weighted_gain = sum(x["improvement_num"] for x in fold_results) / total_den if total_den > 1e-9 else 0.0
    worst_gain = min(gains)
    worst_bias = max(biases)
    replay_pass = (
        cutoffs >= min(3, requested)
        and wins / cutoffs >= replay_min_win
        and median_gain > replay_min_med
        and worst_gain >= replay_min_worst
        and worst_bias <= replay_max_bias
    )
    fmt = lambda xs: ",".join(f"{100.0*x:.2f}%" for x in xs)
    return default.with_columns(
        pl.lit(int(cutoffs), dtype=pl.Int8).alias("v1297_temporal_replay_cutoffs_value"),
        pl.lit(float(wins / cutoffs)).alias("v1297_temporal_replay_win_rate_value"),
        pl.lit(float(median_gain)).alias("v1297_temporal_replay_median_gain_value"),
        pl.lit(float(worst_gain)).alias("v1297_temporal_replay_worst_gain_value"),
        pl.lit(float(weighted_gain)).alias("v1297_temporal_replay_weighted_gain_value"),
        pl.lit(float(worst_bias)).alias("v1297_temporal_replay_bias_worst_value"),
        pl.lit(float(med_share)).alias("v1297_temporal_replay_selected_share_median_value"),
        pl.lit(fmt(gains)).alias("v1297_temporal_replay_fold_gains_value"),
        pl.lit(fmt(shares)).alias("v1297_temporal_replay_fold_shares_value"),
        pl.lit(fmt(biases)).alias("v1297_temporal_replay_fold_bias_value"),
        pl.lit(bool(replay_pass)).alias("v1297_temporal_replay_pass_value"),
    )


def _v1298_sparse_bias_diagnostic(
    history_scores: pl.DataFrame,
    context_scores: pl.DataFrame,
    selections: pl.DataFrame,
) -> pl.DataFrame:
    """Causal v12.9.8 sparse-demand bias-rescue metadata.

    Diagnostic only.  For each target (Qty/Value), learn one robust multiplicative
    ratio per section × sparse sales-frequency bucket (07-09 or 10-13) from
    CLOSED 28-day blocks.  The historical forecast used in each row follows the
    family selected for the target period (v11 incumbent or v12 challenger).
    The latest closed block assigns the current leaf bucket.  No target-period
    actual enters factor construction.
    """
    enabled = bool(getattr(settings, "V1298_SPARSE_BIAS_DIAGNOSTIC_ENABLED", True))
    min_folds = int(getattr(settings, "V1298_SPARSE_BIAS_MIN_FOLDS", 8))
    buckets = tuple(getattr(settings, "V1298_SPARSE_BIAS_BUCKETS", ("07-09", "10-13")))
    min_med = float(getattr(settings, "V1298_SPARSE_BIAS_MIN_MEDIAN_RATIO", 1.05))
    min_p25 = float(getattr(settings, "V1298_SPARSE_BIAS_MIN_P25_RATIO", 1.00))
    min_worst = float(getattr(settings, "V1298_SPARSE_BIAS_MIN_WORST_RATIO", 0.90))
    shrink = float(getattr(settings, "V1298_SPARSE_BIAS_SHRINK", 0.50))
    clip_lo, clip_hi = tuple(getattr(settings, "V1298_SPARSE_BIAS_FACTOR_CLIP", (1.0, 1.15)))

    base = selections.select(
        "unique_id",
        pl.col("v12_selected_y").fill_null(False).alias("_v1298_use12_y"),
        pl.col("v12_selected_value").fill_null(False).alias("_v1298_use12_v"),
    )
    defaults = base.with_columns(
        pl.lit(bool(enabled)).alias("v1298_sparse_bias_enabled"),
        pl.lit(None, dtype=pl.Utf8).alias("v1298_sparse_bucket_y"),
        pl.lit(None, dtype=pl.Utf8).alias("v1298_sparse_bucket_value"),
        pl.lit(0, dtype=pl.Int8).alias("v1298_sparse_folds_y"),
        pl.lit(0, dtype=pl.Int8).alias("v1298_sparse_folds_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1298_sparse_median_ratio_y"),
        pl.lit(None, dtype=pl.Float64).alias("v1298_sparse_median_ratio_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1298_sparse_p25_ratio_y"),
        pl.lit(None, dtype=pl.Float64).alias("v1298_sparse_p25_ratio_value"),
        pl.lit(None, dtype=pl.Float64).alias("v1298_sparse_worst_ratio_y"),
        pl.lit(None, dtype=pl.Float64).alias("v1298_sparse_worst_ratio_value"),
        pl.lit(1.0).alias("v1298_sparse_factor_y"),
        pl.lit(1.0).alias("v1298_sparse_factor_value"),
        pl.lit(False).alias("v1298_sparse_candidate_y"),
        pl.lit(False).alias("v1298_sparse_candidate_value"),
    )
    if (not enabled) or history_scores.height == 0 or context_scores.height == 0 or base.height == 0:
        return defaults.drop("_v1298_use12_y", "_v1298_use12_v")

    # Current bucket is taken from the latest CLOSED block available for the
    # target period (historical block-1 for OOS, observed OOS for forecast-only).
    ctx = context_scores.select("unique_id", "_n_y", "_n_v").with_columns(
        pl.when(pl.col("_n_y") <= 9).then(pl.lit("07-09"))
        .when(pl.col("_n_y") <= 13).then(pl.lit("10-13"))
        .when(pl.col("_n_y") <= 20).then(pl.lit("14-20"))
        .otherwise(pl.lit("21-28")).alias("_v1298_bucket_y"),
        pl.when(pl.col("_n_v") <= 9).then(pl.lit("07-09"))
        .when(pl.col("_n_v") <= 13).then(pl.lit("10-13"))
        .when(pl.col("_n_v") <= 20).then(pl.lit("14-20"))
        .otherwise(pl.lit("21-28")).alias("_v1298_bucket_v"),
    )

    hist = history_scores.join(base, on="unique_id", how="inner").with_columns(
        pl.when(pl.col("_n_y") <= 9).then(pl.lit("07-09"))
        .when(pl.col("_n_y") <= 13).then(pl.lit("10-13"))
        .when(pl.col("_n_y") <= 20).then(pl.lit("14-20"))
        .otherwise(pl.lit("21-28")).alias("_v1298_hist_bucket_y"),
        pl.when(pl.col("_n_v") <= 9).then(pl.lit("07-09"))
        .when(pl.col("_n_v") <= 13).then(pl.lit("10-13"))
        .when(pl.col("_n_v") <= 20).then(pl.lit("14-20"))
        .otherwise(pl.lit("21-28")).alias("_v1298_hist_bucket_v"),
        pl.when(pl.col("_v1298_use12_y"))
        .then(pl.col("_den_y") + pl.col("_se12_y"))
        .otherwise(pl.col("_den_y") + pl.col("_se11_y"))
        .clip(lower_bound=0.0).alias("_v1298_fc_y"),
        pl.when(pl.col("_v1298_use12_v"))
        .then(pl.col("_den_v") + pl.col("_se12_v"))
        .otherwise(pl.col("_den_v") + pl.col("_se11_v"))
        .clip(lower_bound=0.0).alias("_v1298_fc_v"),
    )

    def _stats(target: str) -> pl.DataFrame:
        den = f"_den_{'y' if target == 'y' else 'v'}"
        fc = f"_v1298_fc_{'y' if target == 'y' else 'v'}"
        bucket = f"_v1298_hist_bucket_{'y' if target == 'y' else 'v'}"
        # One portfolio ratio per closed block and bucket; aggregating first
        # avoids tiny leaves dominating the robust factor.
        fold = (
            hist.filter(pl.col(bucket).is_in(list(buckets)))
            .group_by([bucket, "_validation_block"])
            .agg(pl.col(den).sum().alias("_den"), pl.col(fc).sum().alias("_fc"))
            .filter((pl.col("_den") > 0) & (pl.col("_fc") > 0))
            .with_columns((pl.col("_den") / pl.col("_fc")).alias("_ratio"))
        )
        if fold.height == 0:
            return pl.DataFrame(schema={
                f"_v1298_bucket_{target}": pl.Utf8,
                f"v1298_sparse_folds_{'y' if target == 'y' else 'value'}": pl.Int8,
            })
        suf = "y" if target == "y" else "value"
        return (
            fold.group_by(bucket)
            .agg(
                pl.len().cast(pl.Int8).alias(f"v1298_sparse_folds_{suf}"),
                pl.col("_ratio").median().alias(f"v1298_sparse_median_ratio_{suf}"),
                pl.col("_ratio").quantile(0.25, interpolation="linear").alias(f"v1298_sparse_p25_ratio_{suf}"),
                pl.col("_ratio").min().alias(f"v1298_sparse_worst_ratio_{suf}"),
            )
            .rename({bucket: f"_v1298_bucket_{target}"})
            .with_columns(
                (
                    pl.lit(1.0)
                    + pl.lit(shrink) * (pl.col(f"v1298_sparse_median_ratio_{suf}") - 1.0)
                ).clip(lower_bound=float(clip_lo), upper_bound=float(clip_hi)).alias(f"v1298_sparse_factor_{suf}"),
            )
            .with_columns(
                (
                    (pl.col(f"v1298_sparse_folds_{suf}") >= min_folds)
                    & (pl.col(f"v1298_sparse_median_ratio_{suf}") >= min_med)
                    & (pl.col(f"v1298_sparse_p25_ratio_{suf}") >= min_p25)
                    & (pl.col(f"v1298_sparse_worst_ratio_{suf}") >= min_worst)
                    & (pl.col(f"v1298_sparse_factor_{suf}") > 1.0)
                ).alias(f"v1298_sparse_candidate_{suf}")
            )
        )

    sy = _stats("y")
    sv = _stats("v")
    out = defaults.join(ctx, on="unique_id", how="left")
    if sy.height:
        out = out.join(sy, left_on="_v1298_bucket_y", right_on="_v1298_bucket_y", how="left", suffix="_newy")
    if sv.height:
        out = out.join(sv, left_on="_v1298_bucket_v", right_on="_v1298_bucket_v", how="left", suffix="_newv")

    # Coalesce joined statistics over defaults and expose the current causal bucket.
    exprs = [
        pl.col("_v1298_bucket_y").alias("v1298_sparse_bucket_y"),
        pl.col("_v1298_bucket_v").alias("v1298_sparse_bucket_value"),
    ]
    for suf, jsuf in (("y", "_newy"), ("value", "_newv")):
        for stem in ("folds", "median_ratio", "p25_ratio", "worst_ratio", "factor", "candidate"):
            name = f"v1298_sparse_{stem}_{suf}"
            alt = name + jsuf
            if alt in out.columns:
                exprs.append(pl.coalesce([pl.col(alt), pl.col(name)]).alias(name))
    out = out.with_columns(*exprs)
    drop_cols = [c for c in out.columns if c.startswith("_v1298_") or c.endswith("_newy") or c.endswith("_newv")]
    return out.drop(*drop_cols)


def _v1299_value_sparse_segmented_diagnostic(
    history_scores: pl.DataFrame,
    context_scores: pl.DataFrame,
    selections: pl.DataFrame,
) -> pl.DataFrame:
    """v12.9.9 value-only segmented sparse-bias rescue + temporal replay.

    Diagnostic only: never changes ``valuehat``.  The rescue is learned from
    strictly CLOSED 28-day blocks on causal segments
    ``sales-days bucket × level-method × seasonal-flag``.  A discrete uplift
    grid is scored with exact positive-day absolute errors precomputed by
    :func:`_score_closed_block`.  The current OOS is excluded from learning.

    A four-cutoff replay repeats the complete exercise causally: each held-out
    historical block is evaluated using only older blocks.  The existing
    v11/v12 family selector is reconstructed at each pseudo-cutoff before the
    rescue factor is learned, so held-out actuals cannot affect either the
    baseline family or the bias-rescue factor.
    """
    enabled = bool(getattr(settings, "V1299_VALUE_SPARSE_DIAGNOSTIC_ENABLED", True))
    buckets = tuple(getattr(settings, "V1299_VALUE_SPARSE_BUCKETS", ("07-09", "10-13")))
    grid = tuple(sorted({float(x) for x in getattr(
        settings, "V1299_VALUE_SPARSE_FACTOR_GRID",
        (1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15),
    )}))
    min_folds = int(getattr(settings, "V1299_VALUE_SPARSE_MIN_FOLDS", 8))
    min_leaves = int(getattr(settings, "V1299_VALUE_SPARSE_MIN_LEAVES_PER_FOLD", 20))
    min_win = float(getattr(settings, "V1299_VALUE_SPARSE_MIN_WIN_RATE", 0.75))
    min_med = float(getattr(settings, "V1299_VALUE_SPARSE_MIN_MEDIAN_GAIN", 0.0))
    min_worst = float(getattr(settings, "V1299_VALUE_SPARSE_MIN_WORST_GAIN", -0.005))
    max_bias = float(getattr(settings, "V1299_VALUE_SPARSE_MAX_BIAS_WORSEN", 0.01))
    replay_cutoffs_req = int(getattr(settings, "V1299_VALUE_SPARSE_REPLAY_CUTOFFS", 4))
    promote_enabled = bool(getattr(settings, "V12910_VALUE_SPARSE_PROMOTION_ENABLED", True))
    active_min = int(getattr(settings, "V12910_VALUE_SPARSE_ACTIVE_MIN_NONZERO_DAYS", getattr(settings, "OOS_ACTIVE_MIN_NONZERO_DAYS", 7)))
    promote_min_cutoffs = int(getattr(settings, "V12910_VALUE_SPARSE_MIN_REPLAY_CUTOFFS", 4))
    promote_min_win = float(getattr(settings, "V12910_VALUE_SPARSE_MIN_REPLAY_WIN_RATE", 1.0))
    promote_min_med = float(getattr(settings, "V12910_VALUE_SPARSE_MIN_REPLAY_MEDIAN_GAIN", 0.0025))
    promote_min_worst = float(getattr(settings, "V12910_VALUE_SPARSE_MIN_REPLAY_WORST_GAIN", 0.0))
    promote_min_weighted = float(getattr(settings, "V12910_VALUE_SPARSE_MIN_REPLAY_WEIGHTED_GAIN", 0.0025))
    promote_max_bias = float(getattr(settings, "V12910_VALUE_SPARSE_MAX_REPLAY_BIAS_WORSEN", 0.0))
    promote_max_share = float(getattr(settings, "V12910_VALUE_SPARSE_MAX_SELECTED_VOLUME_SHARE", 0.30))

    base = selections.select(
        "unique_id",
        pl.col("v12_selected_value").fill_null(False).alias("_v1299_use12"),
    )
    defaults = base.with_columns(
        pl.lit(bool(enabled)).alias("v1299_value_sparse_enabled"),
        pl.lit(None, dtype=pl.Utf8).alias("v1299_value_sparse_bucket"),
        pl.lit(None, dtype=pl.Utf8).alias("v1299_value_sparse_level_method"),
        pl.lit(False).alias("v1299_value_sparse_seasonal"),
        pl.lit(None, dtype=pl.Utf8).alias("v1299_value_sparse_segment"),
        pl.lit(1.0).alias("v1299_value_sparse_factor"),
        pl.lit(0, dtype=pl.Int8).alias("v1299_value_sparse_folds"),
        pl.lit(0.0).alias("v1299_value_sparse_win_rate"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_median_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_worst_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_weighted_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_bias_worst"),
        pl.lit(False).alias("v1299_value_sparse_candidate"),
        pl.lit(0, dtype=pl.Int8).alias("v1299_value_sparse_replay_cutoffs"),
        pl.lit(0.0).alias("v1299_value_sparse_replay_win_rate"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_replay_median_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_replay_worst_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_replay_weighted_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_replay_bias_worst"),
        pl.lit(None, dtype=pl.Float64).alias("v1299_value_sparse_replay_selected_share_median"),
        pl.lit(None, dtype=pl.Utf8).alias("v1299_value_sparse_replay_fold_gains"),
        pl.lit(None, dtype=pl.Utf8).alias("v1299_value_sparse_replay_fold_shares"),
        pl.lit(None, dtype=pl.Utf8).alias("v1299_value_sparse_replay_fold_bias"),
        pl.lit(False).alias("v1299_value_sparse_replay_pass"),
        pl.lit(bool(promote_enabled)).alias("v12910_value_sparse_promotion_enabled"),
        pl.lit(0, dtype=pl.Int8).alias("v12910_value_sparse_active_replay_cutoffs"),
        pl.lit(0.0).alias("v12910_value_sparse_active_replay_win_rate"),
        pl.lit(None, dtype=pl.Float64).alias("v12910_value_sparse_active_replay_median_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v12910_value_sparse_active_replay_worst_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v12910_value_sparse_active_replay_weighted_gain"),
        pl.lit(None, dtype=pl.Float64).alias("v12910_value_sparse_active_replay_bias_worst"),
        pl.lit(None, dtype=pl.Float64).alias("v12910_value_sparse_active_replay_selected_share_median"),
        pl.lit(None, dtype=pl.Utf8).alias("v12910_value_sparse_active_replay_fold_gains"),
        pl.lit(None, dtype=pl.Utf8).alias("v12910_value_sparse_active_replay_fold_shares"),
        pl.lit(None, dtype=pl.Utf8).alias("v12910_value_sparse_active_replay_fold_bias"),
        pl.lit(False).alias("v12910_value_sparse_promoted"),
        pl.lit("active_replay_not_available").alias("v12910_value_sparse_promotion_reason"),
    )
    if (not enabled) or history_scores.height == 0 or context_scores.height == 0 or base.height == 0:
        return defaults.drop("_v1299_use12")

    def _profile(src: pl.DataFrame) -> pl.DataFrame:
        return src.select(
            "unique_id",
            pl.when(pl.col("_n_v") <= 9).then(pl.lit("07-09"))
            .when(pl.col("_n_v") <= 13).then(pl.lit("10-13"))
            .when(pl.col("_n_v") <= 20).then(pl.lit("14-20"))
            .otherwise(pl.lit("21-28")).alias("_v1299_bucket"),
            pl.col("_hist_level_method_value").fill_null("unknown").cast(pl.Utf8).alias("_v1299_method"),
            ((pl.col("_hist_seasonal_multiplier_value").fill_null(1.0) - 1.0).abs() > 1e-6).alias("_v1299_seasonal"),
        ).with_columns(
            pl.concat_str([
                pl.col("_v1299_bucket"), pl.col("_v1299_method"),
                pl.when(pl.col("_v1299_seasonal")).then(pl.lit("seasonal")).otherwise(pl.lit("plain")),
            ], separator="|").alias("_v1299_segment")
        )

    def _factor_cols(factor: float, use12_col: str = "_v1299_use12") -> tuple[pl.Expr, pl.Expr]:
        if abs(factor - 1.0) < 1e-12:
            ae11, ae12, se11, se12 = "_ae11_v", "_ae12_v", "_se11_v", "_se12_v"
        else:
            tag = str(int(round(factor * 1000)))
            ae11, ae12 = f"_v1299_ae11_v_f{tag}", f"_v1299_ae12_v_f{tag}"
            se11, se12 = f"_v1299_se11_v_f{tag}", f"_v1299_se12_v_f{tag}"
        return (
            pl.when(pl.col(use12_col)).then(pl.col(ae12)).otherwise(pl.col(ae11)),
            pl.when(pl.col(use12_col)).then(pl.col(se12)).otherwise(pl.col(se11)),
        )

    def _learn(train_scores: pl.DataFrame, train_selection: pl.DataFrame) -> pl.DataFrame:
        h = train_scores.join(
            train_selection.select("unique_id", pl.col("v12_selected_value").fill_null(False).alias("_v1299_use12")),
            on="unique_id", how="inner",
        ).with_columns(
            pl.when(pl.col("_n_v") <= 9).then(pl.lit("07-09"))
            .when(pl.col("_n_v") <= 13).then(pl.lit("10-13"))
            .when(pl.col("_n_v") <= 20).then(pl.lit("14-20"))
            .otherwise(pl.lit("21-28")).alias("_v1299_bucket"),
            pl.col("_hist_level_method_value").fill_null("unknown").cast(pl.Utf8).alias("_v1299_method"),
            ((pl.col("_hist_seasonal_multiplier_value").fill_null(1.0) - 1.0).abs() > 1e-6).alias("_v1299_seasonal"),
        ).with_columns(
            pl.concat_str([
                pl.col("_v1299_bucket"), pl.col("_v1299_method"),
                pl.when(pl.col("_v1299_seasonal")).then(pl.lit("seasonal")).otherwise(pl.lit("plain")),
            ], separator="|").alias("_v1299_segment")
        )
        h = h.filter(pl.col("_v1299_bucket").is_in(list(buckets)))
        if h.height == 0:
            return pl.DataFrame()
        factor_rows: list[dict[str, object]] = []
        segments = h.get_column("_v1299_segment").drop_nulls().unique().to_list()
        for seg in segments:
            hs = h.filter(pl.col("_v1299_segment") == seg)
            best: dict[str, object] | None = None
            for factor in grid:
                if factor <= 1.0 + 1e-12:
                    continue
                ae, se = _factor_cols(factor)
                ae0, se0 = _factor_cols(1.0)
                folds = (
                    hs.group_by("_validation_block")
                    .agg(
                        pl.col("_den_v").sum().alias("den"),
                        ae0.sum().alias("ae0"), se0.sum().alias("se0"),
                        ae.sum().alias("ae1"), se.sum().alias("se1"),
                        pl.len().alias("n"),
                    )
                    .filter((pl.col("den") > 0) & (pl.col("n") >= min_leaves))
                    .with_columns(
                        ((pl.col("ae0") - pl.col("ae1")) / pl.col("den")).alias("gain"),
                        ((pl.col("se1") / pl.col("den")).abs() - (pl.col("se0") / pl.col("den")).abs()).alias("bias_worsen"),
                    )
                )
                if folds.height < min_folds:
                    continue
                gains = [float(x) for x in folds.get_column("gain").to_list()]
                wins = sum(x > 0.0 for x in gains)
                med = float(folds.get_column("gain").median())
                worst = min(gains)
                bias_worst = float(folds.get_column("bias_worsen").max())
                den_sum = float(folds.get_column("den").sum() or 0.0)
                weighted = float(((folds.get_column("gain") * folds.get_column("den")).sum() or 0.0) / den_sum) if den_sum > 0 else 0.0
                passed = (
                    wins / len(gains) >= min_win and med > min_med
                    and worst >= min_worst and bias_worst <= max_bias and weighted > 0.0
                )
                if not passed:
                    continue
                rec = {
                    "_v1299_segment": seg, "v1299_value_sparse_factor": float(factor),
                    "v1299_value_sparse_folds": len(gains),
                    "v1299_value_sparse_win_rate": wins / len(gains),
                    "v1299_value_sparse_median_gain": med,
                    "v1299_value_sparse_worst_gain": worst,
                    "v1299_value_sparse_weighted_gain": weighted,
                    "v1299_value_sparse_bias_worst": bias_worst,
                    "v1299_value_sparse_candidate": True,
                }
                if best is None or float(rec["v1299_value_sparse_weighted_gain"]) > float(best["v1299_value_sparse_weighted_gain"]):
                    best = rec
            if best is not None:
                factor_rows.append(best)
        return pl.DataFrame(factor_rows) if factor_rows else pl.DataFrame()

    current_profile = _profile(context_scores)
    learned = _learn(history_scores, selections)
    out = defaults.join(current_profile, on="unique_id", how="left")
    if learned.height:
        out = out.join(learned, on="_v1299_segment", how="left", suffix="_learned")
        updates: list[pl.Expr] = []
        for name in (
            "v1299_value_sparse_factor", "v1299_value_sparse_folds", "v1299_value_sparse_win_rate",
            "v1299_value_sparse_median_gain", "v1299_value_sparse_worst_gain",
            "v1299_value_sparse_weighted_gain", "v1299_value_sparse_bias_worst",
            "v1299_value_sparse_candidate",
        ):
            alt = name + "_learned"
            if alt in out.columns:
                updates.append(pl.coalesce([pl.col(alt), pl.col(name)]).alias(name))
        if updates:
            out = out.with_columns(*updates)
    out = out.with_columns(
        pl.col("_v1299_bucket").alias("v1299_value_sparse_bucket"),
        pl.col("_v1299_method").alias("v1299_value_sparse_level_method"),
        pl.col("_v1299_seasonal").fill_null(False).alias("v1299_value_sparse_seasonal"),
        pl.col("_v1299_segment").alias("v1299_value_sparse_segment"),
    )

    # Strict causal temporal replay: block b is held out; only b+1..N train.
    replay_results: list[dict[str, float]] = []
    active_replay_results: list[dict[str, float]] = []
    max_block = int(history_scores.get_column("_validation_block").max() or 0)
    requested = min(replay_cutoffs_req, max(0, max_block - min_folds))
    for held in range(1, requested + 1):
        older = history_scores.filter(pl.col("_validation_block") > held)
        if older.get_column("_validation_block").n_unique() < min_folds:
            continue
        # Re-index older blocks so all existing selector/stress routines see the
        # newest available training block as 1 at this pseudo-cutoff.
        older = older.with_columns((pl.col("_validation_block") - held).cast(pl.Int8).alias("_validation_block"))
        heldout = history_scores.filter(pl.col("_validation_block") == held)
        if heldout.height == 0:
            continue
        target_profile = heldout.select(
            "unique_id",
            pl.col("_hist_level_method_value").fill_null("unknown").alias("_seg_level_method_value"),
            pl.col("_hist_seasonal_multiplier_value").fill_null(1.0).alias("_seg_seasonal_multiplier_value"),
        )
        recent = older.filter(pl.col("_validation_block") <= 4)
        pseudo_sel = _selection_from_closed_blocks(recent, "out_sample", target_profile)
        pseudo_stress = _value_long_horizon_segment_stress(older, target_profile)
        pseudo_sel = pseudo_sel.join(pseudo_stress, on="unique_id", how="left")
        pseudo_sel = _apply_v1296_sec23_long_horizon_gate(pseudo_sel, recent)
        factors = _learn(older, pseudo_sel)
        held_prof = _profile(heldout)
        ev = heldout.join(
            pseudo_sel.select("unique_id", pl.col("v12_selected_value").fill_null(False).alias("_v1299_use12")),
            on="unique_id", how="left",
        ).join(held_prof, on="unique_id", how="left")
        if factors.height:
            ev = ev.join(factors.select("_v1299_segment", "v1299_value_sparse_factor", "v1299_value_sparse_candidate"), on="_v1299_segment", how="left")
        else:
            ev = ev.with_columns(pl.lit(1.0).alias("v1299_value_sparse_factor"), pl.lit(False).alias("v1299_value_sparse_candidate"))
        ev = ev.with_columns(
            pl.col("v1299_value_sparse_factor").fill_null(1.0),
            pl.col("v1299_value_sparse_candidate").fill_null(False),
        )
        # Evaluate by factor groups so exact precomputed AE/SE columns are used.
        # v12.9.10 repeats the exact same calculation on the official ACTIVE
        # cohort (_n_v >= active_min).  This ACTIVE replay, not current OOS,
        # is the productive promotion gate.
        def _evaluate_replay(frame: pl.DataFrame) -> dict[str, float] | None:
            den_total = float(frame.get_column("_den_v").sum() or 0.0)
            if den_total <= 0:
                return None
            ae0_total = se0_total = ae1_total = se1_total = selected_den = 0.0
            for factor in (1.0,) + tuple(x for x in grid if x > 1.0 + 1e-12):
                part = frame.filter(
                    pl.when(pl.col("v1299_value_sparse_candidate"))
                    .then((pl.col("v1299_value_sparse_factor") - factor).abs() < 1e-9)
                    .otherwise(pl.lit(abs(factor - 1.0) < 1e-12))
                )
                if part.height == 0:
                    continue
                ae0, se0 = _factor_cols(1.0)
                ae1, se1 = _factor_cols(factor)
                vals = part.select(
                    ae0.sum().alias("ae0"), se0.sum().alias("se0"),
                    ae1.sum().alias("ae1"), se1.sum().alias("se1"),
                    pl.col("_den_v").sum().alias("den"),
                    pl.when(pl.col("v1299_value_sparse_candidate")).then(pl.col("_den_v")).otherwise(0.0).sum().alias("sel_den"),
                ).row(0, named=True)
                ae0_total += float(vals["ae0"] or 0.0); se0_total += float(vals["se0"] or 0.0)
                ae1_total += float(vals["ae1"] or 0.0); se1_total += float(vals["se1"] or 0.0)
                selected_den += float(vals["sel_den"] or 0.0)
            return {
                "gain": (ae0_total - ae1_total) / den_total,
                "bias": abs(se1_total / den_total) - abs(se0_total / den_total),
                "share": selected_den / den_total,
                "den": den_total,
            }

        all_result = _evaluate_replay(ev)
        if all_result is not None:
            replay_results.append(all_result)
        active_ev = ev.filter(pl.col("_n_v") >= active_min)
        active_result = _evaluate_replay(active_ev) if active_ev.height else None
        if active_result is not None:
            active_replay_results.append(active_result)

    if replay_results:
        gains = [r["gain"] for r in replay_results]
        biases = [r["bias"] for r in replay_results]
        shares = sorted(r["share"] for r in replay_results)
        cutoffs = len(gains)
        wins = sum(g > 0.0 for g in gains)
        med = sorted(gains)[cutoffs // 2] if cutoffs % 2 else 0.5 * (sorted(gains)[cutoffs//2-1] + sorted(gains)[cutoffs//2])
        share_med = shares[cutoffs // 2] if cutoffs % 2 else 0.5 * (shares[cutoffs//2-1] + shares[cutoffs//2])
        worst = min(gains)
        bias_worst = max(biases)
        den_sum = sum(r["den"] for r in replay_results)
        weighted = sum(r["gain"] * r["den"] for r in replay_results) / den_sum if den_sum > 0 else 0.0
        replay_pass = (
            cutoffs >= min(3, replay_cutoffs_req)
            and wins / cutoffs >= min_win and med > min_med
            and worst >= min_worst and bias_worst <= max_bias
        )
        fmt = lambda xs: ",".join(f"{100.0*x:.2f}%" for x in xs)
        out = out.with_columns(
            pl.lit(cutoffs, dtype=pl.Int8).alias("v1299_value_sparse_replay_cutoffs"),
            pl.lit(wins / cutoffs).alias("v1299_value_sparse_replay_win_rate"),
            pl.lit(med).alias("v1299_value_sparse_replay_median_gain"),
            pl.lit(worst).alias("v1299_value_sparse_replay_worst_gain"),
            pl.lit(weighted).alias("v1299_value_sparse_replay_weighted_gain"),
            pl.lit(bias_worst).alias("v1299_value_sparse_replay_bias_worst"),
            pl.lit(share_med).alias("v1299_value_sparse_replay_selected_share_median"),
            pl.lit(fmt(gains)).alias("v1299_value_sparse_replay_fold_gains"),
            pl.lit(fmt([r["share"] for r in replay_results])).alias("v1299_value_sparse_replay_fold_shares"),
            pl.lit(fmt(biases)).alias("v1299_value_sparse_replay_fold_bias"),
            pl.lit(bool(replay_pass)).alias("v1299_value_sparse_replay_pass"),
        )

    # v12.9.10 productive gate: only historical pseudo-OOS ACTIVE leaves count.
    # A section is promoted as a whole only when every risk constraint passes.
    if active_replay_results:
        agains = [r["gain"] for r in active_replay_results]
        abiases = [r["bias"] for r in active_replay_results]
        ashares = sorted(r["share"] for r in active_replay_results)
        acutoffs = len(agains)
        awins = sum(g > 0.0 for g in agains)
        s_gains = sorted(agains)
        amed = s_gains[acutoffs // 2] if acutoffs % 2 else 0.5 * (s_gains[acutoffs//2-1] + s_gains[acutoffs//2])
        ashare_med = ashares[acutoffs // 2] if acutoffs % 2 else 0.5 * (ashares[acutoffs//2-1] + ashares[acutoffs//2])
        aworst = min(agains)
        abias_worst = max(abiases)
        aden_sum = sum(r["den"] for r in active_replay_results)
        aweighted = sum(r["gain"] * r["den"] for r in active_replay_results) / aden_sum if aden_sum > 0 else 0.0
        promoted = bool(
            promote_enabled
            and acutoffs >= promote_min_cutoffs
            and awins / acutoffs >= promote_min_win
            and amed >= promote_min_med
            and aworst >= promote_min_worst
            and aweighted >= promote_min_weighted
            and abias_worst <= promote_max_bias
            and ashare_med <= promote_max_share
        )
        failed: list[str] = []
        if acutoffs < promote_min_cutoffs: failed.append("cutoffs")
        if awins / acutoffs < promote_min_win: failed.append("win_rate")
        if amed < promote_min_med: failed.append("median_gain")
        if aworst < promote_min_worst: failed.append("worst_gain")
        if aweighted < promote_min_weighted: failed.append("weighted_gain")
        if abias_worst > promote_max_bias: failed.append("bias")
        if ashare_med > promote_max_share: failed.append("exposure")
        if not promote_enabled: failed.append("disabled")
        reason = "active_replay_pass" if promoted else "active_replay_fail:" + ",".join(failed)
        fmt10 = lambda xs: ",".join(f"{100.0*x:.2f}%" for x in xs)
        out = out.with_columns(
            pl.lit(acutoffs, dtype=pl.Int8).alias("v12910_value_sparse_active_replay_cutoffs"),
            pl.lit(awins / acutoffs).alias("v12910_value_sparse_active_replay_win_rate"),
            pl.lit(amed).alias("v12910_value_sparse_active_replay_median_gain"),
            pl.lit(aworst).alias("v12910_value_sparse_active_replay_worst_gain"),
            pl.lit(aweighted).alias("v12910_value_sparse_active_replay_weighted_gain"),
            pl.lit(abias_worst).alias("v12910_value_sparse_active_replay_bias_worst"),
            pl.lit(ashare_med).alias("v12910_value_sparse_active_replay_selected_share_median"),
            pl.lit(fmt10(agains)).alias("v12910_value_sparse_active_replay_fold_gains"),
            pl.lit(fmt10([r["share"] for r in active_replay_results])).alias("v12910_value_sparse_active_replay_fold_shares"),
            pl.lit(fmt10(abiases)).alias("v12910_value_sparse_active_replay_fold_bias"),
            pl.lit(promoted).alias("v12910_value_sparse_promoted"),
            pl.lit(reason).alias("v12910_value_sparse_promotion_reason"),
        )

    drop_cols = [c for c in out.columns if c.startswith("_v1299_") or c.endswith("_learned")]
    return out.drop(*drop_cols)


def _v12911_sec23_value_residual_selector(
    history_scores: pl.DataFrame,
    context_scores: pl.DataFrame,
    selections: pl.DataFrame,
) -> pl.DataFrame:
    """Causal residual v11→v12 challenger for Value Sec23 (diagnostic only).

    v12.9.10 is kept completely frozen.  This layer only asks whether leaves
    that the productive selector still leaves on v11 have enough *leaf-specific*
    long-horizon evidence to justify a future v12 switch.  Current OOS actuals
    are never used to learn or authorize the candidate.

    The target-period gate is learned from closed 28d blocks and is capped by
    volume.  A separate four-cutoff replay rebuilds the leaf gate using only
    blocks older than each pseudo-OOS.  Replay is intentionally relative to the
    v11 incumbent on the selected residual leaves; it measures generalization of
    the incremental family switch, not the already-validated v12.9.6 portfolio.
    """
    enabled = bool(getattr(settings, "V12911_SEC23_VALUE_RESIDUAL_ENABLED", True)) and int(
        getattr(settings, "RLS_BLOCK_DAYS", 28)
    ) == 28
    min_folds = int(getattr(settings, "V12911_SEC23_VALUE_MIN_FOLDS", 8))
    min_win = float(getattr(settings, "V12911_SEC23_VALUE_MIN_WIN_RATE", 0.75))
    min_med = float(getattr(settings, "V12911_SEC23_VALUE_MIN_MEDIAN_GAIN", 0.0))
    min_p25 = float(getattr(settings, "V12911_SEC23_VALUE_MIN_P25_GAIN", -0.005))
    min_weighted = float(getattr(settings, "V12911_SEC23_VALUE_MIN_WEIGHTED_GAIN", 0.005))
    min_recent = float(getattr(settings, "V12911_SEC23_VALUE_MIN_RECENT_GAIN", 0.0))
    min_worst = float(getattr(settings, "V12911_SEC23_VALUE_MIN_WORST_GAIN", -0.10))
    max_bias = float(getattr(settings, "V12911_SEC23_VALUE_MAX_BIAS_WORSEN", 0.05))
    max_share = float(getattr(settings, "V12911_SEC23_VALUE_MAX_VOLUME_SHARE", 0.10))
    top05 = float(getattr(settings, "V12911_SEC23_VALUE_TOP05_PERCENTILE", 0.95))
    top01 = float(getattr(settings, "V12911_SEC23_VALUE_TOP01_PERCENTILE", 0.99))
    requested = int(getattr(settings, "V12911_SEC23_VALUE_REPLAY_CUTOFFS", 4))
    replay_min_win = float(getattr(settings, "V12911_SEC23_VALUE_REPLAY_MIN_WIN_RATE", 0.75))
    replay_min_med = float(getattr(settings, "V12911_SEC23_VALUE_REPLAY_MIN_MEDIAN_GAIN", 0.0))
    replay_min_worst = float(getattr(settings, "V12911_SEC23_VALUE_REPLAY_MIN_WORST_GAIN", -0.0025))
    replay_max_bias = float(getattr(settings, "V12911_SEC23_VALUE_REPLAY_MAX_BIAS_WORSEN", 0.01))

    base = selections.select(
        "unique_id",
        pl.col("v12_selected_value").fill_null(False).alias("_v12911_baseline_v12"),
    ).unique("unique_id")
    defaults = base.with_columns(
        pl.lit(bool(enabled)).alias("v12911_residual_enabled_value"),
        pl.lit(False).alias("v12911_residual_candidate_value"),
        pl.lit(0, dtype=pl.Int8).alias("v12911_residual_folds_value"),
        pl.lit(0.0).alias("v12911_residual_win_rate_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_median_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_p25_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_worst_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_weighted_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_recent_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_bias_worst_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_score_value"),
        pl.lit(0.0).alias("v12911_residual_selected_share_value"),
        pl.lit(0, dtype=pl.Int8).alias("v12911_residual_replay_cutoffs_value"),
        pl.lit(0.0).alias("v12911_residual_replay_win_rate_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_replay_median_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_replay_worst_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_replay_weighted_gain_value"),
        pl.lit(None, dtype=pl.Float64).alias("v12911_residual_replay_bias_worst_value"),
        pl.lit(0.0).alias("v12911_residual_replay_selected_share_median_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v12911_residual_replay_fold_gains_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v12911_residual_replay_fold_shares_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v12911_residual_replay_fold_bias_value"),
        pl.lit(False).alias("v12911_residual_replay_pass_value"),
    )
    if (not enabled) or history_scores.height == 0 or context_scores.height == 0 or base.height == 0:
        return defaults.drop("_v12911_baseline_v12")

    hist = history_scores.filter(
        pl.col("_eligible_v")
        & pl.col("unique_id").str.starts_with("23||")
        & pl.col("unique_id").str.contains("||T:", literal=True)
        & (pl.col("_den_v") > 1e-9)
    ).with_columns(
        ((pl.col("_ae11_v") - pl.col("_ae12_v")) / pl.col("_den_v")).alias("_v12911_gain"),
        (
            (pl.col("_se12_v") / pl.col("_den_v")).abs()
            - (pl.col("_se11_v") / pl.col("_den_v")).abs()
        ).alias("_v12911_bias_worsen"),
    )
    if hist.height == 0:
        return defaults.drop("_v12911_baseline_v12")

    def _stats(train: pl.DataFrame) -> pl.DataFrame:
        if train.height == 0:
            return pl.DataFrame()
        st = train.group_by("unique_id").agg(
            pl.col("_validation_block").n_unique().cast(pl.Int8).alias("_folds"),
            (pl.col("_v12911_gain") > 0.0).cast(pl.Float64).mean().alias("_win"),
            pl.col("_v12911_gain").median().alias("_med"),
            pl.col("_v12911_gain").quantile(0.25, interpolation="linear").alias("_p25"),
            pl.col("_v12911_gain").quantile(0.10, interpolation="linear").alias("_p10"),
            pl.col("_v12911_gain").min().alias("_worst"),
            pl.col("_den_v").sum().alias("_den_all"),
            (pl.col("_ae11_v") - pl.col("_ae12_v")).sum().alias("_imp_num"),
            pl.col("_v12911_bias_worsen").max().alias("_bias_worst"),
            pl.when(pl.col("_validation_block") == 1).then(pl.col("_v12911_gain")).otherwise(None).max().alias("_g1"),
            pl.when(pl.col("_validation_block") == 2).then(pl.col("_v12911_gain")).otherwise(None).max().alias("_g2"),
            pl.when(pl.col("_validation_block") == 3).then(pl.col("_v12911_gain")).otherwise(None).max().alias("_g3"),
        ).with_columns(
            pl.when(pl.col("_den_all") > 1e-9).then(pl.col("_imp_num") / pl.col("_den_all")).otherwise(None).alias("_weighted"),
            (
                0.55 * pl.col("_g1").fill_null(0.0)
                + 0.30 * pl.col("_g2").fill_null(0.0)
                + 0.15 * pl.col("_g3").fill_null(0.0)
            ).alias("_recent"),
        ).with_columns(
            (
                pl.min_horizontal(pl.col("_med").fill_null(-1.0), pl.col("_weighted").fill_null(-1.0))
                + 0.50 * pl.col("_p25").fill_null(-1.0)
                + 0.25 * pl.col("_recent").fill_null(-1.0)
            ).alias("_score")
        )
        return st

    def _select_from_train(train: pl.DataFrame, prior: pl.DataFrame, allowed_ids: set[str] | None = None):
        st = _stats(train)
        if st.height == 0 or prior.height == 0:
            return [], 0.0, st
        prof = prior.select("unique_id", pl.col("_den_v").fill_null(0.0).alias("_weight")).unique("unique_id")
        prof = prof.join(st, on="unique_id", how="left")
        # Exposure cap is always relative to the full Sec23 pseudo-current
        # volume, even when the target-period residual pool is restricted to
        # leaves that the productive selector kept on v11.
        total_w = float(prof.select(pl.col("_weight").sum()).item() or 0.0)
        if allowed_ids is not None:
            prof = prof.filter(pl.col("unique_id").is_in(list(allowed_ids)))
        if prof.height == 0:
            return [], 0.0, st
        n = max(1, prof.height)
        prof = prof.with_columns(
            (pl.col("_weight").rank(method="ordinal", descending=False).cast(pl.Float64) / float(n)).alias("_impact_pct")
        )
        normal = (
            (pl.col("_folds").fill_null(0) >= min_folds)
            & (pl.col("_win").fill_null(0.0) >= min_win)
            & (pl.col("_med").fill_null(-999.0) > min_med)
            & (pl.col("_p25").fill_null(-999.0) >= min_p25)
            & (pl.col("_weighted").fill_null(-999.0) >= min_weighted)
            & (pl.col("_recent").fill_null(-999.0) >= min_recent)
            & (pl.col("_worst").fill_null(-999.0) >= min_worst)
            & (pl.col("_bias_worst").fill_null(999.0) <= max_bias)
        )
        impact_guard = (
            pl.when(pl.col("_impact_pct") >= top01)
            .then((pl.col("_p10").fill_null(-999.0) >= 0.0) & (pl.col("_p25").fill_null(-999.0) > 0.0))
            .when(pl.col("_impact_pct") >= top05)
            .then(pl.col("_p25").fill_null(-999.0) > 0.0)
            .otherwise(True)
        )
        eligible = prof.filter(normal & impact_guard).sort(["_score", "_weight"], descending=[True, False])
        cap = max_share * total_w
        selected: list[str] = []
        used = 0.0
        for r in eligible.iter_rows(named=True):
            w = float(r["_weight"] or 0.0)
            if used + w > cap + 1e-9:
                continue
            selected.append(str(r["unique_id"]))
            used += w
        return selected, (used / total_w if total_w > 1e-9 else 0.0), prof

    # Current target decision: only residual leaves still on v11 are eligible.
    v11_ids = set(
        base.filter((~pl.col("_v12911_baseline_v12")) & pl.col("unique_id").str.starts_with("23||"))
        .get_column("unique_id").to_list()
    )
    selected_ids, target_share, target_prof = _select_from_train(hist, context_scores, v11_ids)
    st_all = _stats(hist)
    target_meta = base.join(st_all, on="unique_id", how="left")
    target_meta = target_meta.with_columns(
        pl.col("unique_id").is_in(selected_ids).alias("v12911_residual_candidate_value"),
        pl.col("_folds").fill_null(0).cast(pl.Int8).alias("v12911_residual_folds_value"),
        pl.col("_win").fill_null(0.0).alias("v12911_residual_win_rate_value"),
        pl.col("_med").alias("v12911_residual_median_gain_value"),
        pl.col("_p25").alias("v12911_residual_p25_gain_value"),
        pl.col("_worst").alias("v12911_residual_worst_gain_value"),
        pl.col("_weighted").alias("v12911_residual_weighted_gain_value"),
        pl.col("_recent").alias("v12911_residual_recent_gain_value"),
        pl.col("_bias_worst").alias("v12911_residual_bias_worst_value"),
        pl.col("_score").alias("v12911_residual_score_value"),
        pl.lit(float(target_share)).alias("v12911_residual_selected_share_value"),
    )

    # Strict multi-cutoff replay.  Cutoff b is evaluated only after learning on
    # b+1..N; b+1 supplies the causal pseudo-current volume profile.
    max_block = int(hist.select(pl.col("_validation_block").max()).item() or 0)
    n_cutoffs = min(max(0, requested), max(0, max_block - min_folds))
    fold_results: list[dict[str, float]] = []
    for cutoff in range(1, n_cutoffs + 1):
        train = hist.filter(pl.col("_validation_block") > cutoff).with_columns(
            (pl.col("_validation_block") - cutoff).cast(pl.Int8).alias("_validation_block")
        )
        prior = hist.filter(pl.col("_validation_block") == cutoff + 1)
        valid = hist.filter(pl.col("_validation_block") == cutoff)
        ids, share, _ = _select_from_train(train, prior, None)
        den = float(valid.select(pl.col("_den_v").sum()).item() or 0.0)
        if den <= 1e-9:
            continue
        if ids:
            sel = valid.filter(pl.col("unique_id").is_in(ids))
            imp = float(sel.select((pl.col("_ae11_v") - pl.col("_ae12_v")).sum()).item() or 0.0)
            se11_sel = float(sel.select(pl.col("_se11_v").sum()).item() or 0.0)
            se12_sel = float(sel.select(pl.col("_se12_v").sum()).item() or 0.0)
        else:
            imp = se11_sel = se12_sel = 0.0
        se11_all = float(valid.select(pl.col("_se11_v").sum()).item() or 0.0)
        final_se = se11_all - se11_sel + se12_sel
        bias_worsen = abs(final_se / den) - abs(se11_all / den)
        fold_results.append({"gain": imp / den, "share": share, "bias": bias_worsen, "den": den, "imp": imp})

    if fold_results:
        gains = [r["gain"] for r in fold_results]
        shares = [r["share"] for r in fold_results]
        biases = [r["bias"] for r in fold_results]
        cutoffs = len(gains)
        wins = sum(g > 0.0 for g in gains) / cutoffs
        sg = sorted(gains); ss = sorted(shares)
        med = sg[len(sg)//2] if len(sg)%2 else 0.5*(sg[len(sg)//2-1]+sg[len(sg)//2])
        share_med = ss[len(ss)//2] if len(ss)%2 else 0.5*(ss[len(ss)//2-1]+ss[len(ss)//2])
        den_all = sum(r["den"] for r in fold_results)
        weighted = sum(r["imp"] for r in fold_results) / den_all if den_all > 1e-9 else 0.0
        worst = min(gains); bworst = max(biases)
        replay_pass = (
            cutoffs >= min(requested, 4)
            and wins >= replay_min_win
            and med > replay_min_med
            and worst >= replay_min_worst
            and bworst <= replay_max_bias
        )
        fmt = lambda xs: ",".join(f"{100.0*x:.2f}%" for x in xs)
        target_meta = target_meta.with_columns(
            pl.lit(cutoffs, dtype=pl.Int8).alias("v12911_residual_replay_cutoffs_value"),
            pl.lit(float(wins)).alias("v12911_residual_replay_win_rate_value"),
            pl.lit(float(med)).alias("v12911_residual_replay_median_gain_value"),
            pl.lit(float(worst)).alias("v12911_residual_replay_worst_gain_value"),
            pl.lit(float(weighted)).alias("v12911_residual_replay_weighted_gain_value"),
            pl.lit(float(bworst)).alias("v12911_residual_replay_bias_worst_value"),
            pl.lit(float(share_med)).alias("v12911_residual_replay_selected_share_median_value"),
            pl.lit(fmt(gains)).alias("v12911_residual_replay_fold_gains_value"),
            pl.lit(fmt(shares)).alias("v12911_residual_replay_fold_shares_value"),
            pl.lit(fmt(biases)).alias("v12911_residual_replay_fold_bias_value"),
            pl.lit(bool(replay_pass)).alias("v12911_residual_replay_pass_value"),
        )

    keep = [c for c in defaults.columns if c != "_v12911_baseline_v12"]
    # Preserve defaults for replay fields if no replay materialized.
    out = defaults.drop("_v12911_baseline_v12").join(
        target_meta.select(["unique_id"] + [c for c in target_meta.columns if c.startswith("v12911_")]),
        on="unique_id", how="left", suffix="_new",
    )
    exprs = []
    for c in keep:
        if c == "unique_id":
            continue
        nc = f"{c}_new"
        if nc in out.columns:
            exprs.append(pl.coalesce([pl.col(nc), pl.col(c)]).alias(c))
    if exprs:
        out = out.with_columns(*exprs)
    return out.select(*keep)


def _v12912_ultra_stable_residual_diagnostic(selections: pl.DataFrame) -> pl.DataFrame:
    """Strict diagnostic-only gate layered on v12.9.11 residual evidence.

    It does not alter v12_selected_value or any productive forecast.  The goal is
    to identify only residual Sec23 Value leaves whose historical evidence is
    unusually stable before any future promotion is considered.
    """
    enabled = bool(getattr(settings, "V12912_ULTRA_STABLE_ENABLED", True))
    defaults = selections.select("unique_id").unique().with_columns(
        pl.lit(bool(enabled)).alias("v12912_ultra_enabled_value"),
        pl.lit(False).alias("v12912_ultra_candidate_value"),
        pl.lit(False).alias("v12912_ultra_replay_pass_value"),
        pl.lit(None, dtype=pl.Utf8).alias("v12912_ultra_reason_value"),
    )
    req = {
        "v12911_residual_candidate_value",
        "v12911_residual_win_rate_value",
        "v12911_residual_median_gain_value",
        "v12911_residual_p25_gain_value",
        "v12911_residual_worst_gain_value",
        "v12911_residual_weighted_gain_value",
        "v12911_residual_recent_gain_value",
        "v12911_residual_bias_worst_value",
        "v12911_residual_selected_share_value",
        "v12911_residual_replay_win_rate_value",
        "v12911_residual_replay_median_gain_value",
        "v12911_residual_replay_worst_gain_value",
        "v12911_residual_replay_weighted_gain_value",
        "v12911_residual_replay_bias_worst_value",
        "v12911_residual_replay_selected_share_median_value",
    }
    if (not enabled) or not req.issubset(set(selections.columns)):
        return defaults

    min_win = float(getattr(settings, "V12912_ULTRA_MIN_WIN_RATE", 1.0))
    min_med = float(getattr(settings, "V12912_ULTRA_MIN_MEDIAN_GAIN", 0.0010))
    min_p25 = float(getattr(settings, "V12912_ULTRA_MIN_P25_GAIN", 0.0005))
    min_worst = float(getattr(settings, "V12912_ULTRA_MIN_WORST_GAIN", 0.0))
    min_weighted = float(getattr(settings, "V12912_ULTRA_MIN_WEIGHTED_GAIN", 0.0010))
    min_recent = float(getattr(settings, "V12912_ULTRA_MIN_RECENT_GAIN", 0.0))
    max_bias = float(getattr(settings, "V12912_ULTRA_MAX_BIAS_WORSEN", 0.0))
    max_share = float(getattr(settings, "V12912_ULTRA_MAX_VOLUME_SHARE", 0.03))

    x = selections.select(["unique_id", *sorted(req)]).unique("unique_id")
    leaf_pass = (
        pl.col("v12911_residual_candidate_value").fill_null(False)
        & (pl.col("v12911_residual_win_rate_value").fill_null(0.0) >= min_win)
        & (pl.col("v12911_residual_median_gain_value").fill_null(-999.0) >= min_med)
        & (pl.col("v12911_residual_p25_gain_value").fill_null(-999.0) >= min_p25)
        & (pl.col("v12911_residual_worst_gain_value").fill_null(-999.0) > min_worst)
        & (pl.col("v12911_residual_weighted_gain_value").fill_null(-999.0) >= min_weighted)
        & (pl.col("v12911_residual_recent_gain_value").fill_null(-999.0) > min_recent)
        & (pl.col("v12911_residual_bias_worst_value").fill_null(999.0) <= max_bias)
        & (pl.col("v12911_residual_selected_share_value").fill_null(1.0) <= max_share)
    )
    replay_pass = (
        (pl.col("v12911_residual_replay_win_rate_value").fill_null(0.0) >= min_win)
        & (pl.col("v12911_residual_replay_median_gain_value").fill_null(-999.0) >= min_med)
        & (pl.col("v12911_residual_replay_worst_gain_value").fill_null(-999.0) > min_worst)
        & (pl.col("v12911_residual_replay_weighted_gain_value").fill_null(-999.0) >= min_weighted)
        & (pl.col("v12911_residual_replay_bias_worst_value").fill_null(999.0) <= max_bias)
        & (pl.col("v12911_residual_replay_selected_share_median_value").fill_null(1.0) <= max_share)
    )
    out = x.with_columns(
        leaf_pass.alias("v12912_ultra_candidate_value"),
        replay_pass.alias("v12912_ultra_replay_pass_value"),
    ).with_columns(
        pl.when(pl.col("v12912_ultra_candidate_value") & pl.col("v12912_ultra_replay_pass_value"))
        .then(pl.lit("ULTRA_STABLE_PASS"))
        .when(pl.col("v12912_ultra_candidate_value"))
        .then(pl.lit("LEAF_PASS_REPLAY_FAIL"))
        .otherwise(pl.lit("NO_ACTION"))
        .alias("v12912_ultra_reason_value"),
        pl.lit(True).alias("v12912_ultra_enabled_value"),
    )
    return out.select(
        "unique_id",
        "v12912_ultra_enabled_value",
        "v12912_ultra_candidate_value",
        "v12912_ultra_replay_pass_value",
        "v12912_ultra_reason_value",
    )

def _selection_from_closed_blocks(
    block_scores: pl.DataFrame,
    target_period_type: str,
    target_profile: pl.DataFrame,
) -> pl.DataFrame:
    """Causal v12.8.3 target-specific selection from strictly closed blocks.

    The original nested discovery/confirmation statistics remain available as
    diagnostics and as a conservative fallback.  Production selection is driven
    by a pooled regularized logistic meta-model trained with a strict temporal
    shift: block 1 winner labels are predicted only from blocks 2/3/4.  The
    target is then scored from blocks 1/2/3. Quantity and value are independent
    magnitude decisions; the occurrence/share event layer remains shared.
    """
    min_improvement = float(getattr(settings, "V12_MIN_IMPROVEMENT", 0.02))
    min_blocks = int(getattr(settings, "V12_SELECTION_MIN_BLOCKS", 2))
    min_wins = int(getattr(settings, "V12_SELECTION_MIN_WINS", 2))
    require_recent_win = bool(getattr(settings, "V12_REQUIRE_RECENT_WIN", True))
    recent_min_improvement = float(getattr(settings, "V12_RECENT_MIN_IMPROVEMENT", min_improvement))
    max_degradation = float(getattr(settings, "V12_MAX_BLOCK_DEGRADATION", 0.08))
    max_bias_worsen = float(getattr(settings, "V12_MAX_ABS_BIAS_WORSEN", 0.10))
    section_min_improvement = float(getattr(settings, "V12_SECTION_MIN_IMPROVEMENT", 0.005))
    section_min_wins = int(getattr(settings, "V12_SECTION_MIN_WINS", 2))
    section_require_recent_win = bool(getattr(settings, "V12_SECTION_REQUIRE_RECENT_WIN", True))
    require_joint = bool(getattr(settings, "V12_REQUIRE_JOINT_TARGET_WIN", True))
    bias_weight = float(getattr(settings, "V12_SELECTOR_BIAS_WEIGHT", 0.25))
    stability_weight = float(getattr(settings, "V12_SELECTOR_STABILITY_WEIGHT", 0.10))
    utility_min = float(getattr(settings, "V12_SELECTOR_MIN_UTILITY_IMPROVEMENT", 0.0025))
    min_wmape_improvement = max(
        min_improvement,
        float(getattr(settings, "V12_SELECTOR_MIN_WMAPE_IMPROVEMENT", min_improvement)),
    )
    eps = 1e-9

    # Block 1 is held out from discovery and acts as the temporal confirmation
    # block.  All leaf pooled statistics below are computed on blocks > 1.
    discovery_scores = block_scores.filter(pl.col("_validation_block") > 1)
    confirm_scores = block_scores.filter(pl.col("_validation_block") == 1)

    discovery = (
        discovery_scores.group_by("unique_id")
        .agg(
            pl.col("_eligible_y").sum().cast(pl.Int16).alias("v12_validation_blocks_y"),
            pl.col("_eligible_v").sum().cast(pl.Int16).alias("v12_validation_blocks_value"),
            (
                pl.col("_eligible_y")
                & (pl.col("_improvement_y") >= min_wmape_improvement)
                & (pl.col("_utility_improvement_y") >= utility_min)
            ).sum().cast(pl.Int16).alias("v12_validation_wins_y"),
            (
                pl.col("_eligible_v")
                & (pl.col("_improvement_v") >= min_wmape_improvement)
                & (pl.col("_utility_improvement_v") >= utility_min)
            ).sum().cast(pl.Int16).alias("v12_validation_wins_value"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_den_y")).otherwise(0.0).sum().alias("_pool_den_y"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_ae11_y")).otherwise(0.0).sum().alias("_pool_ae11_y"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_ae12_y")).otherwise(0.0).sum().alias("_pool_ae12_y"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_se11_y")).otherwise(0.0).sum().alias("_pool_se11_y"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_se12_y")).otherwise(0.0).sum().alias("_pool_se12_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_den_v")).otherwise(0.0).sum().alias("_pool_den_v"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_ae11_v")).otherwise(0.0).sum().alias("_pool_ae11_v"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_ae12_v")).otherwise(0.0).sum().alias("_pool_ae12_v"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_se11_v")).otherwise(0.0).sum().alias("_pool_se11_v"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_se12_v")).otherwise(0.0).sum().alias("_pool_se12_v"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_wmape12_y") - pl.col("_wmape11_y")).otherwise(None).max().alias("v12_validation_max_degradation_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_wmape12_v") - pl.col("_wmape11_v")).otherwise(None).max().alias("v12_validation_max_degradation_value"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_utility_improvement_y")).otherwise(None).mean().alias("v12_validation_utility_mean_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_utility_improvement_v")).otherwise(None).mean().alias("v12_validation_utility_mean_value"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_utility_improvement_y")).otherwise(None).std(ddof=0).alias("v12_validation_utility_std_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_utility_improvement_v")).otherwise(None).std(ddof=0).alias("v12_validation_utility_std_value"),
        )
        .with_columns(
            pl.when(pl.col("_pool_den_y") > eps).then(pl.col("_pool_ae11_y") / pl.col("_pool_den_y")).otherwise(None).alias("v12_validation_wmape_v11_y"),
            pl.when(pl.col("_pool_den_y") > eps).then(pl.col("_pool_ae12_y") / pl.col("_pool_den_y")).otherwise(None).alias("v12_validation_wmape_v12_y"),
            pl.when(pl.col("_pool_den_y") > eps).then(pl.col("_pool_se11_y") / pl.col("_pool_den_y")).otherwise(None).alias("v12_validation_bias_v11_y"),
            pl.when(pl.col("_pool_den_y") > eps).then(pl.col("_pool_se12_y") / pl.col("_pool_den_y")).otherwise(None).alias("v12_validation_bias_v12_y"),
            pl.when(pl.col("_pool_den_v") > eps).then(pl.col("_pool_ae11_v") / pl.col("_pool_den_v")).otherwise(None).alias("v12_validation_wmape_v11_value"),
            pl.when(pl.col("_pool_den_v") > eps).then(pl.col("_pool_ae12_v") / pl.col("_pool_den_v")).otherwise(None).alias("v12_validation_wmape_v12_value"),
            pl.when(pl.col("_pool_den_v") > eps).then(pl.col("_pool_se11_v") / pl.col("_pool_den_v")).otherwise(None).alias("v12_validation_bias_v11_value"),
            pl.when(pl.col("_pool_den_v") > eps).then(pl.col("_pool_se12_v") / pl.col("_pool_den_v")).otherwise(None).alias("v12_validation_bias_v12_value"),
        )
        .with_columns(
            (pl.col("v12_validation_wmape_v11_y") - pl.col("v12_validation_wmape_v12_y")).alias("v12_validation_improvement_y"),
            (pl.col("v12_validation_wmape_v11_value") - pl.col("v12_validation_wmape_v12_value")).alias("v12_validation_improvement_value"),
            (
                pl.col("v12_validation_wmape_v11_y")
                + bias_weight * pl.col("v12_validation_bias_v11_y").abs()
                - pl.col("v12_validation_wmape_v12_y")
                - bias_weight * pl.col("v12_validation_bias_v12_y").abs()
            ).alias("v12_validation_utility_improvement_y"),
            (
                pl.col("v12_validation_wmape_v11_value")
                + bias_weight * pl.col("v12_validation_bias_v11_value").abs()
                - pl.col("v12_validation_wmape_v12_value")
                - bias_weight * pl.col("v12_validation_bias_v12_value").abs()
            ).alias("v12_validation_utility_improvement_value"),
        )
    )

    # The confirmation block is intentionally summarized separately, after the
    # older blocks have formed the discovery candidate set.
    confirmation = (
        confirm_scores.group_by("unique_id")
        .agg(
            pl.col("_eligible_y").any().alias("_confirm_eligible_y"),
            pl.col("_eligible_v").any().alias("_confirm_eligible_v"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_improvement_y")).otherwise(None).max().alias("v12_validation_recent_improvement_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_improvement_v")).otherwise(None).max().alias("v12_validation_recent_improvement_value"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_utility_improvement_y")).otherwise(None).max().alias("v12_validation_recent_utility_improvement_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_utility_improvement_v")).otherwise(None).max().alias("v12_validation_recent_utility_improvement_value"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_bias11_y")).otherwise(None).max().alias("_confirm_bias11_y"),
            pl.when(pl.col("_eligible_y")).then(pl.col("_bias12_y")).otherwise(None).max().alias("_confirm_bias12_y"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_bias11_v")).otherwise(None).max().alias("_confirm_bias11_v"),
            pl.when(pl.col("_eligible_v")).then(pl.col("_bias12_v")).otherwise(None).max().alias("_confirm_bias12_v"),
        )
    )

    leaf = (
        discovery.join(confirmation, on="unique_id", how="left")
        .with_columns(
            (
                (pl.col("v12_validation_blocks_y") >= min_blocks)
                & (pl.col("v12_validation_wins_y") >= min_wins)
                & (pl.col("v12_validation_improvement_y") >= min_wmape_improvement)
                & (pl.col("v12_validation_utility_improvement_y") >= utility_min)
                & (
                    pl.col("v12_validation_utility_mean_y").fill_null(-999.0)
                    >= stability_weight * pl.col("v12_validation_utility_std_y").fill_null(0.0)
                )
                & (pl.col("v12_validation_max_degradation_y").fill_null(999.0) <= max_degradation)
                & (
                    pl.col("v12_validation_bias_v12_y").abs()
                    <= pl.col("v12_validation_bias_v11_y").abs() + max_bias_worsen
                )
                & (
                    (~pl.lit(require_recent_win))
                    | (
                        pl.col("_confirm_eligible_y").fill_null(False)
                        & (pl.col("v12_validation_recent_improvement_y").fill_null(-999.0) >= recent_min_improvement)
                        & (pl.col("v12_validation_recent_utility_improvement_y").fill_null(-999.0) >= utility_min)
                        & (
                            pl.col("_confirm_bias12_y").abs()
                            <= pl.col("_confirm_bias11_y").abs() + max_bias_worsen
                        )
                    )
                )
            ).alias("_rule_preselect_y"),
            (
                (pl.col("v12_validation_blocks_value") >= min_blocks)
                & (pl.col("v12_validation_wins_value") >= min_wins)
                & (pl.col("v12_validation_improvement_value") >= min_wmape_improvement)
                & (pl.col("v12_validation_utility_improvement_value") >= utility_min)
                & (
                    pl.col("v12_validation_utility_mean_value").fill_null(-999.0)
                    >= stability_weight * pl.col("v12_validation_utility_std_value").fill_null(0.0)
                )
                & (pl.col("v12_validation_max_degradation_value").fill_null(999.0) <= max_degradation)
                & (
                    pl.col("v12_validation_bias_v12_value").abs()
                    <= pl.col("v12_validation_bias_v11_value").abs() + max_bias_worsen
                )
                & (
                    (~pl.lit(require_recent_win))
                    | (
                        pl.col("_confirm_eligible_v").fill_null(False)
                        & (pl.col("v12_validation_recent_improvement_value").fill_null(-999.0) >= recent_min_improvement)
                        & (pl.col("v12_validation_recent_utility_improvement_value").fill_null(-999.0) >= utility_min)
                        & (
                            pl.col("_confirm_bias12_v").abs()
                            <= pl.col("_confirm_bias11_v").abs() + max_bias_worsen
                        )
                    )
                )
            ).alias("_rule_preselect_v"),
        )
    )

    # v12.8.3 adaptive causal policy. Quantity is unchanged from v12.8.2; Value
    # adds the closed-block portfolio safety layer.  Threshold and section/target mode are
    # learned on the latest CLOSED block with a meta-model that was itself fit
    # only on older blocks.  The production meta-model is then shifted forward
    # one block.  The legacy section gate below is retained for diagnostics only
    # and no longer vetoes a meta decision.
    meta_enabled = bool(getattr(settings, "V12_META_SELECTOR_ENABLED", True))
    meta_default_threshold = float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))
    meta_max_recent_degradation = float(getattr(
        settings, "V12_META_SELECTOR_MAX_RECENT_DEGRADATION", max_degradation
    ))
    meta_bias_tol = float(getattr(settings, "V12_META_SELECTOR_BIAS_GUARD_TOLERANCE", 0.03))
    qty_blocks = max(2, int(getattr(settings, "V129_QTY_FROZEN_SELECTION_BLOCKS", 4)))
    qty_scores = block_scores.filter(pl.col("_validation_block") <= qty_blocks)
    policy_y = _calibrate_meta_policy(qty_scores, "y")
    policy_v = _calibrate_value_walkforward_policy(block_scores)
    threshold_y = float(policy_y.get("threshold") or meta_default_threshold)
    threshold_v = float(policy_v.get("threshold") or meta_default_threshold)
    mode_y = str(policy_y.get("mode") or "meta_leaf")
    mode_v = str(policy_v.get("mode") or "meta_leaf")

    meta_y = _fit_meta_selector(qty_scores, "y")
    meta_v = _fit_meta_selector(block_scores, "v")
    expected_gain_v = _fit_value_expected_gain_selector(block_scores)
    segment_v = _fit_value_hierarchical_segment_selector(block_scores, target_profile)
    leaf = (
        leaf.join(meta_y, on="unique_id", how="left")
        .join(meta_v, on="unique_id", how="left")
        .join(expected_gain_v, on="unique_id", how="left")
        .join(segment_v, on="unique_id", how="left")
        .with_columns(
            pl.col("v12_meta_model_available_y").fill_null(False),
            pl.col("v12_meta_model_available_value").fill_null(False),
            pl.col("v12_meta_training_rows_y").fill_null(0),
            pl.col("v12_meta_training_rows_value").fill_null(0),
        )
        .with_columns(
            (
                pl.col("v12_meta_bias12_y").abs()
                <= pl.col("v12_meta_bias11_y").abs() + meta_bias_tol
            ).fill_null(False).alias("v12_meta_bias_guard_pass_y"),
            (
                pl.col("v12_meta_bias12_value").abs()
                <= pl.col("v12_meta_bias11_value").abs() + meta_bias_tol
            ).fill_null(False).alias("v12_meta_bias_guard_pass_value"),
        )
    )

    def _adaptive_preselect_expr(suffix: str, mode: str, threshold: float) -> pl.Expr:
        out_suffix = "y" if suffix == "y" else "value"
        if mode == "v11_all":
            return pl.lit(False)
        if mode == "v12_all":
            return pl.lit(True)
        return (
            pl.when(pl.lit(meta_enabled) & pl.col(f"v12_meta_model_available_{out_suffix}"))
            .then(
                (pl.col(f"v12_meta_probability_{out_suffix}").fill_null(0.0) >= float(threshold))
                & (pl.col(f"v12_meta_recent_gain_{out_suffix}").fill_null(-999.0) >= -meta_max_recent_degradation)
                & pl.col(f"v12_meta_bias_guard_pass_{out_suffix}")
            )
            .otherwise(pl.col("_rule_preselect_y" if suffix == "y" else "_rule_preselect_v"))
        )

    # v12.9.6 production Value decision: hierarchical, interpretable
    # segment walk-forward selector.  v12.9.1/2 ridge expected-impact fields
    # stay attached strictly as diagnostics and never authorize production.
    seg_enabled = bool(getattr(settings, "V1293_VALUE_SEGMENT_SELECTOR_ENABLED", True)) and int(
        getattr(settings, "RLS_BLOCK_DAYS", 28)
    ) == 28
    leaf = (
        leaf.with_columns(
            _adaptive_preselect_expr("y", mode_y, threshold_y).alias("_preselect_y"),
            (
                pl.col("v1293_segment_budget_selected_value").fill_null(False)
                if seg_enabled
                else _adaptive_preselect_expr("v", mode_v, threshold_v)
            ).alias("_preselect_v"),
            # Compatibility metadata: v12.9.2 expected-impact is diagnostic only.
            pl.lit(False).alias("v1291_expected_gain_enabled_value"),
            pl.lit(False).alias("v1291_expected_gain_portfolio_pass_value"),
            pl.lit(0.0, dtype=pl.Float64).alias("v1291_expected_gain_portfolio_gain_value"),
            pl.lit(0.0, dtype=pl.Float64).alias("v1291_expected_gain_portfolio_share_value"),
            pl.lit(float(getattr(settings, "V1292_VALUE_MIN_CALIBRATED_GAIN", 0.0040)), dtype=pl.Float64).alias("v1291_expected_gain_min_gain_value"),
            pl.lit(float(getattr(settings, "V1292_VALUE_MIN_CONFIDENCE", 0.15)), dtype=pl.Float64).alias("v1291_expected_gain_min_confidence_value"),
            pl.lit(float(getattr(settings, "V1292_VALUE_PORTFOLIO_MAX_VOLUME_SHARE", 0.45)), dtype=pl.Float64).alias("v1292_portfolio_budget_value"),
            pl.lit(0.0, dtype=pl.Float64).alias("v1292_portfolio_expected_gain_value"),
            pl.lit(0.0, dtype=pl.Float64).alias("v1292_portfolio_selected_share_value"),
            pl.lit(False).alias("v1292_portfolio_pass_value"),
            pl.lit(False).alias("v1292_budget_selected_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_effective_min_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_effective_min_confidence_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_effective_max_leaf_loss_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_effective_min_p10_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v1292_expected_impact_score_value"),
        )
        .with_columns(
            (pl.col("_preselect_y") & pl.col("_preselect_v")).alias("v12_joint_target_preselect"),
            pl.lit(threshold_y).cast(pl.Float64).alias("v12_meta_threshold_y"),
            pl.lit(threshold_v).cast(pl.Float64).alias("v12_meta_threshold_value"),
            pl.lit(mode_y).alias("v12_meta_portfolio_mode_y"),
            pl.lit(mode_v).alias("v12_meta_portfolio_mode_value"),
            pl.lit(bool(policy_y.get("available", False))).alias("v12_meta_policy_available_y"),
            pl.lit(bool(policy_v.get("available", False))).alias("v12_meta_policy_available_value"),
            pl.lit(policy_y.get("utility_gain"), dtype=pl.Float64).alias("v12_meta_policy_utility_gain_y"),
            pl.lit(policy_v.get("utility_gain"), dtype=pl.Float64).alias("v12_meta_policy_utility_gain_value"),
            pl.lit(policy_y.get("wmape_v11"), dtype=pl.Float64).alias("v12_meta_policy_wmape_v11_y"),
            pl.lit(policy_v.get("wmape_v11"), dtype=pl.Float64).alias("v12_meta_policy_wmape_v11_value"),
            pl.lit(policy_y.get("wmape_selected"), dtype=pl.Float64).alias("v12_meta_policy_wmape_selected_y"),
            pl.lit(policy_v.get("wmape_selected"), dtype=pl.Float64).alias("v12_meta_policy_wmape_selected_value"),
            pl.lit(policy_y.get("bias_v11"), dtype=pl.Float64).alias("v12_meta_policy_bias_v11_y"),
            pl.lit(policy_v.get("bias_v11"), dtype=pl.Float64).alias("v12_meta_policy_bias_v11_value"),
            pl.lit(policy_y.get("bias_selected"), dtype=pl.Float64).alias("v12_meta_policy_bias_selected_y"),
            pl.lit(policy_v.get("bias_selected"), dtype=pl.Float64).alias("v12_meta_policy_bias_selected_value"),
            pl.lit(bool(policy_v.get("value_safety_enabled", False))).alias("v12_value_safety_enabled"),
            pl.lit(bool(policy_v.get("value_safety_dominance_pass", False))).alias("v12_value_safety_dominance_pass"),
            pl.lit(int(policy_v.get("value_safety_recent_confirmations", 0) or 0)).cast(pl.Int8).alias("v12_value_safety_recent_confirmations"),
            pl.lit(int(policy_v.get("value_safety_recent_blocks", 0) or 0)).cast(pl.Int8).alias("v12_value_safety_recent_blocks"),
            pl.lit(policy_v.get("value_safety_bias_coverage"), dtype=pl.Float64).alias("v12_value_safety_bias_coverage"),
            pl.lit(policy_v.get("value_safety_bias_coverage_threshold"), dtype=pl.Float64).alias("v12_value_safety_bias_coverage_threshold"),
            pl.lit(bool(policy_v.get("value_safety_bias_coverage_pass", False))).alias("v12_value_safety_bias_coverage_pass"),
            pl.lit(bool(policy_v.get("value_safety_meta_margin_pass", False))).alias("v12_value_safety_meta_margin_pass"),
            pl.lit(policy_v.get("value_safety_best_all_mode"), dtype=pl.Utf8).alias("v12_value_safety_best_all_mode"),
            pl.lit(policy_v.get("value_safety_reason"), dtype=pl.Utf8).alias("v12_value_safety_reason"),
            pl.lit(bool(policy_v.get("v129_value_wf_enabled", False))).alias("v129_value_wf_enabled"),
            pl.lit(bool(policy_v.get("v129_value_wf_available", False))).alias("v129_value_wf_available"),
            pl.lit(int(policy_v.get("v129_value_wf_folds", 0) or 0)).cast(pl.Int8).alias("v129_value_wf_folds"),
            pl.lit(policy_v.get("v129_value_wf_win_rate"), dtype=pl.Float64).alias("v129_value_wf_win_rate"),
            pl.lit(policy_v.get("v129_value_wf_median_gain"), dtype=pl.Float64).alias("v129_value_wf_median_gain"),
            pl.lit(policy_v.get("v129_value_wf_worst_gain"), dtype=pl.Float64).alias("v129_value_wf_worst_gain"),
            pl.lit(policy_v.get("v129_value_wf_weighted_gain"), dtype=pl.Float64).alias("v129_value_wf_weighted_gain"),
            pl.lit(policy_v.get("v129_value_wf_weighted_utility_gain"), dtype=pl.Float64).alias("v129_value_wf_weighted_utility_gain"),
            pl.lit(policy_v.get("v129_value_wf_bias_worsen_max"), dtype=pl.Float64).alias("v129_value_wf_bias_worsen_max"),
            pl.lit(int(policy_v.get("v129_value_wf_meta_folds", 0) or 0)).cast(pl.Int8).alias("v129_value_wf_meta_folds"),
            pl.lit(policy_v.get("v129_value_wf_fold1_gain"), dtype=pl.Float64).alias("v129_value_wf_fold1_gain"),
            pl.lit(policy_v.get("v129_value_wf_fold2_gain"), dtype=pl.Float64).alias("v129_value_wf_fold2_gain"),
            pl.lit(policy_v.get("v129_value_wf_fold3_gain"), dtype=pl.Float64).alias("v129_value_wf_fold3_gain"),
            pl.lit(policy_v.get("v129_value_wf_reason"), dtype=pl.Utf8).alias("v129_value_wf_reason"),
        )
    )

    # v12.8.2 defaults to target-specific magnitude selection. The joint flag is
    # retained only for backwards-compatible diagnostics; occurrence/share is
    # still shared at the retail-event layer.
    ids_y = leaf.filter(pl.col("v12_joint_target_preselect") if require_joint else pl.col("_preselect_y")).select("unique_id")
    ids_v = leaf.filter(pl.col("v12_joint_target_preselect") if require_joint else pl.col("_preselect_v")).select("unique_id")

    def _portfolio_gate(ids: pl.DataFrame, suffix: str) -> tuple[bool, int, float | None, float | None]:
        if ids.height == 0:
            return False, 0, None, None
        elig = f"_eligible_{suffix}"
        den = f"_den_{suffix}"
        ae11 = f"_ae11_{suffix}"
        ae12 = f"_ae12_{suffix}"

        # Discovery portfolio uses only blocks > 1.
        xd = (
            discovery_scores.join(ids, on="unique_id", how="semi")
            .filter(pl.col(elig))
        )
        if xd.height == 0:
            return False, 0, None, None
        by_block = (
            xd.group_by("_validation_block")
            .agg(
                pl.col(den).sum().alias("_den"),
                pl.col(ae11).sum().alias("_ae11"),
                pl.col(ae12).sum().alias("_ae12"),
            )
            .filter(pl.col("_den") > eps)
            .with_columns(
                (pl.col("_ae11") / pl.col("_den")).alias("_w11"),
                (pl.col("_ae12") / pl.col("_den")).alias("_w12"),
            )
        )
        if by_block.height < min_blocks:
            return False, int(by_block.height), None, None
        wins = int(by_block.filter(pl.col("_w12") + section_min_improvement < pl.col("_w11")).height)
        pooled = by_block.select(
            ((pl.col("_ae11").sum() - pl.col("_ae12").sum()) / pl.col("_den").sum()).alias("_gain")
        ).item()
        gain = None if pooled is None else float(pooled)

        # Confirmation portfolio is evaluated on block 1 only and was not used
        # in the discovery pooled/win statistics above.
        xc = (
            confirm_scores.join(ids, on="unique_id", how="semi")
            .filter(pl.col(elig))
        )
        confirm_gain = None
        if xc.height:
            c = xc.select(
                pl.col(den).sum().alias("_den"),
                pl.col(ae11).sum().alias("_ae11"),
                pl.col(ae12).sum().alias("_ae12"),
            )
            cden = c["_den"][0]
            if cden is not None and float(cden) > eps:
                confirm_gain = float((c["_ae11"][0] - c["_ae12"][0]) / cden)

        confirm_ok = (
            (not section_require_recent_win)
            or (confirm_gain is not None and confirm_gain >= section_min_improvement)
        )
        gate = (
            wins >= section_min_wins
            and gain is not None
            and gain >= section_min_improvement
            and confirm_ok
        )
        return bool(gate), wins, gain, confirm_gain

    gate_y_raw, section_wins_y, section_gain_y, section_confirm_gain_y = _portfolio_gate(ids_y, "y")
    gate_v_raw, section_wins_v, section_gain_v, section_confirm_gain_v = _portfolio_gate(ids_v, "v")
    joint_gate = bool(gate_y_raw and gate_v_raw) if require_joint else None
    gate_y = joint_gate if require_joint else gate_y_raw
    gate_v = joint_gate if require_joint else gate_v_raw

    return (
        leaf.with_columns(
            (
                pl.col("v12_joint_target_preselect")
                if require_joint
                else pl.col("_preselect_y")
            ).alias("v12_selected_y"),
            (
                pl.col("v12_joint_target_preselect")
                if require_joint
                else pl.col("_preselect_v")
            ).alias("v12_selected_value"),
            pl.lit(bool(gate_y)).alias("v12_section_portfolio_gate_y"),
            pl.lit(bool(gate_v)).alias("v12_section_portfolio_gate_value"),
            pl.lit(bool(joint_gate) if require_joint else bool(gate_y and gate_v)).alias("v12_section_portfolio_gate_joint"),
            pl.lit(section_wins_y).cast(pl.Int16).alias("v12_section_portfolio_wins_y"),
            pl.lit(section_wins_v).cast(pl.Int16).alias("v12_section_portfolio_wins_value"),
            pl.lit(section_gain_y, dtype=pl.Float64).alias("v12_section_portfolio_gain_y"),
            pl.lit(section_gain_v, dtype=pl.Float64).alias("v12_section_portfolio_gain_value"),
            pl.lit(section_confirm_gain_y, dtype=pl.Float64).alias("v12_section_portfolio_recent_gain_y"),
            pl.lit(section_confirm_gain_v, dtype=pl.Float64).alias("v12_section_portfolio_recent_gain_value"),
            pl.lit(target_period_type).alias("period_type"),
        )
        .select(
            "unique_id", "period_type",
            "v12_selected_y", "v12_selected_value",
            "v12_joint_target_preselect",
            "v12_meta_probability_y", "v12_meta_probability_value",
            "v12_meta_model_available_y", "v12_meta_model_available_value",
            "v12_meta_training_rows_y", "v12_meta_training_rows_value",
            "v12_meta_recent_gain_y", "v12_meta_recent_gain_value",
            "v12_meta_weighted_gain_y", "v12_meta_weighted_gain_value",
            "v12_meta_win_rate_y", "v12_meta_win_rate_value",
            "v12_meta_top_driver_y", "v12_meta_top_driver_value",
            "v12_meta_bias11_y", "v12_meta_bias11_value",
            "v12_meta_bias12_y", "v12_meta_bias12_value",
            "v12_meta_bias_guard_pass_y", "v12_meta_bias_guard_pass_value",
            "v12_meta_threshold_y", "v12_meta_threshold_value",
            "v12_meta_portfolio_mode_y", "v12_meta_portfolio_mode_value",
            "v12_meta_policy_available_y", "v12_meta_policy_available_value",
            "v12_meta_policy_utility_gain_y", "v12_meta_policy_utility_gain_value",
            "v12_meta_policy_wmape_v11_y", "v12_meta_policy_wmape_v11_value",
            "v12_meta_policy_wmape_selected_y", "v12_meta_policy_wmape_selected_value",
            "v12_meta_policy_bias_v11_y", "v12_meta_policy_bias_v11_value",
            "v12_meta_policy_bias_selected_y", "v12_meta_policy_bias_selected_value",
            "v12_value_safety_enabled", "v12_value_safety_dominance_pass",
            "v12_value_safety_recent_confirmations", "v12_value_safety_recent_blocks",
            "v12_value_safety_bias_coverage", "v12_value_safety_bias_coverage_threshold",
            "v12_value_safety_bias_coverage_pass", "v12_value_safety_meta_margin_pass",
            "v12_value_safety_best_all_mode", "v12_value_safety_reason",
            "v129_value_wf_enabled", "v129_value_wf_available", "v129_value_wf_folds",
            "v129_value_wf_win_rate", "v129_value_wf_median_gain", "v129_value_wf_worst_gain",
            "v129_value_wf_weighted_gain", "v129_value_wf_weighted_utility_gain",
            "v129_value_wf_bias_worsen_max", "v129_value_wf_meta_folds",
            "v129_value_wf_fold1_gain", "v129_value_wf_fold2_gain", "v129_value_wf_fold3_gain", "v129_value_wf_reason",
            "v1291_expected_gain_enabled_value", "v1291_expected_gain_available_value",
            "v1291_expected_gain_value", "v1291_expected_gain_confidence_value",
            "v1291_expected_gain_recent_value", "v1291_expected_gain_range_value",
            "v1291_expected_gain_training_rows_value", "v1291_expected_gain_weight_value",
            "v1291_expected_gain_resid_rmse_value", "v1291_expected_gain_top_driver_value",
            "v1291_expected_gain_portfolio_pass_value", "v1291_expected_gain_portfolio_gain_value",
            "v1291_expected_gain_portfolio_share_value", "v1291_expected_gain_min_gain_value",
            "v1291_expected_gain_min_confidence_value",
            "v1292_raw_expected_gain_value", "v1292_calibrated_gain_value",
            "v1292_calibration_bucket_value", "v1292_calibration_rows_value",
            "v1292_bucket_loss_rate_value", "v1292_bucket_p10_gain_value",
            "v1292_leaf_loss_rate_value", "v1292_leaf_p10_gain_value",
            "v1292_impact_percentile_value", "v1292_risk_score_value",
            "v1292_effective_min_gain_value", "v1292_effective_min_confidence_value",
            "v1292_effective_max_leaf_loss_value", "v1292_effective_min_p10_gain_value",
            "v1292_expected_impact_score_value", "v1292_budget_selected_value",
            "v1292_portfolio_budget_value", "v1292_portfolio_expected_gain_value",
            "v1292_portfolio_selected_share_value", "v1292_portfolio_pass_value",
            "v1293_segment_selector_enabled_value", "v1293_segment_prebudget_value",
            "v1293_segment_budget_selected_value", "v1293_segment_level_value",
            "v1293_segment_reason_value", "v1293_segment_key_value",
            "v1293_segment_folds_value", "v1293_segment_win_rate_value",
            "v1293_segment_median_gain_value", "v1293_segment_worst_gain_value",
            "v1293_segment_weighted_gain_value", "v1293_segment_utility_gain_value",
            "v1293_segment_bias_worsen_value", "v1293_segment_min_leaves_value",
            "v1293_segment_selected_volume_share_value", "v1293_segment_section_strong_value",
            "v1293_segment_budget_value",
            "v1294_segment_risk_score_value", "v1294_segment_exposure_budget_value",
            "v1294_impact_percentile_value", "v1294_high_impact_guard_pass_value",
            "v12_validation_blocks_y", "v12_validation_blocks_value",
            "v12_validation_wins_y", "v12_validation_wins_value",
            "v12_validation_improvement_y", "v12_validation_improvement_value",
            "v12_validation_recent_improvement_y", "v12_validation_recent_improvement_value",
            "v12_validation_utility_improvement_y", "v12_validation_utility_improvement_value",
            "v12_validation_recent_utility_improvement_y", "v12_validation_recent_utility_improvement_value",
            "v12_validation_utility_mean_y", "v12_validation_utility_mean_value",
            "v12_validation_utility_std_y", "v12_validation_utility_std_value",
            "v12_section_portfolio_gate_y", "v12_section_portfolio_gate_value",
            "v12_section_portfolio_gate_joint",
            "v12_section_portfolio_wins_y", "v12_section_portfolio_wins_value",
            "v12_section_portfolio_gain_y", "v12_section_portfolio_gain_value",
            "v12_section_portfolio_recent_gain_y", "v12_section_portfolio_recent_gain_value",
            "v12_validation_wmape_v11_y", "v12_validation_wmape_v12_y",
            "v12_validation_bias_v11_y", "v12_validation_bias_v12_y",
            "v12_validation_wmape_v11_value", "v12_validation_wmape_v12_value",
            "v12_validation_bias_v11_value", "v12_validation_bias_v12_value",
        )
    )

def apply_v12_occurrence_share_challenger(
    rows: pl.DataFrame,
    train_obs: pl.DataFrame,
    oos_obs: pl.DataFrame,
    horizons: dict,
    section_id: str,
    n_jobs: int = 1,
) -> pl.DataFrame:
    """Apply v12.9.8: v12.9.7 production policy + causal sparse-bias diagnostic."""
    if rows.height == 0:
        return rows
    if not bool(getattr(settings, "V12_LEAF_CHALLENGER_ENABLED", True)):
        return rows.with_columns(
            pl.col("yhat_raw").alias("v11_yhat_raw_before_v12"),
            pl.col("valuehat_raw").alias("v11_valuehat_raw_before_v12"),
            pl.lit("v11_ses_rls").alias("leaf_model_family_y"),
            pl.lit("v11_ses_rls").alias("leaf_model_family_value"),
            pl.lit(False).alias("v12_selected_y"),
            pl.lit(False).alias("v12_selected_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_candidate_yhat_raw"),
            pl.lit(None, dtype=pl.Float64).alias("v12_candidate_valuehat_raw"),
            pl.lit(None, dtype=pl.Float64).alias("v12_sku_forecast_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_sku_forecast_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_sku_forecast_base_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_sku_forecast_base_value"),
            pl.lit(None, dtype=pl.String).alias("v12_sku_shape_model_y"),
            pl.lit(None, dtype=pl.String).alias("v12_sku_shape_model_value"),
            pl.lit(0, dtype=pl.Int32).alias("v12_sku_shape_training_rows_y"),
            pl.lit(0, dtype=pl.Int32).alias("v12_sku_shape_training_rows_value"),
            pl.lit(False).alias("v12_sku_shape_applied_y"),
            pl.lit(False).alias("v12_sku_shape_applied_value"),
            pl.lit(None, dtype=pl.String).alias("v12_sku_model_y"),
            pl.lit(None, dtype=pl.String).alias("v12_sku_model_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_sku_validation_wmape_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_sku_validation_wmape_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_store_share_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_store_share_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_occurrence_prob_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_occurrence_prob_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_occurrence_prob_joint"),
            pl.lit(False).alias("v12_occurrence_gate_open_y"),
            pl.lit(False).alias("v12_occurrence_gate_open_value"),
            pl.lit(False).alias("v12_occurrence_gate_open_joint"),
            pl.lit(False).alias("v12_candidate_available_y"),
            pl.lit(False).alias("v12_candidate_available_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_wmape_v11_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_wmape_v12_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_bias_v11_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_bias_v12_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_wmape_v11_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_wmape_v12_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_bias_v11_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_bias_v12_value"),
            pl.lit(0, dtype=pl.Int16).alias("v12_validation_blocks_y"),
            pl.lit(0, dtype=pl.Int16).alias("v12_validation_blocks_value"),
            pl.lit(0, dtype=pl.Int16).alias("v12_validation_wins_y"),
            pl.lit(0, dtype=pl.Int16).alias("v12_validation_wins_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_improvement_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_improvement_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_recent_improvement_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_validation_recent_improvement_value"),
            pl.lit(False).alias("v12_section_portfolio_gate_y"),
            pl.lit(False).alias("v12_section_portfolio_gate_value"),
            pl.lit(False).alias("v12_section_portfolio_gate_joint"),
            pl.lit(False).alias("v12_joint_target_preselect"),
            pl.lit(0, dtype=pl.Int16).alias("v12_section_portfolio_wins_y"),
            pl.lit(0, dtype=pl.Int16).alias("v12_section_portfolio_wins_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_section_portfolio_gain_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_section_portfolio_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_section_portfolio_recent_gain_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_section_portfolio_recent_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_probability_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_probability_value"),
            pl.lit(False).alias("v12_meta_model_available_y"),
            pl.lit(False).alias("v12_meta_model_available_value"),
            pl.lit(0, dtype=pl.Int32).alias("v12_meta_training_rows_y"),
            pl.lit(0, dtype=pl.Int32).alias("v12_meta_training_rows_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_recent_gain_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_recent_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_weighted_gain_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_weighted_gain_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_win_rate_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_win_rate_value"),
            pl.lit(None, dtype=pl.Utf8).alias("v12_meta_top_driver_y"),
            pl.lit(None, dtype=pl.Utf8).alias("v12_meta_top_driver_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_bias11_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_bias11_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_bias12_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_bias12_value"),
            pl.lit(False).alias("v12_meta_bias_guard_pass_y"),
            pl.lit(False).alias("v12_meta_bias_guard_pass_value"),
            pl.lit(float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))).alias("v12_meta_threshold_y"),
            pl.lit(float(getattr(settings, "V12_META_SELECTOR_THRESHOLD", 0.55))).alias("v12_meta_threshold_value"),
            pl.lit("v11_all").alias("v12_meta_portfolio_mode_y"),
            pl.lit("v11_all").alias("v12_meta_portfolio_mode_value"),
            pl.lit(False).alias("v12_meta_policy_available_y"),
            pl.lit(False).alias("v12_meta_policy_available_value"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_policy_utility_gain_y"),
            pl.lit(None, dtype=pl.Float64).alias("v12_meta_policy_utility_gain_value"),
            pl.lit(False).alias("v12_value_safety_enabled"),
            pl.lit(False).alias("v12_value_safety_dominance_pass"),
            pl.lit(0, dtype=pl.Int8).alias("v12_value_safety_recent_confirmations"),
            pl.lit(0, dtype=pl.Int8).alias("v12_value_safety_recent_blocks"),
            pl.lit(None, dtype=pl.Float64).alias("v12_value_safety_bias_coverage"),
            pl.lit(None, dtype=pl.Float64).alias("v12_value_safety_bias_coverage_threshold"),
            pl.lit(False).alias("v12_value_safety_bias_coverage_pass"),
            pl.lit(False).alias("v12_value_safety_meta_margin_pass"),
            pl.lit("v11_all").alias("v12_value_safety_best_all_mode"),
            pl.lit("challenger_disabled").alias("v12_value_safety_reason"),
            pl.lit(False).alias("v129_value_wf_enabled"),
            pl.lit(False).alias("v129_value_wf_available"),
            pl.lit(0, dtype=pl.Int8).alias("v129_value_wf_folds"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_win_rate"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_median_gain"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_worst_gain"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_weighted_gain"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_weighted_utility_gain"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_bias_worsen_max"),
            pl.lit(0, dtype=pl.Int8).alias("v129_value_wf_meta_folds"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_fold1_gain"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_fold2_gain"),
            pl.lit(None, dtype=pl.Float64).alias("v129_value_wf_fold3_gain"),
            pl.lit("challenger_disabled").alias("v129_value_wf_reason"),
        )

    block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
    test_start = _as_date(horizons["test_start"])
    test_end = _as_date(horizons["test_end"])
    forecast_start = _as_date(horizons["forecast_start"])
    forecast_end = _as_date(horizons["forecast_end"])
    selection_blocks = int(getattr(settings, "V12_SELECTION_BLOCKS", 4))
    selection_blocks = max(2, selection_blocks)
    # v12.9.6 materializes a deeper, diagnostic-only history. Productive
    # selection continues to see exactly ``selection_blocks``; the extra
    # closed blocks are consumed only by the long-horizon stress audit.
    stress_enabled = bool(getattr(settings, "V1295_STRESS_TEST_ENABLED", True)) and block_days == 28
    stress_requested = int(getattr(settings, "V1295_STRESS_TEST_BLOCKS", 12)) if stress_enabled else selection_blocks
    stress_blocks = max(selection_blocks, stress_requested)
    if stress_enabled and train_obs.height:
        _hist_min_raw = train_obs.select(pl.col("ds").cast(pl.Date).min()).item()
        _hist_min = _as_date(_hist_min_raw) if _hist_min_raw is not None else None
        if _hist_min is not None:
            _full_closed_blocks = max(0, (test_start - _hist_min).days // block_days)
            stress_blocks = max(selection_blocks, min(stress_blocks, _full_closed_blocks))
    validation_windows: list[tuple[dt.date, dt.date]] = []
    for i in range(1, stress_blocks + 1):
        start = test_start - dt.timedelta(days=block_days * i)
        end = start + dt.timedelta(days=block_days - 1)
        validation_windows.append((start, end))

    # Use only actual observations.  forecast-only synthetic rows never enter
    # the history frame.  For forecast-only origin, OOS is already observed.
    history_parts = [
        train_obs.select("unique_id", "ds", "y", "value").with_columns(pl.col("ds").cast(pl.Date))
    ]
    if oos_obs.height:
        history_parts.append(
            oos_obs.select("unique_id", "ds", "y", "value").with_columns(pl.col("ds").cast(pl.Date))
        )
    history_all = pl.concat(history_parts, how="vertical_relaxed")
    prepared_history = _leaf_keys(history_all).with_columns(
        _finite_nonnegative("y").alias("_v12_y"),
        _finite_nonnegative("value").alias("_v12_v"),
        pl.col("ds").dt.weekday().cast(pl.Int8).alias("_v12_dow"),
    )
    sku_daily_all = (
        prepared_history.group_by(["_v12_sku", "ds"])
        .agg(
            pl.col("_v12_y").sum().alias("_sku_day_y"),
            pl.col("_v12_v").sum().alias("_sku_day_v"),
        )
    )
    sku_first_all = (
        prepared_history.group_by("_v12_sku")
        .agg(pl.col("ds").min().alias("_sku_first_ds"))
    )
    uid_first_all = (
        prepared_history.group_by(["unique_id", "_v12_sku"])
        .agg(pl.col("ds").min().alias("_uid_first_ds"))
    )

    ids = (
        _leaf_keys(rows.select("unique_id").unique())
        .select("unique_id", "_v12_sku", "_v12_store_uid")
    )

    # v12.6 pooled SHAPE layer.  The context precomputes the overlapping
    # closed SKU blocks once, then fits one LightGBM per section/target/origin.
    # The base 28-day SKU total is a hard invariant and is never changed.
    shape_context = None
    if bool(getattr(settings, "V12_SKU_SHAPE_LGBM_ENABLED", True)):
        from app.forecasting.sku_shape_lgbm import PooledLGBMShapeContext

        history_blocks = int(getattr(settings, "V12_SKU_SHAPE_LGBM_HISTORY_BLOCKS", 16))
        sku_ids = ids.select("_v12_sku").unique()

        def _base_sku_forecast(start: dt.date, end: dt.date) -> pl.DataFrame:
            return _selected_sku_total_forecast(
                prepared_history=prepared_history,
                sku_daily_all=sku_daily_all,
                sku_first_all=sku_first_all,
                sku_ids=sku_ids,
                incumbent_rows=rows,
                origin=start,
                end=end,
            )

        shape_context = PooledLGBMShapeContext(
            sku_daily_all=sku_daily_all,
            sku_ids=sku_ids,
            section_id=str(section_id),
            target_origins=[start for start, _ in validation_windows] + [test_start, forecast_start],
            block_days=block_days,
            history_blocks=history_blocks,
            n_jobs=max(1, int(n_jobs)),
            base_forecast_builder=_base_sku_forecast,
        )

    def _candidate(start: dt.date, end: dt.date) -> pl.DataFrame:
        sku_override = (
            shape_context.corrected_forecast(start)
            if shape_context is not None
            else None
        )
        candidate = _block_candidate(
            prepared_history, sku_daily_all, sku_first_all, uid_first_all,
            ids, rows, start, end, sku_forecast_override=sku_override,
        )
        # The corrected/base SKU forecast for this origin is already copied into
        # the leaf candidate.  Do not retain one additional full DataFrame per
        # historical origin inside the pooled LightGBM context.
        if shape_context is not None:
            shape_context.release_origin(start)
        del sku_override
        return candidate

    historical_scores: list[pl.DataFrame] = []
    for idx, (start, end) in enumerate(validation_windows, start=1):
        cand = _candidate(start, end)
        historical_scores.append(_score_closed_block(rows, cand, start, end, idx))
        del cand

    # MEMORY-SAFE v12.9.12: all historical candidate scores are now
    # materialized.  Keep only feature blocks that can still enter the OOS or
    # forecast-only LightGBM training windows.  This is an exact causal prune:
    # each target retains itself + the same `history_blocks` closed origins.
    if shape_context is not None:
        shape_context.retain_for_targets([test_start, forecast_start])

    oos_candidate = _candidate(test_start, test_end)
    if shape_context is not None:
        # Forecast-only needs the newly closed OOS plus its older closed
        # history, but no feature block outside that exact window.
        shape_context.retain_for_targets([forecast_start])
    fc_candidate = _candidate(forecast_start, forecast_end)

    # At this point every LightGBM-derived candidate needed downstream has been
    # materialized.  The context can otherwise retain ~30 feature blocks plus
    # the complete SKU history through its closure, overlapping with the much
    # wider final leaf frame and causing the peak-RAM failure observed after
    # section 1.  This is a lifetime-only change; forecast algebra is untouched.
    if shape_context is not None:
        shape_context.close()
        shape_context = None
        gc.collect()

    def _segment_profile(period_type: str) -> pl.DataFrame:
        src = rows.filter(pl.col("period_type") == period_type)
        method_expr = (
            pl.col("leaf_level_method_value").first()
            if "leaf_level_method_value" in src.columns else pl.lit("unknown")
        ).alias("_seg_level_method_value")
        mult_expr = (
            pl.col("sku_seasonal_multiplier_value").first()
            if "sku_seasonal_multiplier_value" in src.columns else pl.lit(1.0)
        ).cast(pl.Float64).alias("_seg_seasonal_multiplier_value")
        return src.group_by("unique_id").agg(method_expr, mult_expr)

    # OOS production selection remains frozen on the original N strictly
    # closed blocks. The deeper v12.9.6 pool is diagnostic only.
    oos_scores = pl.concat(historical_scores[:selection_blocks], how="vertical_relaxed")
    stress_scores = pl.concat(historical_scores, how="vertical_relaxed")
    oos_profile = _segment_profile("out_sample")
    select_oos = _selection_from_closed_blocks(oos_scores, "out_sample", oos_profile)
    stress_oos = _value_long_horizon_segment_stress(stress_scores, oos_profile)
    select_oos = select_oos.join(stress_oos, on="unique_id", how="left")
    replay_oos = _v1297_sec23_temporal_replay(stress_scores, oos_profile)
    select_oos = select_oos.join(replay_oos, on="unique_id", how="left")
    select_oos = _apply_v1296_sec23_long_horizon_gate(select_oos, oos_scores)
    # v12.9.8 diagnostic only: learn sparse-demand uplift from the long CLOSED
    # history and attach the candidate metadata.  It does not change yhat/valuehat.
    sparse_oos = _v1298_sparse_bias_diagnostic(
        stress_scores, historical_scores[0], select_oos
    )
    select_oos = select_oos.join(sparse_oos, on="unique_id", how="left")
    # v12.9.9 learns segmented Value sparse-bias rescue factors; v12.9.10 adds
    # an ACTIVE-only causal replay gate that may promote them productively.
    sparse99_oos = _v1299_value_sparse_segmented_diagnostic(
        stress_scores, historical_scores[0], select_oos
    )
    select_oos = select_oos.join(sparse99_oos, on="unique_id", how="left")
    # v12.9.11 diagnostic-only residual selector-gap challenger.  It targets
    # Sec23 Value leaves still on v11 using leaf-specific long-horizon evidence.
    residual11_oos = _v12911_sec23_value_residual_selector(
        stress_scores, historical_scores[0], select_oos
    )
    select_oos = select_oos.join(residual11_oos, on="unique_id", how="left")
    ultra12_oos = _v12912_ultra_stable_residual_diagnostic(select_oos)
    select_oos = select_oos.join(ultra12_oos, on="unique_id", how="left")

    # Forecast-only selection: OOS is now closed/observed, followed by the
    # N-1 most recent historical blocks. No forecast-only actual is used.
    oos_closed_score = _score_closed_block(rows, oos_candidate, test_start, test_end, 1)
    fc_score_parts = [oos_closed_score]
    for idx, score in enumerate(historical_scores[: selection_blocks - 1], start=2):
        fc_score_parts.append(score.with_columns(pl.lit(idx).cast(pl.Int8).alias("_validation_block")))
    fc_profile = _segment_profile("forecast_only")
    select_fc = _selection_from_closed_blocks(
        pl.concat(fc_score_parts, how="vertical_relaxed"), "forecast_only", fc_profile
    )
    # Keep the current OOS completely outside the long-horizon stress evidence;
    # forecast-only carries the same pre-OOS historical stress test with its own
    # current segment profile.
    stress_fc = _value_long_horizon_segment_stress(stress_scores, fc_profile)
    select_fc = select_fc.join(stress_fc, on="unique_id", how="left")
    replay_fc = _v1297_sec23_temporal_replay(stress_scores, fc_profile)
    select_fc = select_fc.join(replay_fc, on="unique_id", how="left")
    select_fc = _apply_v1296_sec23_long_horizon_gate(
        select_fc, pl.concat(fc_score_parts, how="vertical_relaxed")
    )
    # For forecast-only the just-observed OOS is legitimately the newest CLOSED
    # block.  Shift the older stress history one position and cap at 12 folds.
    sparse_fc_parts = [oos_closed_score.with_columns(pl.lit(1).cast(pl.Int8).alias("_validation_block"))]
    for idx, score in enumerate(historical_scores[: max(0, int(getattr(settings, "V1295_STRESS_TEST_BLOCKS", 12)) - 1)], start=2):
        sparse_fc_parts.append(score.with_columns(pl.lit(idx).cast(pl.Int8).alias("_validation_block")))
    sparse_fc_history = pl.concat(sparse_fc_parts, how="vertical_relaxed")
    sparse_fc = _v1298_sparse_bias_diagnostic(sparse_fc_history, oos_closed_score, select_fc)
    select_fc = select_fc.join(sparse_fc, on="unique_id", how="left")
    sparse99_fc = _v1299_value_sparse_segmented_diagnostic(
        sparse_fc_history, oos_closed_score, select_fc
    )
    select_fc = select_fc.join(sparse99_fc, on="unique_id", how="left")
    residual11_fc = _v12911_sec23_value_residual_selector(
        sparse_fc_history, oos_closed_score, select_fc
    )
    select_fc = select_fc.join(residual11_fc, on="unique_id", how="left")
    ultra12_fc = _v12912_ultra_stable_residual_diagnostic(select_fc)
    select_fc = select_fc.join(ultra12_fc, on="unique_id", how="left")
    selections = pl.concat([select_oos, select_fc], how="vertical_relaxed")

    candidates = pl.concat(
        [
            oos_candidate.with_columns(pl.lit("out_sample").alias("period_type")),
            fc_candidate.with_columns(pl.lit("forecast_only").alias("period_type")),
        ],
        how="vertical_relaxed",
    )

    out = (
        rows.with_columns(
            pl.col("yhat_raw").alias("v11_yhat_raw_before_v12"),
            pl.col("valuehat_raw").alias("v11_valuehat_raw_before_v12"),
        )
        .join(candidates, on=["unique_id", "ds", "period_type"], how="left")
        .join(selections, on=["unique_id", "period_type"], how="left")
        .with_columns(
            (
                pl.col("v12_selected_y").fill_null(False)
                & pl.col("v12_candidate_available_y").fill_null(False)
            ).alias("v12_selected_y"),
            (
                pl.col("v12_selected_value").fill_null(False)
                & pl.col("v12_candidate_available_value").fill_null(False)
            ).alias("v12_selected_value"),
            pl.when(pl.col("v12_selected_y").fill_null(False))
            .then(pl.lit("v12_occ_share"))
            .otherwise(pl.lit("v11_ses_rls"))
            .alias("leaf_model_family_y"),
            pl.when(pl.col("v12_selected_value").fill_null(False))
            .then(pl.lit("v12_occ_share"))
            .otherwise(pl.lit("v11_ses_rls"))
            .alias("leaf_model_family_value"),
        )
        .with_columns(
            pl.when(pl.col("v12_selected_y"))
            .then(pl.col("v12_candidate_yhat_raw"))
            .otherwise(pl.col("yhat_raw"))
            .alias("yhat_raw"),
            pl.when(pl.col("v12_selected_value"))
            .then(pl.col("v12_candidate_valuehat_raw"))
            .otherwise(pl.col("valuehat_raw"))
            .alias("valuehat_raw"),
        )
        .with_columns(
            pl.col("valuehat_raw").alias("v12910_valuehat_raw_before_sparse"),
            (
                pl.col("v12910_value_sparse_promoted").fill_null(False)
                & pl.col("v1299_value_sparse_candidate").fill_null(False)
                & (pl.col("v1299_value_sparse_factor").fill_null(1.0) > 1.0)
            ).alias("v12910_value_sparse_applied"),
        )
        .with_columns(
            pl.when(pl.col("v12910_value_sparse_applied"))
            .then(pl.col("valuehat_raw") * pl.col("v1299_value_sparse_factor").fill_null(1.0))
            .otherwise(pl.col("valuehat_raw"))
            .alias("valuehat_raw"),
        )
        .with_columns(
            pl.col("yhat_raw").clip(lower_bound=0.0).round(0).alias("yhat"),
            pl.col("valuehat_raw").clip(lower_bound=0.0).round(2).alias("valuehat"),
            pl.when(pl.col("v12_selected_y") | pl.col("v12_selected_value"))
            .then(
                pl.concat_str(
                    [
                        pl.lit("v12_occ_share[y="), pl.col("v12_selected_y").cast(pl.Utf8),
                        pl.lit(",v="), pl.col("v12_selected_value").cast(pl.Utf8), pl.lit("]"),
                    ]
                )
            )
            .otherwise(pl.col("modelo_seleccionado"))
            .alias("modelo_seleccionado"),
        )
        .with_columns(
            pl.when(pl.col("v12910_value_sparse_applied"))
            .then(pl.concat_str([
                pl.col("modelo_seleccionado"), pl.lit("+v12910_value_sparse_x"),
                pl.col("v1299_value_sparse_factor").fill_null(1.0).round(3).cast(pl.Utf8),
            ]))
            .otherwise(pl.col("modelo_seleccionado"))
            .alias("modelo_seleccionado")
        )
    )

    selected_counts = out.filter(pl.col("period_type") == "out_sample").select(
        pl.col("unique_id").filter(pl.col("v12_selected_y")).n_unique().alias("y"),
        pl.col("unique_id").filter(pl.col("v12_selected_value")).n_unique().alias("v"),
    ).row(0, named=True)
    selection_gate = select_oos.select(
        pl.col("v12_section_portfolio_gate_y").first().alias("gy"),
        pl.col("v12_section_portfolio_gate_value").first().alias("gv"),
        pl.col("v12_section_portfolio_gate_joint").first().alias("gj"),
        pl.col("v12_section_portfolio_gain_y").first().alias("gain_y"),
        pl.col("v12_section_portfolio_gain_value").first().alias("gain_v"),
        pl.col("v12_section_portfolio_recent_gain_y").first().alias("recent_y"),
        pl.col("v12_section_portfolio_recent_gain_value").first().alias("recent_v"),
        pl.col("v12_meta_portfolio_mode_y").first().alias("mode_y"),
        pl.col("v12_meta_portfolio_mode_value").first().alias("mode_v"),
        pl.col("v12_meta_threshold_y").first().alias("th_y"),
        pl.col("v12_meta_threshold_value").first().alias("th_v"),
        pl.col("v12_meta_policy_utility_gain_y").first().alias("policy_gain_y"),
        pl.col("v12_meta_policy_utility_gain_value").first().alias("policy_gain_v"),
    ).row(0, named=True)
    logger.info(
        "v12.8.3 value-safety+adaptive-meta+SKU-ensemble+LGBM-shape+occurrence+share28x84 | sec=%s | OOS hojas seleccionadas y=%s v=%s | "
        "policy y=%s@%.2f v=%s@%.2f utility_gain y=%s v=%s | legacy_gate(diag) y=%s v=%s joint=%s | "
        "discovery_gain y=%s v=%s | confirm y=%s v=%s",
        section_id,
        int(selected_counts["y"] or 0),
        int(selected_counts["v"] or 0),
        str(selection_gate["mode_y"] or "NA"),
        float(selection_gate["th_y"] or 0.0),
        str(selection_gate["mode_v"] or "NA"),
        float(selection_gate["th_v"] or 0.0),
        "NA" if selection_gate["policy_gain_y"] is None else f"{100.0 * float(selection_gate['policy_gain_y']):.2f}%",
        "NA" if selection_gate["policy_gain_v"] is None else f"{100.0 * float(selection_gate['policy_gain_v']):.2f}%",
        bool(selection_gate["gy"]),
        bool(selection_gate["gv"]),
        bool(selection_gate["gj"]),
        "NA" if selection_gate["gain_y"] is None else f"{100.0 * float(selection_gate['gain_y']):.2f}%",
        "NA" if selection_gate["gain_v"] is None else f"{100.0 * float(selection_gate['gain_v']):.2f}%",
        "NA" if selection_gate["recent_y"] is None else f"{100.0 * float(selection_gate['recent_y']):.2f}%",
        "NA" if selection_gate["recent_v"] is None else f"{100.0 * float(selection_gate['recent_v']):.2f}%",
    )
    return out
