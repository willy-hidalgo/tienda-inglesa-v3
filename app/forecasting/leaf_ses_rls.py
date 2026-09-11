"""Coherent SKU+store model: deseasonalized SES level + one RLS parent.

Statistical contract (v13):

* Section and store forecasts are produced by the same expanding RLS engine.
* A leaf uses exactly one parent (store or section) per target and update block.
  The parent is chosen only from the parent's official positive-demand wMAPE
  accumulated in *previous* closed blocks.
* The parent contributes its RLS forecast as a relative movement around a
  causal recent parent level. A causal calibration center removes persistent
  whole-block offsets so the SES remains owner of the leaf baseline level.
  This avoids transferring either a non-identifiable coefficient split or an
  identifiable-but-level-like parent shift to the leaf.
* The leaf level is SES on positive observations deseasonalized with that
  stable relative parent movement and, when prior-year support is sufficient,
  a causal leaf-specific YoY monthly transition factor. The first state is the median of the first 28
  positive observations' calendar window, which is robust to both zeros and
  peaks.
* SES alpha, RLS dynamics/lambda and Section/Store parent are selected only from
  closed in-sample history. OOS is a holdout: it can advance recursive states
  after a block closes, but its errors never tune those selections.

All tabular work is Polars. Numba is used only for the small numerical SES
recurrence, analogous to the existing RLS numerical kernels.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl
from numba import njit

import settings

_EPS = 1e-12


@dataclass(frozen=True)
class LeafModelConfig:
    block_days: int
    warmup_days: int
    alphas: tuple[float, ...]
    default_alpha: float
    driver_effect_limit: float
    driver_reference_days: int
    driver_reference_min_points: int
    driver_reference_shift_factor: float
    driver_factor_min: float
    driver_factor_max: float
    ses_update_factor_min: float
    ses_update_factor_max: float
    forecast_history_max_multiplier: float
    forecast_history_mean_multiplier: float
    long_gap_forecast_mean_multiplier: float
    forecast_guard_min_positive_points: int
    oos_state_anchor_min_factor: float
    oos_state_anchor_max_factor: float
    oos_recent_positive_window: int
    oos_recent_median_min_points: int
    oos_recent_median_max_factor: float
    gap_aware_enabled: bool
    gap_min_observable_zero_days: int
    gap_decay_alpha_floor: float
    gap_decay_max_observable_days: int
    gap_decay_min_factor: float
    gap_reactivation_robust_bypass_days: int
    gap_reactivation_alpha_max: float
    gap_reactivation_state_factor: float
    gap_dormant_observable_days: int
    gap_dormant_factor: float
    yoy_seasonal_enabled: bool
    yoy_seasonal_min_positive_days: int
    yoy_seasonal_full_reliability_days: int
    yoy_seasonal_factor_min: float
    yoy_seasonal_factor_max: float
    default_parent: str

    @classmethod
    def from_settings(cls) -> "LeafModelConfig":
        alphas = tuple(
            sorted(
                {
                    float(x)
                    for x in getattr(
                        settings,
                        "LEAF_SES_ALPHA_CANDIDATES",
                        (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80),
                    )
                    if 0.0 < float(x) <= 1.0
                }
            )
        )
        default_alpha = float(getattr(settings, "LEAF_SES_ALPHA", 0.10))
        if default_alpha not in alphas:
            alphas = tuple(sorted(set(alphas) | {default_alpha}))
        default_parent = str(getattr(settings, "LEAF_PARENT_DEFAULT", "store"))
        if default_parent not in {"store", "section"}:
            default_parent = "store"
        effect_limit = float(getattr(settings, "LEAF_DRIVER_EFFECT_LIMIT", 20.0))
        reference_days = max(int(getattr(settings, "LEAF_DRIVER_REFERENCE_DAYS", 28)), 1)
        reference_min_points = max(
            int(getattr(settings, "LEAF_DRIVER_REFERENCE_MIN_POINTS", 7)), 1
        )
        reference_shift_factor = float(
            getattr(settings, "LEAF_DRIVER_REFERENCE_SHIFT_FACTOR", 1.50)
        )
        if reference_shift_factor < 1.0:
            raise ValueError("LEAF_DRIVER_REFERENCE_SHIFT_FACTOR debe ser >= 1")
        factor_min = float(getattr(settings, "LEAF_DRIVER_FACTOR_MIN", 0.50))
        factor_max = float(getattr(settings, "LEAF_DRIVER_FACTOR_MAX", 2.50))
        if not (0.0 < factor_min <= 1.0 <= factor_max):
            raise ValueError(
                "LEAF_DRIVER_FACTOR_MIN/MAX inválidos; se requiere 0 < min <= 1 <= max"
            )
        update_factor_min = float(getattr(settings, "LEAF_SES_UPDATE_FACTOR_MIN", 0.50))
        update_factor_max = float(getattr(settings, "LEAF_SES_UPDATE_FACTOR_MAX", 2.00))
        if not (0.0 < update_factor_min <= 1.0 <= update_factor_max):
            raise ValueError(
                "LEAF_SES_UPDATE_FACTOR_MIN/MAX inválidos; se requiere 0 < min <= 1 <= max"
            )
        forecast_history_max_multiplier = float(
            getattr(settings, "LEAF_FORECAST_HISTORY_MAX_MULTIPLIER", 2.00)
        )
        if forecast_history_max_multiplier < 1.0:
            raise ValueError("LEAF_FORECAST_HISTORY_MAX_MULTIPLIER debe ser >= 1")
        forecast_history_mean_multiplier = float(
            getattr(settings, "LEAF_FORECAST_HISTORY_MEAN_MULTIPLIER", 4.00)
        )
        if forecast_history_mean_multiplier < 1.0:
            raise ValueError("LEAF_FORECAST_HISTORY_MEAN_MULTIPLIER debe ser >= 1")
        long_gap_forecast_mean_multiplier = float(
            getattr(settings, "LEAF_LONG_GAP_FORECAST_MEAN_MULTIPLIER", 2.00)
        )
        if long_gap_forecast_mean_multiplier < 1.0:
            raise ValueError("LEAF_LONG_GAP_FORECAST_MEAN_MULTIPLIER debe ser >= 1")
        forecast_guard_min_positive_points = max(
            int(getattr(settings, "LEAF_FORECAST_GUARD_MIN_POSITIVE_POINTS", 7)), 1
        )
        oos_state_anchor_min_factor = float(
            getattr(settings, "LEAF_OOS_STATE_ANCHOR_MIN_FACTOR", 0.50)
        )
        oos_state_anchor_max_factor = float(
            getattr(settings, "LEAF_OOS_STATE_ANCHOR_MAX_FACTOR", 1.50)
        )
        if not (0.0 < oos_state_anchor_min_factor <= 1.0 <= oos_state_anchor_max_factor):
            raise ValueError(
                "LEAF_OOS_STATE_ANCHOR_MIN/MAX inválidos; se requiere 0 < min <= 1 <= max"
            )
        oos_recent_positive_window = max(
            int(getattr(settings, "LEAF_OOS_RECENT_POSITIVE_WINDOW", 28)), 1
        )
        oos_recent_median_min_points = max(
            int(getattr(settings, "LEAF_OOS_RECENT_MEDIAN_MIN_POINTS", 7)), 1
        )
        oos_recent_median_min_points = min(
            oos_recent_median_min_points, oos_recent_positive_window
        )
        oos_recent_median_max_factor = float(
            getattr(settings, "LEAF_OOS_RECENT_MEDIAN_MAX_FACTOR", 1.50)
        )
        if oos_recent_median_max_factor < 1.0:
            raise ValueError("LEAF_OOS_RECENT_MEDIAN_MAX_FACTOR debe ser >= 1")
        gap_aware_enabled = bool(getattr(settings, "LEAF_GAP_AWARE_ENABLED", True))
        gap_min_observable_zero_days = max(
            int(getattr(settings, "LEAF_GAP_MIN_OBSERVABLE_ZERO_DAYS", 1)), 1
        )
        gap_decay_alpha_floor = float(
            getattr(settings, "LEAF_GAP_DECAY_ALPHA_FLOOR", 0.025)
        )
        if not (0.0 < gap_decay_alpha_floor <= 1.0):
            raise ValueError("LEAF_GAP_DECAY_ALPHA_FLOOR debe estar en (0,1]")
        gap_decay_max_observable_days = max(
            int(getattr(settings, "LEAF_GAP_DECAY_MAX_OBSERVABLE_DAYS", 56)), 1
        )
        gap_decay_min_factor = float(
            getattr(settings, "LEAF_GAP_DECAY_MIN_FACTOR", 0.25)
        )
        if not (0.0 < gap_decay_min_factor <= 1.0):
            raise ValueError("LEAF_GAP_DECAY_MIN_FACTOR debe estar en (0,1]")
        gap_reactivation_robust_bypass_days = max(
            int(getattr(settings, "LEAF_GAP_REACTIVATION_ROBUST_BYPASS_DAYS", 28)), 1
        )
        gap_reactivation_alpha_max = float(
            getattr(settings, "LEAF_GAP_REACTIVATION_ALPHA_MAX", 0.20)
        )
        if not (0.0 < gap_reactivation_alpha_max <= 1.0):
            raise ValueError("LEAF_GAP_REACTIVATION_ALPHA_MAX debe estar en (0,1]")
        gap_reactivation_state_factor = float(
            getattr(settings, "LEAF_GAP_REACTIVATION_STATE_FACTOR", 2.00)
        )
        if gap_reactivation_state_factor < 1.0:
            raise ValueError("LEAF_GAP_REACTIVATION_STATE_FACTOR debe ser >= 1")
        gap_dormant_observable_days = max(
            int(getattr(settings, "LEAF_GAP_DORMANT_OBSERVABLE_DAYS", 84)), 1
        )
        gap_dormant_factor = float(
            getattr(settings, "LEAF_GAP_DORMANT_FACTOR", 1e-6)
        )
        if not (0.0 < gap_dormant_factor <= 1.0):
            raise ValueError("LEAF_GAP_DORMANT_FACTOR debe estar en (0,1]")
        yoy_seasonal_enabled = bool(
            getattr(settings, "LEAF_YOY_SEASONAL_ENABLED", True)
        )
        yoy_seasonal_min_positive_days = max(
            int(getattr(settings, "LEAF_YOY_SEASONAL_MIN_POSITIVE_DAYS", 7)), 1
        )
        yoy_seasonal_full_reliability_days = max(
            int(getattr(settings, "LEAF_YOY_SEASONAL_FULL_RELIABILITY_DAYS", 14)),
            yoy_seasonal_min_positive_days,
        )
        yoy_seasonal_factor_min = float(
            getattr(settings, "LEAF_YOY_SEASONAL_FACTOR_MIN", 0.50)
        )
        yoy_seasonal_factor_max = float(
            getattr(settings, "LEAF_YOY_SEASONAL_FACTOR_MAX", 1.50)
        )
        if not (0.0 < yoy_seasonal_factor_min <= 1.0 <= yoy_seasonal_factor_max):
            raise ValueError(
                "LEAF_YOY_SEASONAL_FACTOR_MIN/MAX inválidos; "
                "se requiere 0 < min <= 1 <= max"
            )
        return cls(
            block_days=int(getattr(settings, "RLS_BLOCK_DAYS", 28)),
            warmup_days=int(getattr(settings, "LEAF_INITIAL_LEVEL_DAYS", 28)),
            alphas=alphas,
            default_alpha=default_alpha,
            driver_effect_limit=max(effect_limit, 1.0),
            driver_reference_days=reference_days,
            driver_reference_min_points=min(reference_min_points, reference_days),
            driver_reference_shift_factor=reference_shift_factor,
            driver_factor_min=factor_min,
            driver_factor_max=factor_max,
            ses_update_factor_min=update_factor_min,
            ses_update_factor_max=update_factor_max,
            forecast_history_max_multiplier=forecast_history_max_multiplier,
            forecast_history_mean_multiplier=forecast_history_mean_multiplier,
            long_gap_forecast_mean_multiplier=long_gap_forecast_mean_multiplier,
            forecast_guard_min_positive_points=forecast_guard_min_positive_points,
            oos_state_anchor_min_factor=oos_state_anchor_min_factor,
            oos_state_anchor_max_factor=oos_state_anchor_max_factor,
            oos_recent_positive_window=oos_recent_positive_window,
            oos_recent_median_min_points=oos_recent_median_min_points,
            oos_recent_median_max_factor=oos_recent_median_max_factor,
            gap_aware_enabled=gap_aware_enabled,
            gap_min_observable_zero_days=gap_min_observable_zero_days,
            gap_decay_alpha_floor=gap_decay_alpha_floor,
            gap_decay_max_observable_days=gap_decay_max_observable_days,
            gap_decay_min_factor=gap_decay_min_factor,
            gap_reactivation_robust_bypass_days=gap_reactivation_robust_bypass_days,
            gap_reactivation_alpha_max=gap_reactivation_alpha_max,
            gap_reactivation_state_factor=gap_reactivation_state_factor,
            gap_dormant_observable_days=gap_dormant_observable_days,
            gap_dormant_factor=gap_dormant_factor,
            yoy_seasonal_enabled=yoy_seasonal_enabled,
            yoy_seasonal_min_positive_days=yoy_seasonal_min_positive_days,
            yoy_seasonal_full_reliability_days=yoy_seasonal_full_reliability_days,
            yoy_seasonal_factor_min=yoy_seasonal_factor_min,
            yoy_seasonal_factor_max=yoy_seasonal_factor_max,
            default_parent=default_parent,
        )


def _block_expr(train_start: dt.date, block_days: int) -> pl.Expr:
    return (
        ((pl.col("ds").cast(pl.Date) - pl.lit(train_start)).dt.total_days() // block_days)
        .cast(pl.Int32)
        .alias("_block")
    )


def _period_block_expr(
    train_start: dt.date,
    test_start: dt.date,
    forecast_start: dt.date,
    block_days: int,
) -> pl.Expr:
    """Block id aligned to period boundaries.

    In-sample, OOS and forecast-only never share a block. This matters for
    fixed-origin OOS: no cadence may absorb a partial OOS tail inside the last
    historical block simply because ``block_days`` differs.
    """
    ds = pl.col("ds").cast(pl.Date)
    hist = ((ds - pl.lit(train_start)).dt.total_days() // block_days).cast(pl.Int32)
    oos = (
        ((ds - pl.lit(test_start)).dt.total_days() // block_days).cast(pl.Int32)
        + pl.lit(1_000_000, dtype=pl.Int32)
    )
    fo = (
        ((ds - pl.lit(forecast_start)).dt.total_days() // block_days).cast(pl.Int32)
        + pl.lit(2_000_000, dtype=pl.Int32)
    )
    return (
        pl.when(ds < pl.lit(test_start))
        .then(hist)
        .when(ds < pl.lit(forecast_start))
        .then(oos)
        .otherwise(fo)
        .alias("_block")
    )


def _positive_median_levels(
    train_obs: pl.DataFrame,
    *,
    warmup_days: int,
) -> pl.DataFrame:
    """Per-leaf robust initial level from the first 28 calendar days.

    Only positive-sale observations contribute. The value target uses the same
    sale support (y>0), preserving the client's contract that zero-sale points
    are excluded from the official metric.
    """
    bounds = (
        train_obs.group_by("unique_id")
        .agg(pl.col("ds").cast(pl.Date).min().alias("leaf_start"))
        .with_columns(
            (pl.col("leaf_start") + pl.duration(days=warmup_days - 1)).alias("leaf_warmup_end")
        )
    )
    warm = (
        train_obs.join(bounds, on="unique_id", how="left")
        .filter(pl.col("ds").cast(pl.Date) <= pl.col("leaf_warmup_end"))
        .group_by("unique_id")
        .agg(
            pl.col("y")
            .filter(pl.col("y").is_finite() & (pl.col("y") > 0))
            .median()
            .fill_null(0.0)
            .alias("initial_level_y"),
            pl.col("value")
            .filter(
                pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("value").is_finite()
                & (pl.col("value") > 0)
            )
            .median()
            .fill_null(0.0)
            .alias("initial_level_value"),
            ((pl.col("y").is_finite()) & (pl.col("y") > 0))
            .sum()
            .cast(pl.Int32)
            .alias("warmup_positive_days"),
        )
    )
    return bounds.join(warm, on="unique_id", how="left").with_columns(
        pl.col("initial_level_y").fill_null(0.0).clip(lower_bound=0.0),
        pl.col("initial_level_value").fill_null(0.0).clip(lower_bound=0.0),
        pl.col("warmup_positive_days").fill_null(0),
    )



def _attach_leaf_yoy_monthly_seasonality(
    rows: pl.DataFrame,
    train_obs: pl.DataFrame,
    *,
    enabled: bool,
    min_positive_days: int,
    full_reliability_days: int,
    factor_min: float,
    factor_max: float,
) -> pl.DataFrame:
    """Attach a causal leaf-specific YoY seasonal transition factor.

    For each target month M in year Y, compare the robust positive-demand level
    of M in Y-1 against the immediately preceding month in that historical
    year. Example: Apr-2026 uses Apr-2025 / Mar-2025; May-2026 uses
    May-2025 / Apr-2025. Therefore neither OOS nor forecast-only actuals from
    the target year can enter the factor.

    The raw median ratio is shrunk toward 1.0 using the smaller positive-day
    count of the two historical months and then clipped conservatively. When
    support is insufficient the factor is exactly 1.0. This keeps the feature
    cheap, causal, interpretable and cadence-independent.
    """
    if (not enabled) or rows.height == 0 or train_obs.height == 0:
        return rows.with_columns(
            pl.lit(1.0).alias("leaf_yoy_factor_y"),
            pl.lit(1.0).alias("leaf_yoy_factor_value"),
            pl.lit(1.0).alias("leaf_yoy_raw_factor_y"),
            pl.lit(1.0).alias("leaf_yoy_raw_factor_value"),
            pl.lit(0.0).alias("leaf_yoy_reliability_y"),
            pl.lit(0.0).alias("leaf_yoy_reliability_value"),
        )

    min_days = max(int(min_positive_days), 1)
    full_days = max(int(full_reliability_days), min_days)

    monthly = (
        train_obs.with_columns(
            pl.col("ds").cast(pl.Date),
            pl.col("ds").dt.year().cast(pl.Int32).alias("_hist_year"),
            pl.col("ds").dt.month().cast(pl.Int8).alias("_hist_month"),
        )
        .group_by(["unique_id", "_hist_year", "_hist_month"])
        .agg(
            pl.col("y")
            .filter(pl.col("y").is_finite() & (pl.col("y") > 0))
            .median()
            .alias("_hist_level_y"),
            pl.col("value")
            .filter(
                pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("value").is_finite()
                & (pl.col("value") > 0)
            )
            .median()
            .alias("_hist_level_value"),
            (pl.col("y").is_finite() & (pl.col("y") > 0))
            .sum()
            .cast(pl.Int16)
            .alias("_hist_n_y"),
            (
                pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("value").is_finite()
                & (pl.col("value") > 0)
            )
            .sum()
            .cast(pl.Int16)
            .alias("_hist_n_value"),
        )
    )

    same = monthly.rename(
        {
            "_hist_year": "_same_year",
            "_hist_month": "_same_month",
            "_hist_level_y": "_same_level_y",
            "_hist_level_value": "_same_level_value",
            "_hist_n_y": "_same_n_y",
            "_hist_n_value": "_same_n_value",
        }
    )
    prev = monthly.rename(
        {
            "_hist_year": "_prev_year",
            "_hist_month": "_prev_month",
            "_hist_level_y": "_prev_level_y",
            "_hist_level_value": "_prev_level_value",
            "_hist_n_y": "_prev_n_y",
            "_hist_n_value": "_prev_n_value",
        }
    )

    out = rows.with_columns(
        (pl.col("ds").dt.year() - 1).cast(pl.Int32).alias("_same_year"),
        pl.col("ds").dt.month().cast(pl.Int8).alias("_same_month"),
        pl.when(pl.col("ds").dt.month() == 1)
        .then(pl.col("ds").dt.year() - 2)
        .otherwise(pl.col("ds").dt.year() - 1)
        .cast(pl.Int32)
        .alias("_prev_year"),
        pl.when(pl.col("ds").dt.month() == 1)
        .then(pl.lit(12))
        .otherwise(pl.col("ds").dt.month() - 1)
        .cast(pl.Int8)
        .alias("_prev_month"),
    )
    out = out.join(
        same,
        on=["unique_id", "_same_year", "_same_month"],
        how="left",
    ).join(
        prev,
        on=["unique_id", "_prev_year", "_prev_month"],
        how="left",
    )

    def _factor_expr(target: str) -> list[pl.Expr]:
        same_level = pl.col(f"_same_level_{target}")
        prev_level = pl.col(f"_prev_level_{target}")
        same_n = pl.col(f"_same_n_{target}").fill_null(0)
        prev_n = pl.col(f"_prev_n_{target}").fill_null(0)
        support = pl.min_horizontal(same_n, prev_n)
        reliability = (support.cast(pl.Float64) / float(full_days)).clip(0.0, 1.0)
        valid = (
            (same_n >= min_days)
            & (prev_n >= min_days)
            & same_level.is_finite()
            & prev_level.is_finite()
            & (same_level > 0)
            & (prev_level > 0)
        )
        raw = same_level / prev_level
        shrunk = (1.0 + reliability * (raw - 1.0)).clip(factor_min, factor_max)
        public = "value" if target == "value" else "y"
        return [
            pl.when(valid).then(raw).otherwise(1.0).alias(f"leaf_yoy_raw_factor_{public}"),
            pl.when(valid).then(reliability).otherwise(0.0).alias(f"leaf_yoy_reliability_{public}"),
            pl.when(valid).then(shrunk).otherwise(1.0).alias(f"leaf_yoy_factor_{public}"),
        ]

    out = out.with_columns(*_factor_expr("y"), *_factor_expr("value"))
    return out.drop(
        [
            "_same_year",
            "_same_month",
            "_prev_year",
            "_prev_month",
            "_same_level_y",
            "_same_level_value",
            "_same_n_y",
            "_same_n_value",
            "_prev_level_y",
            "_prev_level_value",
            "_prev_n_y",
            "_prev_n_value",
        ],
        strict=False,
    )


def _store_observable_calendar(
    train_obs: pl.DataFrame,
    oos_obs: pl.DataFrame,
    *,
    oos_observable_store_days: pl.DataFrame | None = None,
    train_start: dt.date,
    test_start: dt.date,
    forecast_start: dt.date,
    forecast_end: dt.date,
    block_days: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a tiny store/day observability spine for gap-aware leaf SES.

    ``selected.parquet`` intentionally contains only positive transactions, so
    a missing SKU/day row is ambiguous by itself.  A day is therefore treated
    as an observable zero for a leaf only when *some* eligible SKU in the same
    Section+Store has a positive transaction that day.  Days with no evidence
    that the store/section was observed do not decay the leaf state.

    The returned cumulative index is causal: at a block origin we use the count
    through the previous calendar day.  OOS store observability can influence
    only later closed blocks, never the block currently being forecast.
    """
    if train_obs.height == 0:
        return pl.DataFrame(), pl.DataFrame()

    def _store_days(df: pl.DataFrame) -> pl.DataFrame:
        if df.height == 0:
            return pl.DataFrame(schema={"_store_uid": pl.Utf8, "ds": pl.Date})
        return (
            df.select(
                pl.col("unique_id")
                .str.replace(r"\|\|S:.*$", "")
                .alias("_store_uid"),
                pl.col("ds").cast(pl.Date),
            )
            .unique()
        )

    stores = _store_days(train_obs).select("_store_uid").unique().sort("_store_uid")
    dates = pl.DataFrame(
        {
            "ds": pl.date_range(
                train_start, forecast_end, interval="1d", eager=True
            )
        }
    ).with_columns(pl.col("ds").cast(pl.Date))

    observed_parts = [_store_days(train_obs)]
    if oos_observable_store_days is not None and oos_observable_store_days.height:
        observed_parts.append(
            oos_observable_store_days.select(
                pl.col("_store_uid").cast(pl.Utf8),
                pl.col("ds").cast(pl.Date),
            ).unique()
        )
    elif oos_obs.height:
        observed_parts.append(_store_days(oos_obs))
    observed = (
        pl.concat(observed_parts, how="vertical_relaxed")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("_store_observed"))
    )

    calendar = (
        stores.join(dates, how="cross")
        .join(observed, on=["_store_uid", "ds"], how="left")
        .with_columns(
            pl.col("_store_observed").fill_null(0).cast(pl.Int8),
            _period_block_expr(train_start, test_start, forecast_start, block_days),
        )
        .sort(["_store_uid", "ds"])
        .with_columns(
            pl.col("_store_observed")
            .cum_sum()
            .over("_store_uid")
            .cast(pl.Int32)
            .alias("_observable_seq")
        )
        .with_columns(
            (pl.col("_observable_seq") - pl.col("_store_observed"))
            .cast(pl.Int32)
            .alias("_observable_seq_before_day")
        )
    )

    origins = (
        calendar.group_by(["_store_uid", "_block"], maintain_order=True)
        .agg(
            pl.col("_observable_seq_before_day")
            .first()
            .cast(pl.Int32)
            .alias("_block_origin_observable_seq")
        )
    )
    return calendar, origins


def _attach_gap_observability(
    rows: pl.DataFrame,
    levels: pl.DataFrame,
    calendar: pl.DataFrame,
    origins: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Attach causal store-observable indexes without densifying every leaf."""
    if calendar.height == 0:
        levels = levels.with_columns(
            pl.lit(0).cast(pl.Int32).alias("warmup_end_observable_seq")
        )
        rows = rows.with_columns(
            pl.lit(0).cast(pl.Int32).alias("_observable_seq"),
            pl.lit(0).cast(pl.Int8).alias("_store_observed"),
            pl.lit(0).cast(pl.Int32).alias("_block_origin_observable_seq"),
        )
        return rows, levels

    warmup_lookup = calendar.select(
        "_store_uid",
        pl.col("ds").alias("leaf_warmup_end"),
        pl.col("_observable_seq").alias("warmup_end_observable_seq"),
    )
    levels = (
        levels.with_columns(
            pl.col("unique_id")
            .str.replace(r"\|\|S:.*$", "")
            .alias("_store_uid")
        )
        .join(warmup_lookup, on=["_store_uid", "leaf_warmup_end"], how="left")
        .with_columns(
            pl.col("warmup_end_observable_seq").fill_null(0).cast(pl.Int32)
        )
        .drop("_store_uid")
    )

    rows = (
        rows.join(
            calendar.select("_store_uid", "ds", "_observable_seq", "_store_observed"),
            on=["_store_uid", "ds"],
            how="left",
        )
        .join(origins, on=["_store_uid", "_block"], how="left")
        .with_columns(
            pl.col("_observable_seq").fill_null(0).cast(pl.Int32),
            pl.col("_store_observed").fill_null(0).cast(pl.Int8),
            pl.col("_block_origin_observable_seq").fill_null(0).cast(pl.Int32),
        )
    )
    return rows, levels

def _relative_parent_forecast_effect_path(
    actual: np.ndarray,
    forecast: np.ndarray,
    blocks: np.ndarray,
    support: np.ndarray,
    *,
    reference_days: int,
    reference_min_points: int,
    reference_shift_factor: float,
    factor_min: float,
    factor_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive a stable, causal and identifiable parent movement for one leaf.

    v13.2.8 correctly stopped importing the non-identifiable split between the
    RLS intercept and the remaining coefficients.  It used the identifiable
    parent forecast relative to a causal recent parent-actual level.  The
    release gate exposed a second-order problem: a persistent calibration/level
    shift of the *whole parent forecast block* was still transferred as a leaf
    driver.  Because the leaf SES is responsible for the absolute leaf level,
    such a block-wide offset can inflate/deflate the SES state and create large
    1d/7d/14d/28d disagreements.

    We therefore keep the identifiable signal but center its block-wide offset
    causally:

        raw_relative_t = log1p(parent_forecast_t)
                         - log1p(parent_actual_reference)
        effect_t       = clip(raw_relative_t - calibration_center, lo, hi)

    ``parent_actual_reference`` uses only positive parent actuals before the
    block origin. ``calibration_center`` is the recent median of previously
    closed identifiable relative effects.  If an entire current block (with at
    least ``reference_min_points`` days) shifts by more than
    ``reference_shift_factor`` versus that history, the current block median is
    used as the center.  This removes a persistent parent-level offset while
    preserving within-block calendar/commercial movement.  The current block's
    forecasts are all known at the origin; no leaf actual is consulted.

    Returns:
      effect, parent_actual_reference_level, raw_log_parent_forecast,
      log_parent_actual_reference, calibration_center.
    """
    yy = np.asarray(actual, dtype=np.float64)
    pp = np.asarray(forecast, dtype=np.float64)
    bb = np.asarray(blocks, dtype=np.int64)
    ss = np.asarray(support, dtype=np.bool_)
    n = pp.size
    effect = np.zeros(n, dtype=np.float64)
    reference_level = np.zeros(n, dtype=np.float64)
    raw_log_forecast = np.zeros(n, dtype=np.float64)
    reference_log_level = np.zeros(n, dtype=np.float64)
    calibration_center = np.zeros(n, dtype=np.float64)
    raw_relative = np.zeros(n, dtype=np.float64)

    lookback = max(int(reference_days), 1)
    min_points = max(min(int(reference_min_points), lookback), 1)
    shift_factor = max(float(reference_shift_factor), 1.0)
    shift_limit = float(np.log(shift_factor))
    lo = float(np.log(max(float(factor_min), 1e-12)))
    hi = float(np.log(max(float(factor_max), 1.0)))

    # First pass: identifiable parent forecast relative to a parent-actual level
    # available at each block origin.
    unique_blocks = np.unique(bb)
    for block_id in unique_blocks:
        idx = np.flatnonzero(bb == block_id)
        if idx.size == 0:
            continue
        first = int(idx[0])
        start = max(0, first - lookback)

        hist_y = yy[start:first]
        hist_support = ss[start:first]
        valid = hist_support & np.isfinite(hist_y) & (hist_y > 0.0)
        vals = hist_y[valid]
        if vals.size < min_points:
            prior_y = yy[:first]
            prior_support = ss[:first]
            valid_all = prior_support & np.isfinite(prior_y) & (prior_y > 0.0)
            vals = prior_y[valid_all]

        ref = float(np.median(vals)) if vals.size else 0.0
        if not np.isfinite(ref) or ref <= 0.0:
            # Defensive seed fallback: only prior parent predictions are used.
            prior_pred = pp[max(0, first - lookback):first]
            prior_pred = prior_pred[np.isfinite(prior_pred) & (prior_pred > 0.0)]
            ref = float(np.median(prior_pred)) if prior_pred.size else 0.0

        reference_level[idx] = max(ref, 0.0)
        ref_log = float(np.log1p(max(ref, 0.0)))
        reference_log_level[idx] = ref_log

        pred = pp[idx].copy()
        pred[~np.isfinite(pred)] = 0.0
        pred = np.clip(pred, 0.0, None)
        raw = np.log1p(pred)
        raw_log_forecast[idx] = raw
        raw_relative[idx] = raw - ref_log if ref > 0.0 else 0.0

    # Second pass: remove only a persistent block-wide calibration/level offset.
    # Historical centers use closed rows only; current block forecasts are known
    # at the origin and may be used solely to detect a persistent whole-block
    # shift when the block is wide enough.
    for block_id in unique_blocks:
        idx = np.flatnonzero(bb == block_id)
        if idx.size == 0:
            continue
        first = int(idx[0])
        start = max(0, first - lookback)

        hist = raw_relative[start:first]
        hist_ok = ss[start:first] & np.isfinite(hist)
        hist = hist[hist_ok]
        block_vals = raw_relative[idx]
        block_vals = block_vals[np.isfinite(block_vals)]
        block_center = float(np.median(block_vals)) if block_vals.size else 0.0

        if hist.size >= min_points:
            prior_center = float(np.median(hist))
            if idx.size >= min_points and abs(block_center - prior_center) > shift_limit:
                center = block_center
            else:
                center = prior_center
        else:
            # With insufficient closed calibration history, avoid importing an
            # arbitrary parent level: center on the current known forecast block.
            center = block_center

        calibration_center[idx] = center
        effect[idx] = np.clip(raw_relative[idx] - center, lo, hi)

    return (
        effect,
        reference_level,
        raw_log_forecast,
        reference_log_level,
        calibration_center,
    )


def _parent_block_table(
    parent_forecasts: pl.DataFrame,
    *,
    train_start: dt.date,
    test_start: dt.date,
    forecast_start: dt.date,
    block_days: int,
    driver_effect_limit: float,
    driver_reference_days: int,
    driver_reference_min_points: int,
    driver_reference_shift_factor: float,
    driver_factor_min: float,
    driver_factor_max: float,
) -> pl.DataFrame:
    """Return causal parent quality and a stable RLS movement for leaf transfer.

    Parent selection still uses official prior-block wMAPE exactly as before.
    Only the *transfer* changes: instead of importing the non-identifiable
    non-intercept coefficient contribution, the leaf receives the selected
    parent's RLS forecast relative to a causal recent parent actual level.
    """
    if parent_forecasts.height == 0:
        return pl.DataFrame()

    lim = float(max(driver_effect_limit, 1.0))
    p = parent_forecasts.with_columns(
        pl.col("ds").cast(pl.Date),
        _period_block_expr(train_start, test_start, forecast_start, block_days),
    ).sort(["unique_id", "ds"])
    eligible = (
        pl.col("rls_metric_eligible").fill_null(False)
        if "rls_metric_eligible" in p.columns
        else pl.col("yhat").is_not_null()
    )

    block = (
        p.group_by(["unique_id", "_block"])
        .agg(
            pl.when(
                eligible
                & (pl.col("period_type") == "in_sample")
                & pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("yhat").is_finite()
            )
            .then((pl.col("y") - pl.col("yhat")).abs())
            .otherwise(0.0)
            .sum()
            .alias("_ae_y"),
            pl.when(
                eligible
                & (pl.col("period_type") == "in_sample")
                & pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("yhat").is_finite()
            )
            .then(pl.col("y").abs())
            .otherwise(0.0)
            .sum()
            .alias("_den_y"),
            pl.when(
                eligible
                & (pl.col("period_type") == "in_sample")
                & pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("value").is_finite()
                & pl.col("valuehat").is_finite()
            )
            .then((pl.col("value") - pl.col("valuehat")).abs())
            .otherwise(0.0)
            .sum()
            .alias("_ae_v"),
            pl.when(
                eligible
                & (pl.col("period_type") == "in_sample")
                & pl.col("y").is_finite()
                & (pl.col("y") > 0)
                & pl.col("value").is_finite()
                & pl.col("valuehat").is_finite()
            )
            .then(pl.col("value").abs())
            .otherwise(0.0)
            .sum()
            .alias("_den_v"),
        )
        .sort(["unique_id", "_block"])
        .with_columns(
            pl.col("_ae_y").cum_sum().shift(1).over("unique_id").fill_null(0.0).alias("_prior_ae_y"),
            pl.col("_den_y").cum_sum().shift(1).over("unique_id").fill_null(0.0).alias("_prior_den_y"),
            pl.col("_ae_v").cum_sum().shift(1).over("unique_id").fill_null(0.0).alias("_prior_ae_v"),
            pl.col("_den_v").cum_sum().shift(1).over("unique_id").fill_null(0.0).alias("_prior_den_v"),
        )
        .with_columns(
            pl.when(pl.col("_prior_den_y") > 0)
            .then(pl.col("_prior_ae_y") / pl.col("_prior_den_y"))
            .otherwise(None)
            .alias("parent_prior_wmape_y"),
            pl.when(pl.col("_prior_den_v") > 0)
            .then(pl.col("_prior_ae_v") / pl.col("_prior_den_v"))
            .otherwise(None)
            .alias("parent_prior_wmape_value"),
        )
    )

    coef_effect_y = pl.col("driver_effect") if "driver_effect" in p.columns else pl.lit(0.0)
    coef_effect_v = (
        pl.col("driver_effect_value")
        if "driver_effect_value" in p.columns
        else pl.lit(0.0)
    )
    base = (
        p.join(
            block.select(
                "unique_id", "_block", "parent_prior_wmape_y", "parent_prior_wmape_value"
            ),
            on=["unique_id", "_block"],
            how="left",
        )
        .with_columns(
            coef_effect_y.fill_nan(0.0).fill_null(0.0).clip(-lim, lim).alias("parent_effect_coef_raw_y"),
            coef_effect_v.fill_nan(0.0).fill_null(0.0).clip(-lim, lim).alias("parent_effect_coef_raw_value"),
            eligible.cast(pl.Boolean).alias("_parent_effect_eligible"),
        )
        .sort(["unique_id", "ds"])
    )

    parts: list[pl.DataFrame] = []
    for uid in base.select("unique_id").unique(maintain_order=True).get_column("unique_id").to_list():
        g = base.filter(pl.col("unique_id") == uid).sort("ds")
        blocks_np = g.get_column("_block").cast(pl.Int64).to_numpy()
        y_np = g.get_column("y").cast(pl.Float64).to_numpy()
        v_np = g.get_column("value").cast(pl.Float64).to_numpy()
        yhat_np = g.get_column("yhat").cast(pl.Float64).to_numpy()
        vhat_np = g.get_column("valuehat").cast(pl.Float64).to_numpy()
        support_y = np.isfinite(y_np) & (y_np > 0.0)
        support_v = support_y & np.isfinite(v_np) & (v_np > 0.0)

        y_effect, y_ref_level, y_raw_log, y_ref_log, y_center = _relative_parent_forecast_effect_path(
            y_np,
            yhat_np,
            blocks_np,
            support_y,
            reference_days=driver_reference_days,
            reference_min_points=driver_reference_min_points,
            reference_shift_factor=driver_reference_shift_factor,
            factor_min=driver_factor_min,
            factor_max=driver_factor_max,
        )
        v_effect, v_ref_level, v_raw_log, v_ref_log, v_center = _relative_parent_forecast_effect_path(
            v_np,
            vhat_np,
            blocks_np,
            support_v,
            reference_days=driver_reference_days,
            reference_min_points=driver_reference_min_points,
            reference_shift_factor=driver_reference_shift_factor,
            factor_min=driver_factor_min,
            factor_max=driver_factor_max,
        )
        parts.append(
            g.with_columns(
                pl.Series("parent_reference_level_y", y_ref_level),
                pl.Series("parent_reference_level_value", v_ref_level),
                pl.Series("parent_effect_raw_y", y_raw_log),
                pl.Series("parent_effect_raw_value", v_raw_log),
                pl.Series("parent_effect_reference_y", y_ref_log),
                pl.Series("parent_effect_reference_value", v_ref_log),
                pl.Series("parent_effect_center_y", y_center),
                pl.Series("parent_effect_center_value", v_center),
                pl.Series("parent_effect_y", y_effect),
                pl.Series("parent_effect_value", v_effect),
            ).with_columns(
                pl.col("parent_effect_y").exp().alias("parent_factor_y"),
                pl.col("parent_effect_value").exp().alias("parent_factor_value"),
            )
        )

    stabilized = pl.concat(parts, how="vertical") if parts else base.head(0)
    return stabilized.select(
        "unique_id", "ds", "_block",
        pl.col("yhat").alias("parent_rls_forecast_y"),
        pl.col("valuehat").alias("parent_rls_forecast_value"),
        "parent_reference_level_y", "parent_reference_level_value",
        "parent_effect_coef_raw_y", "parent_effect_coef_raw_value",
        "parent_effect_raw_y", "parent_effect_raw_value",
        "parent_effect_reference_y", "parent_effect_reference_value",
        "parent_effect_center_y", "parent_effect_center_value",
        "parent_effect_y", "parent_effect_value",
        "parent_factor_y", "parent_factor_value",
        "parent_prior_wmape_y", "parent_prior_wmape_value",
    ).sort(["unique_id", "ds"])

def _attach_selected_parent(
    rows: pl.DataFrame,
    parent_daily: pl.DataFrame,
    *,
    section_id: str,
    default_parent: str,
) -> pl.DataFrame:
    if rows.height == 0:
        return rows
    if parent_daily.height == 0:
        raise RuntimeError("No hay pronósticos RLS de sección/tienda para derivar hojas")

    store = (
        parent_daily.filter(pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 1)
        .select(
            pl.col("unique_id").alias("_store_uid"),
            "ds",
            pl.col("parent_effect_coef_raw_y").alias("_store_effect_coef_raw_y"),
            pl.col("parent_effect_coef_raw_value").alias("_store_effect_coef_raw_v"),
            pl.col("parent_rls_forecast_y").alias("_store_parent_rls_forecast_y"),
            pl.col("parent_rls_forecast_value").alias("_store_parent_rls_forecast_v"),
            pl.col("parent_reference_level_y").alias("_store_reference_level_y"),
            pl.col("parent_reference_level_value").alias("_store_reference_level_v"),
            pl.col("parent_effect_raw_y").alias("_store_effect_raw_y"),
            pl.col("parent_effect_raw_value").alias("_store_effect_raw_v"),
            pl.col("parent_effect_reference_y").alias("_store_effect_ref_y"),
            pl.col("parent_effect_reference_value").alias("_store_effect_ref_v"),
            pl.col("parent_effect_center_y").alias("_store_effect_center_y"),
            pl.col("parent_effect_center_value").alias("_store_effect_center_v"),
            pl.col("parent_effect_y").alias("_store_effect_y"),
            pl.col("parent_effect_value").alias("_store_effect_v"),
            pl.col("parent_factor_y").alias("_store_factor_y"),
            pl.col("parent_factor_value").alias("_store_factor_v"),
            pl.col("parent_prior_wmape_y").alias("_store_wmape_y"),
            pl.col("parent_prior_wmape_value").alias("_store_wmape_v"),
        )
    )
    section = (
        parent_daily.filter(pl.col("unique_id") == str(section_id))
        .select(
            "ds",
            pl.col("parent_effect_coef_raw_y").alias("_section_effect_coef_raw_y"),
            pl.col("parent_effect_coef_raw_value").alias("_section_effect_coef_raw_v"),
            pl.col("parent_rls_forecast_y").alias("_section_parent_rls_forecast_y"),
            pl.col("parent_rls_forecast_value").alias("_section_parent_rls_forecast_v"),
            pl.col("parent_reference_level_y").alias("_section_reference_level_y"),
            pl.col("parent_reference_level_value").alias("_section_reference_level_v"),
            pl.col("parent_effect_raw_y").alias("_section_effect_raw_y"),
            pl.col("parent_effect_raw_value").alias("_section_effect_raw_v"),
            pl.col("parent_effect_reference_y").alias("_section_effect_ref_y"),
            pl.col("parent_effect_reference_value").alias("_section_effect_ref_v"),
            pl.col("parent_effect_center_y").alias("_section_effect_center_y"),
            pl.col("parent_effect_center_value").alias("_section_effect_center_v"),
            pl.col("parent_effect_y").alias("_section_effect_y"),
            pl.col("parent_effect_value").alias("_section_effect_v"),
            pl.col("parent_factor_y").alias("_section_factor_y"),
            pl.col("parent_factor_value").alias("_section_factor_v"),
            pl.col("parent_prior_wmape_y").alias("_section_wmape_y"),
            pl.col("parent_prior_wmape_value").alias("_section_wmape_v"),
        )
    )
    out = rows.join(store, on=["_store_uid", "ds"], how="left").join(section, on="ds", how="left")

    store_available_y = pl.col("_store_factor_y").is_not_null()
    sec_available_y = pl.col("_section_factor_y").is_not_null()
    store_available_v = pl.col("_store_factor_v").is_not_null()
    sec_available_v = pl.col("_section_factor_v").is_not_null()

    def choose_expr(store_score: str, sec_score: str, store_available: pl.Expr, sec_available: pl.Expr) -> pl.Expr:
        store_known = pl.col(store_score).is_not_null() & pl.col(store_score).is_finite()
        sec_known = pl.col(sec_score).is_not_null() & pl.col(sec_score).is_finite()
        fallback = (
            pl.when(store_available).then(pl.lit("store"))
            .when(sec_available).then(pl.lit("section"))
            if default_parent == "store"
            else pl.when(sec_available).then(pl.lit("section")).when(store_available).then(pl.lit("store"))
        )
        return (
            pl.when(store_known & sec_known)
            .then(pl.when(pl.col(store_score) <= pl.col(sec_score)).then(pl.lit("store")).otherwise(pl.lit("section")))
            .when(store_known & store_available).then(pl.lit("store"))
            .when(sec_known & sec_available).then(pl.lit("section"))
            .otherwise(fallback.otherwise(pl.lit("missing")))
        )

    # Candidate parent is always scored from closed in-sample history.  For
    # v13.2.10 we additionally snapshot the last in-sample choice and carry it
    # unchanged through OOS + forecast-only.  The parent RLS *state* may still
    # advance at each operational cadence; only the model selection is frozen.
    out = out.with_columns(
        choose_expr("_store_wmape_y", "_section_wmape_y", store_available_y, sec_available_y).alias("_parent_candidate_y"),
        choose_expr("_store_wmape_v", "_section_wmape_v", store_available_v, sec_available_v).alias("_parent_candidate_v"),
    ).with_columns(
        pl.col("_parent_candidate_y")
        .filter(pl.col("period_type") == "in_sample")
        .last()
        .over("unique_id")
        .alias("_parent_frozen_y"),
        pl.col("_parent_candidate_v")
        .filter(pl.col("period_type") == "in_sample")
        .last()
        .over("unique_id")
        .alias("_parent_frozen_v"),
    ).with_columns(
        pl.when((pl.col("period_type") != "in_sample") & pl.col("_parent_frozen_y").is_not_null())
        .then(pl.col("_parent_frozen_y"))
        .otherwise(pl.col("_parent_candidate_y"))
        .alias("parent_model_y"),
        pl.when((pl.col("period_type") != "in_sample") & pl.col("_parent_frozen_v").is_not_null())
        .then(pl.col("_parent_frozen_v"))
        .otherwise(pl.col("_parent_candidate_v"))
        .alias("parent_model_value"),
    ).with_columns(
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_factor_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_factor_y"))
        .otherwise(None)
        .fill_null(1.0)
        .alias("driver_factor_y"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_factor_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_factor_v"))
        .otherwise(None)
        .fill_null(1.0)
        .alias("driver_factor_value"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_effect_coef_raw_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_effect_coef_raw_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_coef_raw"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_effect_coef_raw_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_effect_coef_raw_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_value_coef_raw"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_parent_rls_forecast_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_parent_rls_forecast_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_parent_rls_forecast_y"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_parent_rls_forecast_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_parent_rls_forecast_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_parent_rls_forecast_value"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_reference_level_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_reference_level_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_reference_level_y"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_reference_level_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_reference_level_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_reference_level_value"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_effect_raw_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_effect_raw_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_raw"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_effect_raw_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_effect_raw_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_value_raw"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_effect_ref_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_effect_ref_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_reference"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_effect_ref_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_effect_ref_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_value_reference"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_effect_center_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_effect_center_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_center"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_effect_center_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_effect_center_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_value_center"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_effect_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_effect_y"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_effect_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_effect_v"))
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("driver_effect_value"),
        pl.when(pl.col("parent_model_y") == "store")
        .then(pl.col("_store_wmape_y"))
        .when(pl.col("parent_model_y") == "section")
        .then(pl.col("_section_wmape_y"))
        .otherwise(None)
        .alias("parent_wmape_y"),
        pl.when(pl.col("parent_model_value") == "store")
        .then(pl.col("_store_wmape_v"))
        .when(pl.col("parent_model_value") == "section")
        .then(pl.col("_section_wmape_v"))
        .otherwise(None)
        .alias("parent_wmape_value"),
    )

    missing = out.filter((pl.col("parent_model_y") == "missing") | (pl.col("parent_model_value") == "missing"))
    if missing.height:
        raise RuntimeError(
            "Hojas sin parent RLS disponible; examples="
            f"{missing.head(5).select('unique_id','ds').to_dicts()}"
        )
    return out


@njit(cache=True)
def _ses_walkforward_kernel(
    uid_codes: np.ndarray,
    blocks: np.ndarray,
    warmup: np.ndarray,
    period_codes: np.ndarray,
    observable_seq: np.ndarray,
    block_origin_observable_seq: np.ndarray,
    warmup_end_observable_seq: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    factor_y: np.ndarray,
    factor_v: np.ndarray,
    init_y: np.ndarray,
    init_v: np.ndarray,
    alphas: np.ndarray,
    productive_alpha_mask: np.ndarray,
    default_alpha_index: int,
    collect_diagnostics: int,
    ses_update_factor_min: float,
    ses_update_factor_max: float,
    forecast_history_max_multiplier: float,
    forecast_guard_min_positive_points: int,
    gap_aware_enabled: int,
    gap_min_observable_zero_days: int,
    gap_reactivation_robust_bypass_days: int,
    gap_decay_alpha_floor: float = 0.025,
    gap_decay_max_observable_days: int = 56,
    gap_decay_min_factor: float = 0.25,
    gap_reactivation_alpha_max: float = 0.20,
    gap_reactivation_state_factor: float = 2.00,
    gap_dormant_observable_days: int = 84,
    gap_dormant_factor: float = 1e-6,
    forecast_history_mean_multiplier: float = 4.00,
    long_gap_forecast_mean_multiplier: float = 2.00,
    oos_state_anchor_min_factor: float = 0.50,
    oos_state_anchor_max_factor: float = 1.50,
    oos_recent_positive_window: int = 28,
    oos_recent_median_min_points: int = 7,
    oos_recent_median_max_factor: float = 1.50,
):
    """Causal walk-forward SES with robust updates and diagnostic gap traces.

    v13.2.17 restores the established v13 walk-forward contract: model
    selection is frozen at OOS origin, but actuals from a CLOSED OOS block may
    update the operational SES state for the NEXT origin of the same cadence.
    Missing SKU/day rows never mutate SES. Gap observability is a causal
    forecast-origin signal only. A recent-positive median robustifies the first
    OOS anchor without using OOS information.
    """
    n = len(uid_codes)
    na = len(alphas)
    yhat = np.zeros(n, dtype=np.float64)
    vhat = np.zeros(n, dtype=np.float64)
    level_y_out = np.zeros(n, dtype=np.float64)
    level_v_out = np.zeros(n, dtype=np.float64)
    alpha_y_out = np.zeros(n, dtype=np.float64)
    alpha_v_out = np.zeros(n, dtype=np.float64)
    forecast_cap_y_out = np.zeros(n, dtype=np.float64)
    forecast_cap_v_out = np.zeros(n, dtype=np.float64)
    gap_days_y_out = np.zeros(n, dtype=np.int32)
    gap_days_v_out = np.zeros(n, dtype=np.int32)
    gap_decay_y_out = np.ones(n, dtype=np.float64)
    gap_decay_v_out = np.ones(n, dtype=np.float64)
    eligible = np.zeros(n, dtype=np.uint8)

    nb = (int(np.max(blocks)) + 1) if (n and collect_diagnostics == 1) else 1
    diag_ae_y = np.zeros((3, nb, na), dtype=np.float64)
    diag_den_y = np.zeros((3, nb, na), dtype=np.float64)
    diag_se_y = np.zeros((3, nb, na), dtype=np.float64)
    diag_ae_v = np.zeros((3, nb, na), dtype=np.float64)
    diag_den_v = np.zeros((3, nb, na), dtype=np.float64)
    diag_se_v = np.zeros((3, nb, na), dtype=np.float64)

    update_min = max(min(float(ses_update_factor_min), 1.0), 1e-12)
    update_max = max(float(ses_update_factor_max), 1.0)
    history_mult = max(float(forecast_history_max_multiplier), 1.0)
    history_mean_mult = max(float(forecast_history_mean_multiplier), 1.0)
    long_gap_mean_mult = max(float(long_gap_forecast_mean_multiplier), 1.0)
    guard_min_points = max(int(forecast_guard_min_positive_points), 1)
    anchor_min = min(max(float(oos_state_anchor_min_factor), 1e-9), 1.0)
    anchor_max = max(float(oos_state_anchor_max_factor), 1.0)
    gap_min = max(int(gap_min_observable_zero_days), 1)
    reactivation_bypass = max(int(gap_reactivation_robust_bypass_days), 1)
    gap_alpha_floor = min(max(float(gap_decay_alpha_floor), 1e-9), 1.0)
    gap_max_days = max(int(gap_decay_max_observable_days), 1)
    gap_factor_floor = min(max(float(gap_decay_min_factor), 1e-9), 1.0)
    reactivation_alpha_max = min(max(float(gap_reactivation_alpha_max), 1e-9), 1.0)
    reactivation_state_factor = max(float(gap_reactivation_state_factor), 1.0)
    dormant_days = max(int(gap_dormant_observable_days), 1)
    dormant_factor = min(max(float(gap_dormant_factor), 1e-15), 1.0)
    recent_window = max(int(oos_recent_positive_window), 1)
    recent_min_points = min(max(int(oos_recent_median_min_points), 1), recent_window)
    recent_median_max_factor = max(float(oos_recent_median_max_factor), 1.0)
    use_gap = int(gap_aware_enabled) == 1

    i = 0
    while i < n:
        uid = uid_codes[i]
        j = i + 1
        while j < n and uid_codes[j] == uid:
            j += 1

        state_y = np.empty(na, dtype=np.float64)
        state_v = np.empty(na, dtype=np.float64)
        state_seq_y = np.empty(na, dtype=np.int64)
        state_seq_v = np.empty(na, dtype=np.int64)
        ae_y = np.zeros(na, dtype=np.float64)
        den_y = np.zeros(na, dtype=np.float64)
        ae_v = np.zeros(na, dtype=np.float64)
        den_v = np.zeros(na, dtype=np.float64)
        init_seq = max(int(warmup_end_observable_seq[i]), 0)
        for a in range(na):
            state_y[a] = max(init_y[i], 0.0)
            state_v[a] = max(init_v[i], 0.0)
            # Initial level is defined at the end of the 28-calendar-day warmup.
            state_seq_y[a] = init_seq
            state_seq_v[a] = init_seq

        # Causal magnitude history shared by all alpha candidates.
        hist_max_y = 0.0
        hist_max_v = 0.0
        hist_sum_y = 0.0
        hist_sum_v = 0.0
        hist_n_y = 0
        hist_n_v = 0

        # Ring buffer of the latest positive DESEASONALIZED magnitudes. It is
        # updated only after a block closes, so the median available at the OOS
        # origin is strictly causal and independent of future OOS observations.
        recent_y = np.zeros(recent_window, dtype=np.float64)
        recent_v = np.zeros(recent_window, dtype=np.float64)
        recent_n_y = 0
        recent_n_v = 0
        recent_pos_y = 0
        recent_pos_v = 0

        # Common OOS-origin state envelope. 1d/7d/14d/28d may close blocks at
        # different times, but recursive SES updates cannot drift arbitrarily
        # far from the same causal state available at the OOS boundary.
        oos_anchor_y = np.zeros(na, dtype=np.float64)
        oos_anchor_v = np.zeros(na, dtype=np.float64)
        oos_anchor_ready = 0

        # Last positive support is only for trace/audit of reactivations.
        last_positive_seq_y = -1
        last_positive_seq_v = -1

        # Explicit model-selection snapshot at OOS origin. -1 means that the
        # holdout boundary has not been reached yet.
        frozen_best_y = -1
        frozen_best_v = -1

        k = i
        while k < j:
            b = blocks[k]
            e = k + 1
            while e < j and blocks[e] == b:
                e += 1

            # Choose alpha strictly from previous closed in-sample blocks.
            best_y = default_alpha_index
            best_v = default_alpha_index
            best_score_y = 1e308
            best_score_v = 1e308
            for a in range(na):
                if productive_alpha_mask[a] != 1:
                    continue
                if den_y[a] > 0.0:
                    score = ae_y[a] / den_y[a]
                    if score < best_score_y - 1e-15:
                        best_score_y = score
                        best_y = a
                if den_v[a] > 0.0:
                    score = ae_v[a] / den_v[a]
                    if score < best_score_v - 1e-15:
                        best_score_v = score
                        best_v = a

            # Freeze the selected alpha at the first non-history origin. OOS
            # can advance state but cannot alter model selection.
            block_period = int(period_codes[k])
            if block_period != 0:
                if frozen_best_y < 0:
                    frozen_best_y = best_y
                    frozen_best_v = best_v
                if oos_anchor_ready == 0:
                    recent_median_y = 0.0
                    recent_median_v = 0.0
                    if recent_n_y >= recent_min_points:
                        tmp_y = np.empty(recent_n_y, dtype=np.float64)
                        for q in range(recent_n_y):
                            tmp_y[q] = recent_y[q]
                        tmp_y.sort()
                        mid_y = recent_n_y // 2
                        recent_median_y = (
                            tmp_y[mid_y]
                            if recent_n_y % 2 == 1
                            else 0.5 * (tmp_y[mid_y - 1] + tmp_y[mid_y])
                        )
                    if recent_n_v >= recent_min_points:
                        tmp_v = np.empty(recent_n_v, dtype=np.float64)
                        for q in range(recent_n_v):
                            tmp_v[q] = recent_v[q]
                        tmp_v.sort()
                        mid_v = recent_n_v // 2
                        recent_median_v = (
                            tmp_v[mid_v]
                            if recent_n_v % 2 == 1
                            else 0.5 * (tmp_v[mid_v - 1] + tmp_v[mid_v])
                        )
                    for a in range(na):
                        ay0 = max(state_y[a], 0.0)
                        av0 = max(state_v[a], 0.0)
                        # Downward-only robust clamp: persistent recent levels
                        # move the median itself; isolated spikes cannot keep the
                        # OOS anchor arbitrarily above recent typical demand.
                        if recent_median_y > 0.0:
                            robust_hi_y = recent_median_y * recent_median_max_factor
                            if ay0 > robust_hi_y:
                                ay0 = robust_hi_y
                        if recent_median_v > 0.0:
                            robust_hi_v = recent_median_v * recent_median_max_factor
                            if av0 > robust_hi_v:
                                av0 = robust_hi_v
                        oos_anchor_y[a] = ay0
                        oos_anchor_v[a] = av0
                        # Align the operational state to the same robust first
                        # OOS origin; later CLOSED OOS blocks may update it for
                        # the next origin of the same cadence.
                        state_y[a] = ay0
                        state_v[a] = av0
                    oos_anchor_ready = 1
                best_y = frozen_best_y
                best_v = frozen_best_v

            origin_seq = max(int(block_origin_observable_seq[k]), 0)

            # v13.2.12: gap staleness is a bounded ORIGIN adjustment, never a
            # destructive mutation of the SES state. Repeated 1d/7d/14d blocks
            # therefore cannot compound the same no-sale gap to ~0 while 28d
            # sees it only once. The total gap is measured from the last closed
            # positive event and the decay is bounded away from zero.
            gap_since_positive_y = 0
            gap_since_positive_v = 0
            if last_positive_seq_y >= 0:
                gap_since_positive_y = max(origin_seq - last_positive_seq_y, 0)
            if last_positive_seq_v >= 0:
                gap_since_positive_v = max(origin_seq - last_positive_seq_v, 0)

            # v13.3.2 hotfix: bounded gap staleness is a FORECAST-ORIGIN
            # adjustment and never mutates the SES state. It is recomputed at
            # EVERY closed block origin from the total observable gap since the
            # latest closed positive. This lets 1d/7d/14d react causally to new
            # no-sale information and lets forecast-only inherit the gap state
            # actually observed during OOS, without compounding state decay.
            candidate_decay_y = 1.0
            candidate_decay_v = 1.0
            if use_gap and gap_since_positive_y >= dormant_days:
                candidate_decay_y = dormant_factor
            elif use_gap and gap_since_positive_y >= gap_min:
                decay_days_y = min(gap_since_positive_y, gap_max_days)
                candidate_decay_y = max(
                    gap_factor_floor, (1.0 - gap_alpha_floor) ** decay_days_y
                )
            if use_gap and gap_since_positive_v >= dormant_days:
                candidate_decay_v = dormant_factor
            elif use_gap and gap_since_positive_v >= gap_min:
                decay_days_v = min(gap_since_positive_v, gap_max_days)
                candidate_decay_v = max(
                    gap_factor_floor, (1.0 - gap_alpha_floor) ** decay_days_v
                )

            selected_decay_y = candidate_decay_y
            selected_decay_v = candidate_decay_v

            block_level_y = state_y[best_y] * selected_decay_y
            block_level_v = state_v[best_v] * selected_decay_v

            # Causal scale-aware forecast cap. v13.2.15 combines the prior
            # 2x-history-maximum guard with a robust positive-history mean cap.
            # The mean term prevents one old large observation from authorizing
            # a stale high forecast indefinitely. During a long observed gap the
            # cap tightens further, without mutating/decaying the SES state.
            cap_y = 0.0
            cap_v = 0.0
            guard_base_y = hist_max_y if hist_n_y > 0 else float(init_y[i])
            guard_base_v = hist_max_v if hist_n_v > 0 else float(init_v[i])
            if guard_base_y > 0.0:
                cap_y = history_mult * guard_base_y
            if guard_base_v > 0.0:
                cap_v = history_mult * guard_base_v
            if hist_n_y > 0:
                hist_mean_y = hist_sum_y / hist_n_y
                if hist_mean_y > 0.0:
                    mean_cap_y = history_mean_mult * hist_mean_y
                    if cap_y <= 0.0 or mean_cap_y < cap_y:
                        cap_y = mean_cap_y
                    if gap_since_positive_y >= reactivation_bypass:
                        long_gap_cap_y = long_gap_mean_mult * hist_mean_y
                        if long_gap_cap_y < cap_y:
                            cap_y = long_gap_cap_y
            if hist_n_v > 0:
                hist_mean_v = hist_sum_v / hist_n_v
                if hist_mean_v > 0.0:
                    mean_cap_v = history_mean_mult * hist_mean_v
                    if cap_v <= 0.0 or mean_cap_v < cap_v:
                        cap_v = mean_cap_v
                    if gap_since_positive_v >= reactivation_bypass:
                        long_gap_cap_v = long_gap_mean_mult * hist_mean_v
                        if long_gap_cap_v < cap_v:
                            cap_v = long_gap_cap_v

            # Forecast every row using only block-origin state/history.
            for r in range(k, e):
                if warmup[r] == 1:
                    ly = init_y[r]
                    lv = init_v[r]
                    ay = alphas[default_alpha_index]
                    av = alphas[default_alpha_index]
                    fy = 1.0
                    fv = 1.0
                    gd_y = 0
                    gd_v = 0
                    dec_y = 1.0
                    dec_v = 1.0
                else:
                    ly = block_level_y
                    lv = block_level_v
                    ay = alphas[best_y]
                    av = alphas[best_v]
                    fy = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                    fv = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                    gd_y = gap_since_positive_y
                    gd_v = gap_since_positive_v
                    dec_y = selected_decay_y
                    dec_v = selected_decay_v
                    if period_codes[r] != 2:
                        eligible[r] = 1

                level_y_out[r] = max(ly, 0.0)
                level_v_out[r] = max(lv, 0.0)
                alpha_y_out[r] = ay
                alpha_v_out[r] = av
                forecast_cap_y_out[r] = cap_y
                forecast_cap_v_out[r] = cap_v
                gap_days_y_out[r] = gd_y
                gap_days_v_out[r] = gd_v
                gap_decay_y_out[r] = dec_y
                gap_decay_v_out[r] = dec_v

                py = max(np.expm1(np.log1p(max(ly, 0.0)) + np.log(fy)), 0.0)
                pv = max(np.expm1(np.log1p(max(lv, 0.0)) + np.log(fv)), 0.0)
                if cap_y > 0.0 and py > cap_y:
                    py = cap_y
                if cap_v > 0.0 and pv > cap_v:
                    pv = cap_v
                yhat[r] = py
                vhat[r] = pv

            # Score each alpha on this closed block. Candidate forecasts use
            # the same origin state/cap as production. Only in-sample errors
            # participate in future alpha selection.
            for a in range(na):
                sy = state_y[a]
                sv = state_v[a]
                sy_seq = int(state_seq_y[a])
                sv_seq = int(state_seq_v[a])
                lp_y = last_positive_seq_y
                lp_v = last_positive_seq_v
                block_reactivated_y = 0
                block_reactivated_v = 0
                for r in range(k, e):
                    if warmup[r] == 1 or period_codes[r] == 2:
                        continue
                    fy = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                    fv = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                    if np.isfinite(y[r]) and y[r] > 0.0:
                        py = max(np.expm1(np.log1p(max(sy, 0.0)) + np.log(fy)), 0.0)
                        if cap_y > 0.0 and py > cap_y:
                            py = cap_y
                        err_y = py - y[r]
                        pc = period_codes[r]
                        if pc == 0:  # only closed in-sample history selects alpha
                            ae_y[a] += abs(err_y)
                            den_y[a] += abs(y[r])
                        if collect_diagnostics == 1 and pc <= 1:
                            diag_ae_y[pc, b, a] += abs(err_y)
                            diag_den_y[pc, b, a] += abs(y[r])
                            diag_se_y[pc, b, a] += err_y
                        if np.isfinite(value[r]):
                            pv = max(np.expm1(np.log1p(max(sv, 0.0)) + np.log(fv)), 0.0)
                            if cap_v > 0.0 and pv > cap_v:
                                pv = cap_v
                            err_v = pv - value[r]
                            if pc == 0:  # same holdout purity for Value
                                ae_v[a] += abs(err_v)
                                den_v[a] += abs(value[r])
                            if collect_diagnostics == 1 and pc <= 1:
                                diag_ae_v[pc, b, a] += abs(err_v)
                                diag_den_v[pc, b, a] += abs(value[r])
                                diag_se_v[pc, b, a] += err_v

                # Update state after the block closes. Between positive leaf
                # events, consume only observable zero days. Then robustify the
                # positive magnitude exactly as in v13.2.10.
                for r in range(k, e):
                    if warmup[r] == 1 or period_codes[r] == 2:
                        continue
                    if np.isfinite(y[r]) and y[r] > 0.0:
                        event_seq = max(int(observable_seq[r]), 0)
                        zeros_before_y = max(event_seq - lp_y - 1, 0) if lp_y >= 0 else 0
                        fy = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                        obs_y = max(np.expm1(np.log1p(y[r]) - np.log(fy)), 0.0)
                        reactivation_gap_y = 0
                        if lp_y >= 0:
                            reactivation_gap_y = max(event_seq - lp_y - 1, 0)
                        if reactivation_gap_y >= reactivation_bypass:
                            block_reactivated_y = 1
                            # A real positive reactivation is now closed and may
                            # re-anchor only the NEXT block. Clamp the stale state
                            # around the observed deseasonalized magnitude; do not
                            # decay beforehand and do not alter the already-issued
                            # forecast for this observation.
                            if obs_y > 0.0:
                                if sy > 0.0:
                                    lo_state_y = obs_y / reactivation_state_factor
                                    hi_state_y = obs_y * reactivation_state_factor
                                    if sy < lo_state_y:
                                        sy = lo_state_y
                                    elif sy > hi_state_y:
                                        sy = hi_state_y
                                else:
                                    sy = obs_y
                            update_alpha_y = min(alphas[a], reactivation_alpha_max)
                        else:
                            if sy > 0.0:
                                lo_y = sy * update_min
                                hi_y = sy * update_max
                                if obs_y < lo_y:
                                    obs_y = lo_y
                                elif obs_y > hi_y:
                                    obs_y = hi_y
                            update_alpha_y = alphas[a]
                        sy = update_alpha_y * obs_y + (1.0 - update_alpha_y) * sy
                        sy_seq = event_seq
                        lp_y = event_seq

                        if np.isfinite(value[r]) and value[r] > 0.0:
                            zeros_before_v = max(event_seq - lp_v - 1, 0) if lp_v >= 0 else 0
                            fv = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                            obs_v = max(np.expm1(np.log1p(value[r]) - np.log(fv)), 0.0)
                            reactivation_gap_v = 0
                            if lp_v >= 0:
                                reactivation_gap_v = max(event_seq - lp_v - 1, 0)
                            if reactivation_gap_v >= reactivation_bypass:
                                block_reactivated_v = 1
                                if obs_v > 0.0:
                                    if sv > 0.0:
                                        lo_state_v = obs_v / reactivation_state_factor
                                        hi_state_v = obs_v * reactivation_state_factor
                                        if sv < lo_state_v:
                                            sv = lo_state_v
                                        elif sv > hi_state_v:
                                            sv = hi_state_v
                                    else:
                                        sv = obs_v
                                update_alpha_v = min(alphas[a], reactivation_alpha_max)
                            else:
                                if sv > 0.0:
                                    lo_v = sv * update_min
                                    hi_v = sv * update_max
                                    if obs_v < lo_v:
                                        obs_v = lo_v
                                    elif obs_v > hi_v:
                                        obs_v = hi_v
                                update_alpha_v = alphas[a]
                            sv = update_alpha_v * obs_v + (1.0 - update_alpha_v) * sv
                            sv_seq = event_seq
                            lp_v = event_seq
                # OOS recurrence is allowed, but it is bounded against the
                # common OOS-origin state rather than compounding a relative
                # guard around each newly-updated state. This is the key
                # cross-cadence stabilizer in v13.2.15.
                if block_period == 1 and oos_anchor_ready == 1:
                    ay0 = oos_anchor_y[a]
                    av0 = oos_anchor_v[a]
                    if ay0 > 0.0:
                        lo_anchor_y = ay0 * anchor_min
                        hi_anchor_y = ay0 * anchor_max
                        # A confirmed long-gap reactivation may legitimately
                        # reset downward; only the upper anchor remains binding
                        # for that closing block. Ordinary OOS updates retain
                        # both sides of the common-origin envelope.
                        if block_reactivated_y == 0 and sy < lo_anchor_y:
                            sy = lo_anchor_y
                        elif sy > hi_anchor_y:
                            sy = hi_anchor_y
                    if av0 > 0.0:
                        lo_anchor_v = av0 * anchor_min
                        hi_anchor_v = av0 * anchor_max
                        if block_reactivated_v == 0 and sv < lo_anchor_v:
                            sv = lo_anchor_v
                        elif sv > hi_anchor_v:
                            sv = hi_anchor_v
                state_y[a] = max(sy, 0.0)
                state_v[a] = max(sv, 0.0)
                state_seq_y[a] = sy_seq
                state_seq_v[a] = sv_seq

            # Update causal raw-scale history and last-positive pointers only
            # AFTER the block closes. Forecast-only never changes either.
            for r in range(k, e):
                if period_codes[r] == 2:
                    continue
                if np.isfinite(y[r]) and y[r] > 0.0:
                    event_seq = max(int(observable_seq[r]), 0)
                    last_positive_seq_y = event_seq
                    hist_n_y += 1
                    hist_sum_y += y[r]
                    if y[r] > hist_max_y:
                        hist_max_y = y[r]
                    fy_hist = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                    obs_hist_y = max(np.expm1(np.log1p(y[r]) - np.log(fy_hist)), 0.0)
                    if obs_hist_y > 0.0:
                        recent_y[recent_pos_y] = obs_hist_y
                        recent_pos_y = (recent_pos_y + 1) % recent_window
                        if recent_n_y < recent_window:
                            recent_n_y += 1
                    if np.isfinite(value[r]) and value[r] > 0.0:
                        last_positive_seq_v = event_seq
                        hist_n_v += 1
                        hist_sum_v += value[r]
                        if value[r] > hist_max_v:
                            hist_max_v = value[r]
                        fv_hist = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                        obs_hist_v = max(np.expm1(np.log1p(value[r]) - np.log(fv_hist)), 0.0)
                        if obs_hist_v > 0.0:
                            recent_v[recent_pos_v] = obs_hist_v
                            recent_pos_v = (recent_pos_v + 1) % recent_window
                            if recent_n_v < recent_window:
                                recent_n_v += 1

            k = e
        i = j

    return (
        yhat, vhat, level_y_out, level_v_out, alpha_y_out, alpha_v_out,
        forecast_cap_y_out, forecast_cap_v_out,
        gap_days_y_out, gap_days_v_out, gap_decay_y_out, gap_decay_v_out,
        eligible,
        diag_ae_y, diag_den_y, diag_se_y, diag_ae_v, diag_den_v, diag_se_v,
    )


def _collect_leaf_parent_candidate_diagnostics(
    rows: pl.DataFrame,
    *,
    yhat_raw: np.ndarray,
    valuehat_raw: np.ndarray,
    level_y: np.ndarray,
    level_value: np.ndarray,
    forecast_cap_y: np.ndarray,
    forecast_cap_value: np.ndarray,
    eligible: np.ndarray,
) -> pl.DataFrame:
    """Score current/Section/Store RLS transfer on each leaf closed history.

    Phase 5 is diagnostic-only. Every candidate reuses the exact productive SES
    level and alpha already emitted for the row; only the *existing* parent RLS
    effect is swapped between Section and Store. ``ses_only`` is a reference and
    can never become a productive parent.

    To protect RAM, daily alternate forecasts are never persisted. We aggregate
    first to leaf×block, compare each candidate to ``current_policy``, then return
    only leaf×period×candidate summaries. This keeps the artifact O(leaves), not
    O(daily rows × candidate parents).
    """
    if rows.height == 0:
        return pl.DataFrame()

    needed = {
        "unique_id", "_block", "period_type", "y", "value",
        "parent_model_y", "parent_model_value",
        "_store_effect_y", "_section_effect_y",
        "_store_effect_v", "_section_effect_v",
    }
    if not needed.issubset(rows.columns):
        return pl.DataFrame()

    base = rows.select(sorted(needed)).with_columns(
        pl.Series("_diag_level_y", level_y).cast(pl.Float64),
        pl.Series("_diag_level_v", level_value).cast(pl.Float64),
        pl.Series("_diag_current_raw_y", yhat_raw).cast(pl.Float64),
        pl.Series("_diag_current_raw_v", valuehat_raw).cast(pl.Float64),
        pl.Series("_diag_cap_y", forecast_cap_y).cast(pl.Float64),
        pl.Series("_diag_cap_v", forecast_cap_value).cast(pl.Float64),
        pl.Series("_diag_eligible", eligible).cast(pl.Boolean),
    ).filter(
        pl.col("_diag_eligible")
        & pl.col("period_type").is_in(["in_sample", "out_sample"])
    )
    if base.height == 0:
        return pl.DataFrame()

    def _candidate_expr(
        level_col: str,
        effect_col: str | None,
        current_raw_col: str | None,
        cap_col: str,
    ) -> pl.Expr:
        if current_raw_col is not None:
            raw = pl.col(current_raw_col).clip(lower_bound=0.0)
        elif effect_col is None:
            raw = pl.col(level_col).clip(lower_bound=0.0)
        else:
            raw = (
                (
                    (pl.col(level_col).clip(lower_bound=0.0) + 1.0).log()
                    + pl.col(effect_col).cast(pl.Float64)
                ).exp()
                - 1.0
            ).clip(lower_bound=0.0)
        cap = pl.col(cap_col).cast(pl.Float64).fill_null(0.0)
        return pl.when(cap > 0.0).then(pl.min_horizontal(raw, cap)).otherwise(raw)

    def _block_metrics(
        *,
        target: str,
        candidate_parent: str,
        actual_col: str,
        level_col: str,
        effect_col: str | None,
        current_raw_col: str | None,
        cap_col: str,
        current_parent_col: str,
        decimals: int,
    ) -> pl.DataFrame:
        frame = base
        if effect_col is not None:
            frame = frame.filter(pl.col(effect_col).is_not_null() & pl.col(effect_col).is_finite())
        if frame.height == 0:
            return pl.DataFrame()
        scored = frame.with_columns(
            _candidate_expr(level_col, effect_col, current_raw_col, cap_col).round(decimals).alias("_candidate_forecast")
        ).filter(
            pl.col("y").is_finite()
            & (pl.col("y") > 0.0)
            & pl.col(actual_col).is_finite()
            & pl.col("_candidate_forecast").is_finite()
        )
        if scored.height == 0:
            return pl.DataFrame()
        return (
            scored.group_by(["unique_id", "period_type", "_block"])
            .agg(
                (pl.col("_candidate_forecast") - pl.col(actual_col)).abs().sum().alias("sum_abs_error"),
                pl.col(actual_col).abs().sum().alias("sum_abs_y"),
                (pl.col("_candidate_forecast") - pl.col(actual_col)).sum().alias("sum_signed_error"),
                pl.len().alias("n_points"),
                pl.col(current_parent_col).drop_nulls().first().alias("current_parent"),
            )
            .filter(pl.col("sum_abs_y") > 0)
            .with_columns(
                pl.lit(str(target)).alias("target"),
                pl.lit(candidate_parent).alias("candidate_parent"),
                (pl.col("sum_abs_error") / pl.col("sum_abs_y")).alias("wmape"),
                (pl.col("sum_signed_error") / pl.col("sum_abs_y")).alias("bias"),
            )
            .rename({"_block": "block"})
        )

    def _period_summary(blocks: pl.DataFrame, current: pl.DataFrame | None) -> pl.DataFrame:
        if blocks.height == 0:
            return pl.DataFrame()
        scored = blocks
        if current is None:
            scored = scored.with_columns(pl.lit(1.0).alias("_fold_nonworse"))
        else:
            cur = current.select(
                "unique_id", "target", "period_type", "block",
                pl.col("wmape").alias("_current_wmape"),
            )
            scored = scored.join(
                cur,
                on=["unique_id", "target", "period_type", "block"],
                how="inner",
            ).with_columns(
                (pl.col("wmape") <= pl.col("_current_wmape") + 1e-15)
                .cast(pl.Float64)
                .alias("_fold_nonworse")
            )
        if scored.height == 0:
            return pl.DataFrame()
        return (
            scored.group_by(["unique_id", "target", "period_type", "candidate_parent"])
            .agg(
                pl.col("sum_abs_error").sum(),
                pl.col("sum_abs_y").sum(),
                pl.col("sum_signed_error").sum(),
                pl.col("n_points").sum(),
                pl.col("block").n_unique().alias("n_folds"),
                pl.col("wmape").median().alias("median_fold_wmape"),
                pl.col("wmape").max().alias("worst_fold_wmape"),
                pl.col("_fold_nonworse").mean().alias("positive_fold_share"),
                pl.col("current_parent").drop_nulls().last().alias("last_current_parent"),
            )
            .with_columns(
                (pl.col("sum_abs_error") / pl.col("sum_abs_y")).alias("wmape"),
                (pl.col("sum_signed_error") / pl.col("sum_abs_y")).alias("bias"),
            )
        )

    summaries: list[pl.DataFrame] = []
    for target in ("Unidades", "Valor ($)"):
        if target == "Unidades":
            actual_col, level_col, current_raw_col, cap_col = (
                "y", "_diag_level_y", "_diag_current_raw_y", "_diag_cap_y"
            )
            current_parent_col = "parent_model_y"
            effect_store, effect_section, decimals = "_store_effect_y", "_section_effect_y", 0
        else:
            actual_col, level_col, current_raw_col, cap_col = (
                "value", "_diag_level_v", "_diag_current_raw_v", "_diag_cap_v"
            )
            current_parent_col = "parent_model_value"
            effect_store, effect_section, decimals = "_store_effect_v", "_section_effect_v", 2

        current = _block_metrics(
            target=target, candidate_parent="current_policy", actual_col=actual_col,
            level_col=level_col, effect_col=None, current_raw_col=current_raw_col,
            cap_col=cap_col, current_parent_col=current_parent_col, decimals=decimals,
        )
        cur_summary = _period_summary(current, None)
        if cur_summary.height:
            summaries.append(cur_summary)

        for candidate_parent, effect_col in (
            ("store", effect_store),
            ("section", effect_section),
            ("ses_only", None),
        ):
            blocks = _block_metrics(
                target=target, candidate_parent=candidate_parent, actual_col=actual_col,
                level_col=level_col, effect_col=effect_col, current_raw_col=None,
                cap_col=cap_col, current_parent_col=current_parent_col, decimals=decimals,
            )
            summary = _period_summary(blocks, current)
            if summary.height:
                summaries.append(summary)

    if not summaries:
        return pl.DataFrame()
    return pl.concat(summaries, how="diagonal_relaxed").sort(
        ["unique_id", "target", "period_type", "candidate_parent"]
    )

def build_leaf_forecasts(
    *,
    train_leaves: pl.DataFrame,
    oos_leaves: pl.DataFrame,
    parent_forecasts: pl.DataFrame,
    oos_observable_store_days: pl.DataFrame | None = None,
    section_id: str,
    horizons: dict,
    meta: dict | None = None,
    diagnostics_out: list[pl.DataFrame] | None = None,
    parent_diagnostics_out: list[pl.DataFrame] | None = None,
    optimization_phase2: bool = False,
) -> pl.DataFrame:
    """Generate coherent leaf forecasts for all three periods."""
    if train_leaves.height == 0:
        return pl.DataFrame()

    cfg = LeafModelConfig.from_settings()
    train_start = horizons["train_start"]
    test_start = horizons["test_start"]
    test_end = horizons["test_end"]
    forecast_start = horizons["forecast_start"]
    forecast_end = horizons["forecast_end"]

    train_obs = train_leaves.with_columns(pl.col("ds").cast(pl.Date)).sort(["unique_id", "ds"])
    oos_obs = (
        oos_leaves.with_columns(pl.col("ds").cast(pl.Date)).sort(["unique_id", "ds"])
        if oos_leaves.height
        else pl.DataFrame()
    )
    levels = _positive_median_levels(train_obs, warmup_days=cfg.warmup_days)
    observable_calendar, observable_origins = _store_observable_calendar(
        train_obs,
        oos_obs,
        oos_observable_store_days=oos_observable_store_days,
        train_start=train_start,
        test_start=test_start,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        block_days=cfg.block_days,
    )

    meta_cols = [c for c in ("sku_desc", "store_name", "seccion", "conteo_sku") if c in train_obs.columns]
    ids = (
        train_obs.select(["unique_id"] + meta_cols)
        .group_by("unique_id")
        .agg([pl.col(c).drop_nulls().first().alias(c) for c in meta_cols])
        if meta_cols
        else train_obs.select("unique_id").unique()
    )

    historical = train_obs.select([c for c in ["unique_id", "ds", "y", "value"] + meta_cols if c in train_obs.columns]).with_columns(
        pl.lit("in_sample").alias("period_type")
    )

    uid_arr = ids["unique_id"].to_numpy()
    oos_dates = pl.date_range(test_start, test_end, interval="1d", eager=True).to_numpy()
    fc_dates = pl.date_range(forecast_start, forecast_end, interval="1d", eager=True).to_numpy()

    oos_grid = pl.DataFrame(
        {"unique_id": np.repeat(uid_arr, len(oos_dates)), "ds": np.tile(oos_dates, len(uid_arr))}
    ).with_columns(pl.col("ds").cast(pl.Date))
    if meta_cols:
        oos_grid = oos_grid.join(ids, on="unique_id", how="left")
    if oos_obs.height:
        oos_grid = oos_grid.join(
            oos_obs.select("unique_id", "ds", "y", "value"), on=["unique_id", "ds"], how="left"
        )
    oos_grid = oos_grid.with_columns(
        (pl.col("y").fill_null(0.0) if "y" in oos_grid.columns else pl.lit(0.0).alias("y")),
        (pl.col("value").fill_null(0.0) if "value" in oos_grid.columns else pl.lit(0.0).alias("value")),
        pl.lit("out_sample").alias("period_type"),
    )

    forecast_grid = pl.DataFrame(
        {
            "unique_id": np.repeat(uid_arr, len(fc_dates)),
            "ds": np.tile(fc_dates, len(uid_arr)),
            "y": np.zeros(len(uid_arr) * len(fc_dates), dtype=np.float64),
            "value": np.zeros(len(uid_arr) * len(fc_dates), dtype=np.float64),
        }
    ).with_columns(pl.col("ds").cast(pl.Date), pl.lit("forecast_only").alias("period_type"))
    if meta_cols:
        forecast_grid = forecast_grid.join(ids, on="unique_id", how="left")

    rows = (
        pl.concat([historical, oos_grid, forecast_grid], how="diagonal_relaxed")
        .with_columns(
            _period_block_expr(train_start, test_start, forecast_start, cfg.block_days),
            pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
        )
    )
    rows = _attach_leaf_yoy_monthly_seasonality(
        rows,
        train_obs,
        enabled=cfg.yoy_seasonal_enabled,
        min_positive_days=cfg.yoy_seasonal_min_positive_days,
        full_reliability_days=cfg.yoy_seasonal_full_reliability_days,
        factor_min=cfg.yoy_seasonal_factor_min,
        factor_max=cfg.yoy_seasonal_factor_max,
    )
    rows, levels = _attach_gap_observability(
        rows, levels, observable_calendar, observable_origins
    )
    rows = rows.join(levels, on="unique_id", how="left")

    parent_daily = _parent_block_table(
        parent_forecasts,
        train_start=train_start,
        test_start=test_start,
        forecast_start=forecast_start,
        block_days=cfg.block_days,
        driver_effect_limit=cfg.driver_effect_limit,
        driver_reference_days=cfg.driver_reference_days,
        driver_reference_min_points=cfg.driver_reference_min_points,
        driver_reference_shift_factor=cfg.driver_reference_shift_factor,
        driver_factor_min=cfg.driver_factor_min,
        driver_factor_max=cfg.driver_factor_max,
    )
    rows = _attach_selected_parent(
        rows,
        parent_daily,
        section_id=str(section_id),
        default_parent=cfg.default_parent,
    ).with_columns(
        (pl.col("ds") <= pl.col("leaf_warmup_end")).alias("_warmup"),
        pl.when(pl.col("period_type") == "in_sample")
        .then(pl.lit(0))
        .when(pl.col("period_type") == "out_sample")
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .cast(pl.Int8)
        .alias("_period_code"),
    ).sort(["unique_id", "_block", "ds"])

    # Código entero estable sin bucles Python por hoja. El lookup tiene una
    # fila por unique_id y el join es pequeño frente al panel diario.
    uid_codes = rows.select("unique_id").unique(maintain_order=True).with_row_index("_uid_code")
    rows = rows.join(uid_codes, on="unique_id", how="left")
    productive_alphas = tuple(cfg.alphas)
    diagnostic_alphas = (
        tuple(
            float(x)
            for x in getattr(settings, "STAT_OPT_SES_EXTRA_ALPHAS", ())
            if 0.0 < float(x) <= 1.0
        )
        if optimization_phase2 and diagnostics_out is not None
        else ()
    )
    all_alphas = tuple(sorted(set(productive_alphas) | set(diagnostic_alphas)))
    alphas = np.asarray(all_alphas, dtype=np.float64)
    productive_alpha_mask = np.asarray(
        [1 if float(a) in set(productive_alphas) else 0 for a in all_alphas],
        dtype=np.uint8,
    )
    default_idx = int(np.argmin(np.abs(alphas - cfg.default_alpha)))

    # Productive leaf factor = centered RLS parent movement × causal leaf YoY
    # transition. The two traces remain separate in the output for auditability.
    rows = rows.with_columns(
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_yoy_factor_y")).alias("leaf_yoy_factor_y"),
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_yoy_factor_value")).alias("leaf_yoy_factor_value"),
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_yoy_raw_factor_y")).alias("leaf_yoy_raw_factor_y"),
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_yoy_raw_factor_value")).alias("leaf_yoy_raw_factor_value"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("leaf_yoy_reliability_y")).alias("leaf_yoy_reliability_y"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("leaf_yoy_reliability_value")).alias("leaf_yoy_reliability_value"),
    ).with_columns(
        (pl.col("driver_factor_y").fill_null(1.0) * pl.col("leaf_yoy_factor_y").fill_null(1.0))
        .clip(lower_bound=1e-9)
        .alias("leaf_total_factor_y"),
        (pl.col("driver_factor_value").fill_null(1.0) * pl.col("leaf_yoy_factor_value").fill_null(1.0))
        .clip(lower_bound=1e-9)
        .alias("leaf_total_factor_value"),
    )

    (
        yhat, vhat, level_y, level_v, alpha_y, alpha_v,
        forecast_cap_y, forecast_cap_v,
        gap_days_y, gap_days_v, gap_decay_y, gap_decay_v, eligible,
        diag_ae_y, diag_den_y, diag_se_y, diag_ae_v, diag_den_v, diag_se_v,
    ) = _ses_walkforward_kernel(
        rows["_uid_code"].to_numpy(),
        rows["_block"].to_numpy(),
        rows["_warmup"].cast(pl.UInt8).to_numpy(),
        rows["_period_code"].to_numpy(),
        rows["_observable_seq"].cast(pl.Int64).to_numpy(),
        rows["_block_origin_observable_seq"].cast(pl.Int64).to_numpy(),
        rows["warmup_end_observable_seq"].cast(pl.Int64).fill_null(0).to_numpy(),
        rows["y"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        rows["value"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        rows["leaf_total_factor_y"].cast(pl.Float64).fill_null(1.0).to_numpy(),
        rows["leaf_total_factor_value"].cast(pl.Float64).fill_null(1.0).to_numpy(),
        rows["initial_level_y"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        rows["initial_level_value"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        alphas,
        productive_alpha_mask,
        default_idx,
        1 if diagnostics_out is not None else 0,
        cfg.ses_update_factor_min,
        cfg.ses_update_factor_max,
        cfg.forecast_history_max_multiplier,
        cfg.forecast_guard_min_positive_points,
        1 if cfg.gap_aware_enabled else 0,
        cfg.gap_min_observable_zero_days,
        cfg.gap_reactivation_robust_bypass_days,
        cfg.gap_decay_alpha_floor,
        cfg.gap_decay_max_observable_days,
        cfg.gap_decay_min_factor,
        cfg.gap_reactivation_alpha_max,
        cfg.gap_reactivation_state_factor,
        cfg.gap_dormant_observable_days,
        cfg.gap_dormant_factor,
        cfg.forecast_history_mean_multiplier,
        cfg.long_gap_forecast_mean_multiplier,
        cfg.oos_state_anchor_min_factor,
        cfg.oos_state_anchor_max_factor,
        cfg.oos_recent_positive_window,
        cfg.oos_recent_median_min_points,
        cfg.oos_recent_median_max_factor,
    )

    if optimization_phase2 and parent_diagnostics_out is not None:
        parent_diag = _collect_leaf_parent_candidate_diagnostics(
            rows,
            yhat_raw=yhat,
            valuehat_raw=vhat,
            level_y=level_y,
            level_value=level_v,
            forecast_cap_y=forecast_cap_y,
            forecast_cap_value=forecast_cap_v,
            eligible=eligible,
        )
        if parent_diag.height:
            parent_diagnostics_out.append(parent_diag)

    if diagnostics_out is not None:
        period_names = ("in_sample", "out_sample")
        records: list[dict] = []
        for pc, period_name in enumerate(period_names):
            for b in range(diag_den_y.shape[1]):
                for a, alpha in enumerate(alphas):
                    den = float(diag_den_y[pc, b, a])
                    if den > 0.0:
                        ae = float(diag_ae_y[pc, b, a])
                        se = float(diag_se_y[pc, b, a])
                        records.append({
                            "component": "SES",
                            "section": str(section_id),
                            "target": "Unidades",
                            "period_type": period_name,
                            "block": int(b),
                            "alpha": float(alpha),
                            "productive_candidate": bool(productive_alpha_mask[a]),
                            "sum_abs_error": ae,
                            "sum_abs_y": den,
                            "sum_signed_error": se,
                            "wmape": ae / den,
                            "bias": se / den,
                        })
                    den = float(diag_den_v[pc, b, a])
                    if den > 0.0:
                        ae = float(diag_ae_v[pc, b, a])
                        se = float(diag_se_v[pc, b, a])
                        records.append({
                            "component": "SES",
                            "section": str(section_id),
                            "target": "Valor ($)",
                            "period_type": period_name,
                            "block": int(b),
                            "alpha": float(alpha),
                            "productive_candidate": bool(productive_alpha_mask[a]),
                            "sum_abs_error": ae,
                            "sum_abs_y": den,
                            "sum_signed_error": se,
                            "wmape": ae / den,
                            "bias": se / den,
                        })
        if records:
            diagnostics_out.append(pl.DataFrame(records))

    out = rows.with_columns(
        pl.Series("yhat_raw", yhat),
        pl.Series("valuehat_raw", vhat),
        pl.Series("ses_level_y", level_y),
        pl.Series("ses_level_value", level_v),
        pl.Series("ses_alpha_y", alpha_y),
        pl.Series("ses_alpha_value", alpha_v),
        pl.Series("leaf_forecast_cap_y", forecast_cap_y),
        pl.Series("leaf_forecast_cap_value", forecast_cap_v),
        pl.Series("leaf_observable_day_index", rows["_observable_seq"].cast(pl.Int32).to_numpy()),
        pl.Series("leaf_gap_observable_days_y", gap_days_y).cast(pl.Int32),
        pl.Series("leaf_gap_observable_days_value", gap_days_v).cast(pl.Int32),
        pl.Series("leaf_gap_decay_factor_y", gap_decay_y),
        pl.Series("leaf_gap_decay_factor_value", gap_decay_v),
        pl.Series("rls_metric_eligible", eligible).cast(pl.Boolean),
    ).with_columns(
        pl.col("yhat_raw").round(0).alias("yhat"),
        pl.col("valuehat_raw").round(2).alias("valuehat"),
        pl.when(pl.col("_warmup"))
        .then(pl.lit("warmup"))
        .otherwise(pl.col("parent_model_y"))
        .alias("parent_model_y"),
        pl.when(pl.col("_warmup"))
        .then(pl.lit("warmup"))
        .otherwise(pl.col("parent_model_value"))
        .alias("parent_model_value"),
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("driver_factor_y")).alias("driver_factor_y"),
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("driver_factor_value")).alias("driver_factor_value"),
        # The kernel always forecasts warm-up rows with factor=1.0. Keep the
        # exported trace algebraically identical after driver_factor_* is reset
        # to 1.0 for warm-up; otherwise leaf_total_factor_* would retain the
        # pre-kernel parent factor and the validator would report a false
        # identity failure on warm-up rows.
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_total_factor_y")).alias("leaf_total_factor_y"),
        pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_total_factor_value")).alias("leaf_total_factor_value"),
        pl.lit("ses_deseasonalized+yoy_monthly").alias("leaf_level_method_y"),
        pl.lit("ses_deseasonalized+yoy_monthly").alias("leaf_level_method_value"),
        pl.lit("rls_parent_forecast_relative_centered_causal").alias("parent_driver_mode_y"),
        pl.lit("rls_parent_forecast_relative_centered_causal").alias("parent_driver_mode_value"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("driver_effect_center")).alias("driver_effect_center"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("driver_effect_value_center")).alias("driver_effect_value_center"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("driver_effect")).alias("driver_effect"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("driver_effect_value")).alias("driver_effect_value"),
        pl.col("_block").alias("rls_block"),
        (pl.col("_block") * cfg.block_days).clip(lower_bound=cfg.warmup_days).cast(pl.Int32).alias("rls_train_days"),
        pl.concat_str(
            [pl.lit("SES+RLS+YoY("), pl.col("parent_model_y"), pl.lit("/"), pl.col("parent_model_value"), pl.lit(")")]
        ).alias("modelo_seleccionado"),
    )

    if meta:
        out = out.with_columns([pl.lit(v).alias(k) for k, v in meta.items()])

    keep = [
        "unique_id", "ds", "value", "valuehat", "valuehat_raw", "y", "yhat", "yhat_raw",
        "period_type", "rls_metric_eligible", "rls_block", "rls_train_days",
        "modelo_seleccionado", "ses_alpha_y", "ses_alpha_value",
        "ses_level_y", "ses_level_value", "initial_level_y", "initial_level_value",
        "warmup_positive_days", "leaf_start", "leaf_warmup_end",
        "parent_model_y", "parent_model_value", "parent_driver_mode_y", "parent_driver_mode_value",
        "parent_wmape_y", "parent_wmape_value", "driver_factor_y", "driver_factor_value",
        "leaf_yoy_factor_y", "leaf_yoy_factor_value",
        "leaf_yoy_raw_factor_y", "leaf_yoy_raw_factor_value",
        "leaf_yoy_reliability_y", "leaf_yoy_reliability_value",
        "leaf_total_factor_y", "leaf_total_factor_value",
        "leaf_forecast_cap_y", "leaf_forecast_cap_value",
        "leaf_observable_day_index",
        "leaf_gap_observable_days_y", "leaf_gap_observable_days_value",
        "leaf_gap_decay_factor_y", "leaf_gap_decay_factor_value",
        "driver_effect", "driver_effect_value",
        "driver_effect_raw", "driver_effect_value_raw",
        "driver_effect_reference", "driver_effect_value_reference",
        "driver_effect_center", "driver_effect_value_center",
        "driver_effect_coef_raw", "driver_effect_value_coef_raw",
        "driver_parent_rls_forecast_y", "driver_parent_rls_forecast_value",
        "driver_reference_level_y", "driver_reference_level_value",
        "leaf_level_method_y", "leaf_level_method_value",
    ]
    keep += [c for c in meta_cols if c not in keep]
    if meta:
        keep += [k for k in meta if k not in keep]
    return out.select([c for c in keep if c in out.columns]).sort(["unique_id", "ds"])
