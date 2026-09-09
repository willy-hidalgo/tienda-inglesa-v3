"""Coherent SKU+store model: deseasonalized SES level + one RLS parent.

Statistical contract (v13):

* Section and store forecasts are produced by the same expanding RLS engine.
* A leaf uses exactly one parent (store or section) per target and update block.
  The parent is chosen only from the parent's official positive-demand wMAPE
  accumulated in *previous* closed blocks.
* The parent contributes a non-intercept log1p driver effect from the selected RLS parent. The SES
  supplies the leaf baseline level and the RLS coefficients supply the same
  calendar/commercial dynamics used at section/store level.
* The leaf level is SES on positive, deseasonalized observations
  the inverse log1p RLS effect. The first state is the median of the first 28
  positive observations' calendar window, which is robust to both zeros and
  peaks.
* SES alpha is selected walk-forward from prior positive-demand wMAPE. The same
  algorithm generates in-sample, OOS and forecast-only; only the information
  available at each origin changes.

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
        return cls(
            block_days=int(getattr(settings, "RLS_BLOCK_DAYS", 28)),
            warmup_days=int(getattr(settings, "LEAF_INITIAL_LEVEL_DAYS", 28)),
            alphas=alphas,
            default_alpha=default_alpha,
            driver_effect_limit=max(effect_limit, 1.0),
            default_parent=default_parent,
        )


def _block_expr(train_start: dt.date, block_days: int) -> pl.Expr:
    return (
        ((pl.col("ds").cast(pl.Date) - pl.lit(train_start)).dt.total_days() // block_days)
        .cast(pl.Int32)
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


def _parent_block_table(
    parent_forecasts: pl.DataFrame,
    *,
    train_start: dt.date,
    block_days: int,
    driver_effect_limit: float,
) -> pl.DataFrame:
    """Return the causal RLS driver signal and prior parent wMAPE by day.

    The RLS parent forecasts an absolute section/store series with a log1p link.
    ``driver_effect`` is the chosen model's non-intercept linear contribution,
    emitted directly by :class:`RLSForecastRunner`.  The leaf model therefore
    reuses the *same* RLS coefficients without importing the parent's absolute
    level::

        log(1 + leaf_forecast) = log(1 + SES_level) + driver_effect

    Parent quality for block ``b`` is cumulative official wMAPE through closed
    blocks before ``b``.  Seed/warm-up rows never enter that score.
    """
    if parent_forecasts.height == 0:
        return pl.DataFrame()

    # Solo guard numérico: el efecto RLS no se calibra ni se reescala.
    # El límite ±20 evita overflow de exp() y es muy superior a cualquier
    # factor comercial razonable (exp(20) ≈ 4.9e8).
    lim = float(max(driver_effect_limit, 1.0))
    p = parent_forecasts.with_columns(
        pl.col("ds").cast(pl.Date),
        _block_expr(train_start, block_days),
    )
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
                & (pl.col("period_type") != "forecast_only")
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
                & (pl.col("period_type") != "forecast_only")
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
                & (pl.col("period_type") != "forecast_only")
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
                & (pl.col("period_type") != "forecast_only")
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

    effect_y = (
        pl.col("driver_effect")
        if "driver_effect" in p.columns
        else pl.lit(0.0)
    )
    effect_v = (
        pl.col("driver_effect_value")
        if "driver_effect_value" in p.columns
        else pl.lit(0.0)
    )
    return (
        p.join(
            block.select(
                "unique_id", "_block", "parent_prior_wmape_y", "parent_prior_wmape_value"
            ),
            on=["unique_id", "_block"],
            how="left",
        )
        .with_columns(
            effect_y.fill_nan(0.0).fill_null(0.0).clip(-lim, lim).alias("parent_effect_y"),
            effect_v.fill_nan(0.0).fill_null(0.0).clip(-lim, lim).alias("parent_effect_value"),
        )
        .with_columns(
            pl.col("parent_effect_y").exp().alias("parent_factor_y"),
            pl.col("parent_effect_value").exp().alias("parent_factor_value"),
        )
        .select(
            "unique_id", "ds", "_block",
            "parent_effect_y", "parent_effect_value",
            "parent_factor_y", "parent_factor_value",
            "parent_prior_wmape_y", "parent_prior_wmape_value",
        )
    )

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

    out = out.with_columns(
        choose_expr("_store_wmape_y", "_section_wmape_y", store_available_y, sec_available_y).alias("parent_model_y"),
        choose_expr("_store_wmape_v", "_section_wmape_v", store_available_v, sec_available_v).alias("parent_model_value"),
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
):
    n = len(uid_codes)
    na = len(alphas)
    yhat = np.zeros(n, dtype=np.float64)
    vhat = np.zeros(n, dtype=np.float64)
    level_y_out = np.zeros(n, dtype=np.float64)
    level_v_out = np.zeros(n, dtype=np.float64)
    alpha_y_out = np.zeros(n, dtype=np.float64)
    alpha_v_out = np.zeros(n, dtype=np.float64)
    eligible = np.zeros(n, dtype=np.uint8)

    # Small aggregate diagnostics for statistical tuning.  They do not alter
    # the productive selection path: for each closed block we accumulate the
    # error that every fixed-alpha SES candidate would have produced under the
    # exact same RLS parent factors.  Dimensions are period(in-sample/OOS/FO)
    # × block × alpha; forecast-only has no actuals and therefore remains zero.
    nb = (int(np.max(blocks)) + 1) if (n and collect_diagnostics == 1) else 1
    diag_ae_y = np.zeros((3, nb, na), dtype=np.float64)
    diag_den_y = np.zeros((3, nb, na), dtype=np.float64)
    diag_se_y = np.zeros((3, nb, na), dtype=np.float64)
    diag_ae_v = np.zeros((3, nb, na), dtype=np.float64)
    diag_den_v = np.zeros((3, nb, na), dtype=np.float64)
    diag_se_v = np.zeros((3, nb, na), dtype=np.float64)

    i = 0
    while i < n:
        uid = uid_codes[i]
        j = i + 1
        while j < n and uid_codes[j] == uid:
            j += 1

        state_y = np.empty(na, dtype=np.float64)
        state_v = np.empty(na, dtype=np.float64)
        ae_y = np.zeros(na, dtype=np.float64)
        den_y = np.zeros(na, dtype=np.float64)
        ae_v = np.zeros(na, dtype=np.float64)
        den_v = np.zeros(na, dtype=np.float64)
        for a in range(na):
            state_y[a] = max(init_y[i], 0.0)
            state_v[a] = max(init_v[i], 0.0)

        k = i
        while k < j:
            b = blocks[k]
            e = k + 1
            while e < j and blocks[e] == b:
                e += 1

            # Choose alpha strictly from previous closed blocks.
            best_y = default_alpha_index
            best_v = default_alpha_index
            best_score_y = 1e308
            best_score_v = 1e308
            for a in range(na):
                # Extra Phase-2 alphas are diagnostic-only: they are scored
                # identically but can never alter the productive forecast.
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

            block_level_y = state_y[best_y]
            block_level_v = state_v[best_v]

            # Forecast all rows using the block-origin state. No actual inside
            # the block may alter that block's forecast.
            for r in range(k, e):
                if warmup[r] == 1:
                    ly = init_y[r]
                    lv = init_v[r]
                    ay = alphas[default_alpha_index]
                    av = alphas[default_alpha_index]
                    fy = 1.0
                    fv = 1.0
                else:
                    ly = block_level_y
                    lv = block_level_v
                    ay = alphas[best_y]
                    av = alphas[best_v]
                    fy = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                    fv = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                    if period_codes[r] != 2:  # in-sample/OOS only
                        eligible[r] = 1
                level_y_out[r] = max(ly, 0.0)
                level_v_out[r] = max(lv, 0.0)
                alpha_y_out[r] = ay
                alpha_v_out[r] = av
                yhat[r] = max(np.expm1(np.log1p(max(ly, 0.0)) + np.log(fy)), 0.0)
                vhat[r] = max(np.expm1(np.log1p(max(lv, 0.0)) + np.log(fv)), 0.0)

            # Score each alpha on this closed block, using official support
            # y>0. Then update all candidate states with positive,
            # deseasonalized actuals for the NEXT block.
            for a in range(na):
                sy = state_y[a]
                sv = state_v[a]
                for r in range(k, e):
                    if warmup[r] == 1 or period_codes[r] == 2:
                        continue
                    fy = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                    fv = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                    if np.isfinite(y[r]) and y[r] > 0.0:
                        py = max(np.expm1(np.log1p(max(sy, 0.0)) + np.log(fy)), 0.0)
                        err_y = py - y[r]
                        ae_y[a] += abs(err_y)
                        den_y[a] += abs(y[r])
                        pc = period_codes[r]
                        if collect_diagnostics == 1 and pc <= 1:
                            diag_ae_y[pc, b, a] += abs(err_y)
                            diag_den_y[pc, b, a] += abs(y[r])
                            diag_se_y[pc, b, a] += err_y
                        if np.isfinite(value[r]):
                            # Official Value metric uses the SAME sale-event
                            # support y>0. A zero monetary actual therefore
                            # still contributes forecast error (denominator 0).
                            pv = max(np.expm1(np.log1p(max(sv, 0.0)) + np.log(fv)), 0.0)
                            err_v = pv - value[r]
                            ae_v[a] += abs(err_v)
                            den_v[a] += abs(value[r])
                            if collect_diagnostics == 1 and pc <= 1:
                                diag_ae_v[pc, b, a] += abs(err_v)
                                diag_den_v[pc, b, a] += abs(value[r])
                                diag_se_v[pc, b, a] += err_v
                # State update is event-time SES on deseasonalized positive
                # magnitudes. Zero-sale days hold the magnitude state.
                for r in range(k, e):
                    if warmup[r] == 1 or period_codes[r] == 2:
                        continue
                    if np.isfinite(y[r]) and y[r] > 0.0:
                        fy = factor_y[r] if np.isfinite(factor_y[r]) and factor_y[r] > 0.0 else 1.0
                        obs_y = max(np.expm1(np.log1p(y[r]) - np.log(fy)), 0.0)
                        sy = alphas[a] * obs_y + (1.0 - alphas[a]) * sy
                        if np.isfinite(value[r]) and value[r] > 0.0:
                            fv = factor_v[r] if np.isfinite(factor_v[r]) and factor_v[r] > 0.0 else 1.0
                            obs_v = max(np.expm1(np.log1p(value[r]) - np.log(fv)), 0.0)
                            sv = alphas[a] * obs_v + (1.0 - alphas[a]) * sv
                state_y[a] = max(sy, 0.0)
                state_v[a] = max(sv, 0.0)

            k = e
        i = j

    return (
        yhat, vhat, level_y_out, level_v_out, alpha_y_out, alpha_v_out, eligible,
        diag_ae_y, diag_den_y, diag_se_y, diag_ae_v, diag_den_v, diag_se_v,
    )



def _collect_leaf_parent_candidate_diagnostics(
    rows: pl.DataFrame,
    *,
    yhat_raw: np.ndarray,
    valuehat_raw: np.ndarray,
    level_y: np.ndarray,
    level_value: np.ndarray,
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
        pl.Series("_diag_eligible", eligible).cast(pl.Boolean),
    ).filter(
        pl.col("_diag_eligible")
        & pl.col("period_type").is_in(["in_sample", "out_sample"])
    )
    if base.height == 0:
        return pl.DataFrame()

    def _candidate_expr(level_col: str, effect_col: str | None, current_raw_col: str | None) -> pl.Expr:
        if current_raw_col is not None:
            return pl.col(current_raw_col).clip(lower_bound=0.0)
        if effect_col is None:
            return pl.col(level_col).clip(lower_bound=0.0)
        return (
            (
                (pl.col(level_col).clip(lower_bound=0.0) + 1.0).log()
                + pl.col(effect_col).cast(pl.Float64)
            ).exp()
            - 1.0
        ).clip(lower_bound=0.0)

    def _block_metrics(
        *,
        target: str,
        candidate_parent: str,
        actual_col: str,
        level_col: str,
        effect_col: str | None,
        current_raw_col: str | None,
        current_parent_col: str,
        decimals: int,
    ) -> pl.DataFrame:
        frame = base
        if effect_col is not None:
            frame = frame.filter(pl.col(effect_col).is_not_null() & pl.col(effect_col).is_finite())
        if frame.height == 0:
            return pl.DataFrame()
        scored = frame.with_columns(
            _candidate_expr(level_col, effect_col, current_raw_col).round(decimals).alias("_candidate_forecast")
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
            actual_col, level_col, current_raw_col = "y", "_diag_level_y", "_diag_current_raw_y"
            current_parent_col = "parent_model_y"
            effect_store, effect_section, decimals = "_store_effect_y", "_section_effect_y", 0
        else:
            actual_col, level_col, current_raw_col = "value", "_diag_level_v", "_diag_current_raw_v"
            current_parent_col = "parent_model_value"
            effect_store, effect_section, decimals = "_store_effect_v", "_section_effect_v", 2

        current = _block_metrics(
            target=target, candidate_parent="current_policy", actual_col=actual_col,
            level_col=level_col, effect_col=None, current_raw_col=current_raw_col,
            current_parent_col=current_parent_col, decimals=decimals,
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
                current_parent_col=current_parent_col, decimals=decimals,
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
            _block_expr(train_start, cfg.block_days),
            pl.col("unique_id").str.replace(r"\|\|S:.*$", "").alias("_store_uid"),
        )
        .join(levels, on="unique_id", how="left")
    )

    parent_daily = _parent_block_table(
        parent_forecasts,
        train_start=train_start,
        block_days=cfg.block_days,
        driver_effect_limit=cfg.driver_effect_limit,
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

    (
        yhat, vhat, level_y, level_v, alpha_y, alpha_v, eligible,
        diag_ae_y, diag_den_y, diag_se_y, diag_ae_v, diag_den_v, diag_se_v,
    ) = _ses_walkforward_kernel(
        rows["_uid_code"].to_numpy(),
        rows["_block"].to_numpy(),
        rows["_warmup"].cast(pl.UInt8).to_numpy(),
        rows["_period_code"].to_numpy(),
        rows["y"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        rows["value"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        rows["driver_factor_y"].cast(pl.Float64).fill_null(1.0).to_numpy(),
        rows["driver_factor_value"].cast(pl.Float64).fill_null(1.0).to_numpy(),
        rows["initial_level_y"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        rows["initial_level_value"].cast(pl.Float64).fill_null(0.0).to_numpy(),
        alphas,
        productive_alpha_mask,
        default_idx,
        1 if diagnostics_out is not None else 0,
    )

    if optimization_phase2 and parent_diagnostics_out is not None:
        parent_diag = _collect_leaf_parent_candidate_diagnostics(
            rows,
            yhat_raw=yhat,
            valuehat_raw=vhat,
            level_y=level_y,
            level_value=level_v,
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
        pl.lit("ses_deseasonalized").alias("leaf_level_method_y"),
        pl.lit("ses_deseasonalized").alias("leaf_level_method_value"),
        pl.lit("rls_non_intercept").alias("parent_driver_mode_y"),
        pl.lit("rls_non_intercept").alias("parent_driver_mode_value"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("driver_effect")).alias("driver_effect"),
        pl.when(pl.col("_warmup")).then(0.0).otherwise(pl.col("driver_effect_value")).alias("driver_effect_value"),
        pl.col("_block").alias("rls_block"),
        (pl.col("_block") * cfg.block_days).clip(lower_bound=cfg.warmup_days).cast(pl.Int32).alias("rls_train_days"),
        pl.concat_str(
            [pl.lit("SES+RLS("), pl.col("parent_model_y"), pl.lit("/"), pl.col("parent_model_value"), pl.lit(")")]
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
        "driver_effect", "driver_effect_value", "leaf_level_method_y", "leaf_level_method_value",
    ]
    keep += [c for c in meta_cols if c not in keep]
    if meta:
        keep += [k for k in meta if k not in keep]
    return out.select([c for c in keep if c in out.columns]).sort(["unique_id", "ds"])
