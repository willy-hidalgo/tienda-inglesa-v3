"""Fast diagnostic lab for SKU+Store exception models.

This module is deliberately OUTSIDE the productive SES+RLS path.  It reads an
existing forecast artifact, classifies leaf demand regimes from CLOSED history,
and benchmarks a small set of cheap challengers only for a bounded candidate
set.  Nothing here changes the production forecast, alpha/lambda/parent/driver
selection, or the official y!=0 metric.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import settings

# Classical Syntetos/Boylan regime cut points are starting diagnostics only.
ADI_CUT = 1.32
CV2_CUT = 0.49

# Diagnostic-only routing thresholds.  These do NOT route production.
HIGH_WMAPE = 3.00
MEDIUM_WMAPE = 1.50
ERROR_PERCENTILE_SCREEN = 0.99
MEDIUM_ERROR_PERCENTILE = 0.95
EXTREME_RELATIVE_WMAPE = 10.0
DEFAULT_FOLDS = 6
DEFAULT_BLOCK_DAYS = 28
DEFAULT_MAX_SERIES = 250
HARD_MAX_SERIES = 2000
MIN_POSITIVE_TRAIN = 2


@dataclass(frozen=True)
class ModelResult:
    forecast: float
    model: str


def _positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


def forecast_zero(train: np.ndarray) -> float:
    return 0.0


def forecast_last_positive(train: np.ndarray) -> float:
    p = _positive(train)
    return float(p[-1]) if p.size else 0.0


def forecast_positive_median(train: np.ndarray, window: int = 28) -> float:
    p = _positive(train)
    if not p.size:
        return 0.0
    return float(np.median(p[-int(window):]))


def forecast_positive_trimmed_mean(train: np.ndarray, window: int = 28) -> float:
    """Cheap robust-positive baseline; 10% trim when enough positives exist."""
    p = _positive(train)
    if not p.size:
        return 0.0
    p = np.sort(p[-int(window):])
    k = int(math.floor(0.10 * p.size)) if p.size >= 10 else 0
    if k > 0 and 2 * k < p.size:
        p = p[k:-k]
    return float(np.mean(p))


def forecast_positive_ses(train: np.ndarray, alpha: float = 0.20) -> float:
    """SES over positive magnitudes only; diagnostic challenger, no alpha tuning."""
    p = _positive(train)
    if not p.size:
        return 0.0
    level = float(np.median(p[: min(7, p.size)]))
    for z in p:
        level = alpha * float(z) + (1.0 - alpha) * level
    return max(0.0, float(level))


def forecast_croston_sba(train: np.ndarray, alpha: float = 0.10) -> float:
    """Croston with SBA bias correction; expected demand per calendar period."""
    x = np.asarray(train, dtype=float)
    idx = np.flatnonzero(np.isfinite(x) & (x > 0))
    if idx.size == 0:
        return 0.0
    z = float(x[idx[0]])
    q = float(idx[0] + 1)
    prev = int(idx[0])
    for j in idx[1:]:
        interval = float(int(j) - prev)
        z += alpha * (float(x[j]) - z)
        q += alpha * (interval - q)
        prev = int(j)
    if q <= 0:
        return 0.0
    return max(0.0, (1.0 - alpha / 2.0) * z / q)


def forecast_tsb(train: np.ndarray, alpha_p: float = 0.10, alpha_z: float = 0.10) -> float:
    """Teunter-Syntetos-Babai expected demand; occurrence x positive magnitude."""
    x = np.asarray(train, dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    if x.size == 0:
        return 0.0
    pos = x > 0
    if not np.any(pos):
        return 0.0
    first = int(np.flatnonzero(pos)[0])
    p = 1.0 / float(first + 1)
    z = float(x[first])
    for val in x[first + 1 :]:
        occurrence = 1.0 if val > 0 else 0.0
        p = p + alpha_p * (occurrence - p)
        if val > 0:
            z = z + alpha_z * (float(val) - z)
    return max(0.0, p * z)


def forecast_hurdle_robust(train: np.ndarray, occurrence_window: int = 84, magnitude_window: int = 28) -> float:
    """Fast hurdle baseline: recent occurrence rate x robust positive magnitude."""
    x = np.asarray(train, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    tail = x[-min(int(occurrence_window), x.size) :]
    p = float(np.mean(tail > 0)) if tail.size else 0.0
    mag = forecast_positive_median(x, magnitude_window)
    return max(0.0, p * mag)


# Keep the productive challenger set deliberately tiny.  Croston/TSB remain
# available as diagnostic functions above, but are not in the default benchmark
# because the first experiment showed that cheaper robust baselines won the
# reviewed leaves while preserving the project's speed contract.
CHALLENGERS: dict[str, Callable[[np.ndarray], float]] = {
    "last_positive_naive": forecast_last_positive,
    "positive_median_28": forecast_positive_median,
    "positive_ses_a20": forecast_positive_ses,
    "hurdle_robust_84x28": forecast_hurdle_robust,
}

# A zero forecast is retained only as a diagnostic lower-complexity baseline.
# It is never eligible for automatic routing/recommendation.
DIAGNOSTIC_BASELINES: dict[str, Callable[[np.ndarray], float]] = {
    "zero_naive": forecast_zero,
}

BENCHMARK_MODELS: dict[str, Callable[[np.ndarray], float]] = {
    **DIAGNOSTIC_BASELINES,
    **CHALLENGERS,
}


def _official_metrics(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float, int]:
    """Official support y!=0 plus absolute components; same sign as production BIAS."""
    y = np.asarray(actual, dtype=float)
    yhat = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat) & (y != 0)
    if not np.any(m):
        return (math.nan, math.nan, 0.0, 0.0, 0)
    yy = y[m]
    pp = yhat[m]
    den = float(np.abs(yy).sum())
    ae = float(np.abs(yy - pp).sum())
    se = float((pp - yy).sum())
    return (ae / den if den else math.nan, se / den if den else math.nan, ae, den, int(m.sum()))


def _zero_demand_forecast(actual: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p) & (y == 0) & (p > 0)
    return float(p[m].sum()) if np.any(m) else 0.0


def classify_regime(adi: float | None, cv2: float | None, days_since_last_positive: int | None) -> str:
    if days_since_last_positive is not None and days_since_last_positive >= 28:
        return "dormant"
    if adi is None or not math.isfinite(float(adi)):
        return "no_positive"
    cv = float(cv2) if cv2 is not None and math.isfinite(float(cv2)) else 0.0
    if adi <= ADI_CUT and cv <= CV2_CUT:
        return "smooth"
    if adi <= ADI_CUT and cv > CV2_CUT:
        return "erratic"
    if adi > ADI_CUT and cv <= CV2_CUT:
        return "intermittent"
    return "lumpy"


def _target_frames(df, pl):
    base_cols = ["unique_id", "ds", "period_type"]
    if "rls_metric_eligible" in df.columns:
        base_cols.append("rls_metric_eligible")
    frames = []
    for target, actual, forecast in (
        ("Unidades", "y", "yhat"),
        ("Valor ($)", "value", "valuehat"),
    ):
        if actual not in df.columns or forecast not in df.columns:
            continue
        f = df.select(base_cols + [actual, forecast]).rename({actual: "actual", forecast: "forecast"})
        frames.append(f.with_columns(pl.lit(target).alias("target")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def build_regime_diagnostics(long_df, pl):
    """One row per leaf/target from closed in-sample history only."""
    leaf = long_df.filter(
        (pl.col("period_type") == "in_sample")
        & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        & pl.col("actual").is_not_null()
        & pl.col("actual").is_finite()
    ).sort(["target", "unique_id", "ds"])
    if "rls_metric_eligible" in leaf.columns:
        score_leaf = leaf.filter(pl.col("rls_metric_eligible") == True)
    else:
        score_leaf = leaf

    # Productive leaf history is intentionally sparse: SKU+Store rows are not
    # densified across the full training calendar for speed.  Regime diagnostics
    # must therefore reconstruct calendar exposure implicitly instead of treating
    # "observed sale rows" as "days".  Start each leaf at its first observed day
    # (do not invent pre-launch zeros) and extend it causally to the latest closed
    # in-sample day of its section/target.  Missing calendar days inside that
    # interval are zero-demand days for ADI/dormancy diagnostics only.
    leaf = leaf.with_columns(
        pl.col("unique_id").str.split("||").list.first().alias("section")
    )
    section_end = leaf.group_by(["section", "target"]).agg(
        pl.col("ds").max().alias("history_end")
    )
    base = (
        leaf.group_by(["unique_id", "target", "section"]).agg(
            pl.col("ds").min().alias("history_start"),
            pl.len().alias("n_observed_rows"),
            (pl.col("actual") > 0).sum().alias("n_positive"),
        )
        .join(section_end, on=["section", "target"], how="left")
        .with_columns(
            (
                (pl.col("history_end") - pl.col("history_start")).dt.total_days()
                + 1
            )
            .cast(pl.Int32)
            .alias("n_days")
        )
    )
    pos = leaf.filter(pl.col("actual") > 0).with_columns(
        (pl.col("ds").diff().over(["target", "unique_id"]).dt.total_days() - 1)
        .cast(pl.Int32)
        .alias("zero_gap_days")
    )
    pos_stats = pos.group_by(["unique_id", "target"]).agg(
        pl.col("actual").mean().alias("positive_mean"),
        pl.col("actual").std().fill_null(0.0).alias("positive_std"),
        pl.col("actual").median().alias("positive_median"),
        pl.col("actual").quantile(0.90, interpolation="nearest").alias("positive_p90"),
        pl.col("actual").max().alias("positive_max"),
        pl.col("ds").last().alias("last_positive_ds"),
        pl.col("zero_gap_days").max().fill_null(0).alias("longest_zero_gap"),
        pl.col("zero_gap_days").median().fill_null(0.0).alias("median_zero_gap"),
        (pl.col("zero_gap_days") >= 28).sum().alias("reactivations_28d"),
        (pl.col("zero_gap_days") >= 56).sum().alias("reactivations_56d"),
        (pl.col("zero_gap_days") >= 84).sum().alias("reactivations_84d"),
    )
    scored = score_leaf.filter(
        pl.col("forecast").is_not_null()
        & pl.col("forecast").is_finite()
        & (pl.col("actual") != 0)
    ).with_columns(
        (pl.col("actual") - pl.col("forecast")).abs().alias("_abs_error"),
        pl.col("actual").abs().alias("_abs_y"),
        (pl.col("forecast") - pl.col("actual")).alias("_signed_error"),
    ).group_by(["unique_id", "target"]).agg(
        pl.col("_abs_error").sum().alias("champion_sum_abs_error"),
        pl.col("_abs_y").sum().alias("champion_sum_abs_y"),
        pl.col("_signed_error").sum().alias("champion_sum_signed_error"),
        pl.len().alias("champion_n_points"),
    ).with_columns(
        (pl.col("champion_sum_abs_error") / pl.col("champion_sum_abs_y").replace(0, None)).alias("champion_wmape"),
        (pl.col("champion_sum_signed_error") / pl.col("champion_sum_abs_y").replace(0, None)).alias("champion_bias"),
    )
    out = base.join(pos_stats, on=["unique_id", "target"], how="left").join(scored, on=["unique_id", "target"], how="left")
    out = out.with_columns(
        (pl.col("n_days") / pl.col("n_positive").replace(0, None)).alias("adi"),
        ((pl.col("positive_std") / pl.col("positive_mean").replace(0, None)) ** 2).alias("cv2_positive"),
        (pl.col("n_positive") / pl.col("n_days").replace(0, None)).alias("positive_share"),
        (pl.col("history_end") - pl.col("last_positive_ds")).dt.total_days().fill_null(pl.col("n_days")).cast(pl.Int32).alias("days_since_last_positive"),
    )
    total_err = out.group_by("target").agg(pl.col("champion_sum_abs_error").sum().alias("_target_abs_error"))
    out = out.join(total_err, on="target", how="left").with_columns(
        (pl.col("champion_sum_abs_error") / pl.col("_target_abs_error").replace(0, None)).alias("abs_error_contribution"),
        pl.col("champion_sum_abs_error").rank(method="max").over("target").alias("_error_rank_target"),
        pl.len().over("target").alias("_error_n_target"),
        pl.col("champion_sum_abs_error").rank(method="max").over(["section", "target"]).alias("_error_rank_st"),
        pl.len().over(["section", "target"]).alias("_error_n_st"),
    ).with_columns(
        (pl.col("_error_rank_target") / pl.col("_error_n_target").replace(0, None)).alias("abs_error_percentile_target"),
        (pl.col("_error_rank_st") / pl.col("_error_n_st").replace(0, None)).alias("abs_error_percentile_section_target"),
        (
            (~pl.col("champion_wmape").is_finite())
            | (pl.col("champion_wmape") >= EXTREME_RELATIVE_WMAPE)
            | (~pl.col("champion_sum_abs_error").is_finite())
        ).alias("extreme_relative_error"),
    ).drop("_target_abs_error", "_error_rank_target", "_error_n_target", "_error_rank_st", "_error_n_st")

    # Regime classification is expressed in Polars for speed and reproducibility.
    out = out.with_columns(
        pl.when(pl.col("n_positive") == 0).then(pl.lit("no_positive"))
        .when(pl.col("days_since_last_positive") >= 28).then(pl.lit("dormant"))
        .when((pl.col("adi") <= ADI_CUT) & (pl.col("cv2_positive").fill_null(0) <= CV2_CUT)).then(pl.lit("smooth"))
        .when((pl.col("adi") <= ADI_CUT) & (pl.col("cv2_positive").fill_null(0) > CV2_CUT)).then(pl.lit("erratic"))
        .when((pl.col("adi") > ADI_CUT) & (pl.col("cv2_positive").fill_null(0) <= CV2_CUT)).then(pl.lit("intermittent"))
        .otherwise(pl.lit("lumpy")).alias("regime")
    )
    return out.sort(["target", "champion_sum_abs_error"], descending=[False, True])


def select_exception_candidates(diag, pl, max_series: int = DEFAULT_MAX_SERIES):
    """Robust bounded screen for diagnostic challengers only.

    Selection is intentionally based on within-section/target error percentiles
    instead of global error contribution.  A single numerical explosion can no
    longer suppress every other candidate.  Regime only controls eligibility;
    it never routes production.
    """
    if diag.height == 0:
        return diag
    max_series = min(int(max_series), HARD_MAX_SERIES)
    d = diag.with_columns(
        pl.col("champion_sum_abs_error")
        .rank(method="max")
        .over(["section", "target"])
        .alias("_error_rank_st"),
        pl.len().over(["section", "target"]).alias("_error_n_st"),
        pl.col("champion_sum_abs_error")
        .rank(method="max")
        .over("target")
        .alias("_error_rank_target"),
        pl.len().over("target").alias("_error_n_target"),
    ).with_columns(
        (pl.col("_error_rank_st") / pl.col("_error_n_st").replace(0, None)).alias("abs_error_percentile_section_target"),
        (pl.col("_error_rank_target") / pl.col("_error_n_target").replace(0, None)).alias("abs_error_percentile_target"),
        (
            (~pl.col("champion_wmape").is_finite())
            | (pl.col("champion_wmape") >= EXTREME_RELATIVE_WMAPE)
            | (~pl.col("champion_sum_abs_error").is_finite())
        ).alias("extreme_relative_error"),
    ).with_columns(
        (
            pl.col("extreme_relative_error")
            | (pl.col("champion_wmape") >= HIGH_WMAPE)
            | (
                (pl.col("champion_wmape") >= MEDIUM_WMAPE)
                & (pl.col("abs_error_percentile_section_target") >= MEDIUM_ERROR_PERCENTILE)
            )
            | (
                pl.col("regime").is_in(["intermittent", "lumpy", "dormant"])
                & (pl.col("abs_error_percentile_section_target") >= ERROR_PERCENTILE_SCREEN)
            )
        ).alias("exception_candidate")
    )
    return (
        d.filter(pl.col("exception_candidate") == True)
        .sort(
            ["extreme_relative_error", "abs_error_percentile_section_target", "champion_sum_abs_error"],
            descending=[True, True, True],
        )
        .head(max_series)
        .drop("_error_rank_st", "_error_n_st", "_error_rank_target", "_error_n_target")
    )

def benchmark_candidates(long_df, candidates, pl, folds: int, block_days: int):
    """Historical closed-fold benchmark on a bounded leaf set, with timings."""
    if candidates.height == 0:
        return pl.DataFrame(), pl.DataFrame()
    ids = candidates.select(["unique_id", "target"]).unique()
    hist = long_df.filter(
        (pl.col("period_type") == "in_sample")
        & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
    ).join(ids, on=["unique_id", "target"], how="inner").sort(["target", "unique_id", "ds"])
    rows: list[dict] = []
    model_seconds = {name: 0.0 for name in BENCHMARK_MODELS}
    model_calls = {name: 0 for name in BENCHMARK_MODELS}
    # Closed folds must see the same implicit zero-demand calendar used by the
    # regime diagnostics.  Densify only the bounded candidate set (<=250 by
    # default), never the full productive leaf panel.  This keeps the lab fast
    # and makes results independent of whether production stores leaf history
    # sparsely or densely.
    hist = hist.with_columns(
        pl.col("unique_id").str.split("||").list.first().alias("_section")
    )
    section_end = (
        long_df.filter(
            (pl.col("period_type") == "in_sample")
            & (pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
        )
        .with_columns(
            pl.col("unique_id").str.split("||").list.first().alias("_section")
        )
        .group_by(["_section", "target"])
        .agg(pl.col("ds").max().alias("_section_history_end"))
    )
    hist = hist.join(section_end, on=["_section", "target"], how="left")

    for part in hist.partition_by(["target", "unique_id"], maintain_order=True):
        uid = str(part["unique_id"][0])
        target = str(part["target"][0])
        sparse_dates = part["ds"].to_numpy().astype("datetime64[D]")
        sparse_actual = part["actual"].to_numpy().astype(float, copy=False)
        sparse_champion = part["forecast"].to_numpy().astype(float, copy=False)
        if sparse_dates.size == 0:
            continue
        start = sparse_dates[0]
        section_end_value = part["_section_history_end"][0]
        end = (
            np.datetime64(section_end_value, "D")
            if section_end_value is not None
            else sparse_dates[-1]
        )
        if end < start:
            continue
        dates = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
        actual = np.zeros(dates.size, dtype=float)
        champion = np.full(dates.size, np.nan, dtype=float)
        pos = (sparse_dates - start).astype("timedelta64[D]").astype(np.int64)
        valid_pos = (pos >= 0) & (pos < dates.size)
        actual[pos[valid_pos]] = sparse_actual[valid_pos]
        champion[pos[valid_pos]] = sparse_champion[valid_pos]
        if actual.size < block_days * 2:
            continue
        for fold in range(int(folds), 0, -1):
            fold_end = end - np.timedelta64((fold - 1) * block_days, "D")
            fold_start = fold_end - np.timedelta64(block_days - 1, "D")
            train_mask = dates < fold_start
            test_mask = (dates >= fold_start) & (dates <= fold_end)
            train = actual[train_mask]
            y = actual[test_mask]
            champ = champion[test_mask]
            if y.size == 0 or np.sum(np.isfinite(train) & (train > 0)) < MIN_POSITIVE_TRAIN:
                continue
            cwm, cbias, cae, cden, cn = _official_metrics(y, champ)
            rows.append(dict(unique_id=uid, target=target, fold=fold, model="ses_rls_champion", wmape=cwm, bias=cbias, sum_abs_error=cae, sum_abs_y=cden, n_points=cn, zero_demand_forecast=_zero_demand_forecast(y, champ)))
            for name, fn in BENCHMARK_MODELS.items():
                tm = time.perf_counter()
                f = float(fn(train))
                model_seconds[name] += time.perf_counter() - tm
                model_calls[name] += 1
                pred = np.full(y.shape, f, dtype=float)
                wm, bias, ae, den, n = _official_metrics(y, pred)
                rows.append(dict(unique_id=uid, target=target, fold=fold, model=name, wmape=wm, bias=bias, sum_abs_error=ae, sum_abs_y=den, n_points=n, zero_demand_forecast=_zero_demand_forecast(y, pred)))
    timing_rows = []
    for name in BENCHMARK_MODELS:
        calls = model_calls[name]
        sec = model_seconds[name]
        timing_rows.append(dict(model=name, calls=calls, model_seconds=sec, mean_microseconds_per_call=(sec / calls * 1e6) if calls else math.nan))
    return (pl.DataFrame(rows) if rows else pl.DataFrame(), pl.DataFrame(timing_rows))

def summarize_benchmark(folds_df, diag, pl):
    if folds_df.height == 0:
        return pl.DataFrame(), pl.DataFrame()
    # Folds with no official support (y != 0) return NaN metrics by design.
    # Treat them as non-evaluable, not as losses and not as NaN contributors.
    # This is essential on sparse/intermittent series where some 28d folds can
    # legitimately contain no positive actuals.
    fold_winners = (
        folds_df
        .with_columns(
            pl.when(pl.col("wmape").is_finite() & (pl.col("n_points") > 0))
            .then(pl.col("wmape"))
            .otherwise(None)
            .alias("_wmape_eval")
        )
        .with_columns(
            pl.col("_wmape_eval")
            .min()
            .over(["unique_id", "target", "fold"])
            .alias("_best_fold")
        )
        .with_columns(
            pl.when(
                pl.col("_wmape_eval").is_not_null()
                & pl.col("_best_fold").is_not_null()
            )
            .then(pl.col("_wmape_eval") <= pl.col("_best_fold") + 1e-12)
            .otherwise(None)
            .alias("fold_win")
        )
    )
    summary = fold_winners.group_by(["unique_id", "target", "model"]).agg(
        pl.col("sum_abs_error").sum(),
        pl.col("sum_abs_y").sum(),
        pl.col("n_points").sum(),
        pl.col("zero_demand_forecast").sum(),
        pl.col("fold")
        .filter(pl.col("_wmape_eval").is_not_null())
        .n_unique()
        .alias("n_folds"),
        pl.col("_wmape_eval").median().alias("median_fold_wmape"),
        pl.col("_wmape_eval").max().alias("worst_fold_wmape"),
        pl.col("fold_win").mean().alias("best_model_fold_share"),
    ).with_columns(
        (pl.col("sum_abs_error") / pl.col("sum_abs_y").replace(0, None)).alias("wmape")
    )
    bias_agg = (
        folds_df
        .with_columns(
            pl.when(
                pl.col("bias").is_finite()
                & (pl.col("sum_abs_y") > 0)
                & (pl.col("n_points") > 0)
            )
            .then(pl.col("bias") * pl.col("sum_abs_y"))
            .otherwise(0.0)
            .alias("_signed_error")
        )
        .group_by(["unique_id", "target", "model"])
        .agg(pl.col("_signed_error").sum())
    )
    summary = summary.join(bias_agg, on=["unique_id", "target", "model"], how="left").with_columns(
        (pl.col("_signed_error") / pl.col("sum_abs_y").replace(0, None)).alias("bias")
    ).drop("_signed_error")

    champ = summary.filter(pl.col("model") == "ses_rls_champion").select(
        "unique_id", "target",
        pl.col("wmape").alias("champion_wmape"),
        pl.col("bias").alias("champion_bias"),
        pl.col("zero_demand_forecast").alias("champion_zero_demand_forecast"),
    )
    champ_fold = folds_df.filter(pl.col("model") == "ses_rls_champion").select(
        "unique_id",
        "target",
        "fold",
        pl.col("wmape").alias("_champ_fold_wmape"),
        pl.col("n_points").alias("_champ_fold_n_points"),
    )
    vs = (
        folds_df
        .join(champ_fold, on=["unique_id", "target", "fold"], how="left")
        .with_columns(
            pl.when(
                pl.col("wmape").is_finite()
                & pl.col("_champ_fold_wmape").is_finite()
                & (pl.col("n_points") > 0)
                & (pl.col("_champ_fold_n_points") > 0)
            )
            .then(pl.col("wmape") <= pl.col("_champ_fold_wmape") + 1e-12)
            .otherwise(None)
            .alias("_win_vs_champ")
        )
        .group_by(["unique_id", "target", "model"])
        .agg(
            pl.col("_win_vs_champ")
            .mean()
            .alias("fold_win_share_vs_champion")
        )
    )
    summary = summary.join(vs, on=["unique_id", "target", "model"], how="left").join(champ, on=["unique_id", "target"], how="left")
    diag_cols = [
        "unique_id", "target", "section", "regime", "adi", "cv2_positive",
        "champion_sum_abs_error", "abs_error_contribution",
        "abs_error_percentile_target", "abs_error_percentile_section_target",
        "extreme_relative_error",
    ]
    summary = summary.join(diag.select(diag_cols), on=["unique_id", "target"], how="left").with_columns(
        (pl.col("champion_wmape") - pl.col("wmape")).alias("wmape_improvement_abs"),
        ((pl.col("champion_wmape") - pl.col("wmape")) / pl.col("champion_wmape").replace(0, None)).alias("wmape_improvement_rel"),
        (pl.col("bias").abs() - pl.col("champion_bias").abs()).alias("abs_bias_deterioration"),
    )
    candidates = summary.filter(pl.col("model") != "ses_rls_champion").with_columns(
        (
            pl.col("model").is_in(list(CHALLENGERS))
            & (pl.col("n_folds") >= 4)
            & (pl.col("fold_win_share_vs_champion") >= 0.70)
            & (pl.col("wmape_improvement_abs") >= 0.20)
            & (pl.col("wmape_improvement_rel") >= 0.20)
            & (pl.col("abs_bias_deterioration") <= 0.10)
        ).alias("eligible_for_review")
    ).sort(["eligible_for_review", "extreme_relative_error", "champion_sum_abs_error", "wmape_improvement_abs"], descending=[True, True, True, True])
    return summary, candidates

def build_recommendations(screen, summary, candidates, runtime, pl):
    """One diagnostic recommendation per screened leaf/target; never routes production."""
    if screen.height == 0:
        return pl.DataFrame()
    eligible = candidates.filter(pl.col("eligible_for_review") == True)
    if eligible.height:
        best = eligible.sort(
            ["unique_id", "target", "wmape", "wmape_improvement_abs"],
            descending=[False, False, False, True],
        ).group_by(["unique_id", "target"], maintain_order=True).first()
        if runtime.height:
            best = best.join(runtime.select(["model", "mean_microseconds_per_call"]), on="model", how="left")
        best = best.select(
            "unique_id", "target",
            pl.col("model").alias("best_challenger"),
            pl.col("wmape").alias("challenger_wmape"),
            pl.col("bias").alias("challenger_bias"),
            "n_folds", "fold_win_share_vs_champion", "wmape_improvement_abs", "wmape_improvement_rel",
            pl.col("mean_microseconds_per_call").alias("estimated_model_us_per_call"),
        )
    else:
        best = pl.DataFrame()

    champ = summary.filter(pl.col("model") == "ses_rls_champion").select(
        "unique_id", "target",
        pl.col("wmape").alias("backtest_champion_wmape"),
        pl.col("bias").alias("backtest_champion_bias"),
        pl.col("n_folds").alias("backtest_n_folds"),
    ) if summary.height else pl.DataFrame()

    rec = screen
    if champ.height:
        rec = rec.join(champ, on=["unique_id", "target"], how="left")
    if best.height:
        rec = rec.join(best, on=["unique_id", "target"], how="left")
    else:
        rec = rec.with_columns(
            pl.lit(None, dtype=pl.String).alias("best_challenger"),
            pl.lit(None, dtype=pl.Float64).alias("challenger_wmape"),
            pl.lit(None, dtype=pl.Float64).alias("challenger_bias"),
            pl.lit(None, dtype=pl.Int64).alias("n_folds"),
            pl.lit(None, dtype=pl.Float64).alias("fold_win_share_vs_champion"),
            pl.lit(None, dtype=pl.Float64).alias("wmape_improvement_abs"),
            pl.lit(None, dtype=pl.Float64).alias("wmape_improvement_rel"),
            pl.lit(None, dtype=pl.Float64).alias("estimated_model_us_per_call"),
        )
    rec = rec.with_columns(
        pl.when(pl.col("extreme_relative_error") == True).then(pl.lit("EXTREME_RELATIVE_ERROR"))
        .when(pl.col("best_challenger").is_not_null()).then(pl.lit("REVIEW_EXCEPTION"))
        .when(pl.col("backtest_n_folds").fill_null(0) >= 4).then(pl.lit("KEEP_SES_RLS"))
        .otherwise(pl.lit("INSUFFICIENT_EVIDENCE"))
        .alias("recommendation")
    )
    keep = [
        "unique_id", "section", "target", "regime", "champion_wmape", "champion_bias",
        "champion_sum_abs_error", "abs_error_percentile_target", "abs_error_percentile_section_target",
        "extreme_relative_error", "backtest_champion_wmape", "backtest_champion_bias",
        "best_challenger", "challenger_wmape", "challenger_bias", "n_folds",
        "fold_win_share_vs_champion", "wmape_improvement_abs", "wmape_improvement_rel",
        "estimated_model_us_per_call", "recommendation",
    ]
    return rec.select([c for c in keep if c in rec.columns]).sort(
        ["extreme_relative_error", "abs_error_percentile_section_target", "champion_sum_abs_error"],
        descending=[True, True, True],
    )


def _write_summary(path: Path, diag, screen, bench, candidates, recommendations, runtime, timings, elapsed: float) -> None:
    counts = diag.group_by("regime").len().sort("len", descending=True).to_dicts() if diag.height else []
    review_leafs = recommendations.filter(recommendations["recommendation"] == "REVIEW_EXCEPTION").height if recommendations.height else 0
    hard_failures = recommendations.filter(recommendations["recommendation"] == "EXTREME_RELATIVE_ERROR").height if recommendations.height else 0
    lines = [
        f"EXCEPTION MODEL LAB V2 | APP_VERSION={settings.APP_VERSION}",
        "STATUS=DIAGNOSTIC_ONLY; production SES+RLS unchanged; no automatic routing/promotion",
        f"elapsed_seconds={elapsed:.3f}",
        f"leaf_target_rows={diag.height}",
        f"benchmarked_leaf_targets={screen.height}",
        f"benchmark_rows={bench.height}",
        f"review_exception_leaf_targets={review_leafs}",
        f"extreme_relative_error_leaf_targets={hard_failures}",
        "regime_counts=" + repr(counts),
        "challengers=" + ",".join(CHALLENGERS.keys()),
        "diagnostic_baselines=" + ",".join(DIAGNOSTIC_BASELINES.keys()),
        "official_metric_support=y!=0",
        f"speed_contract=default max_series={DEFAULT_MAX_SERIES}; hard cap={HARD_MAX_SERIES}; no tuning grids",
        "screening=wmape>=300% OR (wmape>=150% and section-target error percentile>=95%) OR intermittent/lumpy/dormant error percentile>=99%; hard failure wmape>=1000%",
    ]
    if runtime.height:
        lines.append("model_runtime=" + repr(runtime.sort("model").to_dicts()))
    if timings.height:
        lines.append("phase_runtime=" + repr(timings.to_dicts()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(forecast_path: Path, out_dir: Path, folds: int, block_days: int, max_series: int, regimes_only: bool = False) -> int:
    import polars as pl
    t0 = time.perf_counter()
    max_series = min(int(max_series), HARD_MAX_SERIES)
    phase_rows: list[dict] = []
    if not forecast_path.exists():
        raise FileNotFoundError(forecast_path)

    tp = time.perf_counter()
    wanted = ["unique_id", "ds", "period_type", "y", "yhat", "value", "valuehat", "rls_metric_eligible"]
    schema = pl.scan_parquet(str(forecast_path)).collect_schema().names()
    cols = [c for c in wanted if c in schema]
    df = pl.read_parquet(str(forecast_path), columns=cols)
    long_df = _target_frames(df, pl)
    phase_rows.append({"phase": "load_prepare", "seconds": time.perf_counter() - tp})

    tp = time.perf_counter()
    diag = build_regime_diagnostics(long_df, pl)
    phase_rows.append({"phase": "diagnostics", "seconds": time.perf_counter() - tp})
    out_dir.mkdir(parents=True, exist_ok=True)
    diag.write_parquet(out_dir / "leaf_regime_diagnostics.parquet")

    tp = time.perf_counter()
    screen = select_exception_candidates(diag, pl, max_series=max_series)
    phase_rows.append({"phase": "screening", "seconds": time.perf_counter() - tp})
    screen.write_parquet(out_dir / "leaf_exception_screen.parquet")

    if regimes_only:
        elapsed = time.perf_counter() - t0
        timings = pl.DataFrame(phase_rows + [{"phase": "total", "seconds": elapsed}])
        timings.write_parquet(out_dir / "exception_lab_timing.parquet")
        print(f"REGIME LAB V2 v{settings.APP_VERSION}: OK | rows={diag.height:,} candidates={screen.height:,} | {elapsed:.1f}s")
        return 0

    tp = time.perf_counter()
    bench, runtime = benchmark_candidates(long_df, screen, pl, folds=folds, block_days=block_days)
    phase_rows.append({"phase": "benchmark", "seconds": time.perf_counter() - tp})
    bench.write_parquet(out_dir / "leaf_exception_benchmark_folds.parquet")
    runtime.write_parquet(out_dir / "exception_model_runtime.parquet")

    tp = time.perf_counter()
    summary, candidates = summarize_benchmark(bench, diag, pl)
    recommendations = build_recommendations(screen, summary, candidates, runtime, pl)
    phase_rows.append({"phase": "summarize", "seconds": time.perf_counter() - tp})
    summary.write_parquet(out_dir / "leaf_exception_model_summary.parquet")
    candidates.write_parquet(out_dir / "leaf_exception_model_candidates.parquet")
    recommendations.write_parquet(out_dir / "leaf_exception_recommendation.parquet")

    elapsed = time.perf_counter() - t0
    timings = pl.DataFrame(phase_rows + [{"phase": "total", "seconds": elapsed}])
    timings.write_parquet(out_dir / "exception_lab_timing.parquet")
    _write_summary(
        out_dir / "exception_benchmark_summary.txt",
        diag, screen, bench, candidates, recommendations, runtime, timings, elapsed,
    )
    review_n = recommendations.filter(pl.col("recommendation") == "REVIEW_EXCEPTION").height if recommendations.height else 0
    hard_n = recommendations.filter(pl.col("recommendation") == "EXTREME_RELATIVE_ERROR").height if recommendations.height else 0
    print(f"EXCEPTION MODEL LAB V2 v{settings.APP_VERSION}: OK")
    print("- production SES+RLS: UNCHANGED")
    print(f"- leaf/target diagnostics: {diag.height:,}")
    print(f"- benchmarked leaf/targets: {screen.height:,} (default={DEFAULT_MAX_SERIES:,}, hard cap={HARD_MAX_SERIES:,})")
    print(f"- challengers: {len(CHALLENGERS)} cheap models + {len(DIAGNOSTIC_BASELINES)} diagnostic baseline; no tuning grids")
    print(f"- REVIEW_EXCEPTION: {review_n:,}")
    print(f"- EXTREME_RELATIVE_ERROR: {hard_n:,}")
    print(f"- elapsed: {elapsed:.1f}s")
    print(f"- outputs: {out_dir}")
    return 0

def resolve_forecast_path(explicit: str | None, update_block_days: int) -> Path:
    """Resolve existing artifact without forcing users to know directory layout."""
    if explicit:
        return Path(explicit)
    preferred = settings.update_block_forecast_path(update_block_days)
    root = settings.OUT_DIR / "forecast.parquet"
    if preferred.exists():
        return preferred
    if int(update_block_days) == 28 and root.exists():
        return root
    tried = [str(preferred)]
    if int(update_block_days) == 28:
        tried.append(str(root))
    raise FileNotFoundError("No forecast artifact found. Tried: " + "; ".join(tried))


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnostic-only fast exception-model lab V2")
    p.add_argument("--forecast", "--forecast-path", dest="forecast", default=None, help="Existing forecast.parquet; default 28d artifact")
    p.add_argument("--update-block-days", type=int, default=28, choices=list(settings.UPDATE_BLOCK_OPTIONS))
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--block-days", type=int, default=DEFAULT_BLOCK_DAYS)
    p.add_argument("--max-series", type=int, default=DEFAULT_MAX_SERIES, help=f"Diagnostic candidate cap (default {DEFAULT_MAX_SERIES}, hard cap {HARD_MAX_SERIES})")
    p.add_argument("--regimes-only", action="store_true")
    args = p.parse_args()
    fp = resolve_forecast_path(args.forecast, args.update_block_days)
    out = settings.OUT_DIR / "exception_model_lab"
    return run(fp, out, args.folds, args.block_days, args.max_series, args.regimes_only)


if __name__ == "__main__":
    raise SystemExit(main())
