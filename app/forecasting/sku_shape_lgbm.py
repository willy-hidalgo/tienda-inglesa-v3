"""Causal pooled LightGBM correction for SKU-total daily SHAPE (v12.6).

The model is deliberately shape-only:

* it is pooled once per section/target, never SKU×store;
* it is trained only on CLOSED 28-day blocks older than the target origin;
* it predicts a residual of normalized daily SKU shape;
* the corrected path is renormalized to the EXACT 28-day SKU total produced by
  the existing causal v12.4 SKU-total ensemble;
* occurrence and the v12.5 store-share remain downstream and unchanged.

The production hyperparameters and gamma are frozen from the PRE-OOS rolling
backtest.  No target-block actual is used for fitting, tuning, or prediction.
"""
from __future__ import annotations

import datetime as dt
import gc
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

import settings
from app.forecasting.calendar import HolidayCalendar
from app.forecasting.features import CalendarFeatureBuilder

logger = logging.getLogger(__name__)

EPS = 1e-12
LAGS = (28, 56, 84, 364, 365)


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # Polars compatibility
        return lf.collect(streaming=True)


def _require_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "v12.6 requiere LightGBM. Ejecute `uv sync` para instalar las "
            "dependencias declaradas en pyproject.toml."
        ) from exc
    return lgb


def _category_meta(section_id: str, sku_ids: pl.DataFrame) -> pl.DataFrame:
    """Static commercial category mapping; no sales/future target value enters."""
    path = Path(getattr(settings, "SELECTED_PATH"))
    if not path.exists():
        return sku_ids.select("_v12_sku").unique().with_columns(
            pl.lit("__SECTION__").alias("_cat")
        )
    names = set(pl.scan_parquet(str(path)).collect_schema().names())
    cat_col = next(
        (
            c
            for c in (
                "DESC_CATEGORIA",
                "CATEGORIA",
                "DESC_FAMILIA",
                "FAMILIA",
                "DESC_SUBCATEGORIA",
                "SUBCATEGORIA",
            )
            if c in names
        ),
        None,
    )
    if cat_col is None or "SECCION" not in names or "SKU_ID" not in names:
        return sku_ids.select("_v12_sku").unique().with_columns(
            pl.lit("__SECTION__").alias("_cat")
        )
    meta = _collect(
        pl.scan_parquet(str(path))
        .select("SECCION", "SKU_ID", cat_col)
        .with_columns(
            pl.col("SECCION").cast(pl.Utf8),
            pl.col("SKU_ID").cast(pl.Utf8).alias("_v12_sku"),
            pl.col(cat_col)
            .cast(pl.Utf8)
            .fill_null("__UNKNOWN__")
            .alias("_cat"),
        )
        .filter(pl.col("SECCION") == str(section_id))
        .group_by("_v12_sku")
        .agg(pl.col("_cat").first().alias("_cat"))
    )
    return (
        sku_ids.select("_v12_sku")
        .unique()
        .join(meta, on="_v12_sku", how="left")
        .with_columns(pl.col("_cat").fill_null("__UNKNOWN__"))
    )


def _calendar_table(lo: dt.date, hi: dt.date) -> tuple[pl.DataFrame, list[str]]:
    dates = pl.DataFrame(
        {"ds": pl.date_range(lo, hi, interval="1d", eager=True)}
    ).with_columns(pl.col("ds").cast(pl.Date))
    cal = HolidayCalendar(getattr(settings, "HOLIDAYS", {}), max(lo.year, hi.year))
    feats = CalendarFeatureBuilder(cal).extract_drivers(dates)
    out = dates
    event_cols: list[str] = []
    for holiday in getattr(settings, "HOLIDAYS", {}).keys():
        hcols = [c for c in feats.columns if c == holiday or c.startswith(f"{holiday}_")]
        if not hcols:
            continue
        cname = f"_ev_{holiday}"
        event_cols.append(cname)
        out = out.join(
            feats.select(
                "ds",
                pl.max_horizontal(*[pl.col(c).cast(pl.Float64) for c in hcols])
                .fill_null(0.0)
                .alias(cname),
            ),
            on="ds",
            how="left",
        )
    if event_cols:
        out = out.with_columns([pl.col(c).fill_null(0.0) for c in event_cols])
    return out, event_cols


def _price_state(sku_daily: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    r28 = origin - dt.timedelta(days=28)
    r84 = origin - dt.timedelta(days=84)
    x28 = (
        sku_daily.filter((pl.col("ds") >= pl.lit(r28)) & (pl.col("ds") < pl.lit(origin)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_q28"),
            pl.col("_sku_day_v").sum().alias("_v28"),
        )
    )
    x84 = (
        sku_daily.filter((pl.col("ds") >= pl.lit(r84)) & (pl.col("ds") < pl.lit(origin)))
        .group_by("_v12_sku")
        .agg(
            pl.col("_sku_day_y").sum().alias("_q84"),
            pl.col("_sku_day_v").sum().alias("_v84"),
        )
    )
    return (
        x84.join(x28, on="_v12_sku", how="left")
        .with_columns(
            pl.when(pl.col("_q28").fill_null(0.0) > EPS)
            .then(pl.col("_v28").fill_null(0.0) / pl.col("_q28"))
            .otherwise(None)
            .alias("_asp28"),
            pl.when(pl.col("_q84").fill_null(0.0) > EPS)
            .then(pl.col("_v84").fill_null(0.0) / pl.col("_q84"))
            .otherwise(None)
            .alias("_asp84"),
        )
        .with_columns(
            pl.when((pl.col("_asp28") > EPS) & (pl.col("_asp84") > EPS))
            .then((pl.col("_asp28") / pl.col("_asp84")).log().clip(-0.70, 0.70))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("_price_state")
        )
        .select("_v12_sku", "_price_state")
    )


def _lag_table(sku_daily: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    ids = sku_daily.select("_v12_sku").unique()
    days = pl.DataFrame(
        {"ds": pl.date_range(start, end, interval="1d", eager=True)}
    ).with_columns(pl.col("ds").cast(pl.Date))
    out = ids.join(days, how="cross")
    for lag in LAGS:
        shifted = sku_daily.select(
            "_v12_sku",
            (pl.col("ds") + dt.timedelta(days=lag)).alias("ds"),
            pl.col("_sku_day_y").cast(pl.Float64).alias(f"lag{lag}_y"),
            pl.col("_sku_day_v").cast(pl.Float64).alias(f"lag{lag}_v"),
        )
        out = out.join(shifted, on=["_v12_sku", "ds"], how="left")
    return out.with_columns(
        [pl.col(f"lag{lag}_{s}").fill_null(0.0) for lag in LAGS for s in ("y", "v")]
    )


def _state_table(sku_daily: pl.DataFrame, origin: dt.date) -> pl.DataFrame:
    def _sum(lo: int, hi: int, suffix: str, alias: str) -> pl.DataFrame:
        src = "_sku_day_y" if suffix == "y" else "_sku_day_v"
        return (
            sku_daily.filter(
                (pl.col("ds") >= pl.lit(origin - dt.timedelta(days=hi)))
                & (pl.col("ds") < pl.lit(origin - dt.timedelta(days=lo)))
            )
            .group_by("_v12_sku")
            .agg(pl.col(src).sum().cast(pl.Float64).alias(alias))
        )

    base = sku_daily.select("_v12_sku").unique()
    for s in ("y", "v"):
        for lo, hi, name in ((0, 28, "r28"), (28, 56, "p28"), (0, 84, "r84")):
            base = base.join(_sum(lo, hi, s, f"_{name}_{s}"), on="_v12_sku", how="left")
        base = base.with_columns(
            pl.col(f"_r28_{s}").fill_null(0.0),
            pl.col(f"_p28_{s}").fill_null(0.0),
            pl.col(f"_r84_{s}").fill_null(0.0),
        ).with_columns(
            ((pl.col(f"_r28_{s}") + 1.0) / (pl.col(f"_r84_{s}") / 3.0 + 1.0))
            .log()
            .clip(-2.0, 2.0)
            .alias(f"recent_state_{s}"),
            ((pl.col(f"_r28_{s}") + 1.0) / (pl.col(f"_p28_{s}") + 1.0))
            .log()
            .clip(-2.0, 2.0)
            .alias(f"trend_state_{s}"),
        )
    return base.select(
        "_v12_sku", "recent_state_y", "trend_state_y", "recent_state_v", "trend_state_v"
    )


def _attach_features(
    block: pl.DataFrame,
    sku_daily: pl.DataFrame,
    start: dt.date,
    end: dt.date,
    calendar: pl.DataFrame,
) -> pl.DataFrame:
    lags = _lag_table(sku_daily, start, end)
    states = _state_table(sku_daily, start)
    return (
        block.join(lags, on=["_v12_sku", "ds"], how="left")
        .join(states, on="_v12_sku", how="left")
        .join(_price_state(sku_daily, start), on="_v12_sku", how="left")
        .join(calendar, on="ds", how="left")
        .with_columns(
            pl.col("ds").dt.weekday().cast(pl.Int8).alias("_dow"),
            pl.col("ds").dt.month().cast(pl.Int8).alias("_mon"),
            ((pl.col("ds") - pl.lit(start)).dt.total_days()).cast(pl.Int16).alias("_h"),
        )
        .with_columns(
            [pl.col(f"lag{lag}_{s}").fill_null(0.0) for lag in LAGS for s in ("y", "v")]
            + [
                pl.col("recent_state_y").fill_null(0.0),
                pl.col("trend_state_y").fill_null(0.0),
                pl.col("recent_state_v").fill_null(0.0),
                pl.col("trend_state_v").fill_null(0.0),
                pl.col("_price_state").fill_null(0.0),
            ]
        )
    )


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x.astype(np.float64, copy=False), 1e-12, None))


def _sum_over_sku(df: pl.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    calc = df.select(["_v12_sku"] + cols).with_columns(
        [
            pl.col(c).sum().over("_v12_sku").cast(pl.Float64).alias(f"__tot_{c}")
            for c in cols
        ]
    )
    return {
        c: calc.get_column(f"__tot_{c}").to_numpy().astype(np.float64, copy=False)
        for c in cols
    }


def _shape_share(x: np.ndarray, total: np.ndarray) -> np.ndarray:
    smooth_frac = float(getattr(settings, "V12_SKU_SHAPE_LGBM_SMOOTH_FRAC", 0.02))
    pc = np.maximum(1e-9, smooth_frac * np.maximum(total, 0.0) / 28.0)
    den = np.maximum(total + 28.0 * pc, 1e-9)
    return (np.maximum(x, 0.0) + pc) / den


def _cat_values(blocks: list[pl.DataFrame]) -> list[str]:
    vals: set[str] = set()
    for b in blocks:
        if "_cat" in b.columns:
            vals.update(str(x) for x in b.get_column("_cat").drop_nulls().unique().to_list())
    return sorted(vals)


def _feature_names(event_cols: list[str], cats: list[str]) -> list[str]:
    names = [
        "log_base",
        "log_base_total",
        "log_base_share",
        "log_lag28",
        "log_lag56",
        "log_lag84",
        "log_lag364",
        "log_lag365",
        "log_lag28_share",
        "log_lag56_share",
        "log_lag84_share",
        "log_lag364_share",
        "log_lag365_share",
        "recent_state",
        "trend_state",
        "price_state",
        "h_sin",
        "h_cos",
        "dow_sin",
        "dow_cos",
        "mon_sin",
        "mon_cos",
    ]
    names += [f"event:{c.replace('_ev_', '')}" for c in event_cols]
    names += [f"cat:{c}" for c in cats]
    return names


def _shape_matrix(
    df: pl.DataFrame,
    suffix: str,
    event_cols: list[str],
    cats: list[str],
    training: bool,
):
    base = df.get_column(f"base_{suffix}").to_numpy().astype(np.float64, copy=False)
    n = len(base)
    names = _feature_names(event_cols, cats)
    if n == 0:
        return np.empty((0, len(names))), np.empty(0), np.empty(0), names

    total_cols = [f"base_{suffix}"] + [f"lag{lag}_{suffix}" for lag in LAGS]
    if training:
        total_cols.append(f"actual_{suffix}")
    totals = _sum_over_sku(df, total_cols)
    base_total = totals[f"base_{suffix}"]
    base_share = _shape_share(base, base_total)
    cols: list[np.ndarray] = [
        np.log1p(np.maximum(base, 0.0)),
        np.log1p(np.maximum(base_total, 0.0)),
        _safe_log(base_share),
    ]

    lag_values: dict[int, np.ndarray] = {}
    lag_totals: dict[int, np.ndarray] = {}
    for lag in LAGS:
        v = df.get_column(f"lag{lag}_{suffix}").to_numpy().astype(np.float64, copy=False)
        lag_values[lag] = v
        lag_totals[lag] = totals[f"lag{lag}_{suffix}"]
        cols.append(np.log1p(np.maximum(v, 0.0)))
    for lag in LAGS:
        cols.append(_safe_log(_shape_share(lag_values[lag], lag_totals[lag])))

    cols.extend(
        [
            df.get_column(f"recent_state_{suffix}").to_numpy().astype(np.float64, copy=False),
            df.get_column(f"trend_state_{suffix}").to_numpy().astype(np.float64, copy=False),
            df.get_column("_price_state").to_numpy().astype(np.float64, copy=False),
        ]
    )
    h = df.get_column("_h").to_numpy().astype(np.float64, copy=False)
    dow = df.get_column("_dow").to_numpy().astype(np.float64, copy=False)
    mon = df.get_column("_mon").to_numpy().astype(np.float64, copy=False)
    cols.extend(
        [
            np.sin(2 * np.pi * h / 28.0),
            np.cos(2 * np.pi * h / 28.0),
            np.sin(2 * np.pi * (dow - 1.0) / 7.0),
            np.cos(2 * np.pi * (dow - 1.0) / 7.0),
            np.sin(2 * np.pi * (mon - 1.0) / 12.0),
            np.cos(2 * np.pi * (mon - 1.0) / 12.0),
        ]
    )
    for ev in event_cols:
        if ev in df.columns:
            cols.append(df.get_column(ev).to_numpy().astype(np.float64, copy=False))
        else:
            cols.append(np.zeros(n, dtype=np.float64))

    cat_arr = np.array(
        [str(x) if x is not None else "" for x in df.get_column("_cat").to_list()],
        dtype=object,
    )
    for c in cats:
        cols.append((cat_arr == c).astype(np.float64))

    X = np.column_stack(cols).astype(np.float32, copy=False)
    X[~np.isfinite(X)] = 0.0
    if not training:
        return X, np.empty(0), np.empty(0), names

    actual = df.get_column(f"actual_{suffix}").to_numpy().astype(np.float64, copy=False)
    actual_total = totals[f"actual_{suffix}"]
    actual_share = _shape_share(actual, actual_total)
    z_clip = float(getattr(settings, "V12_SKU_SHAPE_LGBM_Z_CLIP", 1.5))
    z = np.clip(_safe_log(actual_share) - _safe_log(base_share), -z_clip, z_clip)

    # Match the official positive-day error support while volume-weighting the
    # pooled fit.  The target block is always older/closed when training=True.
    mask = (
        np.isfinite(actual)
        & (actual > 0)
        & np.isfinite(z)
        & (base_total > EPS)
        & (actual_total > EPS)
    )
    w = np.clip(np.sqrt(np.maximum(actual[mask], 0.0)), 0.1, 1000.0).astype(np.float32)
    return X[mask], z[mask].astype(np.float32), w, names


def _fit_model(
    blocks: list[pl.DataFrame],
    suffix: str,
    event_cols: list[str],
    n_jobs: int,
    seed: int,
):
    lgb = _require_lightgbm()
    cats = _cat_values(blocks)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    names: list[str] | None = None
    for b in blocks:
        X, y, w, nm = _shape_matrix(b, suffix, event_cols, cats, True)
        if X.shape[0]:
            xs.append(X)
            ys.append(y)
            ws.append(w)
            names = nm
    if not xs:
        return None, cats, names or _feature_names(event_cols, cats), 0

    X = np.vstack(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    train_set = lgb.Dataset(X, label=y, weight=w, free_raw_data=True)
    params = {
        "objective": "huber",
        "learning_rate": float(getattr(settings, "V12_SKU_SHAPE_LGBM_LEARNING_RATE", 0.035)),
        "num_leaves": int(getattr(settings, "V12_SKU_SHAPE_LGBM_NUM_LEAVES", 15)),
        "max_depth": int(getattr(settings, "V12_SKU_SHAPE_LGBM_MAX_DEPTH", 4)),
        "min_data_in_leaf": int(getattr(settings, "V12_SKU_SHAPE_LGBM_MIN_DATA_IN_LEAF", 500)),
        "bagging_fraction": float(getattr(settings, "V12_SKU_SHAPE_LGBM_BAGGING_FRACTION", 0.85)),
        "bagging_freq": 1,
        "feature_fraction": float(getattr(settings, "V12_SKU_SHAPE_LGBM_FEATURE_FRACTION", 0.85)),
        "lambda_l1": float(getattr(settings, "V12_SKU_SHAPE_LGBM_L1", 0.20)),
        "lambda_l2": float(getattr(settings, "V12_SKU_SHAPE_LGBM_L2", 8.0)),
        "max_bin": int(getattr(settings, "V12_SKU_SHAPE_LGBM_MAX_BIN", 127)),
        "seed": int(seed),
        "feature_fraction_seed": int(seed) + 1,
        "bagging_seed": int(seed) + 2,
        "data_random_seed": int(seed) + 3,
        "num_threads": max(1, int(n_jobs)),
        "verbosity": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=int(getattr(settings, "V12_SKU_SHAPE_LGBM_ROUNDS", 160)),
    )
    return model, cats, names or [], len(y)


def _predict_shape(
    target: pl.DataFrame,
    suffix: str,
    event_cols: list[str],
    fit,
) -> tuple[np.ndarray, list[str], list[tuple[str, float]]]:
    model, cats, names, _ = fit
    base = target.get_column(f"base_{suffix}").to_numpy().astype(np.float64, copy=False)
    if model is None or target.height == 0:
        return base.copy(), names, []

    X, _, _, _ = _shape_matrix(target, suffix, event_cols, cats, False)
    z = np.asarray(model.predict(X), dtype=np.float64)
    z_clip = float(getattr(settings, "V12_SKU_SHAPE_LGBM_Z_CLIP", 1.5))
    z = np.clip(z, -z_clip, z_clip)
    gamma = float(getattr(settings, "V12_SKU_SHAPE_LGBM_GAMMA", 1.0))
    base_total = _sum_over_sku(target, [f"base_{suffix}"])[f"base_{suffix}"]
    smooth_frac = float(getattr(settings, "V12_SKU_SHAPE_LGBM_SMOOTH_FRAC", 0.02))
    pc = np.maximum(1e-9, smooth_frac * np.maximum(base_total, 0.0) / 28.0)
    raw = (np.maximum(base, 0.0) + pc) * np.exp(gamma * z)
    raw_sum = (
        pl.DataFrame({"_v12_sku": target.get_column("_v12_sku"), "raw": raw})
        .with_columns(pl.col("raw").sum().over("_v12_sku").alias("raw_sum"))
        .get_column("raw_sum")
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    fc = np.where(raw_sum > EPS, raw * base_total / np.maximum(raw_sum, EPS), base)
    fc = np.maximum(fc, 0.0)

    gains = np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float64)
    importance: list[tuple[str, float]] = []
    if gains.size and gains.sum() > 0:
        gains = gains / gains.sum()
        idx = np.argsort(gains)[::-1][:8]
        importance = [(names[i], float(gains[i])) for i in idx if i < len(names)]
    return fc, names, importance


class PooledLGBMShapeContext:
    """Precompute causal SKU blocks and return v12.6 corrected target paths."""

    def __init__(
        self,
        *,
        sku_daily_all: pl.DataFrame,
        sku_ids: pl.DataFrame,
        section_id: str,
        target_origins: list[dt.date],
        block_days: int,
        history_blocks: int,
        n_jobs: int,
        base_forecast_builder: Callable[[dt.date, dt.date], pl.DataFrame],
    ) -> None:
        self.sku_daily = sku_daily_all
        self.sku_ids = sku_ids.select("_v12_sku").unique()
        self.section_id = str(section_id)
        self.block_days = int(block_days)
        self.history_blocks = int(history_blocks)
        self.n_jobs = max(1, int(n_jobs))
        self.base_builder = base_forecast_builder
        self.meta = _category_meta(self.section_id, self.sku_ids)
        self._feature_blocks: dict[dt.date, pl.DataFrame] = {}
        self._base_forecasts: dict[dt.date, pl.DataFrame] = {}
        self._corrected: dict[dt.date, pl.DataFrame] = {}

        targets = sorted(set(target_origins))
        needed: set[dt.date] = set(targets)
        for origin in targets:
            for i in range(1, self.history_blocks + 1):
                needed.add(origin - dt.timedelta(days=self.block_days * i))
        lo = min(needed)
        hi = max(needed) + dt.timedelta(days=self.block_days - 1)
        self.calendar, self.event_cols = _calendar_table(lo, hi)

        logger.info(
            "v12.6 LightGBM shape: precomputando %d bloques SKU para sec=%s (history=%d)",
            len(needed), self.section_id, self.history_blocks,
        )
        for origin in sorted(needed):
            end = origin + dt.timedelta(days=self.block_days - 1)
            base = self.base_builder(origin, end)
            self._base_forecasts[origin] = base
            actual = self.sku_daily.filter(
                (pl.col("ds") >= pl.lit(origin)) & (pl.col("ds") <= pl.lit(end))
            ).select(
                "_v12_sku",
                "ds",
                pl.col("_sku_day_y").cast(pl.Float64).alias("actual_y"),
                pl.col("_sku_day_v").cast(pl.Float64).alias("actual_v"),
            )
            block = (
                base.select(
                    "_v12_sku",
                    "ds",
                    pl.col("v12_sku_forecast_y").cast(pl.Float64).fill_null(0.0).alias("base_y"),
                    pl.col("v12_sku_forecast_value").cast(pl.Float64).fill_null(0.0).alias("base_v"),
                )
                .join(actual, on=["_v12_sku", "ds"], how="left")
                .join(self.meta, on="_v12_sku", how="left")
                .with_columns(
                    pl.col("actual_y").fill_null(0.0),
                    pl.col("actual_v").fill_null(0.0),
                    pl.col("_cat").fill_null("__UNKNOWN__"),
                )
            )
            self._feature_blocks[origin] = _attach_features(
                block, self.sku_daily, origin, end, self.calendar
            )

    def corrected_forecast(self, origin: dt.date) -> pl.DataFrame:
        if origin in self._corrected:
            return self._corrected[origin]
        base = self._base_forecasts[origin]
        target = self._feature_blocks[origin]
        hist_origins = [
            origin - dt.timedelta(days=self.block_days * i)
            for i in range(1, self.history_blocks + 1)
        ]
        hist = [self._feature_blocks[d] for d in hist_origins if d in self._feature_blocks]
        if not hist:
            self._corrected[origin] = base
            return base

        # Base forecasts expose default v12.6 metadata for compatibility when
        # the layer is disabled. Drop those placeholders before replacing the
        # production path with the corrected shape.
        drop_cols = [
            c for c in (
                "v12_sku_forecast_base_y", "v12_sku_forecast_base_value",
                "v12_sku_shape_model_y", "v12_sku_shape_model_value",
                "v12_sku_shape_training_rows_y", "v12_sku_shape_training_rows_value",
                "v12_sku_shape_applied_y", "v12_sku_shape_applied_value",
            ) if c in base.columns
        ]
        out = base.drop(drop_cols).rename(
            {
                "v12_sku_forecast_y": "v12_sku_forecast_base_y",
                "v12_sku_forecast_value": "v12_sku_forecast_base_value",
            }
        )
        for suffix, target_col, base_col, model_col, rows_col, applied_col in (
            (
                "y",
                "v12_sku_forecast_y",
                "v12_sku_forecast_base_y",
                "v12_sku_shape_model_y",
                "v12_sku_shape_training_rows_y",
                "v12_sku_shape_applied_y",
            ),
            (
                "v",
                "v12_sku_forecast_value",
                "v12_sku_forecast_base_value",
                "v12_sku_shape_model_value",
                "v12_sku_shape_training_rows_value",
                "v12_sku_shape_applied_value",
            ),
        ):
            seed = int(getattr(settings, "V12_SKU_SHAPE_LGBM_SEED", 20260831)) + (0 if suffix == "y" else 1000)
            fit = _fit_model(hist, suffix, self.event_cols, self.n_jobs, seed)
            fc, _, importance = _predict_shape(target, suffix, self.event_cols, fit)
            train_rows = int(fit[3])
            model = fit[0]
            applied = model is not None and train_rows > 0
            pred = target.select("_v12_sku", "ds").with_columns(pl.Series(target_col, fc))
            out = out.join(pred, on=["_v12_sku", "ds"], how="left").with_columns(
                pl.col(target_col).fill_null(pl.col(base_col)).clip(lower_bound=0.0),
                pl.lit("lgbm_shape_g1p0" if applied else "base").alias(model_col),
                pl.lit(train_rows).cast(pl.Int32).alias(rows_col),
                pl.lit(bool(applied)).alias(applied_col),
            )
            if importance:
                logger.info(
                    "v12.6 LGBM shape sec=%s origin=%s target=%s train=%d top_gain=%s",
                    self.section_id,
                    origin,
                    "U" if suffix == "y" else "V",
                    train_rows,
                    ", ".join(f"{n}:{100.0*g:.1f}%" for n, g in importance[:5]),
                )

            # MEMORY-SAFE v12.9.12: LightGBM Booster can retain its training
            # Dataset even after prediction.  Nothing downstream needs the
            # fitted model once `pred` has been joined into `out`, so release
            # native Dataset buffers before the next target/model and before
            # the final candidate/invariant allocations.  Forecast algebra is
            # unchanged; this only shortens object lifetime.
            if model is not None:
                try:
                    model.free_dataset()
                except Exception:
                    # Older LightGBM builds may not expose/free it cleanly; the
                    # Python references below are still dropped deterministically.
                    pass
            # The per-SKU total invariant is checked while `fc` is still a
            # narrow NumPy vector.  Do not group the final wide `out` frame:
            # on Sec23/forecast-only that group_by requested ~20 MB at the
            # exact RAM peak after the final LightGBM fit.  The streaming
            # dictionary below needs only O(number of SKU) memory and checks
            # the same algebra before `fc` is discarded.
            tol = float(getattr(settings, "V12_SKU_SHAPE_LGBM_TOTAL_TOLERANCE", 1e-9))
            sums: dict[str, list[float]] = {}
            sku_series = target.get_column("_v12_sku")
            base_series = target.get_column(f"base_{suffix}")
            for sku, base_value, corrected_value in zip(sku_series, base_series, fc, strict=True):
                key = str(sku)
                pair = sums.get(key)
                if pair is None:
                    sums[key] = [float(base_value), float(corrected_value)]
                else:
                    pair[0] += float(base_value)
                    pair[1] += float(corrected_value)
            bad_count = 0
            bad_examples: list[tuple[str, float, float]] = []
            for sku, (base_sum, corrected_sum) in sums.items():
                if abs(base_sum - corrected_sum) > tol * max(1.0, abs(base_sum)):
                    bad_count += 1
                    if len(bad_examples) < 5:
                        bad_examples.append((sku, base_sum, corrected_sum))
            if bad_count:
                raise RuntimeError(
                    f"v12.6 shape invariant violated sec={self.section_id} origin={origin} "
                    f"target={suffix}: {bad_count} SKU no preservan total 28d; "
                    f"examples={bad_examples}"
                )
            del sums, sku_series, base_series, bad_examples
            del pred, fc, fit, model
            gc.collect()

        self._corrected[origin] = out
        return out

    def retain_for_targets(self, target_origins: list[dt.date]) -> None:
        """Prune feature caches that cannot be used by any remaining target.

        Every retained target keeps exactly itself plus its ``history_blocks``
        strictly older closed blocks.  Therefore this changes only object
        lifetime, never the training set or forecast algebra.
        """
        keep: set[dt.date] = set()
        target_set = set(target_origins)
        for origin in target_set:
            keep.add(origin)
            for i in range(1, self.history_blocks + 1):
                keep.add(origin - dt.timedelta(days=self.block_days * i))

        before = len(self._feature_blocks)
        for origin in list(self._feature_blocks):
            if origin not in keep:
                self._feature_blocks.pop(origin, None)
        # Base forecasts are needed only for the remaining target origins;
        # historical bases have already been copied into feature blocks.
        for origin in list(self._base_forecasts):
            if origin not in target_set:
                self._base_forecasts.pop(origin, None)
        dropped = before - len(self._feature_blocks)
        if dropped:
            logger.info(
                "v12.9.12 memory-safe: sec=%s feature blocks %d→%d; remaining targets=%s",
                self.section_id, before, len(self._feature_blocks),
                ",".join(str(x) for x in sorted(target_set)),
            )
        gc.collect()

    def release_origin(self, origin: dt.date, *, drop_base: bool = True) -> None:
        """Release target-only caches after its downstream candidate is materialized.

        Feature blocks are intentionally retained until the whole target set has
        been scored because later origins may still need them as closed-history
        training data.  Corrected/base forecasts, however, are target-local and
        otherwise accumulate one full DataFrame per origin.
        """
        self._corrected.pop(origin, None)
        if drop_base:
            self._base_forecasts.pop(origin, None)
        # Force prompt release between closed-block candidates.  This matters
        # on Windows where large Polars/LightGBM buffers can otherwise overlap
        # with the next origin's allocations until cyclic GC runs.
        gc.collect()

    def close(self) -> None:
        """Drop all cached frames once OOS and forecast-only candidates exist."""
        self._corrected.clear()
        self._base_forecasts.clear()
        self._feature_blocks.clear()
        # Static frames are no longer needed after the caller has materialized
        # its candidates.  Replace them with tiny schema-compatible frames so a
        # lingering closure cannot keep the large history alive.
        self.sku_daily = pl.DataFrame(schema=self.sku_daily.schema)
        self.sku_ids = pl.DataFrame(schema=self.sku_ids.schema)
        self.meta = pl.DataFrame(schema=self.meta.schema)
        self.calendar = pl.DataFrame(schema=self.calendar.schema)
