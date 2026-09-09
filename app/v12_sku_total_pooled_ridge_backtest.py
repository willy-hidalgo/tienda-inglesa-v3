"""v12.5 read-only rolling backtest: pooled supervised SKU-total residual ridge.

Purpose
-------
After the current heuristic candidate set, factorized AR variants, explicit
calendar factors and price-state corrections failed to generalize, this module
moves to a single pooled supervised model at SKU-total level while preserving
interpretability and causal 28-day evaluation.

The model predicts the residual on top of the CURRENT v12.5 SKU-total forecast:

    z = log1p(actual) - log1p(current_forecast)

Features are all known before the target day/block:
* current forecast level;
* exact causal lags 28/56/84/364/365 days;
* recent 28d-vs-84d level state and recent block trend;
* weekday/month/horizon position;
* configured holiday/event flags;
* causal price-state (ASP 28d / ASP 84d);
* causal SKU/category residual priors learned only from older closed blocks.

A weighted ridge model is fitted ONCE per section/target/pseudo-OOS (not per SKU
and never per SKU×store).  Four relative ridge strengths are evaluated.  From
the same prediction we expose three interpretable modes:

* ridge_full  : model may change 28d level and daily shape;
* ridge_level : only the model-implied 28d level correction is used;
* ridge_shape : only daily shape correction is used; current 28d total is kept.

All PRE-OOS evaluations are causal.  Production OOS is used only once as final
holdout after selecting a robust candidate from PRE-OOS rolling history.
Store occurrence and the v12.5 28d×84d store-share are kept fixed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings
from app.forecasting.leaf_v12 import _block_candidate
from app.v12_sku_total_calendar_driver_backtest import (
    EPS,
    _baseline_block,
    _calendar_table,
    _eval_leaf,
    _eval_sku,
    _history_from_selected,
    _incumbent_rows,
    _oos_bounds,
    _pct,
    _pp,
)

RIDGE_REL = (1e-4, 1e-3, 1e-2, 1e-1)
MODES = ("full", "level", "shape")
LAGS = (28, 56, 84, 364, 365)


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x.astype(np.float64, copy=False), 0.0, None))


def _lag_table(sku_daily: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    ids = sku_daily.select("_v12_sku").unique()
    days = pl.DataFrame({"ds": pl.date_range(start, end, interval="1d", eager=True)}).with_columns(
        pl.col("ds").cast(pl.Date)
    )
    grid = ids.join(days, how="cross")
    out = grid
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
            .log().clip(-2.0, 2.0).alias(f"recent_state_{s}"),
            ((pl.col(f"_r28_{s}") + 1.0) / (pl.col(f"_p28_{s}") + 1.0))
            .log().clip(-2.0, 2.0).alias(f"trend_state_{s}"),
        )
    return base.select(
        "_v12_sku", "recent_state_y", "trend_state_y", "recent_state_v", "trend_state_v"
    )


def _attach_causal_features(block: pl.DataFrame, sku_daily: pl.DataFrame, start: dt.date, end: dt.date) -> pl.DataFrame:
    lags = _lag_table(sku_daily, start, end)
    states = _state_table(sku_daily, start)
    return (
        block.join(lags, on=["_v12_sku", "ds"], how="left")
        .join(states, on="_v12_sku", how="left")
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


def _prior_tables(hist: pl.DataFrame, suffix: str):
    actual = f"actual_{suffix}"
    base = f"base_{suffix}"
    z = (
        hist.filter(pl.col(actual) > 0)
        .with_columns(
            ((pl.col(actual) + 1.0).log() - (pl.col(base).clip(lower_bound=0.0) + 1.0).log())
            .clip(-2.0, 2.0).alias("_z"),
            pl.col(actual).sqrt().clip(0.1, 1000.0).alias("_w"),
        )
    )
    if z.height == 0:
        sku = hist.select("_v12_sku").unique().with_columns(pl.lit(0.0).alias("sku_prior"), pl.lit(0.0).alias("sku_hist"))
        cat = hist.select("_cat").unique().with_columns(pl.lit(0.0).alias("cat_prior"), pl.lit(0.0).alias("cat_hist"))
        return sku, cat, 0.0

    sec_row = z.select(
        (pl.col("_z") * pl.col("_w")).sum().alias("num"), pl.col("_w").sum().alias("den")
    ).row(0, named=True)
    sec_prior = float(sec_row["num"] or 0.0) / max(EPS, float(sec_row["den"] or 0.0))
    sec_prior = min(1.5, max(-1.5, sec_prior))

    cat = (
        z.group_by("_cat")
        .agg(
            (pl.col("_z") * pl.col("_w")).sum().alias("num"),
            pl.col("_w").sum().alias("den"),
            pl.len().cast(pl.Float64).alias("n"),
        )
        .with_columns((pl.col("num") / pl.col("den").clip(lower_bound=EPS)).clip(-1.5, 1.5).alias("raw"))
        .with_columns((pl.col("n") / (pl.col("n") + 112.0)).clip(0.0, 1.0).alias("shr"))
        .with_columns((pl.lit(sec_prior) * (1.0 - pl.col("shr")) + pl.col("raw") * pl.col("shr")).alias("cat_prior"))
        .select("_cat", "cat_prior", pl.col("n").alias("cat_hist"))
    )
    sku = (
        z.group_by(["_v12_sku", "_cat"])
        .agg(
            (pl.col("_z") * pl.col("_w")).sum().alias("num"),
            pl.col("_w").sum().alias("den"),
            pl.len().cast(pl.Float64).alias("n"),
        )
        .with_columns((pl.col("num") / pl.col("den").clip(lower_bound=EPS)).clip(-1.5, 1.5).alias("raw"))
        .join(cat.select("_cat", "cat_prior"), on="_cat", how="left")
        .with_columns(
            pl.col("cat_prior").fill_null(sec_prior),
            (pl.col("n") / (pl.col("n") + 56.0)).clip(0.0, 1.0).alias("shr"),
        )
        .with_columns((pl.col("cat_prior") * (1.0 - pl.col("shr")) + pl.col("raw") * pl.col("shr")).alias("sku_prior"))
        .select("_v12_sku", "sku_prior", pl.col("n").alias("sku_hist"))
    )
    return sku, cat, sec_prior


def _with_priors(df: pl.DataFrame, sku_prior: pl.DataFrame, cat_prior: pl.DataFrame, sec_prior: float) -> pl.DataFrame:
    return (
        df.join(sku_prior, on="_v12_sku", how="left")
        .join(cat_prior, on="_cat", how="left")
        .with_columns(
            pl.col("sku_prior").fill_null(sec_prior),
            pl.col("cat_prior").fill_null(sec_prior),
            pl.col("sku_hist").fill_null(0.0),
            pl.col("cat_hist").fill_null(0.0),
        )
    )


def _feature_names(event_cols: list[str]) -> list[str]:
    names = [
        "log_base", "log_base_total",
        "log_lag28", "log_lag56", "log_lag84", "log_lag364", "log_lag365",
        "recent_state", "trend_state", "price_state",
        "sku_prior", "cat_prior", "log_sku_hist", "log_cat_hist",
        "h_sin", "h_cos", "dow_sin", "dow_cos", "mon_sin", "mon_cos",
    ]
    names += [f"event:{c.replace('_ev_', '')}" for c in event_cols]
    return names


def _matrix(df: pl.DataFrame, suffix: str, event_cols: list[str], training: bool):
    base = df.get_column(f"base_{suffix}").to_numpy().astype(np.float64, copy=False)
    n = len(base)
    if n == 0:
        return np.empty((0, len(_feature_names(event_cols)))), np.empty(0), np.empty(0)

    sku_total = df.with_columns(pl.col(f"base_{suffix}").sum().over("_v12_sku").alias("_bt")).get_column("_bt").to_numpy()
    cols = [
        _safe_log1p(base),
        _safe_log1p(sku_total),
    ]
    for lag in LAGS:
        cols.append(_safe_log1p(df.get_column(f"lag{lag}_{suffix}").to_numpy()))
    cols.extend([
        df.get_column(f"recent_state_{suffix}").to_numpy().astype(np.float64, copy=False),
        df.get_column(f"trend_state_{suffix}").to_numpy().astype(np.float64, copy=False),
        df.get_column("_price_state").to_numpy().astype(np.float64, copy=False),
        df.get_column("sku_prior").to_numpy().astype(np.float64, copy=False),
        df.get_column("cat_prior").to_numpy().astype(np.float64, copy=False),
        np.log1p(df.get_column("sku_hist").to_numpy().astype(np.float64, copy=False)),
        np.log1p(df.get_column("cat_hist").to_numpy().astype(np.float64, copy=False)),
    ])
    h = df.get_column("_h").to_numpy().astype(np.float64, copy=False)
    dow = df.get_column("_dow").to_numpy().astype(np.float64, copy=False)
    mon = df.get_column("_mon").to_numpy().astype(np.float64, copy=False)
    cols.extend([
        np.sin(2*np.pi*h/28.0), np.cos(2*np.pi*h/28.0),
        np.sin(2*np.pi*(dow-1.0)/7.0), np.cos(2*np.pi*(dow-1.0)/7.0),
        np.sin(2*np.pi*(mon-1.0)/12.0), np.cos(2*np.pi*(mon-1.0)/12.0),
    ])
    for ev in event_cols:
        if ev in df.columns:
            cols.append(df.get_column(ev).to_numpy().astype(np.float64, copy=False))
        else:
            cols.append(np.zeros(n, dtype=np.float64))
    X = np.column_stack(cols).astype(np.float64, copy=False)
    X[~np.isfinite(X)] = 0.0

    if not training:
        return X, np.empty(0), np.empty(0)
    actual = df.get_column(f"actual_{suffix}").to_numpy().astype(np.float64, copy=False)
    mask = np.isfinite(actual) & (actual > 0) & np.isfinite(base)
    z = np.zeros(n, dtype=np.float64)
    z[mask] = np.clip(_safe_log1p(actual[mask]) - _safe_log1p(base[mask]), -2.0, 2.0)
    w = np.zeros(n, dtype=np.float64)
    w[mask] = np.clip(np.sqrt(actual[mask]), 0.1, 1000.0)
    return X[mask], z[mask], w[mask]


def _weighted_standardizer(blocks: list[pl.DataFrame], suffix: str, event_cols: list[str]):
    p = len(_feature_names(event_cols))
    sw = 0.0
    sx = np.zeros(p, dtype=np.float64)
    sx2 = np.zeros(p, dtype=np.float64)
    nobs = 0
    for b in blocks:
        X, _, w = _matrix(b, suffix, event_cols, True)
        if X.shape[0] == 0:
            continue
        sw += float(w.sum())
        sx += (X * w[:, None]).sum(axis=0)
        sx2 += ((X * X) * w[:, None]).sum(axis=0)
        nobs += X.shape[0]
    if sw <= EPS:
        return np.zeros(p), np.ones(p), 0.0, 0
    mean = sx / sw
    var = np.maximum(sx2 / sw - mean * mean, 1e-8)
    std = np.sqrt(var)
    std[std < 1e-4] = 1.0
    return mean, std, sw, nobs


def _fit_ridge(blocks: list[pl.DataFrame], suffix: str, event_cols: list[str], rel_lambdas=RIDGE_REL):
    names = _feature_names(event_cols)
    mean, std, sw, nobs = _weighted_standardizer(blocks, suffix, event_cols)
    p = len(names)
    xtx = np.zeros((p + 1, p + 1), dtype=np.float64)
    xty = np.zeros(p + 1, dtype=np.float64)
    if sw <= EPS:
        return {lam: np.zeros(p + 1) for lam in rel_lambdas}, mean, std, names, nobs

    for b in blocks:
        X, z, w = _matrix(b, suffix, event_cols, True)
        if X.shape[0] == 0:
            continue
        Xs = (X - mean) / std
        X1 = np.column_stack([np.ones(Xs.shape[0]), Xs])
        # Do not materialize sqrt-weighted copies twice.
        xtx += X1.T @ (X1 * w[:, None])
        xty += X1.T @ (z * w)

    out = {}
    for lam in rel_lambdas:
        reg = np.eye(p + 1, dtype=np.float64) * (float(lam) * sw)
        reg[0, 0] = 0.0  # intercept unpenalized
        try:
            beta = np.linalg.solve(xtx + reg + np.eye(p + 1) * 1e-10, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(xtx + reg + np.eye(p + 1) * 1e-8, xty, rcond=None)[0]
        out[lam] = beta
    return out, mean, std, names, nobs


def _predict_modes(df: pl.DataFrame, suffix: str, event_cols: list[str], fit):
    betas, mean, std, names, _ = fit
    X, _, _ = _matrix(df, suffix, event_cols, False)
    Xs = (X - mean) / std
    X1 = np.column_stack([np.ones(Xs.shape[0]), Xs])
    base = df.get_column(f"base_{suffix}").to_numpy().astype(np.float64, copy=False)
    sku = df.get_column("_v12_sku").to_list()

    out = df.select("_v12_sku", "ds", f"actual_{suffix}", f"base_{suffix}")
    for lam, beta in betas.items():
        pred_z = np.clip(X1 @ beta, -1.5, 1.5)
        raw = np.maximum(0.0, np.expm1(_safe_log1p(base) + pred_z))
        # Avoid explosive single-day forecasts from the linear extrapolation.
        raw = np.minimum(raw, np.maximum(5.0, base * 4.0 + 5.0))
        tmp = (
            df.select("_v12_sku")
            .with_columns(pl.Series("raw", raw), pl.Series("base", base))
            .with_columns(
                pl.col("raw").sum().over("_v12_sku").alias("raw_sum"),
                pl.col("base").sum().over("_v12_sku").alias("base_sum"),
            )
        )
        raw_sum = tmp.get_column("raw_sum").to_numpy()
        base_sum = tmp.get_column("base_sum").to_numpy()
        factor = np.where(base_sum > EPS, raw_sum / np.maximum(base_sum, EPS), 1.0)
        factor = np.clip(factor, 0.50, 2.00)
        level = np.maximum(0.0, base * factor)
        shape = np.where(raw_sum > EPS, raw * base_sum / np.maximum(raw_sum, EPS), base)
        shape = np.maximum(0.0, shape)
        tag = f"{lam:g}".replace(".", "p").replace("-", "m")
        out = out.with_columns(
            pl.Series(f"fc_ridge_full_l{tag}_{suffix}", raw),
            pl.Series(f"fc_ridge_level_l{tag}_{suffix}", level),
            pl.Series(f"fc_ridge_shape_l{tag}_{suffix}", shape),
        )
    return out, names


def _candidate_names():
    out = []
    for lam in RIDGE_REL:
        tag = f"{lam:g}".replace(".", "p").replace("-", "m")
        for mode in MODES:
            out.append(f"ridge_{mode}_l{tag}")
    return tuple(out)


CANDIDATES = _candidate_names()


def _summary(rows: list[dict], sec: str, suffix: str, metric: str):
    base = {r["eval_rank"]: r for r in rows if r["sec"] == sec and r["suffix"] == suffix and r["candidate"] == "current_selector"}
    out = []
    for cand in CANDIDATES:
        rs = [r for r in rows if r["sec"] == sec and r["suffix"] == suffix and r["candidate"] == cand]
        if not rs:
            continue
        ae = sum(float(r[f"{metric}_ae"]) for r in rs); den = sum(float(r[f"{metric}_den"]) for r in rs)
        bae = sum(float(base[r["eval_rank"]][f"{metric}_ae"]) for r in rs); bden = sum(float(base[r["eval_rank"]][f"{metric}_den"]) for r in rs)
        pooled = ae / max(EPS, den); bpooled = bae / max(EPS, bden)
        gains = [base[r["eval_rank"]][f"{metric}_wmape"] - r[f"{metric}_wmape"] for r in rs]
        recent = [g for r, g in zip(rs, gains) if r["eval_rank"] <= 4]
        out.append({
            "candidate": cand, "pooled": pooled, "gain": bpooled - pooled,
            "median": median(gains), "recent4": sum(recent)/len(recent) if recent else float("nan"),
            "win": sum(g > 0 for g in gains)/len(gains), "worst": min(gains),
        })
    return sorted(out, key=lambda r: (-r["gain"], r["pooled"]))


def _print_summary(title: str, rows: list[dict]):
    print(f"\n{title}")
    print("  candidate                    pooled    gain    median  recent4  win%   worst")
    for r in rows:
        print(f"  {r['candidate']:28s} {_pct(r['pooled'])} {_pp(r['gain'])} {_pp(r['median'])} {_pp(r['recent4'])} {100*r['win']:5.0f}% {_pp(r['worst'])}")


def _robust_pick(summary: list[dict]):
    eligible = [r for r in summary if r["gain"] > 0.0025 and r["recent4"] > 0 and r["win"] >= 0.58 and r["worst"] >= -0.02]
    return max(eligible, key=lambda r: (r["gain"], r["recent4"], r["win"])) if eligible else None


def _coef_report(fit, topn: int = 12):
    betas, _, _, names, nobs = fit
    # Report the middle regularization as a stable interpretability reference.
    lam = 1e-2 if 1e-2 in betas else list(betas)[0]
    beta = betas[lam][1:]
    idx = np.argsort(np.abs(beta))[::-1][:topn]
    return lam, nobs, [(names[i], float(beta[i])) for i in idx]


def _run_section(selected_path: Path, forecast_path: Path, sec: str, oos_origin: dt.date, blocks: int, history_max: int, active_min: int, out_dir: Path):
    prepared, sku_daily, sku_first, uid_first, ids, meta, cat_name = _history_from_selected(selected_path, sec)
    incumbent = _incumbent_rows(forecast_path, sec)
    block_days = int(getattr(settings, "RLS_BLOCK_DAYS", 28))
    max_block = blocks + history_max
    all_lo = oos_origin - dt.timedelta(days=block_days * max_block + 370)
    all_hi = oos_origin + dt.timedelta(days=block_days - 1)
    _, event_cols = _calendar_table(all_lo, all_hi)

    print(f"\n=== SECCIÓN {sec} | pooled ridge rolling PRE-OOS={blocks} | history={history_max} | categoria={cat_name} ===")
    print(f"features={20+len(event_cols)} | event drivers={len(event_cols)} | ridge_rel={RIDGE_REL}")
    print(f"precomputando {max_block} bloques current SKU-total + causal lags ...")
    daily_by_id: dict[int, pl.DataFrame] = {}
    for bid in range(1, max_block + 1):
        start = oos_origin - dt.timedelta(days=block_days * bid)
        end = start + dt.timedelta(days=block_days - 1)
        b = _baseline_block(prepared, sku_daily, sku_first, ids, meta, incumbent, start, end, bid)
        daily_by_id[bid] = _attach_causal_features(b, sku_daily, start, end)
        print(f"  block {bid:02d}/{max_block}: {start}→{end} SKU={daily_by_id[bid].select('_v12_sku').n_unique()}")

    rows: list[dict] = []
    for eval_rank in range(blocks, 0, -1):
        start = oos_origin - dt.timedelta(days=block_days * eval_rank)
        end = start + dt.timedelta(days=block_days - 1)
        target0 = daily_by_id[eval_rank]
        hist_ids = list(range(eval_rank + 1, min(max_block, eval_rank + history_max) + 1))
        hist0 = pl.concat([daily_by_id[h] for h in hist_ids], how="vertical_relaxed")
        share_rows = _block_candidate(prepared, sku_daily, sku_first, uid_first, ids, incumbent, start, end).select(
            "unique_id", "_v12_sku", "ds", "v12_store_share_y", "v12_store_share_value"
        )

        for suffix in ("y", "v"):
            sku_prior, cat_prior, sec_prior = _prior_tables(hist0, suffix)
            hist_blocks = [_with_priors(daily_by_id[h], sku_prior, cat_prior, sec_prior) for h in hist_ids]
            target = _with_priors(target0, sku_prior, cat_prior, sec_prior)
            fit = _fit_ridge(hist_blocks, suffix, event_cols)
            pred, _ = _predict_modes(target, suffix, event_cols, fit)
            bsku = _eval_sku(target, f"base_{suffix}", suffix)
            bleaf = _eval_leaf(share_rows, target, f"base_{suffix}", prepared, start, end, suffix, active_min)
            rows.append({
                "sec":sec,"eval_rank":eval_rank,"start":start,"end":end,"suffix":suffix,"candidate":"current_selector",
                "sku_wmape":bsku[0],"sku_bias":bsku[1],"sku_ae":bsku[2],"sku_den":bsku[3],
                "leaf_wmape":bleaf[0],"leaf_bias":bleaf[1],"leaf_ae":bleaf[2],"leaf_den":bleaf[3],"active":bleaf[4],
            })
            for cand in CANDIDATES:
                fc = f"fc_{cand}_{suffix}"
                csku = _eval_sku(pred, fc, suffix)
                cleaf = _eval_leaf(share_rows, pred, fc, prepared, start, end, suffix, active_min)
                rows.append({
                    "sec":sec,"eval_rank":eval_rank,"start":start,"end":end,"suffix":suffix,"candidate":cand,
                    "sku_wmape":csku[0],"sku_bias":csku[1],"sku_ae":csku[2],"sku_den":csku[3],
                    "leaf_wmape":cleaf[0],"leaf_bias":cleaf[1],"leaf_ae":cleaf[2],"leaf_den":cleaf[3],"active":cleaf[4],
                })
        by = next(r for r in rows if r["sec"]==sec and r["eval_rank"]==eval_rank and r["suffix"]=="y" and r["candidate"]=="current_selector")
        bv = next(r for r in rows if r["sec"]==sec and r["eval_rank"]==eval_rank and r["suffix"]=="v" and r["candidate"]=="current_selector")
        print(f"  pseudo-OOS {start}→{end} | current leaf U/V={_pct(by['leaf_wmape'])}/{_pct(bv['leaf_wmape'])}")

    pl.DataFrame(rows).write_csv(out_dir / f"pooled_ridge_rolling_sec_{sec}.csv")
    picks = {}
    print(f"\n--- RESUMEN PRE-OOS sec={sec} ---")
    for suffix, label in (("y","Unidades"),("v","Valor ($)")):
        sku_sum = _summary(rows, sec, suffix, "sku")
        leaf_sum = _summary(rows, sec, suffix, "leaf")
        _print_summary(f"SKU-TOTAL | sec={sec} | {label}", sku_sum[:12])
        _print_summary(f"LEAF share v12.5 fijo | sec={sec} | {label}", leaf_sum[:12])
        pick = _robust_pick(leaf_sum)
        picks[suffix] = pick
        if pick:
            print(f"  ==> PRE-OOS winner causal: {pick['candidate']} | gain={_pp(pick['gain'])} recent4={_pp(pick['recent4'])} win={100*pick['win']:.0f}% worst={_pp(pick['worst'])}")
        else:
            print("  ==> sin pooled-ridge challenger que cumpla guardas robustas")

    # Final production OOS holdout.
    oos_end = oos_origin + dt.timedelta(days=block_days - 1)
    oos0 = _attach_causal_features(_baseline_block(prepared, sku_daily, sku_first, ids, meta, incumbent, oos_origin, oos_end, 0), sku_daily, oos_origin, oos_end)
    hist_ids = list(range(1, min(history_max, max_block)+1))
    hist0 = pl.concat([daily_by_id[h] for h in hist_ids], how="vertical_relaxed")
    oos_share = _block_candidate(prepared, sku_daily, sku_first, uid_first, ids, incumbent, oos_origin, oos_end).select(
        "unique_id", "_v12_sku", "ds", "v12_store_share_y", "v12_store_share_value"
    )
    holdout_rows = []
    print(f"\n=== HOLDOUT OOS PRODUCCIÓN sec={sec} | {oos_origin}→{oos_end} ===")
    for suffix, label in (("y","Unidades"),("v","Valor ($)")):
        sku_prior, cat_prior, sec_prior = _prior_tables(hist0, suffix)
        hist_blocks = [_with_priors(daily_by_id[h], sku_prior, cat_prior, sec_prior) for h in hist_ids]
        target = _with_priors(oos0, sku_prior, cat_prior, sec_prior)
        fit = _fit_ridge(hist_blocks, suffix, event_cols)
        pred, _ = _predict_modes(target, suffix, event_cols, fit)
        bsku = _eval_sku(target, f"base_{suffix}", suffix)
        bleaf = _eval_leaf(oos_share, target, f"base_{suffix}", prepared, oos_origin, oos_end, suffix, active_min)
        pick = picks[suffix]
        lam_ref, nobs, topcoef = _coef_report(fit)
        print(f"  {label:10s}: train rows={nobs:,} | top coef std (lambda={lam_ref:g}): " + ", ".join(f"{n}={v:+.3f}" for n,v in topcoef[:6]))
        if pick is None:
            print(f"  {label:10s}: current SKU={_pct(bsku[0])} LEAF={_pct(bleaf[0])} | sin challenger causal")
            holdout_rows.append({"sec":sec,"suffix":suffix,"candidate":"current_selector","sku_wmape":bsku[0],"sku_bias":bsku[1],"leaf_wmape":bleaf[0],"leaf_bias":bleaf[1],"gain_leaf":0.0})
            continue
        cand = pick["candidate"]; fc=f"fc_{cand}_{suffix}"
        csku=_eval_sku(pred,fc,suffix); cleaf=_eval_leaf(oos_share,pred,fc,prepared,oos_origin,oos_end,suffix,active_min)
        gain=bleaf[0]-cleaf[0]
        print(f"  {label:10s}: PRE-OOS pick={cand:24s} | current LEAF={_pct(bleaf[0])} → candidate={_pct(cleaf[0])} gain={_pp(gain)} | BIAS {_pct(bleaf[1])}→{_pct(cleaf[1])}")
        holdout_rows.append({"sec":sec,"suffix":suffix,"candidate":cand,"sku_wmape":csku[0],"sku_bias":csku[1],"leaf_wmape":cleaf[0],"leaf_bias":cleaf[1],"gain_leaf":gain,"baseline_sku_wmape":bsku[0],"baseline_leaf_wmape":bleaf[0],"baseline_leaf_bias":bleaf[1]})
    pl.DataFrame(holdout_rows).write_csv(out_dir / f"pooled_ridge_holdout_sec_{sec}.csv")
    return rows, holdout_rows


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--blocks",type=int,default=12)
    ap.add_argument("--history-max",type=int,default=16)
    ap.add_argument("--active-min",type=int,default=int(getattr(settings,"OOS_ACTIVE_MIN_NONZERO_DAYS",7)))
    args=ap.parse_args()
    if args.blocks < 8: raise ValueError("--blocks debe ser >=8")
    if args.history_max < 13: raise ValueError("--history-max debe ser >=13")

    selected_path=Path(settings.SELECTED_PATH)
    forecast_path=Path(getattr(settings,"FORECAST_PATH",Path(getattr(settings,"OUT_DIR","data/output"))/"forecast.parquet"))
    if not selected_path.exists(): raise FileNotFoundError(selected_path)
    if not forecast_path.exists(): raise FileNotFoundError(forecast_path)
    out_dir=Path(getattr(settings,"OUT_DIR","data/output"))/"diagnostics"/"v12_sku_total_pooled_ridge"
    out_dir.mkdir(parents=True,exist_ok=True)

    print("DIAGNÓSTICO v12.5 — POOLED SUPERVISED RIDGE SKU-TOTAL")
    print(f"Selected: {selected_path}")
    print(f"Forecast: {forecast_path}")
    print(f"Salida  : {out_dir}")
    print("Read-only. Un modelo pooled por sección/target; nunca SKU×store. OOS solo holdout final.")
    print(f"blocks={args.blocks} | history_max={args.history_max} | ridge_rel={RIDGE_REL} | modes={MODES}")
    print("Target residual: log1p(actual)-log1p(current SKU forecast). Store-share/occurrence v12.5 quedan fijos.")

    bounds=_oos_bounds(forecast_path)
    all_rows=[]; holdouts=[]
    for sec in sorted(bounds):
        origin,_=bounds[sec]
        r,h=_run_section(selected_path,forecast_path,sec,origin,int(args.blocks),int(args.history_max),int(args.active_min),out_dir)
        all_rows.extend(r); holdouts.extend(h)
    if all_rows: pl.DataFrame(all_rows).write_csv(out_dir/"pooled_ridge_rolling_all.csv")
    if holdouts: pl.DataFrame(holdouts).write_csv(out_dir/"pooled_ridge_holdout_all.csv")

    print("\n=== DECISIÓN HOLDOUT ===")
    good=[r for r in holdouts if r.get("candidate")!="current_selector" and float(r.get("gain_leaf",0.0))>0]
    for r in holdouts:
        print(f"  sec={r['sec']} {'U' if r['suffix']=='y' else 'V'} | {r['candidate']} | holdout leaf={_pct(float(r['leaf_wmape']))} gain={_pp(float(r.get('gain_leaf',0.0)))}")
    if holdouts and len(good)==len(holdouts):
        print("- 4/4 picks PRE-OOS mejoran OOS: pooled supervised es candidato real para v12.6.")
    else:
        print(f"- Mejoran {len(good)}/{len(holdouts)} paneles OOS. No promover globalmente con regresiones.")
    print("- ridge_level gana => el residual supervisado corrige principalmente NIVEL 28d.")
    print("- ridge_shape gana => corrige principalmente FORMA diaria manteniendo total 28d.")
    print("- ridge_full gana => hacen falta ambas componentes de forma conjunta.")
    print("- Si tampoco generaliza, el límite ya no es la regla lineal: siguiente candidato debe ser no-lineal pooled (LightGBM/GBDT) con las mismas features y validación causal.")
    print("- OOS nunca participa en entrenamiento, regularización ni selección PRE-OOS.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
