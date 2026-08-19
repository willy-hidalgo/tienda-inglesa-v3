"""Polars adapter for causal SKU-store OOS optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from .leaf_models import causal_candidates, recursive_forecast, select_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LeafOptimizationConfig:
    validation_days: int = 28
    min_scored_points: int = 7
    bias_threshold: float = 0.05
    bias_clip: tuple[float, float] = (0.80, 1.20)


def _store_from_uid(uid: str) -> str:
    for part in uid.split("||"):
        if part.startswith("T:"):
            return part[2:]
    return ""


def _stable_factor(
    records: list[dict], key: str | None, cfg: LeafOptimizationConfig
) -> dict[str, float] | float:
    if not records:
        return {} if key else 1.0
    df = pl.DataFrame(records)
    groups = [(None, df)] if key is None else list(df.partition_by(key, as_dict=False))
    out: dict[str, float] = {}
    for group in groups:
        g = group[1] if key is None else group
        if g.height == 0:
            continue
        g = g.filter(
            pl.col("y").is_not_null()
            & pl.col("yhat").is_not_null()
            & pl.col("y").is_finite()
            & pl.col("yhat").is_finite()
            & (pl.col("y") != 0)
        )
        if g.height == 0:
            continue
        if key is None:
            name = "__all__"
        else:
            name = str(g[key][0])
        dates = g.get_column("ds").unique().sort()
        if len(dates) < 4:
            factor = 1.0
        else:
            cut = dates[len(dates) // 2 - 1]
            halves = [g.filter(pl.col("ds") <= cut), g.filter(pl.col("ds") > cut)]
            biases = []
            for h in halves:
                sy = float(h["y"].sum())
                sh = float(h["yhat"].sum())
                biases.append((sh - sy) / sy if sy else 0.0)
            sy = float(g["y"].sum())
            sh = float(g["yhat"].sum())
            stable = (
                abs(biases[0]) >= cfg.bias_threshold
                and abs(biases[1]) >= cfg.bias_threshold
                and np.sign(biases[0]) == np.sign(biases[1])
                and sh > 0.0
            )
            factor = float(np.clip(sy / sh, *cfg.bias_clip)) if stable else 1.0
        if key is None:
            return factor
        out[name] = factor
    return out if key else 1.0


def optimize_leaf_oos(
    df: pl.DataFrame, cfg: LeafOptimizationConfig | None = None
) -> pl.DataFrame:
    """Replace SKU-store OOS/forecast-only yhat using validation-selected causal models.

    In-sample predictions remain untouched. OOS predictions are one-step-ahead and
    may use actual observations through t-1. Forecast-only predictions are recursive.
    Stable recent bias factors are estimated on validation predictions, first per
    store and then for the section; unstable factors are exactly 1.0.
    """
    cfg = cfg or LeafOptimizationConfig()
    required = {"unique_id", "ds", "y", "yhat", "period_type"}
    if df.height == 0 or not required.issubset(df.columns):
        return df

    updates: list[dict] = []
    validation_records: list[dict] = []
    model_by_uid: dict[str, str] = {}

    for part in df.sort(["unique_id", "ds"]).partition_by("unique_id", as_dict=False):
        uid = str(part["unique_id"][0])
        if uid.count("||") != 2:
            continue
        y = np.asarray(part["y"].fill_null(0.0).to_numpy(), dtype=np.float64)
        dates = part["ds"].to_list()
        weekdays = np.asarray([d.weekday() for d in dates], dtype=np.int8)
        periods = np.asarray(part["period_type"].to_list(), dtype=object)
        predictions = causal_candidates(y, weekdays)

        train_idx = np.flatnonzero(periods == "in_sample")
        if len(train_idx) == 0:
            continue
        val_idx = train_idx[-cfg.validation_days :]
        validation_mask = np.zeros(len(y), dtype=bool)
        validation_mask[val_idx] = True
        selected = select_model(
            y,
            predictions,
            validation_mask,
            min_scored_points=cfg.min_scored_points,
        )
        model_by_uid[uid] = selected.model
        candidate = predictions[selected.model]
        store = _store_from_uid(uid)

        for i in val_idx:
            validation_records.append(
                {
                    "unique_id": uid,
                    "store": store,
                    "ds": dates[i],
                    "y": float(y[i]),
                    "yhat": float(candidate[i]),
                }
            )

        oos_idx = np.flatnonzero(periods == "out_sample")
        for i in oos_idx:
            updates.append(
                {
                    "unique_id": uid,
                    "ds": dates[i],
                    "_leaf_yhat": float(candidate[i]),
                    "leaf_model": selected.model,
                }
            )

        fc_idx = np.flatnonzero(periods == "forecast_only")
        if len(fc_idx):
            first = int(fc_idx[0])
            hist_idx = np.arange(first)
            future = recursive_forecast(
                y[hist_idx], weekdays[hist_idx], weekdays[fc_idx], selected.model
            )
            for i, pred in zip(fc_idx, future, strict=True):
                updates.append(
                    {
                        "unique_id": uid,
                        "ds": dates[i],
                        "_leaf_yhat": float(pred),
                        "leaf_model": selected.model,
                    }
                )

    if not updates:
        return df

    # Bias calibration on validation predictions.
    store_factors = _stable_factor(validation_records, "store", cfg)
    adjusted_validation = []
    for row in validation_records:
        r = dict(row)
        r["yhat"] *= float(store_factors.get(r["store"], 1.0))
        adjusted_validation.append(r)
    section_factor = float(_stable_factor(adjusted_validation, None, cfg))

    for row in updates:
        store = _store_from_uid(row["unique_id"])
        sf = float(store_factors.get(store, 1.0))
        row["leaf_store_factor"] = sf
        row["leaf_section_factor"] = section_factor
        row["_leaf_yhat"] = max(0.0, round(row["_leaf_yhat"] * sf * section_factor))

    upd = pl.DataFrame(updates).with_columns(pl.col("ds").cast(df.schema["ds"]))
    out = df.join(upd, on=["unique_id", "ds"], how="left")
    out = out.with_columns(
        pl.when(pl.col("_leaf_yhat").is_not_null())
        .then(pl.col("yhat"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("yhat_pre_leaf_optimizer"),
        pl.coalesce(pl.col("_leaf_yhat"), pl.col("yhat")).alias("yhat"),
        pl.col("leaf_model").fill_null("rls_ses").alias("leaf_model"),
        pl.col("leaf_store_factor").fill_null(1.0),
        pl.col("leaf_section_factor").fill_null(1.0),
    ).drop("_leaf_yhat")
    logger.info(
        "Leaf OOS optimizer: %d filas actualizadas, %d modelos, section_factor=%.4f",
        len(updates),
        len(model_by_uid),
        section_factor,
    )
    return out
