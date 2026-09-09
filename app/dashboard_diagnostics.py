"""Lightweight dashboard diagnostics for leaf forecast explainability.

This module never changes forecasts or model selection.  It only derives
visual/audit diagnostics from already-produced series using Polars + NumPy and
the existing EDP decomposition helper.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from rls_opt.edp import decompose_price
from app.discrepancy import DiscrepancyRule, discrepancy_points


def add_leaf_edp(df: pl.DataFrame) -> pl.DataFrame:
    """Add ex-post leaf ASP/EDP/discount and causal future carry-forward.

    Historical + OOS actuals are decomposed for diagnosis. Forecast-only has no
    actual price, so the last observed leaf EDP/ASP/discount is carried forward.
    This line is diagnostic and is *not* the parent RLS price driver used by the
    productive model.
    """
    if df.height == 0:
        return df
    needed = {"units_actual", "sales_value_actual"}
    if not needed.issubset(df.columns):
        return df
    work = df.sort("ds")
    if "period_type" in work.columns:
        observed_mask = work["period_type"].to_numpy() != "forecast_only"
    else:
        observed_mask = np.ones(work.height, dtype=bool)
    obs_idx = np.flatnonzero(observed_mask)
    asp_full = np.full(work.height, np.nan, dtype=np.float64)
    edp_full = np.full(work.height, np.nan, dtype=np.float64)
    disc_full = np.full(work.height, np.nan, dtype=np.float64)
    source = np.full(work.height, "", dtype=object)
    if obs_idx.size:
        units = work["units_actual"].cast(pl.Float64).fill_null(0.0).to_numpy()[obs_idx]
        value = work["sales_value_actual"].cast(pl.Float64).fill_null(0.0).to_numpy()[obs_idx]
        asp, edp, discount = decompose_price(
            np.ascontiguousarray(value, dtype=np.float64),
            np.ascontiguousarray(units, dtype=np.float64),
        )
        asp_full[obs_idx] = np.asarray(asp, dtype=np.float64)
        edp_full[obs_idx] = np.asarray(edp, dtype=np.float64)
        disc_full[obs_idx] = np.asarray(discount, dtype=np.float64)
        source[obs_idx] = "actual-derived"
        # Carry the last finite positive state into forecast-only.
        def _last_positive(arr: np.ndarray) -> float:
            vals = arr[np.isfinite(arr) & (arr > 0)]
            return float(vals[-1]) if vals.size else 0.0
        last_asp = _last_positive(asp_full[obs_idx])
        last_edp = _last_positive(edp_full[obs_idx])
        last_disc_vals = disc_full[obs_idx][np.isfinite(disc_full[obs_idx])]
        last_disc = float(last_disc_vals[-1]) if last_disc_vals.size else 0.0
        future_idx = np.flatnonzero(~observed_mask)
        if future_idx.size:
            asp_full[future_idx] = last_asp
            edp_full[future_idx] = last_edp
            disc_full[future_idx] = last_disc
            source[future_idx] = "carry-forward"
    def _finite_or_none(arr: np.ndarray) -> list[float | None]:
        return [float(v) if np.isfinite(v) else None for v in arr]
    return work.with_columns(
        pl.Series("asp_observed", _finite_or_none(asp_full), dtype=pl.Float64),
        pl.Series("edp_observed", _finite_or_none(edp_full), dtype=pl.Float64),
        pl.Series("discount_observed", _finite_or_none(disc_full), dtype=pl.Float64),
        pl.Series("edp_source", source.tolist(), dtype=pl.Utf8),
    )
