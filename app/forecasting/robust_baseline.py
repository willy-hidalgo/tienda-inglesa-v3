"""Robust demand baselines used as a guardrail for leaf-level forecasts.

The functions in this module are deliberately NumPy/Python-only.  They are
small, deterministic and easy to test independently of Polars/Streamlit.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence
import re

import numpy as np

SUPPORTED_METHODS = (
    "median_pos28", "median_pos56", "median_pos84",
    "weekday_pos4", "weekday_pos8", "weekday_pos12",
    "seasonal_naive7",
)


def wmape_np(y: Sequence[float], yhat: Sequence[float]) -> float:
    """Client WMAPE: sum(abs(y-yhat))/sum(abs(y)), excluding y == 0."""
    a = np.asarray(y, dtype=np.float64)
    p = np.asarray(yhat, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(p) & (a != 0.0)
    if not np.any(mask):
        return float("inf")
    denom = float(np.abs(a[mask]).sum())
    return float(np.abs(a[mask] - p[mask]).sum() / denom) if denom else float("inf")


def _positive_median(values: np.ndarray) -> float:
    x = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(x)) if x.size else 0.0


def _method_lookback(method: str, default_days: int) -> int:
    m = re.search(r"(\d+)$", method)
    if not m:
        return int(default_days)
    n = int(m.group(1))
    if method.startswith("weekday_pos"):
        return max(7, n * 7)
    return n


def robust_baseline_forecast(
    y_history: Sequence[float],
    history_dates: Sequence[date],
    future_dates: Sequence[date],
    *,
    method: str,
    lookback_days: int = 56,
) -> np.ndarray:
    """Leakage-free robust forecast from history strictly before the horizon."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported baseline method: {method!r}")
    y = np.asarray(y_history, dtype=np.float64)
    if len(y) != len(history_dates):
        raise ValueError("y_history and history_dates must have the same length")
    if y.size == 0:
        return np.zeros(len(future_dates), dtype=np.float64)

    if method == "seasonal_naive7":
        recent = y[-7:] if y.size >= 7 else y
        recent = np.where(np.isfinite(recent), recent, 0.0)
        if recent.size == 0:
            return np.zeros(len(future_dates), dtype=np.float64)
        return np.asarray(
            [max(0.0, float(recent[i % recent.size])) for i in range(len(future_dates))],
            dtype=np.float64,
        )

    lb = min(_method_lookback(method, lookback_days), y.size)
    recent = y[-lb:]
    recent_dates = history_dates[-lb:]
    fallback = _positive_median(recent)

    if method.startswith("median_pos"):
        return np.full(len(future_dates), fallback, dtype=np.float64)

    by_weekday: dict[int, list[float]] = {i: [] for i in range(7)}
    for d, value in zip(recent_dates, recent):
        if np.isfinite(value) and value > 0.0:
            by_weekday[d.weekday()].append(float(value))
    profile = {
        dow: (float(np.median(vals)) if vals else fallback)
        for dow, vals in by_weekday.items()
    }
    return np.asarray([profile[d.weekday()] for d in future_dates], dtype=np.float64)

def validation_score(
    y_history: Sequence[float],
    history_dates: Sequence[date],
    *,
    method: str,
    validation_days: int = 28,
    lookback_days: int = 56,
) -> float:
    """Pseudo-OOS WMAPE on the tail of train with no validation leakage."""
    y = np.asarray(y_history, dtype=np.float64)
    n_val = min(int(validation_days), max(0, y.size - 1))
    if n_val < 1:
        return float("inf")
    split = y.size - n_val
    pred = robust_baseline_forecast(
        y[:split],
        history_dates[:split],
        history_dates[split:],
        method=method,
        lookback_days=lookback_days,
    )
    return wmape_np(y[split:], pred)
