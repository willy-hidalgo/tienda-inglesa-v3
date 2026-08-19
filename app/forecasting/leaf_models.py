"""Causal leaf-level forecasting primitives.

This module deliberately depends only on NumPy so model-selection logic can be
unit-tested without Polars or Streamlit.  Every prediction at t uses values
strictly before t; this is suitable for honest OOS evaluation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

CANDIDATE_MODELS: tuple[str, ...] = (
    "last_nonzero",
    "mean7",
    "mean14",
    "mean28",
    "median28",
    "weekday4",
    "weekday8",
    "seasonal7",
    "seasonal14",
    "seasonal28",
    "ses10",
    "ses20",
    "ses30",
    "ses50",
    "ses74",
    "last",
    "zero",
)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    model: str
    wmape: float
    n_scored: int


def wmape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Client WMAPE: sum(|y-yhat|)/sum(|y|), excluding y==0/non-finite."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(yhat) & (y != 0.0)
    if not np.any(mask):
        return float("nan")
    denominator = float(np.abs(y[mask]).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(np.abs(y[mask] - yhat[mask]).sum() / denominator)


def _rolling_mean_shifted(y: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(y, dtype=np.float64)))
    for i in range(1, len(y)):
        start = max(0, i - window)
        out[i] = (csum[i] - csum[start]) / (i - start)
    return out


def _rolling_median_shifted(y: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float64)
    for i in range(1, len(y)):
        out[i] = float(np.median(y[max(0, i - window) : i]))
    return out


def _weekday_mean_shifted(
    y: np.ndarray, weekdays: np.ndarray, occurrences: int
) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float64)
    history = [deque(maxlen=occurrences) for _ in range(7)]
    for i, (value, weekday) in enumerate(zip(y, weekdays, strict=True)):
        bucket = history[int(weekday)]
        if bucket:
            out[i] = float(sum(bucket) / len(bucket))
        elif i:
            out[i] = float(y[i - 1])
        bucket.append(float(value))
    return out


def _ses_shifted(y: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float64)
    if len(y) == 0:
        return out
    state = float(y[0])
    out[0] = 0.0
    for i in range(1, len(y)):
        out[i] = state
        state = alpha * float(y[i]) + (1.0 - alpha) * state
    return out


def causal_candidates(y: np.ndarray, weekdays: np.ndarray) -> dict[str, np.ndarray]:
    """Return one-step-ahead causal predictions for all candidate models."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, 0.0, None)
    weekdays = np.asarray(weekdays, dtype=np.int8)
    if len(y) != len(weekdays):
        raise ValueError("y and weekdays must have the same length")

    n = len(y)
    last = np.zeros(n, dtype=np.float64)
    if n > 1:
        last[1:] = y[:-1]
    last_nonzero = np.zeros(n, dtype=np.float64)
    state = 0.0
    for i in range(n):
        last_nonzero[i] = state
        if y[i] > 0.0:
            state = float(y[i])

    out: dict[str, np.ndarray] = {
        "zero": np.zeros(n, dtype=np.float64),
        "last": last,
        "last_nonzero": last_nonzero,
        "mean7": _rolling_mean_shifted(y, 7),
        "mean14": _rolling_mean_shifted(y, 14),
        "mean28": _rolling_mean_shifted(y, 28),
        "median28": _rolling_median_shifted(y, 28),
        "weekday4": _weekday_mean_shifted(y, weekdays, 4),
        "weekday8": _weekday_mean_shifted(y, weekdays, 8),
        "ses10": _ses_shifted(y, 0.10),
        "ses20": _ses_shifted(y, 0.20),
        "ses30": _ses_shifted(y, 0.30),
        "ses50": _ses_shifted(y, 0.50),
        "ses74": _ses_shifted(y, 0.74),
    }
    for lag in (7, 14, 28):
        pred = last.copy()
        if n > lag:
            pred[lag:] = y[:-lag]
        out[f"seasonal{lag}"] = pred
    return out


def select_model(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    validation_mask: np.ndarray,
    *,
    min_scored_points: int = 7,
) -> SelectionResult:
    """Choose the lowest validation WMAPE with deterministic tie breaking."""
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(validation_mask, dtype=bool) & np.isfinite(y) & (y != 0.0)
    n_scored = int(mask.sum())
    if n_scored < min_scored_points:
        # Stable fallback for sparse/new series.
        fallback = "mean28" if "mean28" in predictions else next(iter(predictions))
        return SelectionResult(
            fallback, wmape(y[mask], predictions[fallback][mask]), n_scored
        )

    best_model = CANDIDATE_MODELS[0]
    best_score = float("inf")
    for model in CANDIDATE_MODELS:
        pred = predictions.get(model)
        if pred is None:
            continue
        score = wmape(y[mask], pred[mask])
        if np.isfinite(score) and score < best_score - 1e-12:
            best_model, best_score = model, score
    if not np.isfinite(best_score):
        best_model = "mean28" if "mean28" in predictions else next(iter(predictions))
        best_score = wmape(y[mask], predictions[best_model][mask])
    return SelectionResult(best_model, float(best_score), n_scored)


def _forecast_one(
    history: list[float], weekdays: list[int], target_weekday: int, model: str
) -> float:
    if not history or model == "zero":
        return 0.0
    a = np.asarray(history, dtype=np.float64)
    if model == "last":
        return max(0.0, float(a[-1]))
    if model == "last_nonzero":
        nz = a[a > 0.0]
        return float(nz[-1]) if len(nz) else 0.0
    if model.startswith("mean"):
        n = int(model[4:])
        return max(0.0, float(np.mean(a[-n:])))
    if model == "median28":
        return max(0.0, float(np.median(a[-28:])))
    if model.startswith("seasonal"):
        lag = int(model[8:])
        return max(0.0, float(a[-lag] if len(a) >= lag else a[-1]))
    if model.startswith("weekday"):
        n = int(model[7:])
        vals = [
            v for v, wd in zip(history, weekdays, strict=True) if wd == target_weekday
        ]
        return max(0.0, float(np.mean(vals[-n:]))) if vals else max(0.0, float(a[-1]))
    if model.startswith("ses"):
        alpha = int(model[3:]) / 100.0
        state = float(a[0])
        for value in a[1:]:
            state = alpha * float(value) + (1.0 - alpha) * state
        return max(0.0, state)
    raise ValueError(f"Unknown leaf model: {model}")


def recursive_forecast(
    history_y: np.ndarray,
    history_weekdays: np.ndarray,
    future_weekdays: np.ndarray,
    model: str,
) -> np.ndarray:
    """Recursive forecast for periods where actual future y is unavailable."""
    history = [float(v) for v in np.nan_to_num(history_y, nan=0.0)]
    weekdays = [int(v) for v in history_weekdays]
    out = np.empty(len(future_weekdays), dtype=np.float64)
    for i, weekday in enumerate(future_weekdays):
        pred = _forecast_one(history, weekdays, int(weekday), model)
        out[i] = pred
        history.append(pred)
        weekdays.append(int(weekday))
    return out
