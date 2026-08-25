from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from app.forecasting.robust_baseline import (
    robust_baseline_forecast,
    validation_score,
    wmape_np,
)


def _dates(n: int):
    return [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(n)]


def test_wmape_excludes_zero_actuals():
    assert wmape_np([10, 0, 20], [8, 999, 25]) == pytest.approx(7 / 30)


def test_median_positive_ignores_zero_days():
    y = [0, 10, 0, 20, 30, 0]
    pred = robust_baseline_forecast(y, _dates(6), _dates(8)[6:], method="median_pos56")
    assert pred.tolist() == [20.0, 20.0]


def test_weekday_profile_uses_only_history():
    d = _dates(21)
    y = [0.0] * 14
    # two Mondays in history have positive demand; future Monday must use their median
    for i, day in enumerate(d[:14]):
        if day.weekday() == 0:
            y[i] = 10.0 if i < 7 else 20.0
    p = robust_baseline_forecast(y, d[:14], d[14:21], method="weekday_pos8")
    monday_idx = next(i for i, day in enumerate(d[14:21]) if day.weekday() == 0)
    assert p[monday_idx] == pytest.approx(15.0)


def test_validation_score_is_finite_when_tail_has_sales():
    d = _dates(100)
    y = np.asarray([0 if i % 3 else 10 + (i % 7) for i in range(100)], dtype=float)
    score = validation_score(y, d, method="median_pos56", validation_days=28)
    assert np.isfinite(score)


def test_all_adaptive_baseline_candidates_return_finite_nonnegative_forecasts():
    import datetime as dt
    from app.forecasting.robust_baseline import SUPPORTED_METHODS, robust_baseline_forecast

    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(100)]
    y = [0.0 if i % 5 == 0 else float(5 + (i % 7)) for i in range(100)]
    future = [dates[-1] + dt.timedelta(days=i) for i in range(1, 29)]

    for method in SUPPORTED_METHODS:
        pred = robust_baseline_forecast(y, dates, future, method=method)
        assert len(pred) == len(future)
        assert np.isfinite(pred).all()
        assert (pred >= 0).all()
