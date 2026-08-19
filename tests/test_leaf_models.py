from __future__ import annotations

import numpy as np
import pytest

from app.forecasting.leaf_models import causal_candidates, recursive_forecast, select_model, wmape


def test_wmape_excludes_zero_actuals():
    assert wmape(np.array([10.0, 0.0, 20.0]), np.array([8.0, 999.0, 25.0])) == pytest.approx(7 / 30)


def test_all_candidates_are_strictly_causal():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    weekdays = np.arange(len(y)) % 7
    a = causal_candidates(y, weekdays)
    y_changed = y.copy(); y_changed[-1] = 99999.0
    b = causal_candidates(y_changed, weekdays)
    for model in a:
        assert a[model][-1] == pytest.approx(b[model][-1])


def test_selection_prefers_exact_weekly_pattern():
    y = np.tile(np.array([10., 20., 30., 40., 50., 60., 70.]), 10)
    weekdays = np.arange(len(y)) % 7
    preds = causal_candidates(y, weekdays)
    mask = np.zeros(len(y), dtype=bool); mask[-28:] = True
    result = select_model(y, preds, mask)
    assert result.model in {"seasonal7", "weekday4", "weekday8"}
    assert result.wmape == pytest.approx(0.0)


def test_recursive_forecast_does_not_need_future_actuals():
    history = np.array([10., 20., 30., 40., 50., 60., 70.])
    weekdays = np.arange(7)
    future_weekdays = np.array([0, 1, 2])
    out = recursive_forecast(history, weekdays, future_weekdays, "seasonal7")
    assert out.tolist() == pytest.approx([10., 20., 30.])
