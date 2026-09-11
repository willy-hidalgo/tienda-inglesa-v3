from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _kernel_without_numba():
    import numpy as np
    source = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ses_walkforward_kernel")
    fn.decorator_list = []
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<v13217-kernel>", "exec"), ns)
    return ns["_ses_walkforward_kernel"]


def test_v13217_settings_contract():
    import settings
    assert settings.APP_VERSION == "13.3.3"
    assert settings.LEAF_GAP_DORMANT_OBSERVABLE_DAYS == 28
    assert 0 < settings.LEAF_GAP_DORMANT_FACTOR <= 1e-12
    # Established v13 contract: OOS recurrence is bounded, not frozen.
    assert settings.LEAF_OOS_STATE_ANCHOR_MIN_FACTOR == 0.75
    assert settings.LEAF_OOS_STATE_ANCHOR_MAX_FACTOR == 1.25
    assert settings.LEAF_OOS_RECENT_POSITIVE_WINDOW == 28
    assert settings.LEAF_OOS_RECENT_MEDIAN_MIN_POINTS == 7
    assert settings.LEAF_OOS_RECENT_MEDIAN_MAX_FACTOR == 1.50


def test_period_blocks_never_straddle_period_boundary():
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "def _period_block_expr(" in text
    assert "1_000_000" in text and "2_000_000" in text


def test_rls_oos_uses_each_closed_block_origin_not_fixed_global_origin():
    text = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert "fixed_oos_start" not in text
    assert "fixed_oos_boundary" not in text
    assert "paths_y[c][boundary]" in text
    assert "paths_v[c][boundary]" in text
    assert 'history_mask = source_period[s:e] == "in_sample"' in text


def test_leaf_closed_oos_block_can_update_next_origin_without_retuning():
    import numpy as np
    k = _kernel_without_numba()
    y = np.array([10.0] * 8 + [20.0, 20.0], dtype=float)
    n = len(y)
    out = k(
        np.zeros(n, dtype=np.int64), np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.uint8), np.array([0] * 8 + [1, 1], dtype=np.int8),
        np.arange(1, n + 1, dtype=np.int64), np.arange(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64), y, y * 10.0,
        np.ones(n), np.ones(n), np.full(n, 10.0), np.full(n, 100.0),
        np.array([0.10]), np.array([1], dtype=np.uint8), 0, 0,
        0.50, 2.00, 2.00, 7, 1, 1, 28,
        0.025, 84, 0.10, 0.20, 2.0, 28, 1e-12,
        4.0, 2.0, 0.75, 1.25, 28, 7, 1.5,
    )
    level = out[2]
    # First OOS forecast uses history-only origin; after the first OOS block
    # closes, the next 1d origin may move causally, but alpha remains frozen.
    assert abs(float(level[8]) - 10.0) < 1e-12
    assert float(level[9]) > float(level[8])
    assert float(out[4][9]) == float(out[4][8]) == 0.10


def test_recent_positive_median_caps_only_first_oos_anchor():
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "recent_median_max_factor" in text
    assert "if oos_anchor_ready == 0:" in text
    assert "block_level_y = state_y[best_y] * selected_decay_y" in text
    assert "oos_forecast_state_y" not in text


def test_artifact_version():
    text=(ROOT/"app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 33" in text
