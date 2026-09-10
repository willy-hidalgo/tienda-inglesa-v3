"""v13.2.11 contracts: common history + observable-gap-aware leaf SES."""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _kernel_without_numba():
    import numpy as np

    source = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ses_walkforward_kernel"
    )
    fn.decorator_list = []
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "<ses-gap-kernel>", "exec"),
        namespace,
    )
    return namespace["_ses_walkforward_kernel"]


def test_v13211_gap_settings_and_trace_are_explicit():
    settings_text = (ROOT / "settings.py").read_text(encoding="utf-8")
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    validator = (ROOT / "app/forecasting/validate_v13.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "13.2.11"' in settings_text
    assert "LEAF_GAP_AWARE_ENABLED: bool = True" in settings_text
    assert "LEAF_GAP_REACTIVATION_ROBUST_BYPASS_DAYS: int = 28" in settings_text
    assert "_store_observable_calendar" in leaf
    assert "leaf_observable_day_index" in leaf
    assert "leaf_gap_observable_days_y" in leaf
    assert "leaf_gap_observable_days_value" in leaf
    assert "leaf_gap_decay_factor_y" in validator
    assert "leaf_gap_decay_factor_value" in validator


def test_all_update_cadences_use_the_same_train_window():
    import settings

    old = settings.RLS_BLOCK_DAYS
    try:
        windows = []
        for days in settings.UPDATE_BLOCK_OPTIONS:
            settings.RLS_BLOCK_DAYS = int(days)
            hz = settings.section_horizons(
                "1", first_data=dt.date(2024, 5, 1), last_actual=dt.date(2026, 4, 30)
            )
            windows.append((hz["train_start"], hz["train_end"]))
            train_days = (hz["train_end"] - hz["train_start"]).days + 1
            assert train_days % int(days) == 0
            assert train_days % settings.CANONICAL_HISTORY_BLOCK_DAYS == 0
        assert len(set(windows)) == 1
        assert windows[0] == (dt.date(2024, 5, 3), dt.date(2026, 4, 2))
    finally:
        settings.RLS_BLOCK_DAYS = old


def test_observable_zero_gap_decays_ses_but_unobserved_calendar_gap_does_not():
    import numpy as np

    kernel = _kernel_without_numba()
    uid = np.zeros(4, dtype=np.int64)
    blocks = np.array([0, 1, 2, 63], dtype=np.int64)
    warm = np.zeros(4, dtype=np.uint8)
    periods = np.array([0, 0, 0, 1], dtype=np.int8)
    y = np.full(4, 10.0)
    value = y * 10.0
    common = dict(
        uid_codes=uid,
        blocks=blocks,
        warmup=warm,
        period_codes=periods,
        warmup_end_observable_seq=np.zeros(4, dtype=np.int64),
        y=y,
        value=value,
        factor_y=np.ones(4),
        factor_v=np.ones(4),
        init_y=np.full(4, 10.0),
        init_v=np.full(4, 100.0),
        alphas=np.array([0.10]),
        productive_alpha_mask=np.array([1], dtype=np.uint8),
        default_alpha_index=0,
        collect_diagnostics=0,
        ses_update_factor_min=0.50,
        ses_update_factor_max=2.00,
        forecast_history_max_multiplier=2.00,
        forecast_guard_min_positive_points=7,
        gap_aware_enabled=1,
        gap_min_observable_zero_days=1,
        gap_reactivation_robust_bypass_days=28,
    )

    # Store is observable on 60 intervening days: leaf absence is real zero-sale
    # information and the level must be stale/decayed at the OOS origin.
    with_gap = kernel(
        observable_seq=np.array([1, 2, 3, 64], dtype=np.int64),
        block_origin_observable_seq=np.array([0, 1, 2, 63], dtype=np.int64),
        **common,
    )
    assert with_gap[8][3] == 60
    assert with_gap[10][3] < 0.01
    assert with_gap[0][3] < 1.0

    # Same calendar distance, but no evidence that the store/section was
    # observable in-between: no zero is invented and the SES level stays 10.
    no_observable_gap = kernel(
        observable_seq=np.array([1, 2, 3, 4], dtype=np.int64),
        block_origin_observable_seq=np.array([0, 1, 2, 3], dtype=np.int64),
        **common,
    )
    assert no_observable_gap[8][3] == 0
    assert abs(float(no_observable_gap[0][3]) - 10.0) < 1e-12


def test_first_sale_after_long_gap_can_reactivate_standard_ses():
    import numpy as np

    kernel = _kernel_without_numba()
    # First history sale at observable day 1; next sale reappears after 60
    # observable zero days. The state is decayed before its forecast, but once
    # that actual closes it must be allowed to update with standard SES rather
    # than being clipped to 2x a near-zero stale state.
    result = kernel(
        np.zeros(3, dtype=np.int64),
        np.array([0, 61, 62], dtype=np.int64),
        np.zeros(3, dtype=np.uint8),
        np.array([0, 1, 1], dtype=np.int8),
        np.array([1, 62, 63], dtype=np.int64),
        np.array([0, 61, 62], dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.array([10.0, 10.0, 10.0]),
        np.array([100.0, 100.0, 100.0]),
        np.ones(3),
        np.ones(3),
        np.full(3, 10.0),
        np.full(3, 100.0),
        np.array([0.50]),
        np.array([1], dtype=np.uint8),
        0,
        0,
        0.50,
        2.00,
        2.00,
        7,
        1,
        1,
        28,
    )
    assert result[0][1] < 1.0  # stale level before seeing reactivation actual
    assert result[0][2] > 4.9  # next closed block learned the reactivation


def test_release_gate_audits_long_gap_reactivations_and_common_history():
    text = (ROOT / "app/forecasting/regression_gate.py").read_text(encoding="utf-8")
    assert "_long_gap_reactivation_summary" in text
    assert "observable_zero_days" in text
    assert "reactivaciones tras gap largo fuera de escala" in text
    assert "_train_window_summary" in text
    assert "historia al origen OOS difiere entre cadencias" in text


def test_store_observability_is_built_before_oos_leaf_history_filter():
    pipeline = (ROOT / "app/forecasting/pipeline.py").read_text(encoding="utf-8")
    obs_pos = pipeline.index("oos_observable_store_days =")
    semi_pos = pipeline.index('oos_leaves = oos_leaves.join(leaf_uids, on="unique_id", how="semi")')
    assert obs_pos < semi_pos
    assert "oos_leaves_all.select(" in pipeline[obs_pos - 400 : obs_pos + 500]
