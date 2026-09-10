"""v13.2.17 contracts: common history + observable-gap-aware leaf SES."""
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
    assert 'APP_VERSION: str = "13.2.17"' in settings_text
    assert "LEAF_GAP_AWARE_ENABLED: bool = True" in settings_text
    assert "LEAF_GAP_REACTIVATION_ROBUST_BYPASS_DAYS: int = 28" in settings_text
    assert "LEAF_GAP_DECAY_ALPHA_FLOOR: float = 0.025" in settings_text
    assert "LEAF_GAP_DECAY_MAX_OBSERVABLE_DAYS: int = 84" in settings_text
    assert "LEAF_GAP_DECAY_MIN_FACTOR: float = 0.10" in settings_text
    assert "LEAF_GAP_REACTIVATION_ALPHA_MAX: float = 0.20" in settings_text
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


def test_observable_zero_gap_never_mutates_state_and_oos_factor_is_frozen():
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "frozen_gap_factor_y" in leaf and "frozen_gap_factor_v" in leaf
    assert "block_level_y = state_y[best_y] * selected_decay_y" in leaf
    assert "block_level_v = state_v[best_v] * selected_decay_v" in leaf
    assert "sy *= event_decay" not in leaf
    assert "sv *= event_decay" not in leaf


def test_first_sale_after_long_gap_reanchors_only_after_close():
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "reactivation_gap_y >= reactivation_bypass" in leaf
    assert "lo_state_y = obs_y / reactivation_state_factor" in leaf
    assert "hi_state_y = obs_y * reactivation_state_factor" in leaf


def test_release_gate_audits_long_gap_reactivations_and_common_history():
    text = (ROOT / "app/forecasting/regression_gate.py").read_text(encoding="utf-8")
    assert "_long_gap_reactivation_summary" in text
    assert "observable_zero_days" in text
    assert "reactivaciones tras gap largo fuera de escala" in text
    assert "_train_window_summary" in text
    assert "historia al origen OOS difiere entre cadencias" in text
    assert "forecast_to_actual" in text
    assert "Large upward reactivations" in text


def test_store_observability_is_built_before_oos_leaf_history_filter():
    pipeline = (ROOT / "app/forecasting/pipeline.py").read_text(encoding="utf-8")
    obs_pos = pipeline.index("oos_observable_store_days =")
    semi_pos = pipeline.index('oos_leaves = oos_leaves.join(leaf_uids, on="unique_id", how="semi")')
    assert obs_pos < semi_pos
    assert "oos_leaves_all.select(" in pipeline[obs_pos - 400 : obs_pos + 500]
