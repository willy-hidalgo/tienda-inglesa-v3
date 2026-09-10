"""Non-negotiable production contracts. Keep this file version-agnostic where possible."""
from __future__ import annotations
from pathlib import Path
import settings

ROOT = Path(__file__).resolve().parents[1]


def test_model_family_and_parameter_baseline_are_locked():
    assert tuple(settings.LEAF_SES_ALPHA_CANDIDATES) == (0.005,0.01,0.02,0.05,0.10,0.20,0.40,0.60,0.70,0.80)
    assert tuple(settings.RLS_FORGETTING_FACTOR_CANDIDATES) == (0.970,0.985,0.995)
    assert tuple(settings.RLS_VALUE_PRICE_NODE_IDS) == ()
    assert settings.RLS_DRIVER_GROUP_EXCLUSIONS == {}
    assert settings.STAT_OPTIMIZATION_PROMOTE_AUTOMATICALLY is False


def test_sections_expanding_28_and_update_cadences_are_locked():
    assert tuple(settings.FOCUS_SECTIONS) == ('1','23')
    assert settings.RLS_BLOCK_DAYS == 28
    assert settings.RLS_INITIAL_SEED_DAYS == 28
    assert settings.METRIC_HORIZON_DAYS == 28
    assert tuple(settings.UPDATE_BLOCK_OPTIONS) == (1,7,14,28)


def test_oos_is_holdout_for_selection_but_not_frozen_for_state_recurrence():
    runner=(ROOT/'app/forecasting/runner.py').read_text(encoding='utf-8')
    leaf=(ROOT/'app/forecasting/leaf_ses_rls.py').read_text(encoding='utf-8')
    assert 'history_mask = source_period[s:e] == "in_sample"' in runner
    assert 'fixed_oos_start' not in runner and 'fixed_oos_boundary' not in runner
    assert 'if pc == 0:  # only closed in-sample history selects alpha' in leaf
    assert 'if pc == 0:  # same holdout purity for Value' in leaf
    assert 'block_level_y = state_y[best_y] * selected_decay_y' in leaf
    assert 'oos_forecast_state_y' not in leaf


def test_all_skus_and_dashboard_format_contracts_remain_locked():
    cat=(ROOT/'app/categories_selector.py').read_text(encoding='utf-8')
    dash=(ROOT/'app/dashboard.py').read_text(encoding='utf-8')
    assert 'select_best_skus' not in cat
    assert 'best_skus' not in cat
    assert 'SKU_ID").is_in' not in cat
    assert 'def _dashboard_dataframe' in dash
    assert dash.count('st.dataframe(') == 1


def test_official_metrics_contract_not_changed():
    settings_text=(ROOT/'settings.py').read_text(encoding='utf-8')
    assert settings_text.count('mask = np.isfinite(y) & np.isfinite(yhat) & (y != 0)') >= 2
    # Exact formulas remain covered by test_v13_metrics_contract.py/test_wmape.py.
