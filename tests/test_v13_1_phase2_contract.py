"""Phase-2 optimization contracts: same SES+RLS family, diagnostics only."""
from __future__ import annotations

from pathlib import Path

import settings

ROOT = Path(__file__).resolve().parent.parent


def test_phase2_grid_is_diagnostic_only_after_v13_2_6_regression_freeze():
    assert settings.STAT_OPTIMIZATION_PROMOTE_AUTOMATICALLY is False
    assert 0.30 not in settings.LEAF_SES_ALPHA_CANDIDATES
    assert set(settings.STAT_OPT_SES_EXTRA_ALPHAS) == {0.0025, 0.0075, 0.30, 0.50}
    assert {0.990, 0.9975}.isdisjoint(set(settings.RLS_FORGETTING_FACTOR_CANDIDATES))
    assert set(settings.STAT_OPT_RLS_EXTRA_LAMBDAS) == {0.990, 0.9975, 1.0}
    assert settings.RLS_VALUE_PRICE_NODE_IDS == ()
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    runner = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert "productive_alpha_mask" in leaf
    assert "if productive_alpha_mask[a] != 1" in leaf
    assert "diagnostic_candidates" in runner
    assert "candidates = tuple((mode, lam) for mode in mode_candidates for lam in productive_lambdas)" in runner
    assert "for c in candidates:" in runner  # productive cumulative selector


def test_phase2_refits_only_same_rls_and_existing_driver_groups():
    runner = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8").lower()
    assert "refit_same_rls" in runner
    assert 'test_action="remove"' in runner
    assert 'test_action="add"' in runner
    assert "phase2 value + existing price drivers" in runner
    for forbidden in ("lightgbm", "xgboost", "randomforest", "occurrence/share", "meta-selector"):
        assert forbidden not in runner


def test_phase2_cli_is_explicit_and_dashboard_schema_unchanged():
    cli = (ROOT / "app/forecasts.py").read_text(encoding="utf-8")
    assert '"--optimization-phase2"' in cli
    assert "args.optimization_diagnostics or args.optimization_phase2" in cli
    dash = (ROOT / "app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 33" in dash


def test_value_metric_support_is_y_positive_not_value_positive():
    runner = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "valid_v = valid_y & np.isfinite(v[start:end])" in runner
    assert "valid_v = valid_y & np.isfinite(v[s:e])" in runner
    assert "if np.isfinite(value[r]):" in leaf


def test_phase2_driver_refit_summary_polars_smoke():
    import polars as pl
    from app.forecasting.optimization import summarize_driver_refit

    df = pl.DataFrame(
        {
            "section": ["1", "1"],
            "node_level": ["tienda", "tienda"],
            "target": ["Unidades", "Unidades"],
            "period_type": ["in_sample", "in_sample"],
            "driver_group": ["holiday_other", "holiday_other"],
            "test_action": ["remove", "remove"],
            "dynamics": ["base", "base"],
            "lambda": [0.995, 0.995],
            "sum_abs_y": [100.0, 200.0],
            "n_points": [10, 20],
            "wmape_improvement": [0.01, 0.02],
            "abs_bias_change": [0.0, 0.0],
            "unique_id": ["1||T:00154", "1||T:00154"],
            "block": [1, 2],
        }
    )
    out = summarize_driver_refit(df)
    assert out.height == 1
    assert float(out["weighted_wmape_improvement"][0]) > 0
    assert out["decision_signal"][0] == "candidate_remove"


def test_extra_ses_alphas_cannot_change_productive_selection():
    import numpy as np
    from app.forecasting.leaf_ses_rls import _ses_walkforward_kernel

    # One leaf, three 2-day blocks. First block is warm-up; later blocks have
    # actuals that make very small alpha attractive, but that alpha is marked
    # diagnostic-only in the second run.
    uid = np.zeros(6, dtype=np.int64)
    blocks = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    warm = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
    periods = np.zeros(6, dtype=np.int8)
    y = np.array([10., 10., 20., 20., 20., 20.])
    v = y * 10.0
    factor = np.ones(6, dtype=np.float64)
    init_y = np.full(6, 10.0)
    init_v = np.full(6, 100.0)
    observable_seq = np.arange(1, 7, dtype=np.int64)
    block_origin_seq = np.array([0, 0, 2, 2, 4, 4], dtype=np.int64)
    warmup_end_seq = np.full(6, 2, dtype=np.int64)

    prod_alphas = np.array([0.1, 0.2], dtype=np.float64)
    prod_mask = np.array([1, 1], dtype=np.uint8)
    base = _ses_walkforward_kernel(
        uid, blocks, warm, periods, observable_seq, block_origin_seq, warmup_end_seq,
        y, v, factor, factor, init_y, init_v, prod_alphas, prod_mask, 0, 0,
        0.50, 2.00, 2.00, 7, 1, 1, 28,
    )

    all_alphas = np.array([0.0025, 0.1, 0.2], dtype=np.float64)
    all_mask = np.array([0, 1, 1], dtype=np.uint8)
    phase2 = _ses_walkforward_kernel(
        uid, blocks, warm, periods, observable_seq, block_origin_seq, warmup_end_seq,
        y, v, factor, factor, init_y, init_v, all_alphas, all_mask, 1, 1,
        0.50, 2.00, 2.00, 7, 1, 1, 28,
    )

    np.testing.assert_allclose(base[0], phase2[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(base[1], phase2[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(base[4], phase2[4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(base[5], phase2[5], rtol=0, atol=1e-12)


def test_driver_refit_summary_is_node_specific_not_ambiguous():
    import polars as pl
    from app.forecasting.optimization import summarize_driver_refit

    df = pl.DataFrame(
        {
            "section": ["23", "23"],
            "node_level": ["tienda", "tienda"],
            "unique_id": ["23||T:00005", "23||T:00006"],
            "target": ["Valor ($)", "Valor ($)"],
            "period_type": ["in_sample", "in_sample"],
            "driver_group": ["price", "price"],
            "test_action": ["add", "add"],
            "dynamics": ["base", "base"],
            "lambda": [0.995, 0.995],
            "sum_abs_y": [100.0, 100.0],
            "n_points": [10, 10],
            "wmape_improvement": [0.02, -0.01],
            "abs_bias_change": [-0.01, 0.01],
            "block": [1, 1],
        }
    )
    out = summarize_driver_refit(df)
    assert out.height == 2
    assert set(out["unique_id"].to_list()) == {"23||T:00005", "23||T:00006"}


def test_value_price_driver_support_exists_but_no_node_is_productively_promoted():
    runner = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert 'RLS_VALUE_PRICE_NODE_IDS' in runner
    assert 'use_price = uid in price_nodes' in runner
    assert 'self._driver_cols if use_price else self._driver_cols_value_base' in runner
    assert settings.RLS_VALUE_PRICE_NODE_IDS == ()
