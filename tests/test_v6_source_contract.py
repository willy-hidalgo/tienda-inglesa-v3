from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_dashboard_metrics_are_bottom_up():
    src = _read("app/dashboard_data.py")
    assert src.count("metrics_in_out_bottom_up(") >= 2
    assert 'rolling_metrics and node_kind in ("seccion", "tienda")' not in src


def test_leaf_warmup_and_ses_contract_present():
    src = _read("app/forecasting/runner.py")
    assert "Production leaf model: original-scale SES level + normalized RLS shape" in src
    assert "leaf_warmup_calendar_mean" in src
    assert "leaf_ses_level_driver:" in src
    assert "Initial structural level" in src


def test_official_metrics_filter_warmup():
    assert "rls_metric_eligible" in _read("app/forecasting/metrics.py")
    assert "rls_metric_eligible" in _read("app/backend.py")


def test_sku_ranking_min_nonzero_rule_present():
    assert "RANKING_SKU_MIN_NONZERO_POINTS" in _read("app/dashboard_data.py")
    assert "RANKING_SKU_MIN_NONZERO_POINTS" in _read("app/backend.py")


def test_leaf_parent_selection_is_leaf_level():
    src = _read("app/forecasting/runner.py")
    method = src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]
    assert "STAGE 1: select alpha from PURE SES history only" in method
    assert "STAGE 2: select parent shape with SES level already fixed" in method
    assert "_wmape_ses_y" in method and "_wmape_ses_v" in method
    assert "_wmape_parent_y" in method and "_wmape_parent_v" in method


def test_no_direct_parent_wmape_selection_for_leaf():
    src = _read("app/forecasting/runner.py")
    assert "Small dictionary: only section + stores" not in src


def test_forecast_only_does_not_consume_post_oos_actuals():
    settings_src = _read("settings.py")
    runner = _read("app/forecasting/runner.py")
    assert "forecast_start = oos_end + dt.timedelta(days=1)" in settings_src
    assert "No post-OOS leakage" in runner
    method = runner[runner.index("def _fit_and_predict_expanding_blocks"):runner.index("def fit_and_predict_sections")]
    assert 'target_parts.get("actual_extension")' not in method


def test_adaptive_leaf_components_present():
    src = _read("app/forecasting/runner.py")
    settings_src = _read("settings.py")
    assert "LEAF_SES_ALPHA_CANDIDATES" in src
    assert "LEAF_SES_SCALE" in settings_src
    assert "_level_y" in src
    assert "_level_v" in src
    assert "scale_source" not in src


def test_fast_leaf_has_no_shape_changing_filter_in_alpha_with_columns():
    src = _read("app/forecasting/runner.py")
    start = src.index("    def fast_leaf_forecasts(")
    end = src.index("\n    def derive_sku_store_forecasts(", start)
    method = src[start:end]
    # Regression for the ShapeError seen in v7.1: filter() inside with_columns
    # shortened the expression relative to the DataFrame.
    assert (
        '(pl.col("y") - pl.col("_level_y"))\n'
        '                .abs()\n'
        '                .filter('
    ) not in method
    assert (
        '(pl.col("value") - pl.col("_level_v"))\n'
        '                .abs()\n'
        '                .filter('
    ) not in method
    assert "outer_coalesce" not in method


def test_leaf_ses_level_driver_is_two_stage_without_scale():
    src = _read("app/forecasting/runner.py")
    assert "leaf_ses_level_driver:" in src
    assert "_level_y" in src
    assert "_level_v" in src
    assert "scale_source" not in src
    assert "LEAF_SCALE_CLIP" not in src
    assert "parent/RLS performance can NEVER" in src


def test_rls_ar_is_recursive_and_lambda_is_prior_block_selected():
    src = _read("app/forecasting/runner.py")
    assert "_ar_recursive_row" in src
    assert "hist.append(lp)" in src
    assert "RLS_FORGETTING_FACTOR_CANDIDATES" in src
    assert "choose(cum_ae_y, cum_den_y)" in src
    assert 'target_parts.get("actual_extension")' not in src[
        src.index("def _fit_and_predict_expanding_blocks"):
        src.index("def fit_and_predict_sections")
    ]


def test_v8_removed_leaf_scale_from_forecast_logic():
    src = _read("app/forecasting/runner.py")
    settings_src = _read("settings.py")
    method = src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]
    assert "leaf_scale" not in method
    assert "LEAF_SCALE_CLIP" not in settings_src
    assert "original-scale SES level + normalized RLS shape" in method
