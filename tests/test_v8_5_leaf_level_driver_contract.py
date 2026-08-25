from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]


def test_ses_level_is_original_scale_and_independent_of_parent_effect():
    m = leaf_method()
    assert 'LEAF_SES_SCALE: str = "original"' in read("settings.py")
    assert '_level_log_y' not in m
    assert '_level_log_v' not in m
    ses_segment = m[m.index("# ── Score ALL alpha candidates using SES ONLY"):
                    m.index("# ── Score parent shapes")]
    assert 'pl.col("y").clip(lower_bound=0.0)' in ses_segment
    assert 'pl.col("value").clip(lower_bound=0.0)' in ses_segment
    for forbidden in ('_parent', '_store_ey', '_store_ev', '_sec_ey', '_sec_ev'):
        assert forbidden not in ses_segment


def test_driver_shape_is_level_neutral_by_construction():
    m = leaf_method()
    assert 'pl.col("_py") / pl.col("_mean_py")' in m
    assert 'pl.col("_pv") / pl.col("_mean_pv")' in m
    assert '(pl.col("_ratio_y") - 1.0).alias("_dev_y")' in m
    assert '1.0 + pl.col("_shape_scale_y") * pl.col("_dev_y")' in m
    assert '1.0 + pl.col("_shape_scale_v") * pl.col("_dev_v")' in m
    assert 'mean(factor) stays EXACTLY 1' in m


def test_driver_shape_is_bounded_and_configurable():
    settings = read("settings.py")
    m = leaf_method()
    assert 'FAST_LEAF_DRIVER_FACTOR_CLIP: tuple[float, float] = (0.50, 2.00)' in settings
    assert 'factor_lo, factor_hi' in m
    assert '(factor_hi - 1.0) / pl.col("_max_dy")' in m
    assert '(1.0 - factor_lo) / (-pl.col("_min_dy"))' in m


def test_leaf_forecast_equals_ses_level_times_driver_factor():
    m = leaf_method()
    assert 'pl.col("_level_y")' in m
    assert 'pl.col("_ey").exp()' in m
    assert 'pl.col("_level_v")' in m
    assert 'pl.col("_ev").exp()' in m
    assert '.alias("yhat")' in m
    assert '.alias("valuehat")' in m
    assert 'ses_level_y' in m
    assert 'ses_level_value' in m
    assert 'driver_factor_y' in m
    assert 'driver_factor_value' in m


def test_forecast_only_keeps_driver_shape_without_future_actuals():
    m = leaf_method()
    assert 'reuse the immediately preceding' in m
    assert '_prev_effect_y' in m
    assert '_prev_effect_v' in m
    assert 'actual_extension_leaves' in m  # signature compatibility only
    # It is never appended to actual_parts.
    actual_block = m[m.index('# Observed leaf rows through OOS only.'):m.index('# Dense only for OOS')]
    assert 'actual_extension_leaves' not in actual_block


def test_model_label_describes_new_semantics():
    m = leaf_method()
    assert 'leaf_ses_level_driver:' in m
    assert 'leaf_residual_ses:' not in m
