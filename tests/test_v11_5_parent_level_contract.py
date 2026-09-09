from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_parent_level_shape_is_a_causal_candidate_not_forced():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'LEAF_PARENT_DRIVER_MODES' in settings
    assert '("shape_only", "level_shape")' in settings
    assert '"_driver_mode": driver_mode' in src
    assert 'default_driver_mode' in src
    assert 'sort_by(*sort_y)' in src
    assert 'sort_by(*sort_v)' in src


def test_level_factor_uses_prior_parent_actual_block():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert 'prev_parent_level' in src
    assert '(pl.col("_block") + 1)' in src
    assert 'pl.col("_mean_py_block") / pl.col("_prev_y")' in src
    assert 'pl.col("_mean_pv_block") / pl.col("_prev_v")' in src
    assert 'LEAF_REGIME_TREND_MAX_STEP_RATIO' in src


def test_shape_only_retains_mean_one_and_level_shape_has_explicit_factor():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert '_shape_factor_y' in src
    assert '_level_factor_y' in src
    assert 'driver_level_factor_y' in src
    assert 'parent_driver_mode_y' in src
    assert '_expected_y' in validator
    assert 'parent_driver_mode_y' in validator


def test_forecast_only_borrows_shape_not_level_factor():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert '_prev_effect_y_shape' in src
    assert '_prev_effect_v_shape' in src
    assert '(pl.col("_effect_y_shape") + pl.col("_level_factor_y").log())' in src
