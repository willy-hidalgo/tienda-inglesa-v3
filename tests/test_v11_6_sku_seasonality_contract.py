from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v116_uses_causal_364d_sku_seasonal_factor():
    src = (ROOT / "app" / "forecasting" / "leaf_fallback.py").read_text(encoding="utf-8")
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'LEAF_SKU_SEASONAL_LAG_BLOCKS: int = 13' in settings
    assert '13 x 28 days = 364 days' in settings
    assert 'def _with_sku_yoy_factors' in src
    assert 'pl.col("_season_block") + lag_blocks' in src
    assert 'pl.col("_season_block") + lag_blocks + 1' in src
    assert 'sku_yoy_factor_y' in src
    assert 'sku_yoy_factor_value' in src


def test_v116_seasonal_candidate_is_selected_causally_on_validation():
    src = (ROOT / "app" / "forecasting" / "leaf_fallback.py").read_text(encoding="utf-8")
    assert 'def _score_seasonal' in src
    assert 'validation_start' in src and 'validation_end' in src
    assert 'sku_yoy_seasonal' in src
    assert '((pl.col("_direct_score") - pl.col("_seasonal_score")) >= min_improvement)' in src
    assert 'sku_seasonal_multiplier_y' in src
    assert 'sku_seasonal_multiplier_value' in src


def test_v116_disables_parent_level_shape_in_production_after_v115_ab():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'LEAF_PARENT_LEVEL_SHAPE_SELECTABLE: bool = False' in settings
    assert 'LEAF_PARENT_LEVEL_SHAPE_SELECTABLE' in src
    assert 'driver_modes = tuple(m for m in driver_modes if m == "shape_only")' in src


def test_v116_acceptance_report_surfaces_seasonal_quality():
    src = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert '=== 10) CALIDAD OOS DEL FACTOR ESTACIONAL SKU 364D' in src
    assert 'sku_yoy_seasonal' in src
    assert 'sku_seasonal_multiplier_y' in src
