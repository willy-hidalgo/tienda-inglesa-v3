from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_6_version_and_dependency_contract():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_SKU_SHAPE_LGBM_ENABLED: bool = True' in settings
    assert 'V12_SKU_SHAPE_LGBM_HISTORY_BLOCKS: int = 16' in settings
    assert 'V12_SKU_SHAPE_LGBM_GAMMA: float = 1.00' in settings
    assert '"lightgbm>=4.6.0"' in pyproject


def test_v12_6_shape_layer_is_pooled_not_leaf_model():
    src = (ROOT / "app" / "forecasting" / "sku_shape_lgbm.py").read_text(encoding="utf-8")
    assert 'class PooledLGBMShapeContext' in src
    assert 'objective": "huber"' in src
    assert 'num_boost_round=int(getattr(settings, "V12_SKU_SHAPE_LGBM_ROUNDS", 160))' in src
    assert 'over("_v12_sku")' in src
    assert 'SKU×store' in src


def test_v12_6_preserves_sku_total_and_uses_closed_blocks():
    src = (ROOT / "app" / "forecasting" / "sku_shape_lgbm.py").read_text(encoding="utf-8")
    leaf = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'raw * base_total / np.maximum(raw_sum, EPS)' in src
    assert 'v12.6 shape invariant violated' in src
    assert 'for i in range(1, self.history_blocks + 1)' in src
    assert 'sku_forecast_override' in leaf
    assert 'shape_context.corrected_forecast(start)' in leaf


def test_v12_6_shape_metadata_propagates_to_forecast():
    leaf = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    for col in (
        'v12_sku_forecast_base_y',
        'v12_sku_forecast_base_value',
        'v12_sku_shape_model_y',
        'v12_sku_shape_model_value',
        'v12_sku_shape_training_rows_y',
        'v12_sku_shape_training_rows_value',
    ):
        assert col in leaf


def test_v12_6_validator_and_acceptance_audit_shape_total():
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert 'SKU/horizonte v12.8 violan total28 base == LGBM-shape' in validator
    assert '=== 18) v12.8 POOLED LIGHTGBM — SHAPE SKU-TOTAL ===' in report
    assert 'total28 bad=' in report
