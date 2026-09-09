from pathlib import Path


def test_v1296_settings_contract():
    s = Path('settings.py').read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in s
    assert 'V1296_SEC23_MIN_FOLDS: int = 8' in s
    assert 'V1296_SEC23_MIN_P10_GAIN: float = -0.01' in s
    assert 'V1296_SEC23_MAX_VOLUME_SHARE: float = 0.15' in s
    assert 'V1296_SEC23_ALLOW_L3_BACKOFF: bool = True' in s


def test_v1296_product_gate_contract():
    src = Path('app/forecasting/leaf_v12.py').read_text(encoding="utf-8")
    assert 'def _apply_v1296_sec23_long_horizon_gate' in src
    assert 'v11_broad_level_not_productive' in src
    assert 'v11_long_horizon_tail_risk' in src
    assert 'v11_high_impact_tail_guard' in src
    assert 'v12_long_horizon_specific_pass' in src
    assert 'v11_stress_pass_budget_safety' in src
    assert 'select_oos = _apply_v1296_sec23_long_horizon_gate(select_oos, oos_scores)' in src
    assert 'pl.col("v1295_stress_level_value") == "L4"' in src
    assert 'pl.col("v1295_stress_level_value") == "L3"' in src


def test_v1296_report_and_validator_contract():
    report = Path('app/forecasting_acceptance_report.py').read_text(encoding="utf-8")
    validate = Path('app/forecasting/validate_v11.py').read_text(encoding="utf-8")
    assert '=== 29) v12.9.7 SEC23 L4 LONG-HORIZON STRESS-GATED SELECTOR ===' in report
    assert '_print_v1296_sec23_stress_gate(forecast, metrics)' in report
    assert 'violan stress-gate productivo Sec23' in validate
    assert 'no coinciden entre familia final y budget stress-gate Sec23' in validate
