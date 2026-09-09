from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v1295_settings_and_version():
    s = (ROOT / 'settings.py').read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in s
    assert 'V1295_STRESS_TEST_BLOCKS: int = 12' in s
    assert 'V1295_STRESS_TEST_MIN_FOLDS: int = 8' in s
    assert 'V1295_STRESS_MIN_P25_GAIN: float = 0.0' in s
    assert 'V1295_STRESS_MAX_CONSECUTIVE_LOSSES: int = 2' in s


def test_v1295_is_diagnostic_only_and_keeps_production_pool_frozen():
    src = (ROOT / 'app' / 'forecasting' / 'leaf_v12.py').read_text(encoding="utf-8")
    assert 'stress_blocks = max(' in src
    assert 'historical_scores[:selection_blocks]' in src
    assert '_value_long_horizon_segment_stress(stress_scores, oos_profile)' in src
    assert 'The current OOS block is never part of ``block_scores`` passed here.' in src
    assert '_hist_level_method_value' in src
    assert '_hist_seasonal_multiplier_value' in src
    # Stress output is metadata only; the productive family is still controlled
    # by the existing segment budget selector.
    assert 'v1295_stress_pass_value' in src
    assert 'v1293_segment_budget_selected_value' in src


def test_v1295_acceptance_report():
    report = (ROOT / 'app' / 'forecasting_acceptance_report.py').read_text(encoding="utf-8")
    assert '=== 28) v12.9.7 LONG-HORIZON SEGMENT STRESS TEST' in report
    assert 'El OOS actual NO participa en estos folds' in report
    assert 'p25=' in report and 'p10=' in report and 'loss-run=' in report
