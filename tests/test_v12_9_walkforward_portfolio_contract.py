from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_v129_version_and_walkforward_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_SELECTION_BLOCKS: int = 5' in settings
    assert 'V129_VALUE_WALK_FORWARD_ENABLED: bool = True' in settings
    assert 'V129_VALUE_WF_FOLDS: int = 3' in settings
    assert 'V129_QTY_FROZEN_SELECTION_BLOCKS: int = 4' in settings

def test_v129_walkforward_is_value_only_and_causal():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'def _calibrate_value_walkforward_policy' in src
    assert 'label_block = fold + 1' in src
    assert 'train_blocks = tuple(range(fold + 2' in src
    assert 'qty_scores = block_scores.filter(pl.col("_validation_block") <= qty_blocks)' in src
    assert 'policy_y = _calibrate_meta_policy(qty_scores, "y")' in src
    assert 'policy_v = _calibrate_value_walkforward_policy(block_scores)' in src

def test_v129_stability_guards_and_reporting():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'win_rate + 1e-12 >= min_win_rate' in src
    assert 'median_gain + 1e-12 >= min_median_gain' in src
    assert 'worst_gain + 1e-12 >= -max_worst_degradation' in src
    assert 'bias_worsen_max <= max_bias_worsen + 1e-12' in src
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert '=== 23) v12.9.2 VALUE WALK-FORWARD PORTFOLIO SELECTOR ===' in report
    artifacts = (ROOT / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")
    assert 'ARTIFACT_VERSION = 20' in artifacts
    assert '"v129_value_wf_reason"' in artifacts
