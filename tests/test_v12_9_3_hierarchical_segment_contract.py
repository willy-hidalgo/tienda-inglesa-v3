from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_v1293_version_and_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert "V1293_VALUE_SEGMENT_SELECTOR_ENABLED: bool = True" in settings
    assert "V1293_VALUE_SEGMENT_FOLDS: int = 3" in settings
    assert "V1293_VALUE_SECTION_STRONG_GAIN" in settings

def test_v1293_hierarchy_and_backoff_contract():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert "def _fit_value_hierarchical_segment_selector" in src
    assert '("L4", ["_seg_section", "_seg_level_method_value", "_seg_seasonal", "_seg_sales_bucket"]' in src
    assert '("L3", ["_seg_section", "_seg_level_method_value", "_seg_seasonal"]' in src
    assert '("L2", ["_seg_section", "_seg_level_method_value"]' in src
    assert '("L1", ["_seg_section"]' in src
    assert 'v11_insufficient_or_unstable_segment' in src
    assert 'v1293_segment_budget_selected_value' in src

def test_v1293_impact_percentile_semantics():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert '1.0 = highest-impact leaf' in src
    assert 'pct[order] = (n_pct - np.arange(score.height, dtype=np.float64)) / n_pct' in src

def test_v1293_validator_and_acceptance():
    validate = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert 'segment_ok_v' in validate
    assert 'v1293_segment_selected_volume_share_value' in validate
    assert '=== 26) v12.9.7 HIERARCHICAL SEGMENT WALK-FORWARD SELECTOR ===' in report
    assert 'Top segmentos por impacto OOS' in report
