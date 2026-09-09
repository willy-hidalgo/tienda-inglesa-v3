from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_1_uses_three_closed_blocks_and_wmape_first():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_SELECTION_BLOCKS: int = 5' in settings
    assert 'V129_QTY_FROZEN_SELECTION_BLOCKS: int = 4' in settings
    assert 'V12_SELECTION_MIN_WINS: int = 2' in settings
    assert 'V12_BIAS_WEIGHT: float = 0.00' in settings
    assert 'def _score_closed_block' in src
    assert 'def _selection_from_closed_blocks' in src
    assert 'validation_windows' in src
    assert 'v12_validation_improvement_y' in src
    assert 'v12_validation_improvement_value' in src


def test_v12_1_has_no_harm_stability_and_portfolio_guards():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'v12_validation_max_degradation_y' in src
    assert 'v12_validation_max_degradation_value' in src
    assert 'v12_section_portfolio_gate_y' in src
    assert 'v12_section_portfolio_gate_value' in src
    assert 'section_min_improvement' in src
    assert 'section_min_wins' in src
    assert 'v12_validation_bias_v12_y' in src
    assert 'v12_validation_bias_v12_value' in src


def test_v12_1_forecast_only_uses_closed_oos_plus_prior_blocks():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'oos_closed_score = _score_closed_block' in src
    assert 'fc_score_parts = [oos_closed_score]' in src
    assert 'historical_scores[: selection_blocks - 1]' in src
    assert 'No forecast-only actual is used.' in src


def test_v12_1_validator_and_report_expose_stability():
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert 'política causal meta/portfolio' in validator
    assert '=== 13) HISTÓRICO DE SELECCIÓN + LEGACY GATE DIAGNÓSTICO v12.9.7' in report
    assert 'portfolio=' in report and 'legacy-gate(diag)=' in report


def test_v12_1_1_materializes_improvement_before_preselection():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    improvement = src.index('.alias("v12_validation_improvement_value")')
    preselect = src.index('.alias("_preselect_y")', improvement)
    assert improvement < preselect
    assert 'discovery.join(confirmation' in src
