from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_2_true_hurdle_does_not_multiply_share_by_probability():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'v12_occurrence_gate_open_y' in src
    assert '.then(pl.col("_share_28x84_y"))' in src
    assert '.then(pl.col("_share_28x84_v"))' in src
    assert '_share_28x84_y") * pl.max_horizontal' not in src
    assert '_share_28x84_v") * pl.max_horizontal' not in src
    assert 'V12_OCCURRENCE_WEIGHT_FLOOR' not in src


def test_v12_2_requires_recent_leaf_and_portfolio_win():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'V12_REQUIRE_RECENT_WIN: bool = True' in settings
    assert 'V12_RECENT_MIN_IMPROVEMENT: float = 0.02' in settings
    assert 'V12_SECTION_REQUIRE_RECENT_WIN: bool = True' in settings
    assert 'v12_validation_recent_improvement_y' in src
    assert 'v12_validation_recent_improvement_value' in src
    assert 'v12_section_portfolio_recent_gain_y' in src
    assert 'v12_section_portfolio_recent_gain_value' in src
    assert '_validation_block") == 1' in src


def test_v12_2_validator_checks_true_hurdle_and_recent_win():
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert 'true-hurdle: gate cerrado con share positivo' in validator
    assert 'v12_validation_recent_improvement_y' in validator
    assert 'v12_section_portfolio_recent_gain_y' in validator
    assert 'política causal meta/portfolio' in validator


def test_v12_2_acceptance_exposes_selection_counterfactual():
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert '=== 13) HISTÓRICO DE SELECCIÓN + LEGACY GATE DIAGNÓSTICO v12.9.7' in report
    assert '=== 14) CONTRAFACTUAL SOBRE LAS MISMAS HOJAS SELECCIONADAS v12.8' in report
    assert 'v12_validation_recent_improvement_y' in report
    assert '_print_v12_selected_counterfactual' in report
