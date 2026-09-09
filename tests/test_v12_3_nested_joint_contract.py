from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_3_version_and_four_block_nested_selector():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_SELECTION_BLOCKS: int = 5' in settings
    assert 'V129_QTY_FROZEN_SELECTION_BLOCKS: int = 4' in settings
    assert 'discovery_scores = block_scores.filter(pl.col("_validation_block") > 1)' in src
    assert 'confirm_scores = block_scores.filter(pl.col("_validation_block") == 1)' in src
    assert 'discovery.join(confirmation' in src


def test_v12_8_uses_target_specific_family_switch_with_shared_occurrence():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert 'V12_REQUIRE_JOINT_TARGET_WIN: bool = False' in settings
    assert 'v12_joint_target_preselect' in src
    assert 'v12_section_portfolio_gate_joint' in src
    assert 'política causal meta/portfolio' in validator


def test_v12_3_uses_one_shared_occurrence_gate_for_y_and_value():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert 'v12_occurrence_prob_joint' in src
    assert 'v12_occurrence_gate_open_joint' in src
    assert '(pl.col("v12_occurrence_prob_joint") >= gate).alias("v12_occurrence_gate_open_y")' in src
    assert '(pl.col("v12_occurrence_prob_joint") >= gate).alias("v12_occurrence_gate_open_value")' in src
    assert 'shared hurdle quantity/value' in validator


def test_v12_3_acceptance_reports_selected_only_stability_and_joint_consistency():
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert '=== 13) HISTÓRICO DE SELECCIÓN + LEGACY GATE DIAGNÓSTICO v12.9.7' in report
    assert 'sel = x.filter(pl.col(selected).fill_null(False))' in report
    assert '=== 14) CONTRAFACTUAL SOBRE LAS MISMAS HOJAS SELECCIONADAS v12.8' in report
    assert '=== 15) SELECCIÓN TARGET-SPECIFIC v12.8 + OCCURRENCE COMPARTIDO' in report
