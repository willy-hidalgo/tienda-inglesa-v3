from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_version_and_challenger_settings():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert 'V12_LEAF_CHALLENGER_ENABLED: bool = True' in settings
    assert 'V12_HISTORY_DAYS: int = 84' in settings
    assert 'V12_MIN_VALIDATION_SALES: int = 7' in settings
    assert 'V12_MIN_IMPROVEMENT: float = 0.02' in settings
    assert 'V12_OCCURRENCE_GATE: float = 0.10' in settings


def test_v12_history_is_strictly_prior_to_origin():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'def _history_until' in src
    assert 'pl.col("ds") < pl.lit(origin)' in src
    assert 'validation_windows' in src
    assert 'select_oos = _selection_from_closed_blocks' in src
    assert 'select_fc = _selection_from_closed_blocks' in src
    assert 'V12_SELECTION_BLOCKS' in src


def test_v12_builds_sku_total_occurrence_and_normalized_store_share():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'v12_sku_forecast_y' in src
    assert 'v12_occurrence_prob_y' in src
    assert 'v12_store_share_y' in src
    assert 'pl.col("_alloc_weight_y").sum().over(["_v12_sku", "ds"])' in src
    assert 'pl.col("_share_base_y").sum().over(["_v12_sku", "ds"])' in src
    assert 'v12_candidate_yhat_raw' in src
    assert 'v12_sku_forecast_y") * pl.col("v12_store_share_y' in src


def test_v12_is_challenger_not_blind_replacement():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert 'v11_yhat_raw_before_v12' in src
    assert 'v12_validation_wmape_v11_y' in src
    assert 'v12_validation_wmape_v12_y' in src
    assert 'v12_selected_y' in src
    assert 'v12_validation_improvement_y' in src
    assert 'v12_validation_wins_y' in src
    assert 'v12_section_portfolio_gate_y' in src
    assert 'leaf_model_family_y' in src


def test_runner_integrates_v12_after_v11_fallbacks():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    fallback_pos = src.index('rows = apply_robust_leaf_fallbacks(rows, horizons)')
    v12_pos = src.index('rows = apply_v12_occurrence_share_challenger(')
    assert v12_pos > fallback_pos
    assert 'challenger v12.6 LGBM-shape+occurrence+share' in src
    assert 'v12 store-share invariant violated' in src


def test_validator_and_acceptance_understand_v12_formula():
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert 'leaf_model_family_y' in validator
    assert 'identidad v12 SKU-total × store-share' in validator
    assert 'store-share=1' in validator
    assert '=== 11) CALIDAD OOS POR FAMILIA v11 vs v12' in report
    assert '=== 12) A/B CONTRAFACTUAL OOS v12' in report
    assert '=== 13) HISTÓRICO DE SELECCIÓN + LEGACY GATE DIAGNÓSTICO v12.9.7' in report
    assert 'v12_all' in report and 'final_selected' in report
