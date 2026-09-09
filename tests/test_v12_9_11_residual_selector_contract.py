from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12911_version_and_settings_contract():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings
    assert "V12911_SEC23_VALUE_RESIDUAL_ENABLED: bool = True" in settings
    assert "V12911_SEC23_VALUE_MIN_FOLDS: int = 8" in settings
    assert "V12911_SEC23_VALUE_MAX_VOLUME_SHARE: float = 0.10" in settings
    assert "V12911_SEC23_VALUE_REPLAY_CUTOFFS: int = 4" in settings


def test_v12911_is_diagnostic_only_and_integrated():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert "def _v12911_sec23_value_residual_selector(" in src
    assert "residual11_oos = _v12911_sec23_value_residual_selector(" in src
    assert "residual11_fc = _v12911_sec23_value_residual_selector(" in src
    # v12.9.11 must not write back to the productive raw forecast.
    fn = src.split("def _v12911_sec23_value_residual_selector(", 1)[1].split("def _selection_from_closed_blocks(", 1)[0]
    assert 'alias("valuehat_raw")' not in fn
    assert 'alias("yhat_raw")' not in fn
    assert "current OOS actuals" in fn or "Current OOS actuals" in fn


def test_v12911_replay_is_strictly_older_than_cutoff():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    fn = src.split("def _v12911_sec23_value_residual_selector(", 1)[1].split("def _selection_from_closed_blocks(", 1)[0]
    assert 'pl.col("_validation_block") > cutoff' in fn
    assert 'pl.col("_validation_block") == cutoff + 1' in fn
    assert 'pl.col("_validation_block") == cutoff' in fn
    assert "max_share * total_w" in fn


def test_v12911_acceptance_section_34_present():
    src = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert "=== 34) v12.9.11 SEC23 VALUE RESIDUAL SELECTOR-GAP CHALLENGER" in src
    assert "_print_v12911_residual_selector_gap(forecast, metrics)" in src
    assert "OOS ACTIVE hypothetical" in src


def test_v12911_validator_contract_present():
    src = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert "v12.9.11 tienen metadata residual-selector inválida" in src
    assert "V12911_SEC23_VALUE_MAX_VOLUME_SHARE" in src
