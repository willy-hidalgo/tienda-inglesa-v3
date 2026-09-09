from pathlib import Path
import settings

ROOT = Path(__file__).resolve().parents[1]


def test_v1299_version_and_value_only_sparse_settings():
    assert settings.APP_VERSION == "12.9.12"
    assert settings.V1299_VALUE_SPARSE_DIAGNOSTIC_ENABLED is True
    assert settings.V1299_VALUE_SPARSE_BUCKETS == ("07-09", "10-13")
    assert settings.V1299_VALUE_SPARSE_FACTOR_GRID[0] == 1.0
    assert settings.V1299_VALUE_SPARSE_FACTOR_GRID[-1] == 1.15
    assert settings.V1299_VALUE_SPARSE_MIN_FOLDS >= 8
    assert settings.V1299_VALUE_SPARSE_REPLAY_CUTOFFS == 4


def test_v1299_uses_exact_grid_errors_and_is_non_productive():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert "def _v1299_value_sparse_segmented_diagnostic" in src
    assert "_v1299_ae11_v_f" in src
    assert "_v1299_ae12_v_f" in src
    assert "sparse99_oos = _v1299_value_sparse_segmented_diagnostic" in src
    assert "sparse99_fc = _v1299_value_sparse_segmented_diagnostic" in src
    fn = src[src.index("def _v1299_value_sparse_segmented_diagnostic"):src.index("def _selection_from_closed_blocks")]
    assert 'pl.col("_validation_block") > held' in fn
    assert 'pl.col("_validation_block") == held' in fn
    assert "_selection_from_closed_blocks(recent" in fn
    assert "_apply_v1296_sec23_long_horizon_gate" in fn
    # Diagnostic only: no write-back to yhat/valuehat inside the v12.9.9 routine.
    assert '.alias("valuehat")' not in fn
    assert '.alias("yhat")' not in fn


def test_v1299_report_and_validator_contract():
    report = (ROOT / "app" / "forecasting_acceptance_report.py").read_text(encoding="utf-8")
    validator = (ROOT / "app" / "forecasting" / "validate_v11.py").read_text(encoding="utf-8")
    assert "=== 32) v12.9.9 VALUE SPARSE BIAS RESCUE SEGMENTADO + MULTI-CUTOFF" in report
    assert "_print_v1299_value_sparse_segmented(forecast)" in report
    assert "v1299_value_sparse_replay_pass" in report
    assert "v12.9.9 tienen metadata sparse-segmentada inválida" in validator
