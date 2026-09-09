from pathlib import Path

import settings


def test_v12910_version_and_promotion_settings():
    assert settings.APP_VERSION == "12.9.12"
    assert "ARTIFACT_VERSION = 20" in Path("app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert settings.V12910_VALUE_SPARSE_PROMOTION_ENABLED is True
    assert settings.V12910_VALUE_SPARSE_MIN_REPLAY_CUTOFFS == 4
    assert settings.V12910_VALUE_SPARSE_MIN_REPLAY_WIN_RATE == 1.0
    assert settings.V12910_VALUE_SPARSE_MIN_REPLAY_MEDIAN_GAIN == 0.0025
    assert settings.V12910_VALUE_SPARSE_MIN_REPLAY_WORST_GAIN == 0.0
    assert settings.V12910_VALUE_SPARSE_MIN_REPLAY_WEIGHTED_GAIN == 0.0025
    assert settings.V12910_VALUE_SPARSE_MAX_REPLAY_BIAS_WORSEN == 0.0
    assert settings.V12910_VALUE_SPARSE_MAX_SELECTED_VOLUME_SHARE == 0.30


def test_v12910_active_replay_is_causal_and_productive_only_after_gate():
    src = Path("app/forecasting/leaf_v12.py").read_text(encoding="utf-8")
    assert 'active_ev = ev.filter(pl.col("_n_v") >= active_min)' in src
    assert 'v12910_value_sparse_active_replay_weighted_gain' in src
    assert 'v12910_value_sparse_promoted' in src
    assert 'v12910_value_sparse_promotion_reason' in src
    assert 'v12910_valuehat_raw_before_sparse' in src
    assert 'v12910_value_sparse_applied' in src
    assert 'pl.col("valuehat_raw") * pl.col("v1299_value_sparse_factor")' in src
    # The held-out current OOS is not part of the training/promotion evidence.
    fn = src[src.index("def _v1299_value_sparse_segmented_diagnostic"):src.index("def _selection_from_closed_blocks")]
    assert 'older = history_scores.filter(pl.col("_validation_block") > held)' in fn
    assert 'heldout = history_scores.filter(pl.col("_validation_block") == held)' in fn


def test_v12910_validator_and_report_contract():
    validator = Path("app/forecasting/validate_v11.py").read_text(encoding="utf-8")
    report = Path("app/forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert "v12.9.10 violan promotion-gate sparse ACTIVE" in validator
    assert "v12.9.10 no coinciden entre baseline y write-back sparse" in validator
    assert "=== 33) v12.9.10 VALUE SPARSE RESCUE — ACTIVE PROMOTION GATE (PRODUCTIVO) ===" in report
    assert "_print_v12910_value_sparse_promotion(forecast)" in report
    assert "OOS ACTIVE audit" in report
