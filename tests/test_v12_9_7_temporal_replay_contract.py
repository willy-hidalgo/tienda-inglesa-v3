from pathlib import Path


def test_v1297_version_and_settings_contract():
    s = Path("settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in s
    assert "V1297_TEMPORAL_REPLAY_ENABLED: bool = True" in s
    assert "V1297_TEMPORAL_REPLAY_MAX_CUTOFFS: int = 4" in s
    assert "V1297_TEMPORAL_REPLAY_MIN_TRAIN_FOLDS: int = 8" in s


def test_v1297_strict_temporal_replay_contract():
    src = Path("app/forecasting/leaf_v12.py").read_text(encoding="utf-8")
    assert "def _v1297_sec23_temporal_replay" in src
    assert 'train = hist.filter(pl.col("_validation_block") > cutoff)' in src
    assert 'valid = hist.filter(pl.col("_validation_block") == cutoff)' in src
    assert 'prior = hist.filter(pl.col("_validation_block") == cutoff + 1)' in src
    assert 'pl.col("_den_v").fill_null(0.0).alias("_rp_weight")' in src
    assert 'Never use block `cutoff` to construct a' in src
    assert "v1297_temporal_replay_fold_gains_value" in src
    assert "v1297_temporal_replay_pass_value" in src
    assert "replay_oos = _v1297_sec23_temporal_replay(stress_scores, oos_profile)" in src
    assert "replay_fc = _v1297_sec23_temporal_replay(stress_scores, fc_profile)" in src
    # v12.9.7 is diagnostic only; v12.9.6 gate still owns production selection.
    assert "select_oos = _apply_v1296_sec23_long_horizon_gate(select_oos, oos_scores)" in src


def test_v1297_report_and_validator_contract():
    report = Path("app/forecasting_acceptance_report.py").read_text(encoding="utf-8")
    validate = Path("app/forecasting/validate_v11.py").read_text(encoding="utf-8")
    assert "=== 30) v12.9.7 MULTI-CUTOFF TEMPORAL ROBUSTNESS REPLAY" in report
    assert "_print_v1297_temporal_replay(forecast, metrics)" in report
    assert 'pl.col("metric_cohort") == "active"' in report
    assert "v1297_temporal_replay_cutoffs_value" in validate
    assert "metadata temporal-replay inválida" in validate
    assert 'v1296_stress_gate_enabled_value' in validate
    assert 'v12_candidate_available_value' in validate
