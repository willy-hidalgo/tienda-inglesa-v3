from pathlib import Path

import settings


def test_v13215_dormancy_and_anchor_settings():
    assert settings.APP_VERSION == "13.3.3"
    assert settings.LEAF_GAP_DORMANT_OBSERVABLE_DAYS == 28
    assert 0 < settings.LEAF_GAP_DORMANT_FACTOR <= 1e-12
    assert settings.LEAF_OOS_STATE_ANCHOR_MIN_FACTOR == 0.75
    assert settings.LEAF_OOS_STATE_ANCHOR_MAX_FACTOR == 1.25


def test_v13215_kernel_has_causal_dormancy_release():
    text = Path("app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "gap_since_positive_y >= dormant_days" in text
    assert "candidate_decay_y = dormant_factor" in text
    assert "last_positive_seq_y = event_seq" in text
    assert "last_positive_seq_v = event_seq" in text
    assert "frozen_gap_factor_y" not in text
    assert "frozen_gap_factor_v" not in text


def test_v13215_cross_cadence_separates_long_gap_regime():
    text = Path("app/forecasting/regression_gate.py").read_text(encoding="utf-8")
    assert "cadence_for_cross" in text
    assert "long_gap_keys" in text
    assert 'how="anti"' in text
    assert "dedicated long-gap gate" in text


def test_v13215_artifact_version():
    text = Path("app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 33" in text
