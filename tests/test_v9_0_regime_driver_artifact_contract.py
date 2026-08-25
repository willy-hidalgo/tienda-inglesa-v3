from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]

def test_v9_version_and_driver_grid():
    settings = read("settings.py")
    m = leaf_method()
    assert 'APP_VERSION: str = "9.1"' in settings
    assert 'LEAF_DRIVER_STRENGTH_CANDIDATES' in settings
    assert '{"_parent": "none", "_strength": 0.0}' in m
    for token in ("0.25", "0.50", "0.75", "1.00"):
        assert token in settings

def test_driver_strength_preserves_arithmetic_mean_one():
    factors = [0.68, 0.75, 0.90, 1.00, 1.10, 1.57]
    mean = sum(factors) / len(factors)
    factors = [x / mean for x in factors]
    for strength in (0.0, 0.25, 0.50, 0.75, 1.0):
        adjusted = [1.0 + strength * (f - 1.0) for f in factors]
        assert abs(sum(adjusted) / len(adjusted) - 1.0) < 1e-12

def test_real_587833_oos_full_driver_is_worse_than_pure_ses_shape():
    fixture = json.loads(
        (ROOT / "tests/fixtures/leaf_587833_00001_oos_v8_9.json")
        .read_text(encoding="utf-8")
    )
    rows = fixture["rows"]
    actual = [float(r["value"]) for r in rows]
    level = float(rows[0]["ses_level_value"])
    factors = [float(r["driver_factor_value"]) for r in rows]
    den = sum(abs(y) for y in actual)

    def wmape(strength):
        pred = [level * (1.0 + strength * (f - 1.0)) for f in factors]
        return sum(abs(y - p) for y, p in zip(actual, pred)) / den

    assert wmape(0.0) < wmape(1.0)
    assert wmape(0.0) < wmape(0.25) < wmape(0.50) < wmape(0.75)
    assert 1.70 < wmape(1.0) < 1.80

def test_driver_candidate_score_is_recently_weighted():
    m = leaf_method()
    settings = read("settings.py")
    assert "LEAF_DRIVER_SCORE_DECAY" in settings
    assert 'pl.lit(driver_score_decay)' in m
    assert '* pl.col("_cae_parent_v")' in m

def test_regime_selector_uses_14_28_112_without_future_actuals():
    m = leaf_method()
    settings = read("settings.py")
    for token in (
        "LEAF_SES_SCORE_DECAY",
        "LEAF_REGIME_DENSE_COVERAGE",
        "LEAF_REGIME_SHOCK_RATIO",
        "LEAF_REGIME_DECLINE_RATIO",
        "LEAF_REGIME_RECENT14_WEIGHT",
    ):
        assert token in settings
    for token in (
        "_recent14_y", "_recent28_y", "_reference_y",
        "_regime_anchor_y", "_coverage_y",
        "_recent14_v", "_recent28_v", "_reference_v",
        "_regime_anchor_v", "_coverage_v",
    ):
        assert token in m
    assert m.index("chosen_frames.append(") < m.index("# ── Score ALL alpha candidates")

def test_sparse_surge_uses_long_reference_but_decline_can_move_anchor_down():
    dense_threshold = 0.85
    shock_ratio = 1.50
    decline_ratio = 0.60
    w14 = 0.65

    def anchor(r14, r28, r112, coverage):
        if r112 > 1e-9 and r28 < r112 * decline_ratio:
            return w14 * r14 + (1.0 - w14) * r28
        if coverage >= dense_threshold:
            if r28 > 1e-9 and r14 > r28 * shock_ratio:
                return r28
            return w14 * r14 + (1.0 - w14) * r28
        return r112

    assert anchor(2200, 1430, 512, 11/28) == 512
    falling = anchor(27, 477, 2115, 6/28)
    assert falling < 250

def test_artifact_fingerprint_is_strict_and_versioned():
    art = read("app/dashboard_artifacts.py")
    dash = read("app/dashboard.py")
    audit = read("app/dashboard_consistency.py")
    assert "ARTIFACT_VERSION = 6" in art
    for token in ("mtime_ns", "size_bytes", "app_version", "n_rows"):
        assert token in art
    assert "artifacts_match_source" in art
    assert "st_mtime_ns" in dash
    assert "Por seguridad no se muestran" in dash
    assert "audit_metric_sample" in audit
    assert "forecast row count=" in audit

def test_leaf_outputs_driver_strength_and_regime_audit_columns():
    m = leaf_method()
    for token in (
        "driver_strength_y",
        "driver_strength_value",
        "ses_recent28_y",
        "ses_recent28_value",
        "ses_recent14_y",
        "ses_recent14_value",
        "ses_regime_anchor_y",
        "ses_regime_anchor_value",
    ):
        assert token in m

def test_direction_guard_reduces_conflicting_dense_driver_shape_only():
    m = leaf_method()
    settings = read("settings.py")
    for token in (
        "LEAF_DRIVER_DIRECTION_CONFLICT_RATIO",
        "LEAF_DRIVER_RECENT_TREND_FLOOR",
        "LEAF_DRIVER_RECENT_TREND_CEILING",
        "LEAF_DRIVER_DIRECTION_GUARD_MAX_STRENGTH",
    ):
        assert token in settings
    assert "_driver_trend_y" in m and "_driver_trend_v" in m
    assert "_recent_trend_y" in m and "_recent_trend_v" in m
    assert "_direction_guard_y" in m and "_direction_guard_v" in m
    assert "_effective_strength_y" in m and "_effective_strength_v" in m
    # Level equation remains SES × adjusted factor; the guard never writes _level.
    guard = m[m.index("# Direction guard"):m.index('alias("_factor_y_selected")')]
    assert 'alias("_level_y")' not in guard
    assert 'alias("_level_v")' not in guard

def test_direction_guard_dampens_the_real_587833_declining_shape():
    fixture = json.loads(
        (ROOT / "tests/fixtures/leaf_587833_00001_oos_v8_9.json")
        .read_text(encoding="utf-8")
    )
    factors = [float(r["driver_factor_value"]) for r in fixture["rows"]]
    first7 = sum(factors[:7]) / 7
    last7 = sum(factors[-7:]) / 7
    trend = last7 / first7
    assert trend < 0.85

    # If a dense SKU had a flat/rising recent actual trend, v9.0 would cap
    # a selected 100% shape to 25%, cutting the directional amplitude by 75%.
    selected_strength = 1.0
    effective = min(selected_strength, 0.25)
    full_first = 1 + selected_strength * (first7 - 1)
    full_last = 1 + selected_strength * (last7 - 1)
    damp_first = 1 + effective * (first7 - 1)
    damp_last = 1 + effective * (last7 - 1)
    assert abs(damp_last - damp_first) < abs(full_last - full_first)

