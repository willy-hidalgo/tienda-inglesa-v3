from pathlib import Path
import settings

ROOT = Path(__file__).resolve().parent.parent


def test_v1333_leaf_yoy_settings_are_conservative_and_enabled():
    assert settings.APP_VERSION == "13.3.3"
    assert settings.LEAF_YOY_SEASONAL_ENABLED is True
    assert settings.LEAF_YOY_SEASONAL_MIN_POSITIVE_DAYS == 7
    assert settings.LEAF_YOY_SEASONAL_FULL_RELIABILITY_DAYS == 14
    assert settings.LEAF_YOY_SEASONAL_FACTOR_MIN == 0.50
    assert settings.LEAF_YOY_SEASONAL_FACTOR_MAX == 1.50


def test_v1333_leaf_yoy_is_causal_auditable_and_part_of_leaf_factor():
    text = (ROOT / "app" / "forecasting" / "leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "_attach_leaf_yoy_monthly_seasonality" in text
    assert "leaf_yoy_factor_y" in text
    assert "leaf_yoy_factor_value" in text
    assert "leaf_total_factor_y" in text
    assert "leaf_total_factor_value" in text
    assert 'rows["leaf_total_factor_y"]' in text
    assert 'rows["leaf_total_factor_value"]' in text
    assert "May-2025 / Apr-2025" in text


def test_v1333_dashboard_artifact_schema_bumped():
    text = (ROOT / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 33" in text


def test_warmup_total_factor_trace_matches_kernel_identity():
    text = (ROOT / "app" / "forecasting" / "leaf_ses_rls.py").read_text(encoding="utf-8")
    assert 'pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_total_factor_y"))' in text
    assert 'pl.when(pl.col("_warmup")).then(1.0).otherwise(pl.col("leaf_total_factor_value"))' in text
