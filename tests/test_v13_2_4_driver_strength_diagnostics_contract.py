"""Phase 4 diagnostics tune parameters only; production remains SES+RLS."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

import settings
from app.forecasting.optimization import (
    build_calendar_day_recurrence_diagnostics,
    build_leaf_driver_strength_diagnostics,
    build_leaf_residual_diagnostics,
)

ROOT = Path(__file__).resolve().parent.parent


def _strength_row(period: str, block: int, y: float, effect: float) -> dict:
    return {
        "unique_id": "1||T:00001||S:A",
        "period_type": period,
        "rls_metric_eligible": True,
        "rls_block": block,
        "y": y,
        "value": y,
        "ses_level_y": 100.0,
        "ses_level_value": 100.0,
        "parent_model_y": "store",
        "parent_model_value": "store",
        "driver_effect": effect,
        "driver_effect_value": 0.0,
    }


def test_gamma_diagnostic_can_validate_shrinkage_without_changing_production():
    # gamma=1 doubles a level that is already correct; gamma=0 keeps SES=100.
    effect = 0.6931471805599453
    frame = pl.DataFrame([
        _strength_row("in_sample", 1, 100.0, effect),
        _strength_row("in_sample", 2, 100.0, effect),
        _strength_row("in_sample", 3, 100.0, effect),
        _strength_row("in_sample", 4, 100.0, effect),
        _strength_row("out_sample", 5, 100.0, effect),
    ])
    out = build_leaf_driver_strength_diagnostics(frame)
    rec = out["driver_strength_recommendation"].filter(
        (pl.col("target") == "Unidades") & (pl.col("parent_model") == "store")
    )
    assert rec.height == 1
    assert rec["gamma"][0] == 0.0
    assert rec["history_wmape_improvement"][0] > 0.5
    assert rec["holdout_wmape_improvement"][0] > 0.5
    assert rec["validation_status"][0] == "validated_calibration"


def _residual_row(ds: dt.date, period: str, y: float, yhat: float) -> dict:
    return {
        "unique_id": "1||T:00154||S:455436",
        "ds": ds,
        "period_type": period,
        "rls_metric_eligible": True,
        "y": y,
        "yhat": yhat,
        "yhat_raw": yhat,
        "value": y,
        "valuehat": yhat,
        "valuehat_raw": yhat,
        "ses_level_y": 50.0,
        "ses_level_value": 50.0,
        "initial_level_y": 50.0,
        "initial_level_value": 50.0,
        "ses_alpha_y": 0.01,
        "ses_alpha_value": 0.01,
        "parent_model_y": "store",
        "parent_model_value": "store",
        "parent_wmape_y": 0.10,
        "parent_wmape_value": 0.10,
        "driver_factor_y": 1.0,
        "driver_factor_value": 1.0,
        "driver_effect": 0.0,
        "driver_effect_value": 0.0,
    }


def test_residual_taxonomy_separates_zero_sale_error():
    frame = pl.DataFrame([
        _residual_row(dt.date(2026, 3, 1), "in_sample", 50.0, 50.0),
        _residual_row(dt.date(2026, 4, 3), "out_sample", 0.0, 50.0),
        _residual_row(dt.date(2026, 4, 4), "out_sample", 1000.0, 50.0),
    ])
    out = build_leaf_residual_diagnostics(frame)
    summary = out["leaf_residual_summary"].filter(
        (pl.col("period_type") == "out_sample") & (pl.col("target") == "Unidades")
    )
    assert summary.height == 1
    assert summary["n_zero_sale_extreme"][0] == 1
    assert 0.0 < summary["zero_sale_error_share"][0] < 1.0
    assert summary["wmape_all_points"][0] > summary["wmape_official"][0]


def _calendar_row(ds: dt.date, period: str, sku: str, y: float, yhat: float) -> dict:
    return {
        "unique_id": f"23||T:00005||S:{sku}",
        "ds": ds,
        "period_type": period,
        "rls_metric_eligible": True,
        "y": y,
        "yhat": yhat,
        "value": y,
        "valuehat": yhat,
    }


def test_calendar_recurrence_uses_same_month_day_closed_history():
    frame = pl.DataFrame([
        _calendar_row(dt.date(2025, 12, 31), "in_sample", "A", 100.0, 100.0),
        _calendar_row(dt.date(2025, 12, 31), "in_sample", "B", 0.0, 100.0),
        _calendar_row(dt.date(2026, 12, 31), "out_sample", "A", 100.0, 100.0),
        _calendar_row(dt.date(2026, 12, 31), "out_sample", "B", 0.0, 300.0),
    ])
    out = build_calendar_day_recurrence_diagnostics(frame)
    rec = out["calendar_day_recurrence"].filter(
        (pl.col("section") == "23")
        & (pl.col("target") == "Unidades")
        & (pl.col("month_day") == "12-31")
    )
    assert rec.height == 1
    assert rec["history_years"][0] == 1
    assert rec["holdout_wmape_all_points"][0] > rec["history_median_wmape_all_points"][0]
    assert rec["recurrence_signal"][0] == "holdout_regime_shift"


def test_phase4_is_diagnostic_only():
    text = (ROOT / "app/forecasting/optimization.py").read_text(encoding="utf-8")
    assert "STAT_OPT_DRIVER_STRENGTH_CANDIDATES" in text
    assert "driver_strength_recommendation.parquet" in text
    assert "calendar_day_recurrence.parquet" in text
    assert "LEAF_DRIVER_STRENGTH" not in (ROOT / "settings.py").read_text(encoding="utf-8")
    assert settings.APP_VERSION == "13.3.3"
