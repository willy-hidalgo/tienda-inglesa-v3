"""Residual diagnostics v13.2.3 stay inside the SES+RLS architecture."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

import settings
from app.forecasting.optimization import audit_promoted_exclusions, build_leaf_residual_diagnostics


ROOT = Path(__file__).resolve().parent.parent


def _row(ds: dt.date, period: str, y: float, yhat: float, value: float, valuehat: float) -> dict:
    return {
        "unique_id": "1||T:00154||S:455436",
        "ds": ds,
        "period_type": period,
        "rls_metric_eligible": True,
        "y": y,
        "yhat": yhat,
        "yhat_raw": yhat,
        "value": value,
        "valuehat": valuehat,
        "valuehat_raw": valuehat,
        "ses_level_y": 50.0,
        "ses_level_value": 20.0,
        "initial_level_y": 50.0,
        "initial_level_value": 20.0,
        "ses_alpha_y": 0.01,
        "ses_alpha_value": 0.01,
        "parent_model_y": "store",
        "parent_model_value": "store",
        "parent_wmape_y": 0.10,
        "parent_wmape_value": 0.10,
        "driver_factor_y": 1.0,
        "driver_factor_value": 250.0,
        "driver_effect": 0.0,
        "driver_effect_value": 5.521460917862246,
        "seccion": "1",
        "sku_desc": "SKU test",
        "store_name": "HIPERPREC",
    }


def test_extreme_leaf_residuals_detect_both_under_and_overforecast():
    frame = pl.DataFrame(
        [
            _row(dt.date(2026, 3, 1), "in_sample", 50.0, 50.0, 20.0, 20.0),
            _row(dt.date(2026, 4, 3), "out_sample", 1000.0, 50.0, 20.0, 5000.0),
        ]
    )
    out = build_leaf_residual_diagnostics(frame)
    extremes = out["leaf_residual_extremes"]
    assert extremes.height == 2
    assert set(extremes["target"].to_list()) == {"Unidades", "Valor ($)"}
    assert set(extremes["error_direction"].to_list()) == {"underforecast", "overforecast"}
    assert extremes["unexplained_driver_effect"].is_not_null().all()

    summary = out["leaf_residual_summary"].filter(pl.col("period_type") == "out_sample")
    assert summary.height == 2
    assert summary["n_extreme"].to_list() == [1, 1]


def test_phase3_is_diagnostic_only_and_report_version_is_dynamic():
    text = (ROOT / "app/forecasting/optimization.py").read_text(encoding="utf-8")
    assert "settings.APP_VERSION" in text
    assert "leaf_residual_extremes.parquet" in text
    assert "required_driver_effect" in text
    assert "unexplained_driver_effect" in text
    assert "settings.RLS_DRIVER_GROUP_EXCLUSIONS =" not in text
    assert settings.APP_VERSION == "13.2.6"


def test_promoted_exclusion_is_reaudited_after_parameter_reselection():
    rows = []
    for period, improvement in (("in_sample", -0.012), ("out_sample", -0.004)):
        rows.append({
            "section": "1",
            "node_level": "tienda",
            "unique_id": "1||T:00211",
            "target": "Unidades",
            "period_type": period,
            "driver_group": "month",
            "test_action": "add",
            "dynamics": "base",
            "lambda": 0.9975,
            "weighted_wmape_improvement": improvement,
            "positive_fold_share": 0.20,
            "weighted_abs_bias_change": 0.0,
            "decision_signal": "keep_current",
            "n_folds": 12 if period == "in_sample" else 1,
        })
    out = audit_promoted_exclusions(pl.DataFrame(rows))
    assert out.height == 1
    assert out["reaudit_status"][0] == "promotion_stable"
