"""Phase 5: leaf-specific Section-vs-Store diagnostics, same SES+RLS family."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

import settings
from app.forecasting.optimization import (
    build_leaf_zero_rate_diagnostics,
    recommend_leaf_parent_switches,
    summarize_leaf_parent_by_section,
)

ROOT = Path(__file__).resolve().parent.parent


def _candidate_row(
    *,
    period: str,
    candidate: str,
    wmape: float,
    bias: float = 0.0,
    n_folds: int = 4,
    positive_fold_share: float = 1.0,
    current_parent: str = "store",
) -> dict:
    den = 400.0 if period == "in_sample" else 100.0
    return {
        "unique_id": "1||T:00154||S:455436",
        "target": "Unidades",
        "period_type": period,
        "candidate_parent": candidate,
        "last_current_parent": current_parent,
        "sum_abs_error": wmape * den,
        "sum_abs_y": den,
        "sum_signed_error": bias * den,
        "n_points": 80 if period == "in_sample" else 20,
        "n_folds": n_folds,
        "median_fold_wmape": wmape,
        "worst_fold_wmape": wmape,
        "positive_fold_share": positive_fold_share,
        "wmape": wmape,
        "bias": bias,
    }


def test_parent_switch_must_be_created_by_closed_history_then_validated_by_oos():
    rows = [
        _candidate_row(period="in_sample", candidate="current_policy", wmape=0.20),
        _candidate_row(period="in_sample", candidate="section", wmape=0.10),
        _candidate_row(period="in_sample", candidate="store", wmape=0.20),
        _candidate_row(period="in_sample", candidate="ses_only", wmape=0.30),
        _candidate_row(period="out_sample", candidate="current_policy", wmape=0.20, n_folds=1),
        _candidate_row(period="out_sample", candidate="section", wmape=0.12, n_folds=1),
        _candidate_row(period="out_sample", candidate="store", wmape=0.20, n_folds=1),
        _candidate_row(period="out_sample", candidate="ses_only", wmape=0.28, n_folds=1),
    ]
    rec = recommend_leaf_parent_switches(pl.DataFrame(rows))
    assert rec.height == 1
    assert rec["candidate_parent"][0] == "section"
    assert rec["history_n_folds"][0] == 4
    assert rec["history_wmape_improvement"][0] >= 0.09
    assert rec["validation_status"][0] == "validated_parent_switch"
    assert rec["history_ses_only_wmape"][0] == 0.30


def test_oos_cannot_create_parent_candidate():
    rows = [
        _candidate_row(period="in_sample", candidate="current_policy", wmape=0.20, n_folds=1),
        _candidate_row(period="in_sample", candidate="section", wmape=0.05, n_folds=1),
        _candidate_row(period="in_sample", candidate="store", wmape=0.20, n_folds=1),
        _candidate_row(period="out_sample", candidate="current_policy", wmape=0.40, n_folds=1),
        _candidate_row(period="out_sample", candidate="section", wmape=0.01, n_folds=1),
        _candidate_row(period="out_sample", candidate="store", wmape=0.40, n_folds=1),
    ]
    rec = recommend_leaf_parent_switches(pl.DataFrame(rows))
    assert rec.height == 0


def test_parent_section_summary_is_bottom_up_from_leaf_candidate_errors():
    rows = [
        _candidate_row(period="in_sample", candidate="current_policy", wmape=0.20),
        _candidate_row(period="in_sample", candidate="section", wmape=0.10),
        _candidate_row(period="in_sample", candidate="store", wmape=0.20),
    ]
    out = summarize_leaf_parent_by_section(pl.DataFrame(rows))
    sec = out.filter(
        (pl.col("section") == "1")
        & (pl.col("target") == "Unidades")
        & (pl.col("candidate_parent") == "section")
    )
    assert sec.height == 1
    assert sec["wmape"][0] == 0.10


def _zero_row(ds: dt.date, period: str, y: float, block: int) -> dict:
    return {
        "unique_id": "1||T:00154||S:455436",
        "ds": ds,
        "period_type": period,
        "y": y,
        "rls_metric_eligible": True,
        "rls_block": block,
    }


def test_zero_rate_is_diagnostic_and_keeps_leaf_dimensions():
    frame = pl.DataFrame(
        [
            _zero_row(dt.date(2026, 1, 5), "in_sample", 0.0, 1),
            _zero_row(dt.date(2026, 1, 6), "in_sample", 2.0, 1),
            _zero_row(dt.date(2026, 4, 3), "out_sample", 0.0, 5),
            _zero_row(dt.date(2026, 4, 4), "out_sample", 0.0, 5),
            _zero_row(dt.date(2026, 4, 5), "out_sample", 5.0, 5),
        ]
    )
    out = build_leaf_zero_rate_diagnostics(frame)
    summary = out["leaf_zero_rate_summary"].filter(pl.col("period_type") == "out_sample")
    assert summary.height == 1
    assert summary["n_zero_sale"][0] == 2
    assert abs(summary["zero_rate"][0] - 2 / 3) < 1e-12
    patterns = out["leaf_zero_rate_patterns"]
    assert set(patterns["dimension"].unique().to_list()) == {"weekday", "month", "block"}


def test_phase5_is_diagnostic_only_and_does_not_add_new_model_family():
    opt = (ROOT / "app/forecasting/optimization.py").read_text(encoding="utf-8")
    leaf = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    settings_text = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert "leaf_parent_recommendation.parquet" in opt
    assert "ses_only" in opt
    assert "parent_diagnostics_out" in leaf
    assert "STAT_OPT_PARENT_MIN_WMAPE_IMPROVEMENT" in settings_text
    assert "LEAF_PARENT_OVERRIDE" not in settings_text
    assert settings.APP_VERSION == "13.3.3"
