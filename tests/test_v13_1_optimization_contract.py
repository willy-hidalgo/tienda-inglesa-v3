"""Contracts for v13.1 statistical tuning inside the same SES+RLS model."""
from pathlib import Path
import settings

ROOT = Path(__file__).resolve().parent.parent


def test_optimization_never_auto_promotes_or_adds_model_families():
    assert settings.STAT_OPTIMIZATION_ENABLED is True
    assert settings.STAT_OPTIMIZATION_PROMOTE_AUTOMATICALLY is False
    text = (ROOT / "app/forecasting/optimization.py").read_text(encoding="utf-8").lower()
    for forbidden in ("lightgbm", "occurrence/share", "meta-selector", "randomforest", "xgboost"):
        assert forbidden not in text


def test_forecast_cli_exposes_optional_diagnostics_without_changing_default_path():
    text = (ROOT / "app/forecasts.py").read_text(encoding="utf-8")
    assert '"--optimization-diagnostics"' in text
    assert "optimization_diagnostics=(" in text
    assert "STAT_OPTIMIZATION_ENABLED" in text
    assert "write_optimization_artifacts" in text


def test_ses_diagnostics_use_same_walkforward_kernel_and_holdout_is_separate():
    text = (ROOT / "app/forecasting/leaf_ses_rls.py").read_text(encoding="utf-8")
    assert "diag_ae_y" in text
    assert "diag_den_y" in text
    assert 'period_names = ("in_sample", "out_sample")' in text
    # Forecast-only has no actual support for tuning.
    assert "collect_diagnostics == 1 and pc <= 1" in text
    # Productive forecast remains the same SES+RLS identity, with the v13.2.17
    # causal leaf magnitude guard applied after reconstruction.
    assert "py = max(np.expm1(np.log1p(max(ly, 0.0)) + np.log(fy)), 0.0)" in text
    assert "if cap_y > 0.0 and py > cap_y" in text


def test_rls_diagnostics_score_only_existing_candidates_and_existing_driver_groups():
    text = (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    assert "for c in candidates:" in text
    assert 'for period_name in ("in_sample", "out_sample")' in text
    assert 'group = "price"' in text
    assert 'group = "weekday"' in text
    assert 'group = "month"' in text
    assert 'group = "holiday_other"' in text
    assert '"zero_contribution_no_refit"' in text


def test_optimization_artifacts_do_not_touch_dashboard_or_pandas():
    text = (ROOT / "app/forecasting/optimization.py").read_text(encoding="utf-8").lower()
    assert "pandas" not in text
    assert "streamlit" not in text
    assert "promote" not in text or "no se promueve" in text or "never changes" in text
    dash = (ROOT / "app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 30" in dash


def test_optimization_summary_polars_smoke():
    import polars as pl
    from app.forecasting.optimization import (
        recommend_ses,
        summarize_driver_screen,
        summarize_ses,
    )

    ses = pl.DataFrame(
        {
            "section": ["1", "1", "1", "1"],
            "target": ["Unidades"] * 4,
            "period_type": ["in_sample"] * 4,
            "block": [1, 1, 2, 2],
            "alpha": [0.1, 0.2, 0.1, 0.2],
            "sum_abs_error": [10.0, 8.0, 12.0, 15.0],
            "sum_abs_y": [100.0] * 4,
            "sum_signed_error": [-5.0, -3.0, -4.0, -7.0],
            "wmape": [0.10, 0.08, 0.12, 0.15],
            "bias": [-0.05, -0.03, -0.04, -0.07],
        }
    )
    summary = summarize_ses(ses)
    assert summary.height == 2
    assert summary.sort("wmape")["alpha"][0] == 0.1
    assert summary.filter(pl.col("alpha") == 0.1)["fold_wins"][0] == 1
    assert summary.filter(pl.col("alpha") == 0.2)["fold_wins"][0] == 1
    rec = recommend_ses(summary)
    assert rec.height == 1
    assert rec["alpha"][0] == 0.1

    drv = pl.DataFrame(
        {
            "section": ["1", "1"],
            "node_level": ["tienda", "tienda"],
            "target": ["Unidades", "Unidades"],
            "period_type": ["in_sample", "in_sample"],
            "driver_group": ["weekday", "weekday"],
            "sum_abs_y": [100.0, 200.0],
            "n_points": [10, 20],
            "wmape_delta_if_removed": [0.01, 0.02],
            "mean_abs_log_contribution": [0.1, 0.2],
            "unique_id": ["1||T:1", "1||T:1"],
            "block": [1, 2],
        }
    )
    dsum = summarize_driver_screen(drv)
    assert dsum.height == 1
    assert dsum["weighted_wmape_delta_if_removed"][0] > 0
    assert dsum["screen_class"][0] == "useful_signal"


def test_driver_groups_only_classify_columns_already_in_design():
    from app.forecasting.runner import RLSForecastRunner

    columns = ["intercept", "Tue", "Dec", "asp", "xmas_-1"]
    groups = RLSForecastRunner._driver_groups(columns)
    flattened = sorted(i for idxs in groups.values() for i in idxs)
    assert flattened == [1, 2, 3, 4]
    assert set(groups) == {"weekday", "month", "price", "holiday_other"}
