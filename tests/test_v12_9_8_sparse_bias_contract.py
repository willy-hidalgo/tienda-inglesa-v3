from pathlib import Path
import settings


def test_v1298_version_and_sparse_bias_is_diagnostic_only():
    assert settings.APP_VERSION == "12.9.12"
    from app import dashboard_artifacts
    assert dashboard_artifacts.ARTIFACT_VERSION == 20
    assert settings.V1298_SPARSE_BIAS_DIAGNOSTIC_ENABLED is True
    assert settings.V1298_SPARSE_BIAS_MIN_FOLDS >= 8
    assert tuple(settings.V1298_SPARSE_BIAS_BUCKETS) == ("07-09", "10-13")
    assert settings.V1298_SPARSE_BIAS_FACTOR_CLIP[1] <= 1.15
    src = Path("app/forecasting/leaf_v12.py").read_text(encoding="utf-8")
    # Metadata is attached after the frozen v12.9.6 production gate.
    assert "_v1298_sparse_bias_diagnostic" in src
    assert "select_oos = _apply_v1296_sec23_long_horizon_gate" in src
    assert "sparse_oos = _v1298_sparse_bias_diagnostic" in src
    # The diagnostic must not overwrite final forecasts.
    fn = src[src.index("def _v1298_sparse_bias_diagnostic"):src.index("def _selection_from_closed_blocks")]
    assert '.alias("yhat_raw")' not in fn
    assert '.alias("valuehat_raw")' not in fn


def test_v1298_forecast_finalizes_before_dashboard_build():
    src = Path("app/forecasts.py").read_text(encoding="utf-8")
    save = src.index("pipeline.save(res_df, wmapes_df, build_dashboard=False)")
    success = src.index("mark_success(forecast_path)")
    artifacts = src.index("from app.dashboard_artifacts import build_artifacts", success)
    assert save < success < artifacts
    assert "del res_df, wmapes_df" in src[save:success]


def test_v1298_acceptance_has_sparse_bias_section():
    report = Path("app/forecasting_acceptance_report.py").read_text(encoding="utf-8")
    assert "=== 31) v12.9.8 SPARSE-DEMAND BIAS RESCUE" in report
    assert "_print_v1298_sparse_bias_diagnostic(forecast)" in report
