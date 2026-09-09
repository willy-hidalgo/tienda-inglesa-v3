from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fallback_is_sequential_and_active_only():
    src = (ROOT / "app" / "forecasting" / "leaf_fallback.py").read_text(encoding="utf-8")
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert "LEAF_FALLBACK_MIN_VALIDATION_SALES" in settings
    assert 'pl.col("_direct_n_sales") >= int(min_validation_sales)' in src
    assert "robust_needed_ids" in src
    assert 'pl.col("_risk_eligible") & ~pl.col("_use_ses")' in src
    assert 'pl.col("_selected_method") == "deses_ses"' in src
    assert 'pl.col("_selected_method") == "deses_robust_mean"' in src


def test_densified_intercept_is_always_one():
    src = (ROOT / "app" / "forecasting" / "panel.py").read_text(encoding="utf-8")
    assert 'pl.col("intercept").fill_null(1).cast(pl.Int8)' in src
    assert 'pl.lit(1).cast(pl.Int8).alias("intercept")' in src


def test_dashboard_reads_projection_and_uses_linear_label_lookup():
    src = (ROOT / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "required_cols = set(SERIES_COLS_PREFERRED)" in src
    assert "scan.select(load_cols)" in src
    assert "def _leaf_description_lookup" in src
    assert "desc_lookup.get" in src
    assert "next(" not in src[src.index("4/6 Materializando series slim + SKU puro"):src.index("5/6 Construyendo index + labels")]


def test_excel_uid_parsing_is_vectorized():
    src = (ROOT / "app" / "forecasting" / "pipeline.py").read_text(encoding="utf-8")
    block = src[src.index("def _export_forecast_excel"):]
    assert '.str.extract(r"\\|\\|T:([^|]+)", 1).alias("Local")' in block
    assert '.str.extract(r"\\|\\|S:([^|]+)", 1).alias("SKU")' in block
    assert "parsed = [settings.split_unique_id" not in block


def test_bias_correction_runs_before_leaf_concat():
    src = (ROOT / "app" / "forecasting" / "pipeline.py").read_text(encoding="utf-8")
    assert "bias correction padres OOS" in src
    assert "RLSForecastRunner.apply_bias_correction(parent_res)" in src
    assert "RLSForecastRunner.apply_bias_correction(res_df)" not in src


def test_fallback_uses_slim_work_frame():
    src = (ROOT / "app" / "forecasting" / "leaf_fallback.py").read_text(encoding="utf-8")
    assert "work_cols = [" in src
    assert "work = rows.select" in src
    assert "work, target=target" in src


def test_full_section_checkpoints_are_opt_in():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    pipe = (ROOT / "app" / "forecasting" / "pipeline.py").read_text(encoding="utf-8")
    assert "WRITE_SECTION_CHECKPOINTS: bool = False" in settings
    assert 'getattr(settings, "WRITE_SECTION_CHECKPOINTS", False)' in pipe


def test_parent_recursive_prediction_is_numba_compiled():
    kernels = (ROOT / "rls_opt" / "kernels.py").read_text(encoding="utf-8")
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "def _predict_loglink_base" in kernels
    assert "def _predict_loglink_ar" in kernels
    assert "return _predict_loglink_ar(xrows, c, tail)" in runner
    assert "np.concatenate([Xbase" not in runner
