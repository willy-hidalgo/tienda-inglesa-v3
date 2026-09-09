from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_official_metrics_exclude_zero_actual_days():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    metrics = (ROOT / "app" / "forecasting" / "metrics.py").read_text(encoding="utf-8")
    backend = (ROOT / "app" / "backend.py").read_text(encoding="utf-8")
    assert "& (y != 0)" in settings
    assert 'pl.when(pl.col("y") != 0)' in metrics
    assert 'pl.when(pl.col("y") != 0)' in backend


def test_oos_is_last_28_days_and_no_post_oos_actual_path():
    settings = (ROOT / "settings.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "app" / "forecasting" / "pipeline.py").read_text(encoding="utf-8")
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "oos_end = last_actual" in settings
    assert "oos_start = oos_end - dt.timedelta(days=metric_days - 1)" in settings
    assert "update_days = int(RLS_BLOCK_DAYS)" in settings
    assert '"actual_extension_start": None' in settings
    assert '"observed_tail_start": None' in settings
    assert 'targets["actual_extension"]' not in pipeline
    assert 'target_parts.get("actual_extension")' not in runner


def test_high_error_leaf_fallbacks_are_available_and_causal():
    fallback = (ROOT / "app" / "forecasting" / "leaf_fallback.py").read_text(encoding="utf-8")
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert '"deses_ses"' in fallback
    assert '"deses_robust_mean"' in fallback
    assert "validation_history_end" in fallback
    assert "apply_robust_leaf_fallbacks(rows, horizons)" in runner


def test_store_rls_is_batched_and_partitioned_once():
    pipeline = (ROOT / "app" / "forecasting" / "pipeline.py").read_text(encoding="utf-8")
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "RLS tiendas batch" in pipeline
    assert 'train.partition_by("unique_id", as_dict=True)' in runner
    assert "ThreadPoolExecutor" in runner
    assert "_process_one_store" not in pipeline


def test_rls_hot_kernel_is_numba_compiled_and_cached():
    kernels = (ROOT / "rls_opt" / "kernels.py").read_text(encoding="utf-8")
    assert '@jit(nopython=True, cache=True, nogil=True)\ndef _rls(' in kernels
    assert '@jit(nopython=True, cache=True, nogil=True)\ndef _numba_outer(' in kernels
    assert '@jit(nopython=True, cache=True, nogil=True)\ndef _rls_predict(' in kernels


def test_leaf_fallback_has_no_python_loop_per_uid():
    fallback = (ROOT / "app" / "forecasting" / "leaf_fallback.py").read_text(encoding="utf-8")
    assert 'partition_by(' not in fallback
    assert 'for key, g in' not in fallback
    assert 'deses_ses' in fallback
    assert 'deses_robust_mean' in fallback
    assert '_deses_ses_levels' in fallback


def test_rls_runner_fails_fast_without_numba_dispatcher():
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "RLS performance invariant violated" in runner
    assert "CPUDispatcher" in runner
    assert "self._ensure_numba_rls_kernel()" in runner


def test_rls_hot_kernel_is_blas_free():
    kernels = (ROOT / "rls_opt" / "kernels.py").read_text(encoding="utf-8")
    hot = kernels[kernels.index("def _rls("):kernels.index("def _log_weighting")]
    pred = kernels[kernels.index("def _rls_predict("):kernels.index("def _numba_outer")]
    assert " @ " not in hot
    assert "np.dot" not in hot
    assert " @ " not in pred
    assert "np.dot" not in pred
    assert "_matvec_inplace" in kernels
    assert "_vecmat_inplace" in kernels
    assert "_dot_1d" in kernels
