"""Contratos del runner productivo v13."""
from pathlib import Path


def test_runner_exposes_only_coherent_leaf_path():
    text = Path("app/forecasting/runner.py").read_text(encoding="utf-8")
    assert "def predict_leaf_series(" in text
    assert "build_leaf_forecasts(" in text
    assert "leaf_v12" not in text
    assert "leaf_fallback" not in text
    assert "sku_shape_lgbm" not in text


def test_pipeline_uses_section_store_rls_then_leaf_ses_rls():
    text = Path("app/forecasting/pipeline.py").read_text(encoding="utf-8")
    assert "RLS sección" in text
    assert "RLS tiendas batch" in text
    assert "SKU+tienda SES+RLS" in text
    assert "runner.predict_leaf_series(" in text
    assert "occurrence" not in text.lower()
    assert "lightgbm" not in text.lower()
