"""Higiene de código y dependencias de la base v13."""
from __future__ import annotations
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def _python_sources():
    for base in (ROOT / "app", ROOT / "rls_opt"):
        yield from base.rglob("*.py")
    yield ROOT / "settings.py"


def test_no_pandas_anywhere_in_product_code_or_dependencies():
    bad = []
    pattern = re.compile(r"\b(import pandas|from pandas|pd\.|to_pandas\s*\()")
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            bad.append(str(path.relative_to(ROOT)))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "pandas" not in pyproject
    assert bad == []


def test_legacy_model_modules_are_physically_absent():
    forbidden = [
        "app/forecasting/leaf_v12.py",
        "app/forecasting/leaf_fallback.py",
        "app/forecasting/sku_shape_lgbm.py",
        "app/v12_sku_total_calendar_driver_backtest.py",
        "app/v12_sku_total_pooled_ridge_backtest.py",
        "app/v12_sku_total_pooled_lgbm_shape_backtest.py",
    ]
    assert [p for p in forbidden if (ROOT / p).exists()] == []


def test_product_pipeline_has_no_legacy_model_references():
    text = "\n".join(p.read_text(encoding="utf-8") for p in _python_sources())
    # El validador puede nombrar tokens legacy para detectar artefactos viejos.
    product = (ROOT / "app/forecasting/pipeline.py").read_text(encoding="utf-8") + (ROOT / "app/forecasting/runner.py").read_text(encoding="utf-8")
    for token in ("leaf_v12", "leaf_fallback", "sku_shape_lgbm", "LightGBM", "occurrence/share", "meta-selector"):
        assert token.lower() not in product.lower()
