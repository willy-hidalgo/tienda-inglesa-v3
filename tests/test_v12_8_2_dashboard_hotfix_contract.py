from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_fast_dashboard_uses_metrics_artifact_as_oos_source_of_truth():
    data = _text("app/dashboard_data.py")
    assert "def _official_oos_metrics_from_artifact" in data
    assert "metrics.parquet``" in data
    assert "model_compare = None" in data
    assert "load_model_compare_fast" in data


def test_dashboard_model_compare_is_lazy():
    dashboard = _text("app/dashboard.py")
    assert "Cargar diagnóstico avanzado v11/v12" in dashboard
    assert "_cached_model_compare" in dashboard
    assert "OOS e in-sample oficiales deben venir de los artefactos de métricas." in dashboard


def test_bottom_up_kpi_excludes_zero_sales_rows():
    backend = _text("app/backend.py")
    assert 'metric_rows = (' in backend
    assert 'else frame.filter(pl.col("y") != 0)' in backend
    assert 'include_zero_error=True' in backend
