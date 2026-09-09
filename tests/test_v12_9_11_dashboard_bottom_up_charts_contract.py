from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bottom_up_aggregate_series_contract():
    src = (ROOT / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "def _build_bottom_up_aggregate_series_vectorized" in src
    assert "leaf_source = sec_source.filter" in src
    assert "agregados BU=" in src
    assert "bu_abs_error_daily_y" in src
    assert "bu_abs_error_daily_value" in src


def test_official_metric_is_not_redefined():
    src = (ROOT / "app" / "dashboard_data.py").read_text(encoding="utf-8")
    assert "_official_oos_metrics_from_artifact" in src
    assert "metrics.parquet" in src


def test_dashboard_explains_bottom_up_error():
    src = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert "forecast agregado bottom-up desde SKU+Tienda" in src
    assert "Cómo se forma el wMAPE bottom-up" in src
    assert "Error absoluto BU · OOS" in src
