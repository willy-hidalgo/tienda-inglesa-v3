from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_release_versions_are_consistent():
    assert 'APP_VERSION: str = "9.2"' in read("settings.py")
    assert 'version = "9.2.0"' in read("pyproject.toml")
    assert "ARTIFACT_VERSION = 7" in read("app/dashboard_artifacts.py")
    assert 'Pipeline Tienda Inglesa v{settings.APP_VERSION}' in read("app/main.py")

def test_dashboard_stops_on_metric_series_inconsistency():
    src = read("app/dashboard.py")
    assert "if view.consistency_warnings:" in src
    assert "Dashboard detenido:" in src
    assert "st.stop()" in src

def test_artifact_source_identity_includes_row_count():
    src = read("app/dashboard_artifacts.py")
    assert '"mtime_ns"' in src
    assert '"size_bytes"' in src
    assert '"n_rows"' in src
    assert "pl.scan_parquet(path)" in src
    assert "current_rows == expected_rows" in src

def test_v9_driver_grid_contains_pure_ses_and_shrunk_rls():
    src = read("app/forecasting/runner.py")
    assert '{"_parent": "none", "_strength": 0.0}' in src
    assert "LEAF_DRIVER_STRENGTH_CANDIDATES" in src
    assert "_effective_strength_y" in src
    assert "_effective_strength_v" in src
