"""Contratos operativos de la entrega estable multi-bloque v13.2.5-r1."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_main_exposes_one_click_dashboard_ready_all():
    text = (ROOT / "app/main.py").read_text(encoding="utf-8")
    assert "PREPARAR DASHBOARD COMPLETO · 1d/7d/14d/28d + auditorías" in text
    assert '"dashboard-ready-all": "15"' in text
    assert '"prepare-dashboard-all": "15"' in text
    assert "range(1, 16)" in text


def test_dashboard_ready_pipeline_is_non_optimization_and_validates_everything():
    text = (ROOT / "app/dashboard_ready.py").read_text(encoding="utf-8")
    assert '"--all-update-blocks"' in text
    assert 'forecast_cmd.append("--skip-existing")' in text
    assert '"app.dashboard_artifacts"' in text
    assert '"app.forecasting.validate_v13"' in text
    assert '"app.dashboard_consistency"' in text
    assert "--optimization-phase2" not in text
    assert "--optimization-diagnostics" not in text
    assert "DASHBOARD READY" in text


def test_dashboard_only_exposes_current_ready_multiblock_scenarios():
    text = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    assert "_run_status_errors(_fp)" in text
    assert "artifacts.artifacts_exist(_fp)" in text
    assert "Escenarios todavía no listos:" in text
    assert "dashboard-ready-all" in text
    assert "Escenarios listos:" in text
