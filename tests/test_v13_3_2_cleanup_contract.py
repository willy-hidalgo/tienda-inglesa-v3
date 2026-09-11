"""v13.3.3 cleanup contracts: production is pure SES+RLS."""
from __future__ import annotations
from pathlib import Path
import settings

ROOT = Path(__file__).resolve().parents[1]


def test_versions_are_v1332_artifact32():
    assert settings.APP_VERSION == "13.3.3"
    assert 'version = "13.3.3"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 33" in (ROOT / "app" / "dashboard_artifacts.py").read_text(encoding="utf-8")


def test_exception_routing_productive_code_is_removed():
    removed = [
        "app/forecasting/exception_routing.py",
        "app/forecasting/exception_routing_gate.py",
        "app/forecasting/freeze_exception_routing.py",
        "app/forecasting/exception_routing_candidate_registry.csv",
    ]
    assert [p for p in removed if (ROOT / p).exists()] == []
    runner = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "apply_exception_routing" not in runner
    assert "build_leaf_forecasts(" in runner


def test_exception_lab_is_diagnostic_only_and_not_productive():
    lab = (ROOT / "app" / "forecasting" / "exception_lab.py").read_text(encoding="utf-8")
    assert "STATUS=DIAGNOSTIC_ONLY" in lab
    assert "EXTREME_RELATIVE_ERROR" in lab
    for rel in [
        "app/forecasting/runner.py",
        "app/forecasting/pipeline.py",
        "app/forecasts.py",
    ]:
        assert "exception_lab" not in (ROOT / rel).read_text(encoding="utf-8")
