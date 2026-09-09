from __future__ import annotations

import sys
import types
from pathlib import Path

import settings
from app.forecasting.preflight import ensure_selected_input


def _patch_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    input_dir = tmp_path / "data" / "input"
    out_dir = tmp_path / "data" / "output"
    input_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    selected = out_dir / "selected.parquet"
    master = out_dir / "master.parquet"
    sales = out_dir / "sales.parquet"
    monkeypatch.setattr(settings, "INPUT_DIR", input_dir)
    monkeypatch.setattr(settings, "OUT_DIR", out_dir)
    monkeypatch.setattr(settings, "SELECTED_PATH", selected)
    monkeypatch.setattr(settings, "MASTER_PATH", master)
    monkeypatch.setattr(settings, "SALES_PATH", sales)
    monkeypatch.setattr(settings, "MASTER_XLSX_FILENAME", "master.xlsx")
    monkeypatch.setattr(settings, "SALES_FILES_GLOB", "sales_*")
    return input_dir, selected, master, sales


def test_preflight_is_noop_when_selected_exists(monkeypatch, tmp_path):
    _, selected, _, _ = _patch_paths(monkeypatch, tmp_path)
    selected.write_bytes(b"ok")
    assert ensure_selected_input(selected) == selected


def test_preflight_builds_selection_from_existing_intermediates(monkeypatch, tmp_path):
    _, selected, master, sales = _patch_paths(monkeypatch, tmp_path)
    master.write_bytes(b"master")
    sales.write_bytes(b"sales")

    module = types.ModuleType("app.categories_selector")
    class DummySelector:
        def run(self):
            selected.write_bytes(b"selected")
    module.DemandAnalysisPipeline = DummySelector
    monkeypatch.setitem(sys.modules, "app.categories_selector", module)

    assert ensure_selected_input(selected) == selected
    assert selected.read_bytes() == b"selected"


def test_preflight_builds_ingestion_then_selection_from_raw(monkeypatch, tmp_path):
    input_dir, selected, master, sales = _patch_paths(monkeypatch, tmp_path)
    (input_dir / "master.xlsx").write_bytes(b"xlsx")
    (input_dir / "sales_01.dat").write_bytes(b"sales")

    ingest_module = types.ModuleType("app.ingestor")
    class DummyIngestor:
        def run(self):
            master.write_bytes(b"master")
            sales.write_bytes(b"sales")
    ingest_module.DataIngestionPipeline = DummyIngestor
    monkeypatch.setitem(sys.modules, "app.ingestor", ingest_module)

    select_module = types.ModuleType("app.categories_selector")
    class DummySelector:
        def run(self):
            selected.write_bytes(b"selected")
    select_module.DemandAnalysisPipeline = DummySelector
    monkeypatch.setitem(sys.modules, "app.categories_selector", select_module)

    assert ensure_selected_input(selected) == selected
    assert selected.exists()


def test_preflight_fails_fast_with_actionable_missing_sources(monkeypatch, tmp_path):
    _, selected, _, _ = _patch_paths(monkeypatch, tmp_path)
    try:
        ensure_selected_input(selected)
    except FileNotFoundError as exc:
        text = str(exc)
        assert "data/input" in text.replace("\\", "/")
        assert "master.xlsx" in text
        assert "sales_*" in text
    else:
        raise AssertionError("Expected FileNotFoundError")
