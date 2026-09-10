"""Contrato de selección completa: no existe muestreo/whitelist de SKU."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "app" / "categories_selector.py"
SETTINGS = ROOT / "settings.py"


def _function(tree: ast.AST, class_name: str, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == function_name:
                    return item
    raise AssertionError(f"No se encontró {class_name}.{function_name}")


def test_categories_selector_has_no_demo_or_best_sku_sampling() -> None:
    text = SELECTOR.read_text(encoding="utf-8")
    settings = SETTINGS.read_text(encoding="utf-8")

    forbidden = (
        "select_best_skus",
        "best_skus",
        "DEMO_MODE",
        "TI_DEMO_MODE",
        "demo_mode",
        'pl.col("SKU_ID").is_in',
        ".sample(",
        ".head(",
        ".top_k(",
    )
    for token in forbidden:
        assert token not in text
        assert token not in settings


def test_focus_sections_are_exactly_1_and_23() -> None:
    tree = ast.parse(SETTINGS.read_text(encoding="utf-8"))
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "FOCUS_SECTIONS" for t in node.targets):
                value = ast.literal_eval(node.value)
                break
    assert value == ["1", "23"]


def test_pipeline_persists_loader_output_without_second_sku_selector() -> None:
    tree = ast.parse(SELECTOR.read_text(encoding="utf-8"))
    run = _function(tree, "DemandAnalysisPipeline", "run")
    run_text = ast.get_source_segment(SELECTOR.read_text(encoding="utf-8"), run) or ""

    assert "SalesCatalogLoader(self._cfg).load()" in run_text
    assert "df_selected.write_parquet(" in run_text
    assert "SectionDemandSelector" not in run_text
    assert ".filter(" not in run_text
    assert ".is_in(" not in run_text
