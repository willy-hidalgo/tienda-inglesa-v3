"""Contratos funcionales del dashboard v13."""
from pathlib import Path


def test_dashboard_is_multiblock_and_acid_metrics_are_visibility_only():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert '"Bloque de actualización"' in text
    assert '"Mostrar métricas incluyendo y=0"' in text
    assert 'preferred = ["Código", "Descripción", "wMAPE (%)", "BIAS (%)"]' in text
    assert 'acid = ["wMAPE incl. y=0 (%)", "BIAS incl. y=0 (%)"]' in text
    assert "sort([\"wMAPE (%)\"" in text or ".sort([\"wMAPE (%)\"" in text


def test_dashboard_artifacts_support_all_update_blocks_and_v22():
    text = Path("app/dashboard_artifacts.py").read_text(encoding="utf-8")
    assert "ARTIFACT_VERSION = 24" in text
    assert '"--all-update-blocks"' in text
    assert "settings.UPDATE_BLOCK_OPTIONS" in text


def test_dashboard_has_leaf_excel_audit_without_pandas():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert "Preparar auditoría Excel SKU+Tienda" in text
    assert "Descargar auditoría Excel SKU+Tienda" in text
    assert "xlsxwriter" in text
    assert "pandas" not in text.lower()
    assert ".to_pandas(" not in text


def test_main_menu_builds_all_existing_dashboard_artifacts():
    text = Path("app/main.py").read_text(encoding="utf-8")
    assert '"--all-update-blocks"' in text
    assert 'construir artefactos dashboard para todos los bloques existentes' in text

def test_acid_metrics_align_with_official_insample_oos_columns():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert '_acid_in, _acid_oos, _acid_spacer_1, _acid_spacer_2 = st.columns([1.15, 1.15, 1, 1])' in text
    assert 'with _acid_in:' in text
    assert 'st.metric("wMAPE in-sample · incl. y=0"' in text
    assert '_bias_badge("BIAS in-sample · incl. y=0"' in text
    assert 'with _acid_oos:' in text
    assert 'st.metric("wMAPE OOS · incl. y=0"' in text
    assert '_bias_badge("BIAS OOS · incl. y=0"' in text

