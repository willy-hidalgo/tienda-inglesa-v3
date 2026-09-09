"""Contratos operativos del launcher y ranking dashboard v13.2.2."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_main_exposes_all_required_operational_processes():
    text = (ROOT / "app/main.py").read_text(encoding="utf-8")
    required = [
        "ingestar / actualizar datos",
        "seleccionar categorías / secciones",
        "crear pronósticos · un bloque",
        "crear pronósticos · todos los bloques 1/7/14/28d",
        "optimización estadística · SES+RLS + refit drivers + residuos · un bloque",
        "construir / reparar artefactos dashboard · un bloque",
        "construir artefactos dashboard para todos los bloques existentes",
        "validar modelo v13 · un bloque",
        "validar modelo v13 · todos los bloques",
        "validar consistencia dashboard · un bloque",
        "validar consistencia dashboard · todos los bloques",
        "reporte de aceptación estadística",
        "ejecutar suite de pruebas pytest",
        "cargar dashboard Streamlit",
    ]
    for label in required:
        assert label in text
    assert '"--optimization-phase2"' in text
    assert '"--all-update-blocks"' in text
    assert '"--update-block-days"' in text
    assert 'cwd=self._project_root' in text


def test_root_main_remains_single_launcher():
    text = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from app.main import main" in text


def test_leaf_ranking_keeps_description_compact_before_metrics():
    text = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    assert '"SKU descripción": st.column_config.TextColumn("SKU descripción", width="medium")' in text
    assert '"wMAPE (%)": st.column_config.NumberColumn("wMAPE (%)", format="%.2f%%", width="small")' in text
    assert '"BIAS (%)": st.column_config.NumberColumn("BIAS (%)", format="%.2f%%", width="small")' in text
