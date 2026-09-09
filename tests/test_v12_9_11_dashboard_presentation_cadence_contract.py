from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
DATA = (ROOT / "app" / "dashboard_data.py").read_text(encoding="utf-8")


def test_ranking_presentation_contract():
    assert 'font-size", "13px"' in DASH
    assert '_ranking_styler(_global_table_view, yellow=False)' in DASH
    assert 'height=_RANK_HEIGHT' in DASH
    assert '"Rotación"' in DASH
    assert '"Rotación"' in DATA
    assert '{:,.2f}' in DASH
    assert '{:,.0f}' in DASH


def test_ranking_period_selector_is_not_sidebar():
    assert 'st.segmented_control(\n    "Período de métricas de ranking"' in DASH
    assert 'st.sidebar.segmented_control(\n    "Período de métricas de ranking"' not in DASH


def test_cadence_compare_is_section_level_and_demo_text_removed():
    assert 'pl.col("unique_id") == seccion' in DASH
    assert 'Cadencias precalculadas · nivel Sección' in DASH
    assert 'Demo multi-cadencia:' not in DASH
    assert 'Escenario: actualización cada' not in DASH
    assert '📍 Tienda:' not in DASH
