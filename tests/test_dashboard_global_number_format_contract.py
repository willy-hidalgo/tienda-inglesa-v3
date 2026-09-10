"""Contrato global: toda tabla usa separador de miles en numéricos no-% ."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_all_dashboard_tables_go_through_global_dataframe_wrapper():
    text = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    # El único st.dataframe permitido es el encapsulado por _dashboard_dataframe.
    assert text.count("st.dataframe(") == 1
    assert text.count("_dashboard_dataframe(") >= 6
    assert "def _dashboard_column_config(" in text
    assert "def _dashboard_dataframe(" in text


def test_non_percent_numeric_formats_have_thousands_separator():
    text = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    assert 'return "%,d"' in text
    assert 'return "%,.2f"' in text
    assert 'return "%" in str(name)' in text
    # Los porcentajes existentes conservan formato porcentual y no usan coma de miles.
    assert 'format="%.2f%%"' in text


def test_identifiers_are_never_treated_as_numeric_measures():
    text = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    for token in ["codigo", "código", "tienda", "store_id", "sku", "sku_id", "seccion", "sección", "unique_id"]:
        assert f'"{token}"' in text
    assert "st.column_config.TextColumn(name)" in text


def test_detail_backend_preserves_numeric_dtypes_for_global_formatter():
    text = (ROOT / "app/backend.py").read_text(encoding="utf-8")
    start = text.index("def format_detail_display")
    end = text.index("def metrics_rolling28", start)
    body = text[start:end]
    assert "return df" in body
    assert "map_elements" not in body
    assert "pl.Utf8" not in body
