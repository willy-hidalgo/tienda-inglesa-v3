from pathlib import Path


def test_all_dashboard_numeric_nonpercent_tables_use_thousands_separator():
    src = (Path(__file__).parents[1] / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'numeric_formats[c] = "{:,.0f}"' in src
    assert 'numeric_formats[c] = "{:,.2f}"' in src
    assert 'numeric_formats[c] = "{:.0f}"' not in src
    assert 'numeric_formats[c] = "{:.2f}"' not in src
    # Non-percent table columns must not override the Styler with ungrouped NumberColumn formats.
    assert '"Rotación": st.column_config.NumberColumn("Rotación", format="%.2f")' not in src
    assert '"Volumen OOS": st.column_config.NumberColumn("Volumen OOS", format="%.2f")' not in src
    assert '_ranking_styler(_na.select([c for c in _na_visible if c != "unique_id"]))' in src
    assert '_ranking_styler(view.detail)' in src
