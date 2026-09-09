from pathlib import Path


def test_metric_period_is_compact_segmented_control():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert 'st.segmented_control(' in text
    assert 'options=["OOS", "In-sample"]' in text
    assert 'key="ranking_metric_period_selector"' in text
    assert 'st.session_state["ranking_metric_period_selector"] = "OOS"' in text


def test_cadence_section_follows_dashboard_reading_message():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    reading = text.index("**Lectura del dashboard:**")
    cadence = text.index("#### Cadencias precalculadas · nivel Sección")
    rankings = text.index('st.markdown("#### Rankings")')
    assert reading < cadence < rankings


def test_global_ranking_restores_reliable_yellow_background_and_plain_numeric_format():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert '_ranking_styler(_global_table_view, yellow=False)' in text
    assert 'numeric_formats[c] = "{:,.0f}"' in text
    assert 'numeric_formats[c] = "{:,.2f}"' in text
    assert 'font-size", "13px"' in text
    assert '"Rotación": st.column_config.NumberColumn' not in text


def test_chart_periods_have_distinct_palettes():
    text = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert '_in_actual_fill = "rgba(147,197,253,0.20)"' in text
    assert '_in_forecast_line = "#1d4ed8"' in text
    assert '_oos_actual_fill = "rgba(254,215,170,0.28)"' in text
    assert '_oos_forecast_line = "#ea580c"' in text
    assert '_forecast_only_line = "#7c3aed"' in text
    assert 'name="Actual · in-sample"' in text
    assert 'name="Actual · OOS"' in text
    assert 'name="Forecast · in-sample"' in text
    assert 'name="Forecast · OOS"' in text
