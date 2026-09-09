from pathlib import Path


def test_dashboard_verification_ui_contracts():
    text = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert 'st.sidebar.select_slider(' in text
    assert '"Bloque de actualización"' in text
    assert 'col_t, col_s, col_g = st.columns(3' in text
    assert 'preferred = ["Código", "Descripción", "wMAPE (%)", "BIAS (%)"]' in text
    assert 'SKU+Tienda no activos · historia + forecast' in text
    assert 'wMAPE in-sample' in text
    assert '_bias_badge("BIAS in-sample"' in text
    assert 'Marcar outliers robustos' in text
    assert 'los **wMAPE oficiales son bottom-up**' in text
    assert 'with st.sidebar.expander("Cómo generar 1d / 7d / 14d / 28d"' not in text
    assert '_cached_global_ranking_excel' in text


def test_section_horizon_contract_is_index_first():
    text = Path('app/dashboard_data.py').read_text(encoding='utf-8')
    assert 'Horizontes del dashboard SIEMPRE a nivel sección' in text
    assert 'raw = (index.get("horizons_by_sec") or {}).get(seccion)' in text
    assert 'cutoff = train_end or cutoff_date' in text


def test_series_artifact_is_tuned_for_selection_refresh():
    text = Path('app/dashboard_artifacts.py').read_text(encoding='utf-8')
    assert 'row_group_size=4096' in text
    assert 'collect(engine="streaming")' in text
