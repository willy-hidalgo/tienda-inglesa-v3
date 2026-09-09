from pathlib import Path


def test_ranking_period_toggle_and_highlight():
    text = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert 'st.segmented_control(' in text
    assert 'st.sidebar.select_slider(\n    "Período de métricas de ranking"' not in text
    assert 'st.session_state["ranking_metric_period_selector"] = "OOS"' in text
    assert 'options=["OOS", "In-sample"]' in text
    assert 'Mostrando métricas: **{_ranking_period_label.upper()}**' in text


def test_global_active_table_is_complete_selectable_and_counted():
    text = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert '(pl.col("Cohort") == "active")' in text
    assert 'selection_mode="single-row"' in text
    assert 'st.session_state["_pending_tienda"]' in text
    assert 'st.session_state["_pending_sku"]' in text
    assert 'SKU+Tienda Active ·' in text
    assert 'height=_RANK_HEIGHT' in text
    assert '.head(' not in text[text.index('_global_rank_display'):text.index('hz = view.horizons')]


def test_sort_columns_remain_numeric_and_selection_does_not_reorder():
    data = Path('app/dashboard_data.py').read_text(encoding='utf-8')
    dash = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert '"wMAPE (%)": pl.Float64' in data
    assert '"BIAS (%)": pl.Float64' in data
    assert 'label_rot: pl.Float64' in data
    assert '"% ≠0": pl.Float64' in data
    assert '.sort("_selected"' not in data
    assert 'st.column_config.NumberColumn("wMAPE (%)"' in dash
    assert '_ranking_styler(' in dash
    assert '"{:,.2f}"' in dash


def test_in_sample_metrics_are_precomputed_and_scoped_for_speed():
    art = Path('app/dashboard_artifacts.py').read_text(encoding='utf-8')
    dash = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert 'metrics_in_sample.parquet' in art
    assert 'period_type="in_sample"' in art
    assert 'def load_metrics_scope(' in art
    assert 'collect(engine="streaming")' in art
    assert '_load_metrics_scope(' in dash
    assert 'Preparar Excel del ranking global SKU+Tienda' in dash


def test_removed_generation_expander_control():
    text = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert 'with st.sidebar.expander("Cómo generar 1d / 7d / 14d / 28d"' not in text
