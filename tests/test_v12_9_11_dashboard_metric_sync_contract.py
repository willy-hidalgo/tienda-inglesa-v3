from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
DATA = (ROOT / "app" / "dashboard_data.py").read_text(encoding="utf-8")


def test_metric_order_and_bias_pill_contract():
    assert DASH.index('st.markdown("**In-sample**")') < DASH.index('st.markdown("**Out-of-sample**")')
    assert 'border-radius:999px' in DASH
    assert '"↑"' in DASH and '"↓"' in DASH
    assert '#dcfce7' in DASH and '#fee2e2' in DASH


def test_selected_leaf_rankings_sync_to_exact_oos_metric():
    assert "def _sync_selected_leaf_metric(" in DATA
    assert 'uid = settings.make_unique_id(seccion, store=str(store), sku=str(sku))' in DATA
    assert 'wm_checked, bias_checked = _metric_values_from_row(row)' in DATA
    assert 'ranking_tiendas = _sync_selected_leaf_metric(' in DATA
    assert 'ranking_skus = _sync_selected_leaf_metric(' in DATA


def test_metric_identity_is_verified_from_oos_sums():
    assert 'wm_calc = float(row["sum_abs_error"][0] or 0.0) / den' in DATA
    assert 'bi_calc = float(row["sum_signed_error"][0] or 0.0) / den' in DATA


def test_nonactive_chart_marks_oos_and_forecast_only_start():
    assert 'annotation_text="Inicio OOS"' in DASH
    assert 'annotation_text="Inicio forecast-only"' in DASH
