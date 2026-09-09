from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_global_ranking_uses_same_algebra_as_kpi():
    src = (ROOT / "app" / "dashboard_data.py").read_text(encoding="utf-8")
    assert 'den = pl.col("sum_abs_y").cast(pl.Float64).fill_null(0.0)' in src
    assert 'pl.col("sum_abs_error").cast(pl.Float64).fill_null(0.0) / den' in src
    assert 'pl.col("sum_signed_error").cast(pl.Float64).fill_null(0.0) / den' in src


def test_visible_global_ranking_is_scoped_to_current_section_and_unit():
    src = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert '(pl.col("Unidad") == view.unidad)' in src
    assert 'pl.col("Sección").cast(pl.Utf8) == str(view.seccion)' in src


def test_bias_badge_is_not_bold():
    src = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    block = src[src.index("def _bias_badge"):src.index("def _robust_outliers")]
    assert "font-weight:700" not in block
    assert "font-weight:600" not in block
    assert "font-weight:400" in block
