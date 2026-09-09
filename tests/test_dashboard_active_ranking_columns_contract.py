from pathlib import Path


def test_active_leaf_ranking_columns_and_order_contract():
    src = (Path(__file__).parents[1] / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert '.with_row_index("Rank", offset=1)' in src
    assert '(pl.col("Cohort") == "active")' in src
    expected = [
        '"Tienda",',
        '"SKU",',
        '"SKU descripción",',
        '"wMAPE (%)",',
        '"BIAS (%)",',
        '"Rotación",',
        '"Rank",',
        '"Unidad",',
        '"N puntos",',
        '"Días con venta",',
        '"% ≠0",',
        '"Cohort",',
    ]
    start = src.index('.select([\n            "Tienda",')
    block = src[start:start+900]
    positions = [block.index(token) for token in expected]
    assert positions == sorted(positions)
    assert '_vol_col, _fc_col, _err_col, "unique_id"' in block


def test_active_leaf_ranking_percent_formats_contract():
    src = (Path(__file__).parents[1] / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert 'NumberColumn("wMAPE (%)", format="%.2f%%"' in src
    assert 'NumberColumn("BIAS (%)", format="%.2f%%"' in src
    assert 'NumberColumn("% ≠0", format="%.1f%%"' in src
