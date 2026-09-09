from pathlib import Path


def test_global_leaf_ranking_visible_is_active_only():
    src = (Path(__file__).parents[1] / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert '(pl.col("Cohort") == "active")' in src
    assert 'SKU+Tienda · Active' in src
    assert 'SKU+Tienda Active ·' in src


def test_nonactive_expander_still_uses_nonactive_cohorts():
    src = (Path(__file__).parents[1] / "app" / "dashboard.py").read_text(encoding="utf-8")
    assert '(pl.col("Cohort") != "active")' in src
