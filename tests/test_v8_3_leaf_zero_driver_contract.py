from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]

def test_leaf_initial_window_is_leaf_specific():
    m = leaf_method()
    assert 'pl.col("ds").min().alias("_leaf_start")' in m
    assert 'alias("_leaf_warmup_end")' in m
    assert 'pl.col("ds") <= pl.col("_leaf_warmup_end")' in m
    assert 'pl.col("ds") > pl.col("_leaf_warmup_end")' in m

def test_wmape_keeps_zero_actual_days_in_numerator():
    metrics = read("app/forecasting/metrics.py")
    backend = read("app/backend.py")
    settings = read("settings.py")
    # Official scoring filters finite pairs, not y != 0.
    scored = metrics[metrics.index("leaves = leaves.filter("):metrics.index("if (", metrics.index("leaves = leaves.filter("))]
    assert '(pl.col("y") != 0)' not in scored
    assert 'WMAPE_INCLUDE_ZERO_ACTUAL_DAYS: bool = True' in settings
    assert 'Los días cero penalizan sobreforecast' in backend

def test_ranking_sales_count_still_counts_nonzero_days_only():
    backend = read("app/backend.py")
    assert 'pl.col("ds").filter(pl.col("y") != 0).n_unique()' in backend

def test_direct_ses_zero_strength_candidate_is_available():
    m = leaf_method()
    grid = m[m.index("parent_candidates ="):m.index("# Pure SES state")]
    assert '{"_parent": "none", "_strength": 0.0}' in grid
    assert 'for parent_name in ("store", "section")' in grid
    assert "_none_ey" not in m
    assert "_none_ev" not in m


def test_forecast_only_has_driver_shape_fallback():
    m = leaf_method()
    assert 'Forecast-only must preserve a driver shape' in m
    assert '_prev_effect_y' in m and '_prev_effect_v' in m
    assert 'pl.duration(days=block_days)' in m
