from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(rel): return (ROOT/rel).read_text(encoding='utf-8')

def test_rls_dynamics_is_prior_block_selected_not_forced_ar():
    s=read('settings.py'); r=read('app/forecasting/runner.py')
    assert 'RLS_DYNAMICS_CANDIDATES' in s
    assert '("base", "ar")' in s
    assert 'cum_ae_y' in r and 'cum_den_y' in r
    assert 'mode == "ar"' in r
    assert 'rls_dynamics_y' in r and 'rls_forecast_origin' in r

def test_leaf_has_direct_ses_challenger_and_parent_effects():
    r=read('app/forecasting/runner.py')
    method=r[r.index('def fast_leaf_forecasts'):r.index('def derive_sku_store_forecasts')]
    assert 'for parent_name in ("store", "section", "none")' in method
    assert 'pl.lit(0.0).alias("_none_ey")' in method
    assert 'pl.lit(0.0).alias("_none_ev")' in method
    assert 'pl.col("_parent") == "store"' in method
    assert 'pl.col("_parent") == "section"' in method
    assert "_store_ey" in method and "_store_ev" in method
    assert "_sec_ey" in method and "_sec_ev" in method
    assert "choices_block" in method
    assert "cumulative errors only" in method

def test_expanding28_and_no_actual_extension_leakage_preserved():
    r=read('app/forecasting/runner.py')
    m=r[r.index('def _fit_and_predict_expanding_blocks'):r.index('def fit_and_predict_sections')]
    assert 'range(seed_n, n, block_days)' in m
    assert 'boundary = s - 1' in m
    assert 'target_parts.get("actual_extension")' not in m
