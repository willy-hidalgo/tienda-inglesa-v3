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

def test_leaf_can_choose_pure_ses_or_both_parent_shapes():
    r=read('app/forecasting/runner.py')
    method=r[r.index('def fast_leaf_forecasts'):r.index('def derive_sku_store_forecasts')]
    assert 'parent_params = pl.DataFrame' in method
    assert '{"_parent": "none", "_strength": 0.0}' in method
    assert 'for parent_name in ("store", "section")' in method
    assert 'LEAF_DRIVER_STRENGTH_CANDIDATES' in method
    assert 'pl.col("_parent") == "store"' in method
    assert "_store_ey" in method and "_store_ev" in method
    assert "_sec_ey" in method and "_sec_ev" in method
    assert "parent_choices" in method
    assert "_cae_parent_y" in method and "_cden_parent_y" in method


def test_expanding28_and_no_actual_extension_leakage_preserved():
    r=read('app/forecasting/runner.py')
    m=r[r.index('def _fit_and_predict_expanding_blocks'):r.index('def fit_and_predict_sections')]
    assert 'range(seed_n, n, block_days)' in m
    assert 'boundary = s - 1' in m
    assert 'target_parts.get("actual_extension")' not in m
