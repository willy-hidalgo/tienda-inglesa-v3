from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]

def test_leaf_candidates_are_vectorized_by_block():
    m = leaf_method()
    assert '.join(alpha_params, how="cross")' in m
    assert '.join(parent_params, how="cross")' in m
    assert "obs_by_block" in m
    assert "chosen_states" in m
    assert "leaf 2-etapas SES puro" in m


def test_leaf_does_not_run_candidate_specific_block_groupbys():
    m = leaf_method()
    old = (
        'for parent_name, (ey_col, ev_col) in effect_cols.items():',
        'for alpha_c in alpha_candidates:',
    )
    for token in old:
        assert token not in m

def test_leaf_keeps_alpha_grid_and_compact_driver_strength_grid():
    m = leaf_method()
    assert 'alpha_params = pl.DataFrame' in m
    assert 'parent_params = pl.DataFrame' in m
    assert 'for a in alpha_candidates' in m
    assert 'for parent_name in ("store", "section")' in m
    assert 'LEAF_DRIVER_STRENGTH_CANDIDATES' in m
    assert '{"_parent": "none", "_strength": 0.0}' in m


def test_leaf_only_retains_selected_states_for_output():
    m = leaf_method()
    assert "chosen_frames" in m
    assert "chosen_states" in m
    assert "state_frames" not in m
    assert "score_frames" not in m
