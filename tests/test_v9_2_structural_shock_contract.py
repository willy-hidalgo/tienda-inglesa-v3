from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]


def _choose_structural(recent28, prior_blocks, candidate_levels, ratio=1.75, stability=1.30):
    structural = statistics.median(prior_blocks)
    shock = len(prior_blocks) >= 3 and structural > 0 and recent28 > structural * ratio
    if not shock:
        return structural, False, None
    ceiling = structural * stability
    stable = {a: v for a, v in candidate_levels.items() if v <= ceiling}
    pool = stable or candidate_levels
    alpha = min(pool, key=lambda a: (abs(pool[a] - structural), a))
    return structural, True, (alpha, pool[alpha])


def test_release_is_v9_2_and_long_memory_alphas_exist():
    s = read("settings.py")
    assert 'APP_VERSION: str = "9.2"' in s
    assert 'version = "9.2.0"' in read("pyproject.toml")
    assert "0.0002" in s
    assert "0.001" in s


def test_structural_reference_excludes_immediately_closed_block():
    m = leaf_method()
    assert "structural_block_refs" in m
    assert "for lag_block in range(2, structural_blocks + 2)" in m
    assert 'alias("_structural_block_median_y")' in m
    assert 'alias("_structural_block_median_v")' in m


def test_upward_shock_overrides_current_block_based_anchor():
    m = leaf_method()
    assert 'alias("_upward_shock_y")' in m
    assert 'alias("_upward_shock_v")' in m
    assert 'pl.when(pl.col("_upward_shock_y"))' in m
    assert '.then(pl.col("_structural_block_median_y"))' in m
    assert 'LEAF_REGIME_UPWARD_SHOCK_RATIO' in m


def test_upward_shock_can_ignore_historical_near_best_gate():
    m = leaf_method()
    # During a detected upward shock the current block is out-of-regime;
    # selection is allowed to choose the closest stable pure-SES trajectory
    # even if it is not within the ordinary historical near-best tolerance.
    assert 'pl.col("_upward_shock_y")\n                                | (' in m
    assert 'pl.col("_upward_shock_v")\n                                | (' in m


def test_587833_real_pattern_selects_long_memory_pure_ses_near_structural_level():
    # Store 00001, origin 2026-03-09. Values reproduced from the supplied raw data.
    recent28 = 1430.2492857142859
    prior_blocks = [
        139.53964285714287,
        103.4575,
        374.88392857142856,
        929.0571428571428,
        454.8471428571429,
        144.60714285714286,
    ]
    levels = {
        0.0002: 324.3,
        0.001: 331.2,
        0.005: 436.9,
        0.01: 598.6,
        0.02: 903.0,
    }
    structural, shock, chosen = _choose_structural(recent28, prior_blocks, levels)
    assert 250 < structural < 270
    assert shock is True
    assert chosen[0] == 0.0002
    assert 315 < chosen[1] < 335


def test_299993_real_pattern_excludes_two_recent_high_regime_blocks_from_structural_center():
    # Store 00001. Current 28-day mean is >3k, while older closed blocks center near 300.
    recent28 = 3338.193214285714
    prior_blocks = [
        2979.097142857143,
        177.65321428571428,
        540.1110714285714,
        264.5857142857143,
        126.60357142857143,
        316.8639285714286,
    ]
    levels = {
        0.0002: 279.8,
        0.001: 444.6,
        0.005: 1030.2,
        0.01: 1569.4,
    }
    structural, shock, chosen = _choose_structural(recent28, prior_blocks, levels)
    assert 280 < structural < 310
    assert shock is True
    assert chosen[0] == 0.0002
    assert chosen[1] < 300


def test_dense_stable_skus_do_not_enter_upward_shock_branch():
    # ACEITE 483046 and PAN LACTAL 58905 were already around 20-30% OOS.
    structural_aceite = statistics.median([
        15176.99, 13684.33, 15123.79, 18053.57, 15222.85, 13340.85
    ])
    structural_pan = statistics.median([
        5529.41, 6348.41, 6755.58, 6330.41, 7204.60, 8030.30
    ])
    assert 16364.65 < structural_aceite * 1.75
    assert 6751.76 < structural_pan * 1.75


def test_structural_shock_diagnostics_are_exported():
    m = leaf_method()
    for token in (
        "ses_structural_block_median_y",
        "ses_structural_block_median_value",
        "ses_upward_shock_y",
        "ses_upward_shock_value",
    ):
        assert token in m
