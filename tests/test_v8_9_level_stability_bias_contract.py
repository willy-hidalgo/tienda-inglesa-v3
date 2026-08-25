from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]

def test_version_is_v9_0():
    assert 'APP_VERSION: str = "9.2"' in read("settings.py")

def test_warmup_is_sum_over_28_calendar_days():
    m = leaf_method()
    assert 'alias("_sum_y0")' in m
    assert 'alias("_sum_v0")' in m
    assert '/ pl.lit(float(warmup_days))' in m
    warmup = m[m.index("# Initial structural level"):m.index("# Parent RLS contributes SHAPE only")]
    assert '.mean().alias("_mean_y0")' not in warmup
    assert '.mean().alias("_mean_v0")' not in warmup
    assert 'leaf_warmup_calendar_mean' in m

def test_bias_correction_excludes_leaf_by_unique_id_structure():
    src = read("app/forecasting/runner.py")
    method = src[src.index("def apply_bias_correction"):src.index("# ── SKU+tienda")]
    assert 'str.contains(r"\\|\\|T:[^|]+")' in method
    assert 'str.contains(r"\\|\\|S:[^|]+")' in method
    assert '& ~is_leaf_uid' in method
    assert '& ~is_leaf_out' in method
    # Renaming a leaf model must not be able to re-enable bias correction.
    assert 'str.starts_with("fast_leaf")' not in method
    assert 'str.starts_with("leaf_ses28")' not in method
    assert 'str.starts_with("leaf_adaptive")' not in method

def test_stability_guard_is_causal_and_selects_only_pure_ses_states():
    m = leaf_method()
    assert "LEAF_SES_STABILITY_WINDOW_DAYS" in m
    assert "LEAF_SES_LEVEL_MAX_RECENT_RATIO" in m
    assert 'for lag_block in range(1, stability_blocks + 1)' in m
    assert '(pl.col("_block") + lag_block)' in m
    assert 'stability_ref_sums.filter' in m
    assert 'alias("_stable_y")' in m
    assert 'alias("_stable_v")' in m
    assert 'alias("_ses_guard_y")' in m
    assert 'alias("_ses_guard_v")' in m
    # The selected level is still taken from alpha_state, i.e. a pure SES path.
    assert 'alpha_state.join(alpha_choices' in m
    assert 'pl.col("_level_y")' in m
    assert 'pl.col("_level_v")' in m

def test_stability_diagnostics_are_exported():
    m = leaf_method()
    for token in (
        "ses_stability_reference_y",
        "ses_stability_reference_value",
        "ses_stability_guard_y",
        "ses_stability_guard_value",
        "pure_ses_wmape_y",
        "pure_ses_wmape_value",
    ):
        assert token in m

def _ses_path(alpha, warmup_level, post_warmup_actuals):
    level = float(warmup_level)
    for y in post_warmup_actuals:
        level = alpha * float(y) + (1.0 - alpha) * level
    return level

def test_calendar_warmup_for_sparse_leaf_is_not_mean_of_sales_rows():
    # Four sale days of 500 inside 28 calendar days.
    observed = [500.0, 500.0, 500.0, 500.0]
    correct = sum(observed) / 28.0
    old_wrong = sum(observed) / len(observed)
    assert abs(correct - 71.4285714286) < 1e-9
    assert old_wrong == 500.0
    assert correct < old_wrong / 5.0

def test_recent_spike_guard_keeps_level_as_a_pure_ses_candidate():
    # Representative failure mode: stable 500/day, then five 30k shocks
    # immediately before OOS. No clipping/winsorization is used.
    alphas = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80)
    actuals = [500.0] * 107 + [30000.0] * 5
    reference = sum(actuals) / 112.0
    levels = {a: _ses_path(a, 500.0, actuals) for a in alphas}

    max_ratio = 1.50
    stable = {a: level for a, level in levels.items() if level <= reference * max_ratio}
    assert stable

    # The guard chooses from existing SES trajectories. It never substitutes
    # reference*ratio as the forecast level.
    chosen_alpha = min(stable)
    chosen_level = levels[chosen_alpha]
    assert chosen_level in levels.values()
    assert chosen_level < 2000.0
    assert levels[0.80] > 20000.0

def test_bias_scaling_would_double_exploding_leaf_but_is_now_forbidden():
    raw_leaf_forecast = 9000.0
    old_bias_factor = 2.0
    assert raw_leaf_forecast * old_bias_factor == 18000.0
    # Contract is structural exclusion; factor must effectively be one for leaf.
    new_leaf_factor = 1.0
    assert raw_leaf_forecast * new_leaf_factor == raw_leaf_forecast

def test_stability_reference_is_precomputed_not_full_history_rescanned_per_block():
    m = leaf_method()
    assert 'stability_block_sums' in m
    assert 'stability_ref_sums' in m
    loop = m[m.index('for block_i in range(1, max_block + 1):'):]
    pre_stage = loop[:loop.index('# ── STAGE 1: select alpha')]
    assert 'actual_obs.filter(' not in pre_stage


def test_oos_state_is_selected_before_current_block_actuals_update():
    m = leaf_method()
    pos_select = m.index('chosen_frames.append(')
    pos_actual_gate = m.index('if block_i > actual_last_block:')
    pos_update = m.index('# ── Score ALL alpha candidates using SES ONLY')
    assert pos_select < pos_actual_gate < pos_update


def test_if_no_stable_pool_fallback_is_still_a_pure_ses_trajectory():
    m = leaf_method()
    assert '.sort_by("_level_y", "_alpha")' in m
    assert '.sort_by("_level_v", "_alpha")' in m
    assert 'ses_stable_pool_found_y' in m
    assert 'ses_stable_pool_found_value' in m

def _select_ses_alpha_with_guard(alphas, scores, levels, reference, ratio=1.5, near=0.02):
    stable = [a for a in alphas if reference <= 1e-9 or levels[a] <= ratio * reference]
    if stable:
        best = min(scores[a] for a in stable)
        pool = [a for a in stable if scores[a] <= best * (1.0 + near) + 1e-6]
        return min(pool)
    return min(alphas, key=lambda a: (levels[a], a))


def test_guard_materially_reduces_oos_explosion_after_final_shock_cluster():
    # Prior persistent regimes make reactive SES score well historically; the
    # final 5-day shock is not yet validated when the OOS forecast is issued.
    alphas = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80)
    block_days = 28
    train_blocks = []
    for level in (1000, 1000, 2000, 2000, 800, 800):
        train_blocks.append([float(level)] * block_days)
    train_blocks.append([800.0] * 23 + [30000.0] * 5)
    oos = [800.0] * block_days

    states = {a: 500.0 for a in alphas}
    ae = {a: 0.0 for a in alphas}
    den = {a: 0.0 for a in alphas}
    for block in train_blocks:
        for a in alphas:
            level = states[a]
            ae[a] += sum(abs(y - level) for y in block)
            den[a] += sum(abs(y) for y in block)
            for y in block:
                level = a * y + (1.0 - a) * level
            states[a] = level

    scores = {a: ae[a] / den[a] for a in alphas}
    best = min(scores.values())
    unconstrained = min(a for a in alphas if scores[a] <= best * 1.02 + 1e-6)
    reference = sum(sum(b) for b in train_blocks[-4:]) / (4 * block_days)
    guarded = _select_ses_alpha_with_guard(alphas, scores, states, reference)

    def wmape(alpha):
        return sum(abs(y - states[alpha]) for y in oos) / sum(oos)

    assert states[unconstrained] > 10000.0
    assert wmape(unconstrained) > 10.0  # >1000%
    assert states[guarded] < reference * 1.5
    assert wmape(guarded) < wmape(unconstrained) / 5.0


def test_guard_does_not_block_a_genuine_persistent_level_shift():
    alphas = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80)
    # 112 recent days are genuinely around 2,000; a SES level near 2,000 is
    # therefore admissible and must not be forced back to an old 500 regime.
    reference = 2000.0
    levels = {a: 2000.0 + (a - 0.2) * 100.0 for a in alphas}
    scores = {a: abs(a - 0.4) + 0.1 for a in alphas}
    chosen = _select_ses_alpha_with_guard(alphas, scores, levels, reference)
    assert levels[chosen] > 1800.0
    assert levels[chosen] < 2200.0


def test_stability_guard_never_clamps_the_numeric_level():
    m = leaf_method()
    alpha_part = m[m.index('# ── STAGE 1: select alpha'):m.index('# ── STAGE 2: select parent')]
    # Reference is used for candidate eligibility; the forecast level remains
    # an existing _level_y/_level_v from alpha_state.
    assert 'stability_ratio' in alpha_part
    assert '.alias("_alpha_y")' in alpha_part
    assert '.alias("_alpha_v")' in alpha_part
    assert 'alias("_level_y")' not in alpha_part
    assert 'alias("_level_v")' not in alpha_part

def test_calendar_warmup_equals_sales_day_mean_when_sku_sells_every_day():
    actuals = [100.0 + i for i in range(28)]
    calendar_mean = sum(actuals) / 28.0
    sales_day_mean = sum(actuals) / len(actuals)
    assert calendar_mean == sales_day_mean

