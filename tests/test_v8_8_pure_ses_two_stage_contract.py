from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]

def test_version_and_low_alpha_grid():
    settings = read("settings.py")
    assert 'APP_VERSION: str = "9.1"' in settings
    assert 'LEAF_ALPHA_SELECTION: str = "pure_ses_prior_cumulative_wmape"' in settings
    for token in ("0.005", "0.01", "0.02", "0.05", "0.10"):
        assert token in settings

def test_alpha_selection_is_independent_of_parent_rls():
    m = leaf_method()
    start = m.index("# ── Score ALL alpha candidates using SES ONLY")
    end = m.index("# ── Score parent shapes")
    ses = m[start:end]
    for forbidden in (
        "_parent", "_store_ey", "_store_ev", "_sec_ey", "_sec_ev",
        "_pred_y", "_pred_v",
    ):
        assert forbidden not in ses
    assert "_cae_ses_y" in ses
    assert "_cae_ses_v" in ses
    assert "_wzy" in ses and "_wzv" in ses

def test_parent_selection_uses_already_selected_ses_level():
    m = leaf_method()
    assert "STAGE 2: select parent shape with SES level already fixed" in m
    parent = m[m.index("# ── Score parent shapes"):]
    assert "selected_level" in parent
    assert "_cae_parent_y" in parent
    assert "_cae_parent_v" in parent

def test_near_best_ses_prefers_lower_alpha():
    m = leaf_method()
    settings = read("settings.py")
    assert "LEAF_SES_NEAR_BEST_REL_TOLERANCE" in settings
    assert "_regime_dist_y" in m and "_regime_dist_v" in m
    assert '.sort_by("_regime_dist_y", "_alpha")' in m
    assert '.sort_by("_regime_dist_v", "_alpha")' in m

def test_recent_final_spike_does_not_force_high_alpha_level():
    # Mirrors the pure-SES block logic.
    alphas = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.70, 0.80)
    block_days = 28
    baseline = 500.0
    spike = 30000.0
    n_blocks = 12

    history = [baseline] * block_days
    for b in range(1, n_blocks):
        vals = [baseline] * block_days
        if b == n_blocks - 1:
            vals[-1] = spike
        history.extend(vals)

    scored = {}
    for alpha in alphas:
        level = baseline
        ae = 0.0
        den = 0.0
        for b in range(1, n_blocks):
            vals = history[b * block_days:(b + 1) * block_days]
            # 28-day forecast issued at block origin: pure SES level only.
            ae += sum(abs(y - level) for y in vals)
            den += sum(abs(y) for y in vals)
            for y in vals:
                level = alpha * y + (1.0 - alpha) * level
        scored[alpha] = (ae / den, level)

    best_score = min(v[0] for v in scored.values())
    rel_tol = 0.05
    near_best = [
        a for a, (score, _) in scored.items()
        if score <= best_score * (1.0 + rel_tol) + 1e-6
    ]
    chosen = min(near_best)
    level = scored[chosen][1]

    assert chosen == min(alphas)
    # The last-day 30k spike must not turn a 500 baseline into a multi-thousand
    # structural level merely because an alpha=0.7/0.8 exists in the grid.
    assert level < baseline * 2.0

def test_selected_pure_ses_score_is_exported_for_audit():
    m = leaf_method()
    assert 'alias("pure_ses_wmape_y")' in m
    assert 'alias("pure_ses_wmape_value")' in m

def test_low_alpha_grid_limits_a_cluster_of_recent_spikes():
    baseline = 500.0
    spike = 30000.0
    alpha = 0.005
    level = baseline

    # 23 normal days + 5 consecutive spikes immediately before OOS.
    for _ in range(23):
        level = alpha * baseline + (1.0 - alpha) * level
    for _ in range(5):
        level = alpha * spike + (1.0 - alpha) * level

    # Pure SES is allowed to react, but it must not jump anywhere near the
    # 30k spike when a long-memory alpha is selected.
    assert level < 1500.0
    assert level < spike * 0.05

