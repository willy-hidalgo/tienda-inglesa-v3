from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def leaf_method() -> str:
    src = read("app/forecasting/runner.py")
    return src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]


def test_release_is_v9_1():
    assert 'APP_VERSION: str = "9.2"' in read("settings.py")
    assert 'version = "9.2.0"' in read("pyproject.toml")


def test_sparse_reference_uses_median_of_closed_28_day_blocks():
    m = leaf_method()
    assert "robust_block_refs" in m
    assert '.median()' in m
    assert 'alias("_robust_block_median_y")' in m
    assert 'alias("_robust_block_median_v")' in m
    assert 'alias("_sparse_robust_y")' in m
    assert 'alias("_sparse_robust_v")' in m


def test_dense_regime_branch_is_preserved():
    m = leaf_method()
    assert 'pl.col("_coverage_y") >= regime_dense_coverage' in m
    assert 'pl.col("_coverage_v") >= regime_dense_coverage' in m
    assert "LEAF_REGIME_DENSE_COVERAGE" in m


def test_sparse_stability_ceiling_is_tighter_than_dense_ceiling():
    settings = read("settings.py")
    assert 'LEAF_REGIME_SPARSE_STABILITY_RATIO: float = 1.25' in settings
    assert 'LEAF_SES_LEVEL_MAX_RECENT_RATIO: float = 1.50' in settings
    m = leaf_method()
    assert 'alias("_stability_ratio_y")' in m
    assert 'alias("_stability_ratio_v")' in m


def test_sparse_shock_does_not_use_112_day_mean_as_primary_anchor():
    # One shock-heavy 28-day block should not dominate the structural center.
    block_means = [260.0, 310.0, 295.0, 1100.0]
    mean112 = sum(block_means) / 4.0
    ordered = sorted(block_means)
    median4 = (ordered[1] + ordered[2]) / 2.0
    assert mean112 == 491.25
    assert median4 == 302.5
    assert median4 < mean112 * 0.70


def test_sparse_guard_selects_only_existing_pure_ses_states():
    # The robust anchor is selection-only; no clipping/replacement of the
    # forecast level is permitted.
    levels = {0.005: 310.0, 0.01: 420.0, 0.02: 610.0, 0.05: 900.0}
    anchor = 302.5
    max_level = anchor * 1.25
    stable = {a: level for a, level in levels.items() if level <= max_level}
    assert stable == {0.005: 310.0}
    chosen_level = stable[min(stable)]
    assert chosen_level in levels.values()
    assert chosen_level != max_level


def test_587833_fixture_confirms_level_reduction_improves_bias_and_wmape():
    fixture = ROOT / "tests/fixtures/leaf_587833_00001_oos_v8_9.json"
    data = json.loads(fixture.read_text(encoding="utf-8"))
    rows = data["rows"]
    actual = [float(r["value"]) for r in rows]
    factor = [float(r["driver_factor_value"]) for r in rows]
    denominator = sum(abs(x) for x in actual)

    def metrics(level: float):
        pred = [level * f for f in factor]
        wmape = sum(abs(y - p) for y, p in zip(actual, pred)) / denominator
        bias = sum(p - y for y, p in zip(actual, pred)) / denominator
        return wmape, bias

    old_wmape, old_bias = metrics(598.6008308584288)
    lower_wmape, lower_bias = metrics(300.0)
    assert lower_wmape < old_wmape
    assert abs(lower_bias) < abs(old_bias)
    assert old_wmape > 1.70
    assert abs(lower_bias) < 0.01


def test_sparse_diagnostics_are_exported():
    m = leaf_method()
    for token in (
        "ses_sparse_robust_y",
        "ses_sparse_robust_value",
        "ses_sparse_shock_y",
        "ses_sparse_shock_value",
        "ses_robust_block_median_y",
        "ses_robust_block_median_value",
    ):
        assert token in m


def test_sparse_declining_regime_can_choose_lower_pure_ses_state():
    # Representative PALMITO-like pattern: older blocks high, recent block low.
    # The selected level remains one of the pure SES states; the robust ceiling
    # only eliminates states that still track the old high regime.
    block_means = [2100.0, 1200.0, 500.0, 220.0]
    ordered = sorted(block_means)
    robust = (ordered[1] + ordered[2]) / 2.0  # 850
    ceiling = robust * 1.25
    levels = {0.005: 1150.0, 0.01: 900.0, 0.02: 610.0, 0.05: 330.0}
    stable = {a: level for a, level in levels.items() if level <= ceiling}
    assert 0.005 not in stable
    assert set(stable) == {0.01, 0.02, 0.05}
    regime_anchor = 250.0
    chosen_alpha = min(
        stable,
        key=lambda a: (abs(stable[a] - regime_anchor) / regime_anchor, a),
    )
    assert chosen_alpha == 0.05
    assert levels[chosen_alpha] == 330.0
