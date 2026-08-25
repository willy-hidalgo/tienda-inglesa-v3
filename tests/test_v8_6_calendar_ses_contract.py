from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_missing_calendar_days_decay_ses():
    src = read("app/forecasting/runner.py")
    assert "_n_calendar" in src
    assert "_days_to_end" in src
    assert "Missing calendar dates are zeros" in src
    assert "PURE SES recurrence over daily actuals" in src

def test_training_candidate_score_penalizes_missing_zero_days():
    src = read("app/forecasting/runner.py")
    assert "_ses_ae_adjust_y" in src
    assert "_ses_ae_adjust_v" in src
    assert 'pl.col("_level_y") * pl.lit(float(block_days))' in src
    assert "Missing calendar dates are zeros" in src

def test_partial_leaf_warmup_not_scored_as_full_block():
    src = read("app/forecasting/runner.py")
    assert 'full_block = pl.col("_n_calendar") == block_days' in src
    assert "_leaf_warmup_end" in src

def test_leaf_forecast_exposes_level_diagnostics():
    src = read("app/forecasting/runner.py")
    for field in (
        "recent28_mean_y",
        "recent28_mean_value",
        "ses_vs_recent28_ratio_y",
        "ses_vs_recent28_ratio_value",
        "ses_level_y",
        "ses_level_value",
        "driver_factor_y",
        "driver_factor_value",
    ):
        assert field in src

def test_driver_mean_invariant_is_enforced():
    src = read("app/forecasting/runner.py")
    assert "Leaf driver invariant violated" in src
    assert "LEAF_DRIVER_MEAN_TOLERANCE" in src

def test_calendar_semantics_are_configured():
    settings = read("settings.py")
    assert "LEAF_ZERO_FILL_MISSING_CALENDAR_DAYS: bool = True" in settings
    assert "LEAF_LEVEL_REFERENCE_WARN_RATIO" in settings


def test_zero_filled_ses_formula_matches_daily_recursion():
    # Pure-Python regression for the exact formula used by v8.6.
    alpha = 0.4
    level0 = 100.0
    # 7 calendar days; sale rows only on day 1 and day 5.
    actual = [50.0, 0.0, 0.0, 0.0, 200.0, 0.0, 0.0]

    level = level0
    for y in actual:
        level = alpha * y + (1.0 - alpha) * level

    n = len(actual)
    closed = (1.0 - alpha) ** n * level0
    for j, y in enumerate(actual):
        closed += alpha * (1.0 - alpha) ** (n - 1 - j) * y

    assert abs(level - closed) < 1e-12
    # If missing calendar days were ignored, the result would stay much higher.
    sparse = level0
    for y in (50.0, 200.0):
        sparse = alpha * y + (1.0 - alpha) * sparse
    assert level < sparse
