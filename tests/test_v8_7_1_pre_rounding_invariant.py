from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_version_is_patch_release():
    assert 'APP_VERSION: str = "9.1"' in read("settings.py")

def test_hard_level_invariant_uses_raw_forecast():
    src = read("app/forecasting/runner.py")
    assert 'alias("_yhat_raw")' in src
    assert 'alias("_valuehat_raw")' in src
    assert 'pl.col("_mean_yhat_raw") / pl.col("_ses_y")' in src
    assert 'pl.col("_mean_vhat_raw") / pl.col("_ses_v")' in src
    assert 'pl.col("_mean_yhat") / pl.col("_ses_y")' not in src
    assert 'pl.col("_mean_vhat") / pl.col("_ses_v")' not in src

def test_rounded_and_raw_forecasts_are_both_available():
    src = read("app/forecasting/runner.py")
    assert 'pl.col("_yhat_raw").round(0).alias("yhat")' in src
    assert 'pl.col("_valuehat_raw").round(2).alias("valuehat")' in src
    assert 'pl.col("_yhat_raw").alias("yhat_raw")' in src
    assert 'pl.col("_valuehat_raw").alias("valuehat_raw")' in src

def test_quantization_can_exceed_two_percent_without_model_error():
    # This is exactly the class of false positive that stopped v8.7.
    ses = 1.6
    factors = [1.0] * 28
    raw = [ses * f for f in factors]
    rounded = [round(x) for x in raw]

    raw_ratio = sum(raw) / len(raw) / ses
    rounded_ratio = sum(rounded) / len(rounded) / ses

    assert abs(raw_ratio - 1.0) < 1e-12
    assert abs(rounded_ratio - 1.0) > 0.02

def test_diagnostic_cli_exposes_pre_rounding_forecast():
    src = read("app/forecasting/diagnose_leaf.py")
    assert '"yhat_raw"' in src
    assert '"valuehat_raw"' in src
