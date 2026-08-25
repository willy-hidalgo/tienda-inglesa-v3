from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_app_version_is_centralized_and_shown_in_main_menu():
    settings = read("settings.py")
    main = read("app/main.py")
    assert 'APP_VERSION: str = "9.2"' in settings
    assert '=== Pipeline Tienda Inglesa v{settings.APP_VERSION} ===' in main

def test_leaf_diagnostic_chain_is_logged():
    src = read("app/forecasting/runner.py")
    for token in (
        "_actual_mean_v",
        "_recent_v",
        "_ses_v",
        "_min_factor_v",
        "_mean_factor_v",
        "_max_factor_v",
        "_min_vhat",
        "_mean_vhat",
        "_max_vhat",
        "DIAGNÓSTICO LEAF",
    ):
        assert token in src

def test_mean_forecast_must_match_ses_level():
    src = read("app/forecasting/runner.py")
    settings = read("settings.py")
    assert "Leaf level invariant violated" in src
    assert "_forecast_ses_ratio_v" in src
    assert "LEAF_FORECAST_LEVEL_MEAN_TOLERANCE" in settings

def test_leaf_diagnose_cli_exists():
    src = read("app/forecasting/diagnose_leaf.py")
    assert "--uid" in src
    assert "ses_level_value" in src
    assert "driver_factor_value" in src
    assert "recent28_mean_value" in src
