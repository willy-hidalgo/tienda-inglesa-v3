from pathlib import Path
import numpy as np

from app.forecasting.exception_lab import (
    CHALLENGERS,
    DIAGNOSTIC_BASELINES,
    DEFAULT_MAX_SERIES,
    HARD_MAX_SERIES,
    classify_regime,
    forecast_croston_sba,
    forecast_hurdle_robust,
    forecast_last_positive,
    forecast_positive_median,
    forecast_tsb,
    forecast_zero,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exception_lab_is_not_imported_by_productive_forecast_path():
    for rel in ("app/forecasting/leaf_ses_rls.py", "app/forecasting/runner.py", "app/forecasting/pipeline.py", "app/forecasts.py"):
        assert "exception_lab" not in (ROOT / rel).read_text(encoding="utf-8")


def test_candidate_set_is_small_and_fixed_no_parameter_grid():
    assert set(CHALLENGERS) == {
        "last_positive_naive",
        "positive_median_28",
        "positive_ses_a20",
        "hurdle_robust_84x28",
    }
    assert set(DIAGNOSTIC_BASELINES) == {"zero_naive"}
    assert DEFAULT_MAX_SERIES == 250
    assert HARD_MAX_SERIES == 2000


def test_simple_models_behave_on_intermittent_series():
    x = np.array([0, 0, 10, 0, 0, 10, 0, 0, 10], dtype=float)
    assert forecast_zero(x) == 0.0
    assert forecast_last_positive(x) == 10.0
    assert forecast_positive_median(x) == 10.0
    assert 0.0 < forecast_croston_sba(x) < 10.0
    assert 0.0 < forecast_tsb(x) < 10.0
    assert 0.0 < forecast_hurdle_robust(x) < 10.0


def test_regime_taxonomy_and_dormancy_override():
    assert classify_regime(1.1, 0.2, 0) == "smooth"
    assert classify_regime(1.1, 0.8, 0) == "erratic"
    assert classify_regime(2.0, 0.2, 0) == "intermittent"
    assert classify_regime(2.0, 0.8, 0) == "lumpy"
    assert classify_regime(1.0, 0.1, 40) == "dormant"


def test_lab_v2_contract_has_robust_screening_and_recommendations():
    src = (ROOT / "app/forecasting/exception_lab.py").read_text(encoding="utf-8")
    assert "abs_error_percentile_section_target" in src
    assert "extreme_relative_error" in src
    assert "leaf_exception_recommendation.parquet" in src
    assert "exception_model_runtime.parquet" in src
    assert '"--forecast", "--forecast-path"' in src
    assert "EXTREME_RELATIVE_ERROR" in src
    assert "REVIEW_EXCEPTION" in src
    assert "KEEP_SES_RLS" in src
    assert "INSUFFICIENT_EVIDENCE" in src


def test_exception_lab_module_has_cli_entrypoint():
    from pathlib import Path
    src = Path("app/forecasting/exception_lab.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src
    assert 'raise SystemExit(main())' in src


def test_exception_lab_ignores_non_evaluable_folds_in_bias_and_win_share():
    src = (ROOT / "app/forecasting/exception_lab.py").read_text(encoding="utf-8")
    assert 'pl.col("bias").is_finite()' in src
    assert 'pl.col("_wmape_eval").is_not_null()' in src
    assert 'pl.col("_champ_fold_wmape").is_finite()' in src
    assert '.otherwise(None)' in src
