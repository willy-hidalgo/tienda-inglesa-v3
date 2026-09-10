from pathlib import Path


def test_holdout_validation_is_diagnostic_only_and_explicit():
    text = Path("app/forecasting/optimization.py").read_text(encoding="utf-8")
    assert "def validate_driver_refit_candidates" in text
    assert '"validated_candidate"' in text
    assert '"holdout_veto"' in text
    assert '"mixed_holdout"' in text
    assert '"driver_refit_validation.parquet"' in text
    assert "El OOS actual solo puede VALIDAR o VETAR" in text


def test_report_balances_holdout_by_group():
    text = Path("app/forecasting/optimization.py").read_text(encoding="utf-8")
    assert "def _top_per_group" in text
    assert '["section", "target"], 6' in text
    assert '["section", "node_level", "target"], 4' in text
