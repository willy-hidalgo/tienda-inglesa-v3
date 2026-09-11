from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "app" / "forecasting" / "oos_boundary_audit.py").read_text(encoding="utf-8")
LEAF = (ROOT / "app" / "forecasting" / "leaf_ses_rls.py").read_text(encoding="utf-8")


def test_boundary_audit_is_diagnostic_only():
    assert "write_parquet(out_summary)" in SRC
    assert "write_parquet(out_leaves)" in SRC
    assert "forecast.parquet" in SRC
    assert "forecast.write_parquet" not in SRC
    assert "settings.APP_VERSION =" not in SRC


def test_boundary_audit_reconstructs_productive_leaf_identity():
    assert "first_oos_uncapped_reconstructed" in SRC
    assert "first_oos_capped_reconstructed" in SRC
    assert "driver_factor_y" in SRC
    assert "driver_factor_value" in SRC
    assert "leaf_forecast_cap_y" in SRC
    assert "leaf_forecast_cap_value" in SRC
    assert "np.expm1(np.log1p(max(ly, 0.0)) + np.log(fy))" in LEAF


def test_boundary_audit_reports_level_factor_and_final_ratios():
    for token in (
        "ratio_level_vs_recent",
        "ratio_after_factor_vs_level",
        "ratio_final_vs_recent",
        "aggregate_boundary_delta",
        "boundary_rank",
    ):
        assert token in SRC
