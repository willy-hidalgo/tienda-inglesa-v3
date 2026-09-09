from pathlib import Path


def test_validator_uses_lazy_projection_and_streaming_collect():
    src = Path("app/forecasting/validate_v11.py").read_text(encoding="utf-8")
    validate_body = src[src.index("def validate("):src.index("def main()")]
    assert "pl.scan_parquet(path)" in validate_body
    assert "scan.select(load_cols)" in validate_body
    assert '.filter(_leaf_expr())' in validate_body
    assert 'is_in(["out_sample", "forecast_only"])' in validate_body
    assert 'collect(engine="streaming")' in validate_body
    assert "pl.read_parquet(path)" not in validate_body


def test_validator_summary_does_not_reload_full_forecast():
    src = Path("app/forecasting/validate_v11.py").read_text(encoding="utf-8")
    main_body = src[src.index("def main()") :]
    assert "summary_scan = pl.scan_parquet(path)" in main_body
    assert "summary_scan.select(summary_cols)" in main_body
    assert 'collect(engine="streaming")' in main_body
    assert "pl.read_parquet(path)" not in main_body
