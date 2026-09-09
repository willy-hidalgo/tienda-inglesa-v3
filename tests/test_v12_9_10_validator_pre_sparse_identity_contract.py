from pathlib import Path


def test_validator_checks_family_identities_against_pre_sparse_value_baseline():
    src = Path("app/forecasting/validate_v11.py").read_text(encoding="utf-8")
    assert 'v12910_valuehat_raw_before_sparse' in src
    assert 'value_family_raw = pl.coalesce' in src
    assert 'value_family_raw.alias("_valuehat_family_raw")' in src
    assert 'pl.col("_valuehat_family_raw") - pl.col("v12_candidate_valuehat_raw")' in src
    assert 'pl.col("_dv") > scale_tol * pl.max_horizontal(pl.col("_valuehat_family_raw").abs(), pl.lit(1.0))' in src
