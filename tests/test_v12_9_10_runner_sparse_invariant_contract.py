from pathlib import Path


def test_v12910_runner_audits_family_baseline_before_sparse_writeback():
    src = Path("app/forecasting/runner.py").read_text(encoding="utf-8")
    assert 'value_identity_col = (' in src
    assert '"v12910_valuehat_raw_before_sparse"' in src
    assert '(pl.col(value_identity_col) - pl.col("_expected_v11_valuehat_raw")).abs()' in src
    assert '(pl.col(value_identity_col) - pl.col("v12_candidate_valuehat_raw")).abs()' in src


def test_v12910_runner_has_separate_sparse_writeback_invariant():
    src = Path("app/forecasting/runner.py").read_text(encoding="utf-8")
    assert 'sparse10_required = {' in src
    assert 'expected_sparse_apply = (' in src
    assert 'expected_sparse_raw = pl.when(expected_sparse_apply).then(' in src
    assert 'v12.9.10 sparse Value invariant violated' in src
