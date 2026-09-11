from pathlib import Path


def test_exception_lab_reconstructs_calendar_exposure_for_sparse_leaf_history():
    text = Path("app/forecasting/exception_lab.py").read_text(encoding="utf-8")
    assert 'alias("history_start")' in text
    assert 'alias("n_observed_rows")' in text
    assert 'alias("n_days")' in text
    assert '_section_history_end' in text
    assert 'np.arange(start, end + np.timedelta64(1, "D")' in text
    assert 'actual = np.zeros(dates.size, dtype=float)' in text
    assert 'champion = np.full(dates.size, np.nan, dtype=float)' in text
