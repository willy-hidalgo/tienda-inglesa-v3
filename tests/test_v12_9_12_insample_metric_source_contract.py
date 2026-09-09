from pathlib import Path


def test_current_insample_kpi_uses_metric_artifact_not_visual_series():
    text = Path('app/dashboard.py').read_text(encoding='utf-8')
    assert '_load_metrics_scope(' in text
    assert '"in_sample"' in text
    assert '_m_in_row = _metrics_in_scope.filter' in text
    # Regression guard: the KPI must not be rebuilt from the aggregated chart series.
    block = text[text.index('# OOS e in-sample oficiales'):text.index('mins, moos, c3, c4')]
    assert 'metrics_in_out_total(' not in block
    assert '_cached_series(' not in block


def test_version_is_v12912():
    text = Path('settings.py').read_text(encoding='utf-8')
    assert 'APP_VERSION: str = "12.9.12"' in text
