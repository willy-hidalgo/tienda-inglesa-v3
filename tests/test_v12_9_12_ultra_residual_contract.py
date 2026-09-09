from pathlib import Path


def test_ultra_residual_is_diagnostic_only():
    text = Path('app/forecasting/leaf_v12.py').read_text(encoding='utf-8')
    assert 'def _v12912_ultra_stable_residual_diagnostic' in text
    assert 'v12912_ultra_candidate_value' in text
    assert 'v12912_ultra_replay_pass_value' in text
    # It is joined as metadata; productive selection remains v12_selected_value.
    fn = text[text.index('def _v12912_ultra_stable_residual_diagnostic'):text.index('def _selection_from_closed_blocks')]
    assert 'valuehat' not in fn
    assert 'yhat' not in fn


def test_ultra_gate_is_strict_and_small_budget():
    text = Path('settings.py').read_text(encoding='utf-8')
    assert 'V12912_ULTRA_MIN_WIN_RATE: float = 1.00' in text
    assert 'V12912_ULTRA_MIN_MEDIAN_GAIN: float = 0.0010' in text
    assert 'V12912_ULTRA_MIN_P25_GAIN: float = 0.0005' in text
    assert 'V12912_ULTRA_MIN_WORST_GAIN: float = 0.0' in text
    assert 'V12912_ULTRA_MIN_WEIGHTED_GAIN: float = 0.0010' in text
    assert 'V12912_ULTRA_MAX_BIAS_WORSEN: float = 0.0' in text
    assert 'V12912_ULTRA_MAX_VOLUME_SHARE: float = 0.03' in text
