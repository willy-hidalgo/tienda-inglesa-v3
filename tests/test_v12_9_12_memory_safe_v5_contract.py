from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shape_invariant_is_streaming_not_wide_groupby():
    src = (ROOT / "app" / "forecasting" / "sku_shape_lgbm.py").read_text(encoding="utf-8")
    body = src[src.index("def corrected_forecast"):src.index("def retain_for_targets")]
    assert 'for sku, base_value, corrected_value in zip(' in body
    assert 'v12.6 shape invariant violated' in body
    assert 'out.group_by("_v12_sku")' not in body
    assert '20 MB' in body


def test_shape_context_prunes_to_remaining_exact_causal_windows():
    src = (ROOT / "app" / "forecasting" / "sku_shape_lgbm.py").read_text(encoding="utf-8")
    leaf = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    assert "def retain_for_targets(" in src
    assert "for i in range(1, self.history_blocks + 1)" in src
    assert "self._feature_blocks.pop(origin, None)" in src
    first = leaf.index("shape_context.retain_for_targets([test_start, forecast_start])")
    oos = leaf.index("oos_candidate = _candidate(test_start, test_end)")
    second = leaf.index("shape_context.retain_for_targets([forecast_start])")
    fc = leaf.index("fc_candidate = _candidate(forecast_start, forecast_end)")
    assert first < oos < second < fc
