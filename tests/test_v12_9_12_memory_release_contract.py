from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lgbm_shape_context_releases_target_local_frames():
    src = (ROOT / "app" / "forecasting" / "sku_shape_lgbm.py").read_text(encoding="utf-8")
    assert "def release_origin(" in src
    assert "self._corrected.pop(origin, None)" in src
    assert "self._base_forecasts.pop(origin, None)" in src
    assert "def close(" in src
    assert "self._feature_blocks.clear()" in src


def test_leaf_closes_shape_context_before_wide_final_join():
    src = (ROOT / "app" / "forecasting" / "leaf_v12.py").read_text(encoding="utf-8")
    close_pos = src.index("shape_context.close()")
    final_join_pos = src.index("rows.with_columns(", close_pos)
    assert close_pos < final_join_pos
    assert "shape_context.release_origin(start)" in src
    assert "gc.collect()" in src


def test_runner_never_filters_full_wide_leaf_frame_for_target_audits():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    assert "target_leaf_rows = rows.filter(" not in src
    assert "def _target_audit_rows(" in src
    assert '.select(keep)' in src
    assert 'collect(engine="streaming")' in src
    assert "10,503,360 bytes" in src


def test_runner_drops_finally_unused_temp_columns_before_target_audits():
    src = (ROOT / "app" / "forecasting" / "runner.py").read_text(encoding="utf-8")
    challenger = src.index("challenger v12.6 LGBM-shape+occurrence+share")
    drop_pos = src.index("rows = rows.drop(drop_tmp)", challenger)
    audit_pos = src.index("def _target_audit_rows(", challenger)
    assert challenger < drop_pos < audit_pos
