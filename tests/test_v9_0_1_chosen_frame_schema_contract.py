from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_release_version_9_0_1():
    assert 'APP_VERSION: str = "9.1"' in read("settings.py")
    assert 'version = "9.1.0"' in read("pyproject.toml")

def test_parent_choices_are_normalized_before_chosen_frame_concat():
    src = read("app/forecasting/runner.py")
    method = src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]
    assert 'parent_choices = parent_choices.select(' in method
    for token in ('"_parent_y"', '"_parent_v"', '"_strength_y"', '"_strength_v"'):
        assert token in method[method.index('parent_choices = parent_choices.select('):method.index('chosen_frame = (')]
    assert method.index("parent_choices = parent_choices.select(") < method.index("chosen_frames.append(chosen_frame)")

def test_chosen_frame_schema_is_guarded_before_vertical_concat():
    src = read("app/forecasting/runner.py")
    method = src[src.index("def fast_leaf_forecasts"):src.index("def derive_sku_store_forecasts")]
    assert "expected_cols = chosen_frames[0].columns" in method
    assert "set(chosen_frame.columns) != set(expected_cols)" in method
    assert "chosen_frame = chosen_frame.select(expected_cols)" in method
    assert "Leaf chosen-frame schema mismatch before concat" in method
    assert method.index("chosen_frame = chosen_frame.select(expected_cols)") < method.index("chosen_states = pl.concat(")
