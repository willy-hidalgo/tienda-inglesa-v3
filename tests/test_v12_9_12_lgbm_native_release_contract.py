from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lgbm_booster_dataset_is_released_after_each_target_fit():
    src = (ROOT / "app" / "forecasting" / "sku_shape_lgbm.py").read_text(encoding="utf-8")
    assert "import gc" in src
    assert "model.free_dataset()" in src
    assert "del pred, fc, fit, model" in src
    assert "gc.collect()" in src


def test_memory_fix_does_not_change_app_version():
    settings_src = (ROOT / "settings.py").read_text(encoding="utf-8")
    assert 'APP_VERSION: str = "12.9.12"' in settings_src
