from pathlib import Path

import settings


def test_memory_safe_settings_contract():
    assert settings.APP_VERSION == "12.9.12"
    assert settings.MULTIBLOCK_MEMORY_SAFE is True
    assert settings.MULTIBLOCK_MEMORY_SAFE_THRESHOLD_DAYS == 28
    assert settings.MULTIBLOCK_MAX_JOBS[1] == 1
    assert settings.MULTIBLOCK_MAX_JOBS[7] == 2
    assert settings.MULTIBLOCK_MAX_JOBS[14] == 4
    assert settings.MULTIBLOCK_MAX_JOBS[28] == 8


def test_leaf_state_spill_contract():
    src = Path("app/forecasting/runner.py").read_text(encoding="utf-8")
    assert "TemporaryDirectory" in src
    assert "chosen_state_" in src
    assert "pl.scan_parquet" in src
    assert 'collect(engine="streaming")' in src
    assert "MULTIBLOCK_MEMORY_SAFE_THRESHOLD_DAYS" in src


def test_pipeline_section_spill_contract():
    src = Path("app/forecasting/pipeline.py").read_text(encoding="utf-8")
    assert 'section_spill_dir = self._cfg.out_dir / "_section_spill"' in src
    assert "section_res_paths" in src
    assert "gc.collect()" in src


def test_cli_caps_jobs_and_can_skip_existing():
    src = Path("app/forecasts.py").read_text(encoding="utf-8")
    assert "MULTIBLOCK_MAX_JOBS" in src
    assert '"--skip-existing"' in src
    assert "effective_jobs = min" in src
    assert "run_status_errors(existing)" in src
    assert "estadísticamente stale" in src


def test_artifacts_are_built_after_releasing_forecast_frames():
    src = Path("app/forecasts.py").read_text(encoding="utf-8")
    assert "build_dashboard=not memory_safe_run" in src
    assert "del res_df, wmapes_df" in src
    assert src.index("del res_df, wmapes_df") < src.index("adir = build_artifacts(forecast_path)")
