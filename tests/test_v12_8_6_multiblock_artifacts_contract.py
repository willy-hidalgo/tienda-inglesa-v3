from pathlib import Path


def _text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_each_update_block_has_own_dashboard_dir_contract():
    src = _text("app/dashboard_artifacts.py")
    assert 'Path(settings.UPDATE_BLOCKS_DIR).resolve()' in src
    assert 'str(rel.parts[0]).startswith("block_")' in src
    assert 'return fpath.parent / "dashboard"' in src


def test_forecasts_skip_existing_repairs_dashboard_only():
    src = _text("app/forecasts.py")
    assert "regenerando SOLO artefactos dashboard" in src
    assert "artifacts_exist(existing)" in src
    assert "build_artifacts(existing)" in src


def test_dashboard_artifacts_cli_can_skip_synced_scenarios():
    src = _text("app/dashboard_artifacts.py")
    assert '"--skip-existing"' in src
    assert "args.skip_existing and artifacts_exist(fp)" in src


def test_dashboard_multiblock_never_falls_back_to_full_parquet():
    src = _text("app/dashboard.py")
    assert "No se cargará `forecast.parquet` completo" in src
    assert "--update-block-days {selected_update_days}" in src
