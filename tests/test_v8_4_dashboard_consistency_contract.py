from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")

def test_selected_rows_are_not_excluded_from_rankings():
    src = read("app/dashboard_data.py")
    assert "exclude=store" not in src
    assert "exclude=sku" not in src
    assert "_pin_selected" in src
    assert '"Sel."' in src

def test_sidebar_filters_are_bidirectionally_compatible():
    src = read("app/dashboard.py")
    assert "Coherencia bidireccional" in src
    assert "stores_for_sku" in src
    assert "skus_for_store" in src

def test_official_metric_caption_is_always_bottom_up():
    src = read("app/dashboard.py")
    assert "Métricas RLS =" not in src
    assert "WMAPE bottom-up = Σ|y−ŷ|/Σ|y|" in src

def test_dashboard_detects_metric_artifact_mismatch():
    src = read("app/dashboard_data.py")
    assert "_metric_consistency_warnings" in src
    assert "Inconsistencia detectada entre ranking/metrics.parquet y KPI OOS" in src

def test_ranking_percentage_uses_oos_horizon_not_full_history():
    data_src = read("app/dashboard_data.py")
    art_src = read("app/dashboard_artifacts.py")
    dash_src = read("app/dashboard.py")
    assert "ranking_days" in data_src
    assert '"ranking_days": int(ranking_days)' in art_src
    assert 'pl.col("ranking_days").fill_null(0).alias("n_points")' in art_src
    assert "Horizonte OOS:" in dash_src

def test_artifact_version_forces_rebuild_after_semantic_changes():
    src = read("app/dashboard_artifacts.py")
    assert "ARTIFACT_VERSION = 7" in src
    assert 'payload.get("version")' in src


def test_store_ranking_honors_selected_sku_in_fast_and_legacy_paths():
    fast = read("app/dashboard_data.py")
    legacy = read("app/backend.py")
    assert 'pl.col("sku") == peer' in fast
    assert 'pl.col("_sku") == peer' in legacy

def test_ranking_exposes_error_contribution():
    fast = read("app/dashboard_data.py")
    legacy = read("app/backend.py")
    dash = read("app/dashboard.py")
    for src in (fast, legacy):
        assert '"Impacto error (%)"' in src
        assert "scope_abs_error" in src
    assert "participación del nodo en el error absoluto OOS" in dash

def test_offline_dashboard_audit_checks_bottom_up_identities():
    src = read("app/dashboard_consistency.py")
    assert "secciones no cumplen identidad bottom-up" in src
    assert "tiendas no cumplen identidad bottom-up" in src
    assert "SKU puros no cumplen identidad bottom-up" in src


def test_current_selection_is_preserved_even_if_it_fails_new_scope_threshold():
    fast = read("app/dashboard_data.py")
    legacy = read("app/backend.py")
    for src in (fast, legacy):
        assert "selected_row" in src
        assert "fuera criterio" in src
        assert "_eligible" in src
