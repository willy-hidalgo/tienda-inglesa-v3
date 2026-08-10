"""Tests del backend y del dashboard (sin Streamlit)."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import backend
from backend import build_dashboard_context, prepare_dashboard_state


def _sample_forecast() -> pl.DataFrame:
    rows = []
    base = dt.date(2025, 11, 1)
    for i in range(40):
        d = base + dt.timedelta(days=i)
        period = "out_sample" if d >= dt.date(2025, 12, 1) else "in_sample"
        if d > dt.date(2025, 12, 7):
            period = "forecast_only"
            y, yhat = 0.0, 12.0
        else:
            y, yhat = 10.0 + i % 5, 9.0 + i % 4
        rows.append(
            {
                "unique_id": "23",
                "ds": d,
                "y": y,
                "yhat": yhat,
                "value": y * 10,
                "valuehat": yhat * 10,
                "period_type": period,
                "sku_desc": "",
                "store_name": "",
                "seccion": "23",
                "train_end": dt.date(2025, 11, 30),
                "test_start": dt.date(2025, 12, 1),
                "test_end": dt.date(2025, 12, 7),
                "forecast_start": dt.date(2025, 12, 8),
                "forecast_end": dt.date(2026, 2, 1),
            }
        )
        # child series for ranking
        rows.append(
            {
                **{k: v for k, v in rows[-1].items() if k != "unique_id"},
                "unique_id": "23||00155",
                "store_name": "DELSOL",
            }
        )
    return pl.DataFrame(rows)


def test_detail_view_excludes_horizon_cols():
    df = _sample_forecast()
    out = backend.detail_view(df)
    for c in backend.HORIZON_COLUMNS:
        assert c not in out.columns
    assert "abs_error" in out.columns
    assert set(out.columns) <= set(backend.DETAIL_COLUMNS)


def test_wmape_excluye_y_cero():
    df = pl.DataFrame(
        {
            "unique_id": ["1||a", "1||a", "1||b"],
            "y": [10.0, 0.0, 20.0],
            "yhat": [8.0, 5.0, 18.0],
            "period_type": ["out_sample"] * 3,
        }
    )
    w = backend.wmape_por_id(["1||a", "1||b"], df)
    r = w.filter(pl.col("unique_id") == "1||a")
    assert abs(r["wmape"][0] - 0.2) < 1e-9
    # n_with_sales = puntos y≠0; n_points = spine (si no se pasa, n_unique ds)
    assert r["n_with_sales"][0] == 1


def test_ranking_wmape_table_columns():
    base = pl.DataFrame(
        {
            "unique_id": ["1||00122", "1||00154"],
            "wmape": [0.1, 0.2],
            "sum_y": [100.0, 50.0],
            "n_with_sales": [3, 5],
            "n_ds": [10, 10],
        }
    )
    desc = {"1||00122": "CENTRAL", "1||00154": "HIPER"}
    t = backend.ranking_wmape_table(base, "1", desc, n_points_total=10)
    # tabla única con los nombres EXACTOS pedidos
    assert list(t.columns) == [
        "Código",
        "Descripción",
        "wMAPE (%)",
        "Rotación",
        "N puntos ≠0",
        "% ≠0",
        "unique_id",
    ]
    assert t.height == 2
    # N puntos ≠0 = períodos con venta; % ≠0 sobre el total (spine)
    assert t["N puntos ≠0"].to_list() == [3, 5]
    assert t["% ≠0"].to_list() == ["30.0%", "50.0%"]
    assert t["Rotación"].to_list() == ["100.00", "50.00"]


def test_ranking_wmape_excluye_cero_y_seleccionado():
    base = pl.DataFrame(
        {
            "unique_id": ["1||a", "1||b", "1||c"],
            "wmape": [0.0, 0.25, 0.3],
            "sum_y": [0.0, 80.0, 60.0],
            "n_with_sales": [0, 4, 3],
            "n_ds": [10, 10, 10],
        }
    )
    t = backend.ranking_wmape_table(base, "1||b", {}, 10)
    # excluye el seleccionado y las series sin ventas (wmape=0)
    assert t["unique_id"].to_list() == ["1||c"]


def test_ranking_wmape_hierarchical_only_direct_children():
    """La tabla de ranking es JERÁRQUICA: solo los hijos directos del nivel
    seleccionado; nunca series de otros niveles ni de otras secciones."""
    base = pl.DataFrame(
        {
            "unique_id": [
                "1",
                "1||00122",
                "1||00063",
                "1||00122||SKU_A",
                "23",
                "23||00155",
            ],
            "wmape": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            "sum_y": [1000.0, 500.0, 300.0, 50.0, 900.0, 400.0],
            "n_with_sales": [10, 9, 8, 7, 10, 6],
            "n_ds": [30, 30, 30, 30, 30, 30],
        }
    )
    # nivel Sección 1 → SOLO sus tiendas (nada de sección 23, ni SKUs)
    t = backend.ranking_wmape_table(
        base, "1", {}, 30, candidatos=["1||00122", "1||00063"]
    )
    assert set(t["unique_id"].to_list()) == {"1||00122", "1||00063"}
    # nivel tienda 00122 → SOLO sus SKUs
    t2 = backend.ranking_wmape_table(
        base, "1||00122", {}, 30, candidatos=["1||00122||SKU_A"]
    )
    assert t2["unique_id"].to_list() == ["1||00122||SKU_A"]


def test_build_chart_series_keys():
    df = _sample_forecast().filter(pl.col("unique_id") == "23")
    ch = backend.build_chart_series(
        df,
        test_end=dt.date(2025, 12, 7),
        forecast_start=dt.date(2025, 12, 8),
        forecast_end=dt.date(2026, 2, 1),
        cutoff=dt.date(2025, 11, 30),
        has_period=True,
    )
    assert "hist_ds" in ch and "fcst_yhat" in ch
    assert "cutoff" in ch


def test_prepare_dashboard_state():
    df = _sample_forecast()
    view = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="23",
        cutoff_date=dt.date(2025, 11, 30),
        candidatos=["23", "23||00155"],
        nombre_nivel="seccion",
    )
    assert view.selected_id == "23"
    assert "in" in view.metrics and "out" in view.metrics
    assert view.detail.height > 0
    assert "train_end" not in view.detail.columns
    assert view.chart["hist_ds"] is not None
    # sin columnas de rolling → has_rolling False y métricas_28 en cero
    assert view.has_rolling is False
    assert view.metrics_28 == {"wmape_28": 0.0, "bias_28": 0.0, "n": 0}
    assert view.ranking_total_points > 0
    assert {
        "Código",
        "Descripción",
        "wMAPE (%)",
        "Rotación",
        "N puntos ≠0",
        "% ≠0",
        "unique_id",
    } <= set(view.ranking_wmape.columns)
    # la API con contexto es equivalente a la de compatibilidad
    ctx = build_dashboard_context(df, "Unidades")
    view2 = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="23",
        cutoff_date=dt.date(2025, 11, 30),
        candidatos=["23", "23||00155"],
        nombre_nivel="seccion",
    )
    assert view2.ranking_wmape.height == view.ranking_wmape.height


def test_build_dashboard_context_preagregado():
    df = _sample_forecast()
    ctx = build_dashboard_context(df, "Unidades")
    # per_id: una fila por unique_id; wmape de la sección 23 (sin el child… ambos)
    ids = ctx.per_id["unique_id"].to_list()
    assert "23" in ids and "23||00155" in ids
    assert {"wmape", "sum_y", "n_with_sales", "n_ds"} <= set(ctx.per_id.columns)
    # all_ids ordenado
    assert ctx.all_ids == sorted(ctx.all_ids)
    assert ctx.all_ids == ["23", "23||00155"]
    assert ctx.has_rolling is False
    assert "23" in ctx.label_map


def test_dashboard_context_has_rolling_when_columns_present():
    df = _sample_forecast().with_columns(
        (pl.col("yhat") * 0.9).alias("yhat28"),
        (pl.col("valuehat") * 0.9).alias("valuehat28"),
    )
    ctx = build_dashboard_context(df, "Unidades")
    assert ctx.has_rolling is True
    view = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="23",
        cutoff_date=dt.date(2025, 11, 30),
        candidatos=["23", "23||00155"],
        nombre_nivel="seccion",
    )
    assert view.has_rolling is True


def test_dashboard_sku_level_show_store():
    """A nivel SKU (sec||store||sku) el view expone store_id / store_name."""
    df = pl.DataFrame(
        {
            "unique_id": "1||00001||SKU_A",
            "ds": [dt.date(2026, 3, 1) + dt.timedelta(days=i) for i in range(40)],
            "y": [1.0] * 40,
            "yhat": [0.9] * 40,
            "value": [10.0] * 40,
            "valuehat": [9.0] * 40,
            "period_type": ["in_sample"] * 40,
            "seccion": ["1"] * 40,
        }
    )
    view = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="1||00001||SKU_A",
        cutoff_date=None,
        candidatos=["1||00001||SKU_A"],
        nombre_nivel="sku",
    )
    assert view.store_id == "00001"
    assert view.store_name == "CENTRAL"  # settings.SECCIONES['1']['local_names']
    # sin nivel SKU no hay contexto de tienda
    root = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="1||00001",
        cutoff_date=None,
        candidatos=["1||00001"],
        nombre_nivel="store",
    )
    assert root.store_id is None and root.store_name is None


def test_format_detail_display():
    df = pl.DataFrame(
        {
            "unique_id": ["1"],
            "ds": [dt.date(2024, 1, 1)],
            "y": [12345.2],
            "yhat": [1000.9],
            "value": [999999.4],
            "valuehat": [0.4],
            "abs_error": [234.6],
        }
    )
    out = backend.format_detail_display(df)
    assert out["y"][0] == "12,345"
    assert out["yhat"][0] == "1,001"
    assert out["value"][0] == "999,999"
    assert out["valuehat"][0] == "0"
    assert out["abs_error"][0] == "235"
def test_dashboard_effects_section_only():
    """Efectos causales (ef_*): se exponen SOLO a nivel sección (gráfico y
    detalle); a nivel tienda/SKU no aparecen en el detalle."""
    df = _sample_forecast().with_columns(
        pl.when(pl.col("unique_id") == "23")
        .then(pl.lit(6.2))
        .otherwise(None)
        .alias("ef_level"),
        pl.when(pl.col("unique_id") == "23")
        .then(pl.lit(-0.35))
        .otherwise(None)
        .alias("ef_trend"),
        pl.when(pl.col("unique_id") == "23")
        .then(pl.lit(0.05))
        .otherwise(None)
        .alias("ef_seasonality"),
    )
    ctx = build_dashboard_context(df, "Unidades")
    assert ctx.has_effects is True

    kwargs = dict(
        unidad="Unidades",
        freq="Diario",
        cutoff_date=dt.date(2025, 11, 30),
        candidatos=["23", "23||00155"],
        nombre_nivel="seccion",
    )
    view = prepare_dashboard_state(
        df, selected_id="23", **{k: v for k, v in kwargs.items()}
    )
    assert view.has_effects is True
    assert "ef_level" in view.chart["effects"]
    assert "effect_names" in view.chart
    assert "ef_level" in view.detail.columns

    # A nivel tienda: sin efectos en la tabla detallada
    child_view = prepare_dashboard_state(
        df,
        unidad="Unidades",
        freq="Diario",
        selected_id="23||00155",
        cutoff_date=dt.date(2025, 11, 30),
        candidatos=["23||00155"],
        nombre_nivel="store",
    )
    assert child_view.has_effects is True
    assert "ef_level" not in child_view.detail.columns


def test_format_detail_display_effects():
    df = pl.DataFrame(
        {
            "unique_id": ["1"],
            "ds": [dt.date(2024, 1, 1)],
            "y": [100.0],
            "yhat": [110.0],
            "ef_level": [4.6052],
            "ef_trend": [-0.1],
            "ef_price": [0.0],
        }
    )
    out = backend.format_detail_display(df)
    assert out["ef_level"][0] == "+4.6052"
    assert out["ef_trend"][0] == "-0.1000"
    assert out["ef_price"][0] == "+0.0000"
    assert out["y"][0] == "100"
