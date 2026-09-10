"""Tests del backend y dashboard_data (sin Streamlit)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import backend
import settings
from dashboard_data import prepare_dashboard_state


def _sample_forecast() -> pl.DataFrame:
    """
    Sección "23" con: nodo sección, tienda 00155, sku SKU1 (todas las
    tiendas) y tienda 00155 + sku SKU1 — cubre los 4 tipos de nodo del
    nuevo esquema de filtros independientes.
    """
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
        base_row = {
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
        rows.append({**base_row, "unique_id": "23"})
        rows.append({**base_row, "unique_id": "23||T:00155", "store_name": "DELSOL"})
        rows.append({**base_row, "unique_id": "23||S:SKU1", "sku_desc": "LECHE"})
        rows.append(
            {
                **base_row,
                "unique_id": "23||T:00155||S:SKU1",
                "store_name": "DELSOL",
                "sku_desc": "LECHE",
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
            "unique_id": ["1||T:a", "1||T:a", "1||T:b"],
            "y": [10.0, 0.0, 20.0],
            "yhat": [8.0, 5.0, 18.0],
            "period_type": ["out_sample"] * 3,
        }
    )
    w = backend.wmape_por_id(["1||T:a", "1||T:b"], df)
    r = w.filter(pl.col("unique_id") == "1||T:a")
    assert abs(r["wmape"][0] - 0.2) < 1e-9
    assert r["n_with_sales"][0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Filtros cruzados (all_stores_in_section, skus_for_store, etc.)
# ─────────────────────────────────────────────────────────────────────────────
_ALL_IDS = [
    "1",
    "1||T:A",
    "1||T:B",
    "1||S:X",
    "1||S:Y",
    "1||T:A||S:X",
    "1||T:B||S:X",
    "23",
    "23||T:C",
]


def test_secciones_disponibles():
    assert backend.secciones_disponibles(_ALL_IDS) == ["1", "23"]


def test_all_stores_and_skus_in_section():
    assert backend.all_stores_in_section(_ALL_IDS, "1") == ["A", "B"]
    assert backend.all_skus_in_section(_ALL_IDS, "1") == ["X", "Y"]
    assert backend.all_stores_in_section(_ALL_IDS, "23") == ["C"]
    assert backend.all_skus_in_section(_ALL_IDS, "23") == []


def test_stores_for_sku_and_skus_for_store():
    # SKU X existe en tiendas A y B; Y no existe en ninguna combinación
    assert backend.stores_for_sku(_ALL_IDS, "1", "X") == ["A", "B"]
    assert backend.stores_for_sku(_ALL_IDS, "1", "Y") == []
    # Tienda A solo tiene combinación con SKU X
    assert backend.skus_for_store(_ALL_IDS, "1", "A") == ["X"]
    assert backend.skus_for_store(_ALL_IDS, "1", "B") == ["X"]


# ─────────────────────────────────────────────────────────────────────────────
# ranking_table (2 ejes, filtro cruzado)
# ─────────────────────────────────────────────────────────────────────────────
def _tabla_base_ranking() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": [
                "1||T:A", "1||T:B", "1||S:X", "1||S:Y",
                "1||T:A||S:X", "1||T:B||S:X", "1||T:B||S:Y",
            ],
            "wmape": [0.1, 0.2, 0.15, 0.3, 0.05, 0.25, 0.3],
            "sum_y": [100.0, 50.0, 80.0, 20.0, 60.0, 40.0, 20.0],
            "n_points": [10] * 7,
            "n_with_sales": [8, 4, 7, 2, 6, 3, 2],
        }
    )


def test_ranking_table_columns():
    t = backend.ranking_table(
        _tabla_base_ranking(), seccion="1", axis="store", fixed_peer=None,
        exclude=None, desc_map={}, unidad="Unidades",
    )
    assert list(t.columns) == [
        "Código", "Descripción", "wMAPE (%)", "Rotación (unid.)",
        "N puntos", "% ≠0", "unique_id",
    ]


def test_ranking_table_no_filtro_devuelve_nodos_puros():
    tabla = _tabla_base_ranking()
    t_store = backend.ranking_table(
        tabla, seccion="1", axis="store", fixed_peer=None, exclude=None,
        desc_map={}, unidad="Unidades",
    )
    assert sorted(t_store["Código"].to_list()) == ["A", "B"]
    t_sku = backend.ranking_table(
        tabla, seccion="1", axis="sku", fixed_peer=None, exclude=None,
        desc_map={}, unidad="Unidades",
    )
    assert sorted(t_sku["Código"].to_list()) == ["X", "Y"]


def test_ranking_table_order_asc_by_wmape():
    """Ranking ordenado ASCENDENTE por wMAPE (mejor primero)."""
    tabla = _tabla_base_ranking()
    t_store = backend.ranking_table(
        tabla, seccion="1", axis="store", fixed_peer=None, exclude=None,
        desc_map={}, unidad="Unidades",
    )
    # A=0.1, B=0.2 → A primero
    assert t_store["Código"].to_list() == ["A", "B"]
    t_sku = backend.ranking_table(
        tabla, seccion="1", axis="sku", fixed_peer=None, exclude=None,
        desc_map={}, unidad="Unidades",
    )
    # X=0.15, Y=0.3 → X primero
    assert t_sku["Código"].to_list() == ["X", "Y"]


def test_ranking_table_fixed_peer_narrows_to_combo_nodes():
    tabla = _tabla_base_ranking()
    # Tabla tiendas: siempre nodos tienda puros (fixed_peer SKU no mezcla hojas)
    t = backend.ranking_table(
        tabla, seccion="1", axis="store", fixed_peer="X", exclude=None,
        desc_map={}, unidad="Unidades",
    )
    assert t["Código"].to_list() == ["A", "B"]  # ASC por wmape
    # Tienda A fija → SKU en A (hojas tienda+sku), orden ASC
    t = backend.ranking_table(
        tabla, seccion="1", axis="sku", fixed_peer="A", exclude=None,
        desc_map={}, unidad="Unidades",
    )
    assert t["Código"].to_list() == ["X"]


def test_ranking_table_exclude_omits_selected_node():
    tabla = _tabla_base_ranking()
    t = backend.ranking_table(
        tabla, seccion="1", axis="store", fixed_peer=None, exclude="A",
        desc_map={}, unidad="Unidades",
    )
    assert "A" not in t["Código"].to_list()
    assert "B" in t["Código"].to_list()


def test_ranking_table_pct_and_n_puntos():
    tabla = _tabla_base_ranking()
    t = backend.ranking_table(
        tabla, seccion="1", axis="store", fixed_peer=None, exclude=None,
        desc_map={}, unidad="Unidades",
    )
    row_a = t.filter(pl.col("Código") == "A")
    assert row_a["N puntos"][0] == 8
    assert row_a["% ≠0"][0] == "80.0"


def test_ranking_table_valor_label():
    t = backend.ranking_table(
        _tabla_base_ranking(), seccion="1", axis="store", fixed_peer=None,
        exclude=None, desc_map={}, unidad="Valor ($)",
    )
    assert "Rotación ($)" in t.columns


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


# ─────────────────────────────────────────────────────────────────────────────
# prepare_dashboard_state (filtros independientes seccion/store/sku)
# ─────────────────────────────────────────────────────────────────────────────
def test_prepare_dashboard_state_seccion():
    df = _sample_forecast()
    view = prepare_dashboard_state(
        df, unidad="Unidades", freq="Diario", seccion="23", store=None, sku=None,
        cutoff_date=dt.date(2025, 11, 30),
    )
    assert view.selected_id == "23"
    assert view.node_kind == "seccion"
    assert "in" in view.metrics and "out" in view.metrics
    assert view.detail.height > 0
    assert "train_end" not in view.detail.columns
    assert view.chart["hist_ds"] is not None
    # _sample_forecast() no incluye yhat28 → rolling28 no calculado
    assert view.has_rolling28 is False
    assert view.store_context is None
    assert view.n_spine > 0


def test_prepare_dashboard_state_tienda_y_sku_independientes():
    df = _sample_forecast()

    v_store = prepare_dashboard_state(
        df, unidad="Unidades", freq="Diario", seccion="23", store="00155", sku=None,
        cutoff_date=dt.date(2025, 11, 30),
    )
    assert v_store.selected_id == "23||T:00155"
    assert v_store.node_kind == "tienda"
    assert v_store.store_context is None  # sin sku, no hay contexto de tienda

    v_sku = prepare_dashboard_state(
        df, unidad="Unidades", freq="Diario", seccion="23", store=None, sku="SKU1",
        cutoff_date=dt.date(2025, 11, 30),
    )
    assert v_sku.selected_id == "23||S:SKU1"
    assert v_sku.node_kind == "sku"
    assert v_sku.store_context is None  # sku sin tienda = cross-tienda

    v_both = prepare_dashboard_state(
        df, unidad="Unidades", freq="Diario", seccion="23", store="00155", sku="SKU1",
        cutoff_date=dt.date(2025, 11, 30),
    )
    assert v_both.selected_id == "23||T:00155||S:SKU1"
    assert v_both.node_kind == "tienda_sku"
    assert v_both.store_context is not None
    assert "00155" in v_both.store_context
    assert "DELSOL" in v_both.store_context


def test_has_rolling28_true_when_column_present_with_data():
    df = _sample_forecast().with_columns(
        pl.when((pl.col("unique_id") == "23") & (pl.col("period_type") == "in_sample"))
        .then(pl.col("yhat") * 1.05)
        .otherwise(None)
        .alias("yhat28")
    )
    view = prepare_dashboard_state(
        df, unidad="Unidades", freq="Diario", seccion="23", store=None, sku=None,
        cutoff_date=dt.date(2025, 11, 30),
    )
    assert view.has_rolling28 is True

    # Un nodo tienda sin yhat28 propio (columna presente globalmente pero
    # nula para este nodo, tal como ocurre con rolling28 limitado a sección)
    view_store = prepare_dashboard_state(
        df, unidad="Unidades", freq="Diario", seccion="23", store="00155", sku=None,
        cutoff_date=dt.date(2025, 11, 30),
    )
    assert view_store.has_rolling28 is False


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
    # El backend conserva números; la coma de miles es responsabilidad del
    # formatter global de Streamlit, no de una conversión irreversible a texto.
    assert out["y"][0] == 12345.2
    assert out["yhat"][0] == 1000.9
    assert out["value"][0] == 999999.4
    assert out["valuehat"][0] == 0.4
    assert out["abs_error"][0] == 234.6
    assert out.schema["y"] == pl.Float64
