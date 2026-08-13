"""Tests unitarios del DataAggregator (sección / tienda / sku / tienda+sku — filtros independientes)."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from forecasts import DataAggregator  # noqa: E402
import settings  # noqa: E402


@pytest.fixture
def sample_sales() -> pl.DataFrame:
    rows = []
    base = dt.date(2024, 5, 1)
    for sec in ("1", "23"):
        locales = settings.SECCIONES[sec]["locales"][:2]
        for sku in ("SKU_A", "SKU_B"):
            for store in locales:
                for day in range(5):
                    rows.append(
                        {
                            "SALES_DAY": base + dt.timedelta(days=day),
                            "SECCION": sec,
                            "SKU_ID": sku,
                            "STORE_ID": store,
                            "SLS_QTY": 10.0,
                            "SLS_VAL": 100.0,
                            "DESCRIPCION": f"Desc {sku}",
                        }
                    )
    return pl.DataFrame(rows)


def _agg():
    return DataAggregator("SALES_DAY", "SLS_QTY", "SLS_VAL", settings.AGGREGATION_LEVELS)


def test_aggregate_four_levels(sample_sales):
    """sección, sección+tienda, sección+sku, sección+tienda+sku."""
    out = _agg().aggregate(sample_sales)
    kinds = set()
    for uid in out["unique_id"].unique().to_list():
        p = settings.split_unique_id(uid)
        if p["store"] is None and p["sku"] is None:
            kinds.add("seccion")
        elif p["store"] is not None and p["sku"] is None:
            kinds.add("tienda")
        elif p["store"] is None and p["sku"] is not None:
            kinds.add("sku")
        else:
            kinds.add("tienda_sku")
    assert kinds == {"seccion", "tienda", "sku", "tienda_sku"}


def test_section_ids(sample_sales):
    out = _agg().aggregate(sample_sales)
    sections = sorted(
        settings.split_unique_id(uid)["seccion"]
        for uid in out["unique_id"].unique().to_list()
        if settings.split_unique_id(uid)["store"] is None
        and settings.split_unique_id(uid)["sku"] is None
    )
    assert sections == ["1", "23"]


def test_id_format_uses_prefixes(sample_sales):
    """unique_id: sección||T:tienda||S:sku (ver settings.make_unique_id)."""
    out = _agg().aggregate(sample_sales)
    store = settings.SECCIONES["1"]["locales"][0]
    assert f"1||T:{store}" in out["unique_id"].to_list()
    assert "1||S:SKU_A" in out["unique_id"].to_list()
    assert f"1||T:{store}||S:SKU_A" in out["unique_id"].to_list()
    # no debe existir el formato viejo sin prefijos
    assert f"1||{store}" not in out["unique_id"].to_list()


def test_sku_aggregated_across_stores(sample_sales):
    """El nodo sección+sku (sin tienda) suma TODAS las tiendas de la sección."""
    out = _agg().aggregate(sample_sales)
    store = settings.SECCIONES["1"]["locales"][0]
    day0 = dt.date(2024, 5, 1)
    per_store = out.filter(
        (pl.col("unique_id") == f"1||T:{store}||S:SKU_A") & (pl.col("ds") == day0)
    )["y"][0]
    across_stores = out.filter(
        (pl.col("unique_id") == "1||S:SKU_A") & (pl.col("ds") == day0)
    )["y"][0]
    # 2 tiendas configuradas en el fixture, cada una con 10 unidades → 20
    assert across_stores == pytest.approx(2 * per_store)


def test_descriptions_attached(sample_sales):
    out = _agg().aggregate(sample_sales)
    assert "sku_desc" in out.columns
    assert "store_name" in out.columns
    store = settings.SECCIONES["1"]["locales"][0]

    store_row = out.filter(pl.col("unique_id") == f"1||T:{store}")
    expected_name = settings.SECCIONES["1"]["local_names"][store]
    assert store_row["store_name"][0] == expected_name
    assert store_row["sku_desc"][0] == ""

    sku_row = out.filter(pl.col("unique_id") == f"1||T:{store}||S:SKU_A")
    assert sku_row["sku_desc"][0] == "Desc SKU_A"
    assert sku_row["store_name"][0] == expected_name

    sku_only_row = out.filter(pl.col("unique_id") == "1||S:SKU_A")
    assert sku_only_row["sku_desc"][0] == "Desc SKU_A"
    assert sku_only_row["store_name"][0] == ""


def test_y_sum_section(sample_sales):
    """2 SKU × 2 stores × 10 = 40 por día por sección."""
    out = _agg().aggregate(sample_sales)
    sec1 = out.filter(pl.col("unique_id") == "1").sort("ds")
    assert sec1["y"].to_list() == [40.0] * 5


def test_aggregation_levels_order():
    keys = list(settings.AGGREGATION_LEVELS.keys())
    assert keys == ["SECCION", "STORE_ID", "SKU_ID"]
    assert settings.FORECAST_LEVELS == ["seccion", "store", "sku"]
