"""Tests unitarios del DataAggregator (sección → tienda → SKU)."""
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


def test_aggregate_three_depths(sample_sales):
    agg = DataAggregator(
        "SALES_DAY", "SLS_QTY", "SLS_VAL", settings.AGGREGATION_LEVELS
    )
    out = agg.aggregate(sample_sales)
    depths = (
        out.with_columns(
            pl.col("unique_id").str.count_matches(r"\|\|", literal=False).alias("d")
        )
        .select("d")
        .unique()
        .sort("d")["d"]
        .to_list()
    )
    assert depths == [0, 1, 2]


def test_section_ids(sample_sales):
    agg = DataAggregator(
        "SALES_DAY", "SLS_QTY", "SLS_VAL", settings.AGGREGATION_LEVELS
    )
    out = agg.aggregate(sample_sales)
    sections = (
        out.filter(pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 0)[
            "unique_id"
        ]
        .unique()
        .sort()
        .to_list()
    )
    assert sections == ["1", "23"]


def test_store_then_sku_format(sample_sales):
    """unique_id: seccion || store || sku"""
    agg = DataAggregator(
        "SALES_DAY", "SLS_QTY", "SLS_VAL", settings.AGGREGATION_LEVELS
    )
    out = agg.aggregate(sample_sales)
    store = settings.SECCIONES["1"]["locales"][0]
    assert f"1||{store}" in out["unique_id"].to_list()
    assert f"1||{store}||SKU_A" in out["unique_id"].to_list()
    # no debe existir el orden viejo sku||store en el medio
    assert "1||SKU_A" not in out["unique_id"].to_list()


def test_descriptions_attached(sample_sales):
    agg = DataAggregator(
        "SALES_DAY", "SLS_QTY", "SLS_VAL", settings.AGGREGATION_LEVELS
    )
    out = agg.aggregate(sample_sales)
    assert "sku_desc" in out.columns
    assert "store_name" in out.columns
    store = settings.SECCIONES["1"]["locales"][0]
    store_row = out.filter(pl.col("unique_id") == f"1||{store}")
    expected_name = settings.SECCIONES["1"]["local_names"][store]
    assert store_row["store_name"][0] == expected_name
    sku_row = out.filter(pl.col("unique_id") == f"1||{store}||SKU_A")
    assert sku_row["sku_desc"][0] == "Desc SKU_A"


def test_y_sum_section(sample_sales):
    """2 SKU × 2 stores × 10 = 40 por día por sección."""
    agg = DataAggregator(
        "SALES_DAY", "SLS_QTY", "SLS_VAL", settings.AGGREGATION_LEVELS
    )
    out = agg.aggregate(sample_sales)
    sec1 = out.filter(pl.col("unique_id") == "1").sort("ds")
    assert sec1["y"].to_list() == [40.0] * 5


def test_aggregation_levels_order():
    keys = list(settings.AGGREGATION_LEVELS.keys())
    assert keys == ["SECCION", "STORE_ID", "SKU_ID"]
    assert settings.FORECAST_LEVELS == ["seccion", "store", "sku"]
