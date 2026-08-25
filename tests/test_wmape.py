"""
WMAPE según fórmula del cliente:

  WMAPE = Σ |y − ŷ| / Σ |y|   (excluye y = 0 y yhat no finito)

Rankings / backend: bottom-up desde hojas sku+tienda.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import backend  # noqa: E402
import settings  # noqa: E402
from app.forecasting.runner import RLSForecastRunner  # noqa: E402
from app.forecasting.metrics import compute_wmape  # noqa: E402

# Ejemplo (hoja "datos"): local 1 sku 44 con y=0 → excluido
EJEMPLO = [
    # sku, local, yhat, y, abs
    (11, 1, 10, 13, 3),
    (22, 1, 15, 9, 6),
    (33, 1, 3, 10, 7),
    (44, 1, 5, 0, 5),  # y=0 → excluir
    (11, 2, 20, 1, 19),
    (22, 2, 27, 40, 13),
    (33, 2, 7, 2, 5),
    (44, 2, 0, 10, 10),
]


def _ejemplo_df() -> pl.DataFrame:
    rows = [
        {
            "unique_id": f"1||T:{local:05d}||S:{sku}",
            "y": float(y),
            "yhat": float(yhat),
            "period_type": "out_sample",
            "ds": None,
        }
        for sku, local, yhat, y, _ in EJEMPLO
    ]
    return pl.DataFrame(rows)


# ── settings.compute_wmape (listas) ──────────────────────────────────────────


def test_compute_wmape_local_1():
    rows = [r for r in EJEMPLO if r[1] == 1]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    assert abs(settings.compute_wmape(y, yhat) - 0.5) < 1e-9


def test_compute_wmape_local_2():
    rows = [r for r in EJEMPLO if r[1] == 2]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    assert abs(settings.compute_wmape(y, yhat) - 47 / 53) < 1e-9


def test_compute_wmape_total():
    y = [r[3] for r in EJEMPLO]
    yhat = [r[2] for r in EJEMPLO]
    assert abs(settings.compute_wmape(y, yhat) - 63 / 85) < 1e-9


def test_compute_bias_local_1():
    rows = [r for r in EJEMPLO if r[1] == 1 and r[3] != 0]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    assert abs(settings.compute_bias(y, yhat) - (-0.125)) < 1e-9


def test_compute_bias_total():
    rows = [r for r in EJEMPLO if r[3] != 0]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    assert abs(settings.compute_bias(y, yhat) - (82 - 85) / 85) < 1e-9


def test_compute_wmape_all_zero_y():
    assert settings.compute_wmape([0, 0], [1, 2]) == 0.0


# ── RLSForecastRunner._compute_wmape (por period_type) ───────────────────────


def test_runner_wmape_bottom_up_levels():
    w = compute_wmape(_ejemplo_df())

    sku11_l1 = w.filter(pl.col("unique_id") == "1||T:00001||S:11")
    assert sku11_l1.height == 1
    assert abs(sku11_l1["wmape"][0] - 3 / 13) < 1e-9

    store1 = w.filter(pl.col("unique_id") == "1||T:00001")
    assert store1.height == 1
    assert store1["nivel"][0] == "tienda"
    assert abs(store1["wmape"][0] - 16 / 32) < 1e-9

    sec = w.filter(pl.col("unique_id") == "1")
    assert sec.height == 1
    assert sec["nivel"][0] == "seccion"
    assert abs(sec["wmape"][0] - 63 / 85) < 1e-9


# ── backend.wmape_bottom_up / wmape_por_id ────────────────────────────────────


def test_backend_bottom_up_leaf_store_section():
    df = _ejemplo_df()
    bu = backend.wmape_bottom_up(df)

    leaf = bu.filter(pl.col("unique_id") == "1||T:00001||S:11")
    assert leaf.height == 1
    assert abs(leaf["wmape"][0] - 3 / 13) < 1e-9

    store = bu.filter(pl.col("unique_id") == "1||T:00001")
    assert store.height == 1
    assert abs(store["wmape"][0] - 16 / 32) < 1e-9
    assert store["sum_y"][0] == 32.0

    sec = bu.filter(pl.col("unique_id") == "1")
    assert sec.height == 1
    assert abs(sec["wmape"][0] - 63 / 85) < 1e-9
    assert sec["sum_y"][0] == 85.0


def test_backend_bottom_up_pure_sku():
    """SKU puro = suma de hojas de ese SKU en todas las tiendas."""
    df = _ejemplo_df()
    bu = backend.wmape_bottom_up(df)
    # sku 11 en local 1 y 2: abs 3+19=22, y 13+1=14
    pure = bu.filter(pl.col("unique_id") == "1||S:11")
    assert pure.height == 1
    assert abs(pure["wmape"][0] - 22 / 14) < 1e-9


def test_backend_bottom_up_ignores_aggregate_series_yhat():
    """Tienda no debe usar yhat de la serie agregada del nodo tienda."""
    leaves = _ejemplo_df()
    # Nodo tienda con yhat deliberadamente absurdo
    bogus = pl.DataFrame(
        {
            "unique_id": ["1||T:00001"],
            "y": [32.0],
            "yhat": [0.0],
            "period_type": ["out_sample"],
            "ds": [None],
        }
    )
    df = pl.concat([leaves, bogus], how="diagonal_relaxed")
    bu = backend.wmape_bottom_up(df)
    store = bu.filter(pl.col("unique_id") == "1||T:00001")
    assert abs(store["wmape"][0] - 16 / 32) < 1e-9  # sigue siendo bottom-up


def test_wmape_por_id_matches_bottom_up():
    df = _ejemplo_df()
    ids = ["1||T:00001||S:11", "1||T:00001", "1"]
    a = backend.wmape_por_id(ids, df, fill_missing=False)
    b = backend.wmape_bottom_up(df).filter(pl.col("unique_id").is_in(ids))
    for uid in ids:
        wa = float(a.filter(pl.col("unique_id") == uid)["wmape"][0])
        wb = float(b.filter(pl.col("unique_id") == uid)["wmape"][0])
        assert abs(wa - wb) < 1e-12


def test_wmape_por_id_excludes_nan_yhat():
    rows = [
        {
            "unique_id": "1||T:00001||S:11",
            "y": 13.0,
            "yhat": 10.0,
            "period_type": "in_sample",
        },
        {
            "unique_id": "1||T:00001||S:11",
            "y": 9.0,
            "yhat": float("nan"),
            "period_type": "in_sample",
        },
        {
            "unique_id": "1||T:00001||S:11",
            "y": 10.0,
            "yhat": 3.0,
            "period_type": "in_sample",
        },
    ]
    df = pl.DataFrame(rows)
    res = backend.wmape_por_id(["1||T:00001||S:11"], df, n_fechas_spine=3)
    row = res.filter(pl.col("unique_id") == "1||T:00001||S:11")
    assert row.height == 1
    wm = row["wmape"][0]
    assert isinstance(wm, float)
    assert not math.isnan(wm)
    assert abs(wm - (10 / 23)) < 1e-9
    assert row["sum_y"][0] == 23.0
    assert row["n_with_sales"][0] == 2


def test_wmape_por_id_excludes_forecast_only():
    df = pl.DataFrame(
        {
            "unique_id": ["1||T:00001||S:11"] * 3,
            "y": [10.0, 10.0, 10.0],
            "yhat": [8.0, 12.0, 0.0],
            "period_type": ["in_sample", "out_sample", "forecast_only"],
        }
    )
    res = backend.wmape_por_id(["1||T:00001||S:11"], df)
    row = res.filter(pl.col("unique_id") == "1||T:00001||S:11")
    # solo in+out: abs 2+2=4, sum_y=20 → 0.2
    assert abs(row["wmape"][0] - 0.2) < 1e-9
    assert row["n_with_sales"][0] == 2


def test_nesting_sums_leaf_le_store_le_section():
    bu = backend.wmape_bottom_up(_ejemplo_df())
    leaf_y = float(bu.filter(pl.col("unique_id") == "1||T:00001||S:11")["sum_y"][0])
    store_y = float(bu.filter(pl.col("unique_id") == "1||T:00001")["sum_y"][0])
    sec_y = float(bu.filter(pl.col("unique_id") == "1")["sum_y"][0])
    assert leaf_y <= store_y + 1e-9
    assert store_y <= sec_y + 1e-9
