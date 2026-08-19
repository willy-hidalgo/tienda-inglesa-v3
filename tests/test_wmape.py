"""
WMAPE según fórmula del cliente (ejemplo.xlsx):

  WMAPE = Σ_i |y_i − ŷ_i| / Σ_j y_j

Se excluyen filas con y = 0.
Equivale a Σ_i (|err_i|/y_i) · (y_i / Σ y).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from forecasts import RLSForecastRunner
from app.forecasting.metrics import compute_wmape

import settings

# Datos del ejemplo (hoja "datos"), excluyendo ventas=0
# local 1: sku 11,22,33  (44 tiene ventas=0 → excluido)
# local 2: sku 11,22,33,44
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


def test_compute_wmape_local_1():
    rows = [r for r in EJEMPLO if r[1] == 1]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    # excluye y=0 → abs 3+6+7=16, sum y=13+9+10=32 → 0.5
    assert abs(settings.compute_wmape(y, yhat) - 0.5) < 1e-9


def test_compute_wmape_local_2():
    rows = [r for r in EJEMPLO if r[1] == 2]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    # 47/53
    expected = 47 / 53
    assert abs(settings.compute_wmape(y, yhat) - expected) < 1e-9


def test_compute_wmape_total():
    y = [r[3] for r in EJEMPLO]
    yhat = [r[2] for r in EJEMPLO]
    # excluye y=0 de local1 sku44 → 63/85
    expected = 63 / 85
    assert abs(settings.compute_wmape(y, yhat) - expected) < 1e-9


def test_compute_bias_local_1():
    rows = [r for r in EJEMPLO if r[1] == 1 and r[3] != 0]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    # (28-32)/32 = -0.125
    assert abs(settings.compute_bias(y, yhat) - (-0.125)) < 1e-9


def test_compute_bias_total():
    rows = [r for r in EJEMPLO if r[3] != 0]
    y = [r[3] for r in rows]
    yhat = [r[2] for r in rows]
    # (82-85)/85 ≈ -0.035294
    expected = (82 - 85) / 85
    assert abs(settings.compute_bias(y, yhat) - expected) < 1e-9


def test_compute_wmape_all_zero_y():
    assert settings.compute_wmape([0, 0], [1, 2]) == 0.0


def test_runner_wmape_matches_formula():
    import polars as pl

    # Formato actual: "sec||T:local||S:sku"
    rows = []
    for sku, local, yhat, y, _ in EJEMPLO:
        rows.append(
            {
                "unique_id": f"1||T:{local:05d}||S:{sku}",
                "y": float(y),
                "yhat": float(yhat),
                "period_type": "out_sample",
            }
        )
    df = pl.DataFrame(rows)
    w = compute_wmape(df)

    # Hoja: local 1 sku 11 → e=|13-10|=3, sum_y=13
    sku11_l1 = w.filter(pl.col("unique_id") == "1||T:00001||S:11")
    assert sku11_l1.height == 1
    assert abs(sku11_l1["wmape"][0] - abs(13 - 10) / 13) < 1e-9

    # Tienda 00001 (bottom-up, excluye y=0): |3+6+7|/(13+9+10)=16/32=0.5
    store1 = w.filter(pl.col("unique_id") == "1||T:00001")
    assert store1.height == 1
    assert store1["nivel"][0] == "tienda"
    assert abs(store1["wmape"][0] - 16 / 32) < 1e-9

    # Sección 1 (bottom-up sobre hojas con y != 0)
    sec = w.filter(pl.col("unique_id") == "1")
    assert sec.height == 1
    assert sec["nivel"][0] == "seccion"
    y = [r[3] for r in EJEMPLO if r[3] != 0]
    yhat = [r[2] for r in EJEMPLO if r[3] != 0]
    expected_sec = sum(abs(a - b) for a, b in zip(y, yhat)) / sum(y)
    assert abs(sec["wmape"][0] - expected_sec) < 1e-9



def test_wmape_por_id_excludes_nan_yhat():
    """Regresión: yhat NaN (p. ej. parquet stale en in_sample) no debe contagiar
    el WMAPE. Las filas con yhat no finito se ignoran; el resto se calcula
    normalmente (no sale wmape = NaN)."""
    import polars as pl
    import math
    import backend

    rows = [
        {"unique_id": "1||T:00001||S:11", "y": 13.0, "yhat": 10.0,
         "period_type": "in_sample"},
        {"unique_id": "1||T:00001||S:11", "y": 9.0, "yhat": float("nan"),
         "period_type": "in_sample"},   # NaN → excluir del cálculo
        {"unique_id": "1||T:00001||S:11", "y": 10.0, "yhat": 3.0,
         "period_type": "in_sample"},
    ]
    df = pl.DataFrame(rows)
    res = backend.wmape_por_id(["1||T:00001||S:11"], df, n_fechas_spine=3)
    row = res.filter(pl.col("unique_id") == "1||T:00001||S:11")
    assert row.height == 1
    wm = row["wmape"][0]
    assert isinstance(wm, float)
    assert not math.isnan(wm)                      # no se propaga NaN
    # abs_error = |13-10| + |10-3| = 10 ; sum_y = 13+10 = 23
    assert abs(wm - (10 / 23)) < 1e-9
    assert row["sum_y"][0] == 23.0                # solo filas finitas
    assert row["n_with_sales"][0] == 2            # NaN yhat descartado
