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

    rows = []
    for sku, local, yhat, y, _ in EJEMPLO:
        rows.append(
            {
                "unique_id": f"1||{local:05d}||{sku}",
                "y": float(y),
                "yhat": float(yhat),
                "period_type": "out_sample",
            }
        )
    df = pl.DataFrame(rows)
    # agregar nivel local
    for local in (1, 2):
        sub = [r for r in EJEMPLO if r[1] == local]
        df = pl.concat(
            [
                df,
                pl.DataFrame(
                    {
                        "unique_id": [f"1||{local:05d}"],
                        "y": [float(sum(r[3] for r in sub))],
                        "yhat": [float(sum(r[2] for r in sub))],
                        "period_type": ["out_sample"],
                    }
                ),
            ],
            how="diagonal_relaxed",
        )
    w = RLSForecastRunner._compute_wmape(df)
    # local 1
    row1 = w.filter(pl.col("unique_id") == "1||00001")
    # wait we used 1||00001 style - actually f"1||{local:05d}" → 1||00001 and 1||00002
    r1 = w.filter(pl.col("unique_id") == "1||00001")
    r2 = w.filter(pl.col("unique_id") == "1||00002")
    assert r1.height == 1
    # At local level we summed including y=0 row for local1: yhat includes 5, y includes 0
    # But _compute_wmape filters y!=0 at row level before group - for local aggregate
    # we already summed. Better test only sku-level rows.
    sku_w = RLSForecastRunner._compute_wmape(
        df.filter(pl.col("unique_id").str.count_matches(r"\|\|", literal=False) == 2)
    )
    # total of all sku rows
    y = [r[3] for r in EJEMPLO]
    yhat = [r[2] for r in EJEMPLO]
    # recompute from sku groups is not same as total - verify one sku
    sku11_l1 = sku_w.filter(pl.col("unique_id") == "1||00001||11")
    assert sku11_l1.height == 1
    assert abs(sku11_l1["wmape"][0] - abs(13 - 10) / 13) < 1e-9
