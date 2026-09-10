"""Métricas oficiales y diagnósticas: roles separados."""
from __future__ import annotations
import polars as pl
from app import backend


def test_official_metric_excludes_zero_but_all_points_penalizes_it():
    df = pl.DataFrame({
        "unique_id": ["1||T:A||S:X"] * 2,
        "period_type": ["out_sample", "out_sample"],
        "y": [10.0, 0.0],
        "yhat": [8.0, 5.0],
    })
    official = backend.wmape_por_id(["1||T:A||S:X"], df, fill_missing=False)
    assert abs(float(official["wmape"][0]) - 0.2) < 1e-12
    # El cálculo all-points conserva el mismo denominador de volumen real, pero
    # agrega el error del punto y=0 al numerador: (2+5)/10 = 0.7.
    allp = backend.wmape_bottom_up(df)
    leaf = allp.filter(pl.col("unique_id") == "1||T:A||S:X")
    assert abs(float(leaf["wmape_all_points"][0]) - 0.7) < 1e-12
