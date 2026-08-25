"""
Test de equivalencia del refactor de Fase 2 (EDP vectorizado).

`decompose_price` ya soportaba `indexors: Sequence[slice]` para procesar
varias series en una sola llamada numba. Antes, `_calculate_edp` no lo
usaba: partía el panel en un DataFrame por serie (`partition_by`) y llamaba
`decompose_price` una vez por serie desde Python. Ahora se construye la
lista de slices por grupo sobre el panel ya ordenado y se hace una única
llamada batched.

Este test verifica que ambos caminos (batched vs loop por serie) producen
exactamente los mismos `asp` / `edp` / `discount`, fila por fila.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from app.forecasting.pipeline import RLSForecastPipeline

try:
    from rls_opt.edp import decompose_price
except ImportError:  # pragma: no cover
    decompose_price = None


def _synthetic_panel() -> pl.DataFrame:
    """3 series de largo distinto, con algunos días sin venta (y=0)."""
    rows = []
    rng = np.random.default_rng(0)
    specs = {"1||a": 40, "1||b": 25, "1||c": 60}
    for uid, n in specs.items():
        y = rng.integers(0, 6, size=n).astype(float)
        # forzar algunos ceros explícitos (día sin venta)
        y[::7] = 0.0
        price = np.where(y > 0, 9.99 + rng.normal(0, 1.5, size=n).clip(min=0.5), 0.0)
        for i in range(n):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": i,  # orden relativo alcanza para este test
                    "y": float(y[i]),
                    "value": float(price[i] * y[i]) if y[i] > 0 else 0.0,
                }
            )
    return pl.DataFrame(rows)


@pytest.mark.skipif(decompose_price is None, reason="rls_opt no disponible")
def test_edp_batched_matches_per_series_loop():
    panel = _synthetic_panel()

    batched = RLSForecastPipeline._calculate_edp(panel)
    looped = RLSForecastPipeline._calculate_edp_per_series(
        panel.sort(["unique_id", "ds"])
    )

    batched_sorted = batched.sort(["unique_id", "ds"])
    looped_sorted = looped.sort(["unique_id", "ds"])

    assert batched_sorted.height == looped_sorted.height
    for col in ("asp", "edp", "discount"):
        a = np.asarray(batched_sorted[col].to_list(), dtype=np.float64)
        b = np.asarray(looped_sorted[col].to_list(), dtype=np.float64)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9), f"columna '{col}' difiere"


@pytest.mark.skipif(decompose_price is None, reason="rls_opt no disponible")
def test_edp_batch_falls_back_on_failure(monkeypatch):
    """Si la llamada batched con indexors falla, debe caer al loop por serie
    en vez de propagar la excepción (robustez de producción)."""
    import forecasts

    panel = _synthetic_panel()
    original = forecasts.decompose_price

    def _boom_only_for_indexors(*args, **kwargs):
        # Simula un fallo específico del camino batched (p.ej. numba no
        # soporta una lista reflejada de slices en esta versión), dejando
        # intacto el comportamiento por-serie (sin indexors) que usa el
        # fallback.
        if kwargs.get("indexors") is not None:
            raise RuntimeError("numba boom (batched indexors)")
        return original(*args, **kwargs)

    monkeypatch.setattr(forecasts, "decompose_price", _boom_only_for_indexors)
    out = RLSForecastPipeline._calculate_edp(panel)
    assert out.height == panel.height
    assert {"asp", "edp", "discount"} <= set(out.columns)


def test_edp_fast_fallback_when_many_series():
    """Con > 5000 series, se usa el modo rápido vectorizado (aproximado),
    sin llamar a decompose_price en absoluto."""
    n_series = 5001
    panel = pl.DataFrame(
        {
            "unique_id": [f"1||{i}" for i in range(n_series)],
            "ds": [0] * n_series,
            "y": [1.0] * n_series,
            "value": [10.0] * n_series,
        }
    )
    out = RLSForecastPipeline._calculate_edp(panel)
    assert out.height == n_series
    assert out["discount"].to_list() == [0.0] * n_series
