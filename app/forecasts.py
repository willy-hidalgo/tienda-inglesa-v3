"""
Pipeline de forecasting jerárquico con Regresión RLS
=====================================================
Niveles: sección, tienda, sku-tienda
  (filtros independientes tienda/SKU — ver settings.make_unique_id).

Modelo por periodo y nivel:
  - In-sample (train) — los 3 niveles usan RLS:
      · sección / tienda: modelo RLS propio
      · sku+tienda: coeficientes de sección o tienda (mejor WMAPE in-sample)
  - OOS (test) y forecast-only:
      · sección / tienda: sigue RLS (mismos coeficientes de train)
      · sku+tienda: efecto(coefs seleccionados) + SES no causal del residuo

Ventanas por sección:
  - train / OOS / forecast-only (ver settings.section_horizons)

Rendimiento:
  - Carga lazy + collect streaming; agregación lazy.
  - Derivación SKU+tienda vectorizada por tienda.
  - Checkpoint por sección (`forecast_seccion_<n>_partial.parquet`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from app.forecasting.config import ForecastConfig
from app.forecasting.pipeline import RLSForecastPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Orquestador – split por sección
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RLS sección→SKU→local")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Threads paralelos (p. ej. RLS por tienda). Default: secuencial.",
    )
    parser.add_argument(
        "--limit-series",
        type=int,
        default=None,
        help=(
            "Debug/benchmark: muestrea N SKUs por sección (conserva niveles "
            "sección/tienda) para iterar rápido. NO usar en producción."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ForecastConfig.from_settings()
    pipeline = RLSForecastPipeline(
        config, n_jobs=args.n_jobs, limit_series=args.limit_series
    )
    res_df, wmapes_df = pipeline.run()
    print(res_df.head())
    print(wmapes_df.head())
    pipeline.save(res_df, wmapes_df)


if __name__ == "__main__":
    main()
