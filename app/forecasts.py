"""Compatibility facade and CLI for the forecasting package.

Production code lives under :mod:`app.forecasting`. Existing imports from
``forecasts`` are kept for backwards compatibility.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.forecasting import (
    CalendarFeatureBuilder, DataAggregator, ForecastConfig, HolidayCalendar,
    RLSForecastPipeline, RLSForecastRunner, compute_wmape, densify_section_panel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RLS jerárquico")
    parser.add_argument("--n-jobs", type=int, default=None, help="Threads paralelos por tienda")
    parser.add_argument("--limit-series", type=int, default=None, help="Debug: limitar SKU por sección")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    config = ForecastConfig.from_settings()
    pipeline = RLSForecastPipeline(config, n_jobs=args.n_jobs, limit_series=args.limit_series)
    res_df, wmapes_df = pipeline.run()
    print(res_df.head())
    print(wmapes_df.head())
    pipeline.save(res_df, wmapes_df)

if __name__ == "__main__":
    main()
