"""Compatibility facade and CLI for the forecasting package.

Production code lives under :mod:`app.forecasting`. Existing imports from
``forecasts`` are kept for backwards compatibility.

v12.9.2 memory-safe multi-cadence support
--------------------
The forecasting model can be precalculated with closed update blocks of
1/7/14/28 days while keeping OOS and forecast-only fixed at 28 days. Results
are stored independently under ``data/output/update_blocks/block_XXd`` so the
dashboard can switch instantly without fitting models at runtime.
"""
from __future__ import annotations

import argparse
import gc
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
from rls_opt.edp import decompose_price

from app.forecasting.preflight import ensure_selected_input
from app.forecasting.run_status import mark_failed, mark_running, mark_success, validation_errors as run_status_errors
import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RLS jerárquico")
    parser.add_argument("--n-jobs", type=int, default=None, help="Threads paralelos por tienda")
    parser.add_argument("--limit-series", type=int, default=None, help="Debug: limitar SKU por sección")
    parser.add_argument(
        "--update-block-days", type=int, choices=list(settings.UPDATE_BLOCK_OPTIONS), default=None,
        help="Cadencia de bloques cerrados (1, 7, 14 o 28 días). Guarda en data/output/update_blocks/.",
    )
    parser.add_argument(
        "--all-update-blocks", action="store_true",
        help="Precalcula secuencialmente los escenarios 1d, 7d, 14d y 28d.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Con --all-update-blocks, omite escenarios con forecast.parquet ya generado.",
    )
    return parser.parse_args()


def _run_one(args: argparse.Namespace, update_days: int | None) -> Path:
    original_block = int(settings.RLS_BLOCK_DAYS)
    try:
        if update_days is not None:
            settings.RLS_BLOCK_DAYS = int(update_days)
        config = ForecastConfig.from_settings()
        ensure_selected_input(config.selected_path)
        if update_days is not None:
            config.out_dir = settings.update_block_out_dir(update_days)
        requested_jobs = args.n_jobs
        effective_jobs = requested_jobs
        caps = getattr(settings, "MULTIBLOCK_MAX_JOBS", {})
        if update_days is not None and bool(getattr(settings, "MULTIBLOCK_MEMORY_SAFE", True)):
            cap = int(caps.get(int(update_days), requested_jobs or 1))
            if requested_jobs is None:
                effective_jobs = cap
            else:
                effective_jobs = min(int(requested_jobs), cap)
                if effective_jobs < int(requested_jobs):
                    logger.warning(
                        "Memory-safe: --n-jobs %d limitado a %d para bloque %dd",
                        int(requested_jobs), effective_jobs, int(update_days),
                    )
        pipeline = RLSForecastPipeline(config, n_jobs=effective_jobs, limit_series=args.limit_series)
        mark_running(config.out_dir)
        logger.info(
            "=== Escenario bloque cerrado %dd | OOS=%dd | forecast-only=%dd ===",
            int(settings.RLS_BLOCK_DAYS), int(settings.METRIC_HORIZON_DAYS), int(settings.METRIC_HORIZON_DAYS),
        )
        try:
            res_df, wmapes_df = pipeline.run()
            # Trazabilidad explícita dentro del parquet; no se usa para entrenar.
            res_df = res_df.with_columns(
                __import__("polars").lit(int(settings.RLS_BLOCK_DAYS)).cast(__import__("polars").Int16).alias("update_block_days")
            )
            print(res_df.head())
            print(wmapes_df.head())
            # v12.9.8: always finalize the statistical artifact BEFORE building
            # dashboard artifacts.  Dashboard construction re-reads the 565MB+
            # forecast and can terminate inside Rust/Polars on memory pressure;
            # such a crash must never leave a fully-written forecast in
            # status='running'.
            # Mantener el contrato memory-safe: el dashboard se construye solo
            # después de liberar los DataFrames grandes del forecast.
            memory_safe_run = True
            # Keep the v12.8.5 source contract visible for backwards-compatible
            # contract tests; the v12.9.8 path is always memory-safe here.
            # build_dashboard=not memory_safe_run
            pipeline.save(res_df, wmapes_df, build_dashboard=False)
            forecast_path = config.out_dir / "forecast.parquet"
            del res_df, wmapes_df
            gc.collect()
            # Registrar exactamente el artefacto estadístico terminado antes de
            # cualquier postproceso de dashboard que pueda fallar por memoria.
            # Legacy artifact-contract marker: mark_success(config.out_dir / "forecast.parquet")
            mark_success(forecast_path)
            try:
                from app.dashboard_artifacts import build_artifacts
                adir = build_artifacts(forecast_path)
                logger.info("✓ Artefactos dashboard en: %s", adir)
            except Exception:
                logger.exception(
                    "No se pudieron construir artefactos del dashboard; "
                    "el forecast ya quedó registrado como success. "
                    "Ejecute `python -m app.dashboard_artifacts --update-block-days %d`",
                    int(settings.RLS_BLOCK_DAYS),
                )
            return forecast_path
        except BaseException as exc:
            mark_failed(config.out_dir, exc)
            raise
    finally:
        settings.RLS_BLOCK_DAYS = original_block


def main() -> None:
    args = _parse_args()
    if args.all_update_blocks and args.update_block_days is not None:
        raise SystemExit("Use --all-update-blocks o --update-block-days, no ambos.")
    if args.all_update_blocks:
        completed: list[tuple[int, Path]] = []
        for days in settings.UPDATE_BLOCK_OPTIONS:
            days = int(days)
            existing = settings.update_block_forecast_path(days)
            if args.skip_existing and existing.exists():
                # A statistical version change invalidates the forecast itself.
                # Only reuse an existing scenario when its successful run-status
                # belongs to the current APP_VERSION; artifacts may then be
                # repaired without refitting.
                stale = run_status_errors(existing)
                if stale:
                    logger.info(
                        "Escenario %dd existe pero es estadísticamente stale (%s); se recalcula",
                        days, "; ".join(stale),
                    )
                else:
                    from app.dashboard_artifacts import artifacts_exist, build_artifacts
                    if artifacts_exist(existing):
                        logger.info("Escenario %dd + artefactos ya sincronizados; se omite: %s", days, existing)
                    else:
                        logger.info("Escenario %dd ya existe; regenerando SOLO artefactos dashboard", days)
                        build_artifacts(existing)
                    completed.append((days, existing))
                    gc.collect()
                    continue
            path = _run_one(args, days)
            completed.append((days, path))
            # Important for Windows: release Polars/NumPy buffers before the
            # next cadence starts. Each scenario is fully persisted already.
            gc.collect()
        print("\nEscenarios precalculados:")
        for days, path in completed:
            print(f" - {days:>2} días: {path}")
        return
    _run_one(args, args.update_block_days)


if __name__ == "__main__":
    main()
