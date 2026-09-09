"""CLI and compatibility facade for the v13 coherent forecasting pipeline.

The production model is deliberately small and explainable:
RLS at section/store level and one SES+RLS composition at SKU+store level.
The same code path generates in-sample, OOS and forecast-only.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.forecasting import ForecastConfig, RLSForecastPipeline
from app.forecasting.preflight import ensure_selected_input
from app.forecasting.run_status import (
    mark_failed,
    mark_running,
    mark_success,
    validation_errors as run_status_errors,
)
import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline v13: RLS sección/tienda + SES SKU+tienda")
    parser.add_argument("--n-jobs", type=int, default=None, help="Threads paralelos para nodos RLS")
    parser.add_argument("--limit-series", type=int, default=None, help="Debug: limitar hojas SKU+tienda por sección")
    parser.add_argument(
        "--update-block-days",
        type=int,
        choices=list(settings.UPDATE_BLOCK_OPTIONS),
        default=None,
        help="Cadencia cerrada 1/7/14/28d; OOS y forecast-only permanecen en 28d.",
    )
    parser.add_argument(
        "--all-update-blocks",
        action="store_true",
        help="Precalcula secuencialmente los escenarios 1d, 7d, 14d y 28d.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Con --all-update-blocks, reutiliza solo forecasts de la misma APP_VERSION y repara artefactos si hace falta.",
    )
    parser.add_argument(
        "--optimization-diagnostics",
        action="store_true",
        help=(
            "Recolecta diagnósticos SES/RLS/driver de la MISMA arquitectura v13. "
            "No modifica ni promueve parámetros automáticamente."
        ),
    )
    parser.add_argument(
        "--optimization-phase2",
        action="store_true",
        help=(
            "Amplía SOLO el diagnóstico: grilla SES/RLS más fina, refit exacto "
            "de drivers existentes y análisis de residuos extremos SKU+Tienda. "
            "Implica --optimization-diagnostics y no altera forecast."
        ),
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
        if update_days is not None and settings.MULTIBLOCK_MEMORY_SAFE:
            cap = int(settings.MULTIBLOCK_MAX_JOBS.get(int(update_days), requested_jobs or 1))
            effective_jobs = cap if requested_jobs is None else min(int(requested_jobs), cap)
            if requested_jobs is not None and effective_jobs < int(requested_jobs):
                logger.warning(
                    "Memory-safe: --n-jobs %d limitado a %d para bloque %dd",
                    int(requested_jobs), effective_jobs, int(update_days),
                )

        pipeline = RLSForecastPipeline(
            config,
            n_jobs=effective_jobs,
            limit_series=args.limit_series,
            optimization_diagnostics=(
                bool(args.optimization_diagnostics or args.optimization_phase2)
                and bool(getattr(settings, "STAT_OPTIMIZATION_ENABLED", True))
            ),
            optimization_phase2=(
                bool(args.optimization_phase2)
                and bool(getattr(settings, "STAT_OPTIMIZATION_ENABLED", True))
            ),
        )
        mark_running(config.out_dir)
        logger.info(
            "=== v%s | bloque=%dd | OOS=%dd | forecast-only=%dd | modelo=RLS+SES ===",
            settings.APP_VERSION,
            int(settings.RLS_BLOCK_DAYS),
            int(settings.METRIC_HORIZON_DAYS),
            int(settings.METRIC_HORIZON_DAYS),
        )
        try:
            res_df, wmapes_df = pipeline.run()
            res_df = res_df.with_columns(
                pl.lit(int(settings.RLS_BLOCK_DAYS)).cast(pl.Int16).alias("update_block_days")
            )
            pipeline.save(res_df, wmapes_df, build_dashboard=False)
            forecast_path = config.out_dir / "forecast.parquet"

            if args.optimization_diagnostics or args.optimization_phase2:
                try:
                    from app.forecasting.optimization import write_optimization_artifacts

                    opt_dir = write_optimization_artifacts(
                        out_dir=config.out_dir,
                        update_block_days=int(settings.RLS_BLOCK_DAYS),
                        diagnostics=pipeline.optimization_diagnostics(),
                        forecast=res_df,
                    )
                    logger.info("✓ Diagnósticos optimización estadística: %s", opt_dir)
                    summary_path = opt_dir / "optimization_summary.txt"
                    if summary_path.exists():
                        print("\n" + summary_path.read_text(encoding="utf-8"))
                except Exception:
                    logger.exception(
                        "Forecast correcto, pero falló la escritura de diagnósticos de optimización."
                    )

            del res_df, wmapes_df
            gc.collect()

            # The statistical file is complete before the optional dashboard
            # post-process.  A dashboard failure never invalidates the forecast.
            mark_success(forecast_path)
            try:
                from app.dashboard_artifacts import build_artifacts

                adir = build_artifacts(forecast_path)
                logger.info("✓ Artefactos dashboard en: %s", adir)
            except Exception:
                logger.exception(
                    "Forecast correcto, pero faltan artefactos dashboard. "
                    "Ejecute `uv run python -m app.dashboard_artifacts --update-block-days %d`",
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

    if not args.all_update_blocks:
        _run_one(args, args.update_block_days)
        return

    completed: list[tuple[int, Path]] = []
    for days in settings.UPDATE_BLOCK_OPTIONS:
        days = int(days)
        existing = settings.update_block_forecast_path(days)
        if args.skip_existing and existing.exists() and not run_status_errors(existing):
            from app.dashboard_artifacts import artifacts_exist, build_artifacts

            if not artifacts_exist(existing):
                logger.info("Escenario %dd vigente; regenerando solo artefactos dashboard", days)
                build_artifacts(existing)
            else:
                logger.info("Escenario %dd + artefactos vigentes; se omite", days)
            completed.append((days, existing))
            gc.collect()
            continue

        completed.append((days, _run_one(args, days)))
        gc.collect()

    print("\nEscenarios precalculados:")
    for days, path in completed:
        print(f" - {days:>2} días: {path}")


if __name__ == "__main__":
    main()
