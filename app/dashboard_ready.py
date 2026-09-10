"""Preparación transaccional del dashboard multi-cadencia.

Objetivo operativo: dejar disponibles y validados los cuatro escenarios
1d/7d/14d/28d sin ejecutar diagnósticos de optimización. Cada bloque se
calcula en un subproceso independiente para liberar memoria entre escenarios.

Uso recomendado:
    uv run python -m app.dashboard_ready --n-jobs 8

Por defecto reutiliza únicamente escenarios de la MISMA APP_VERSION cuya
corrida figure como exitosa; cualquier bloque ausente, stale o fallido se
recalcula. Después repara artefactos, ejecuta la auditoría estructural, el gate end-to-end de no-regresión y la auditoría del dashboard.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import settings
from app import dashboard_artifacts as artifacts
from app.forecasting.run_status import validation_errors as run_status_errors

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, command: list[str]) -> None:
    logger.info("=== %s ===", label)
    logger.info("Ejecutando: %s", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode:
        raise SystemExit(
            f"{label} falló con código {result.returncode}. "
            "La preparación se detuvo para no abrir un dashboard parcialmente validado."
        )
    logger.info("✓ %s", label)


def _readiness_errors() -> dict[int, list[str]]:
    problems: dict[int, list[str]] = {}
    for days in settings.UPDATE_BLOCK_OPTIONS:
        days = int(days)
        fp = settings.update_block_forecast_path(days)
        errs: list[str] = []
        if not fp.exists():
            errs.append(f"falta forecast.parquet: {fp}")
        else:
            errs.extend(run_status_errors(fp))
            if not artifacts.artifacts_exist(fp):
                errs.append("artefactos dashboard ausentes o desactualizados")
        if errs:
            problems[days] = errs
    return problems


def prepare_dashboard_all(*, n_jobs: int | None = None, force_rebuild: bool = False) -> None:
    python = sys.executable

    forecast_cmd = [python, str(PROJECT_ROOT / "app" / "forecasts.py"), "--all-update-blocks"]
    if not force_rebuild:
        forecast_cmd.append("--skip-existing")
    if n_jobs and n_jobs > 0:
        forecast_cmd.extend(["--n-jobs", str(int(n_jobs))])

    _run("Forecasts 1d/7d/14d/28d", forecast_cmd)
    _run(
        "Artefactos dashboard 1d/7d/14d/28d",
        [python, "-m", "app.dashboard_artifacts", "--all-update-blocks", "--skip-existing"],
    )
    _run(
        "Auditoría modelo 1d/7d/14d/28d",
        [python, "-m", "app.forecasting.validate_v13", "--all-update-blocks"],
    )
    _run(
        "Gate end-to-end de no-regresión 1d/7d/14d/28d",
        [python, "-m", "app.forecasting.regression_gate", "--all-update-blocks"],
    )
    _run(
        "Auditoría dashboard 1d/7d/14d/28d",
        [python, "-m", "app.dashboard_consistency", "--all-update-blocks"],
    )

    problems = _readiness_errors()
    if problems:
        lines = ["La preparación terminó, pero aún hay escenarios no listos:"]
        for days, errs in sorted(problems.items()):
            lines.append(f"  {days}d:")
            lines.extend(f"    - {err}" for err in errs)
        raise SystemExit("\n".join(lines))

    print("\nDASHBOARD READY · APP_VERSION=" + str(settings.APP_VERSION))
    print("Escenarios disponibles y validados: 1d, 7d, 14d, 28d")
    print("Abrir con: uv run python main.py --run dashboard")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera/repara/valida todos los escenarios necesarios para el dashboard."
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recalcula incluso escenarios vigentes de la misma APP_VERSION.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    prepare_dashboard_all(n_jobs=args.n_jobs, force_rebuild=bool(args.force_rebuild))


if __name__ == "__main__":
    main()
