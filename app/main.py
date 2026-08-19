"""
CLI del pipeline Tienda Inglesa
================================

Menú interactivo que ejecuta cada etapa como subproceso con el mismo
intérprete (`sys.executable`).

Forecasts acepta paralelismo por serie:
  - variable de entorno FORECAST_N_JOBS
  - argumento CLI: python -m app.main --n-jobs 4
  - prompt en el menú al elegir "crear pronósticos"

Uso:
  python -m app.main
  python -m app.main --n-jobs 4
  uv run app/main.py --n-jobs 8
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MenuOption:
    key: str
    label: str
    command: list[str]
    # Si True, se pueden añadir flags de paralelismo (--n-jobs)
    supports_parallel: bool = False

    def script_path(self) -> Path | None:
        """Path del script .py en el comando (último arg que termine en .py)."""
        for arg in reversed(self.command):
            if arg.endswith(".py"):
                return Path(arg)
        return None

    def script_exists(self) -> bool:
        p = self.script_path()
        return p is not None and p.exists()


class PipelineCLI:
    _EXIT_KEY = "6"

    def __init__(
        self,
        options: list[MenuOption],
        default_n_jobs: int | None = None,
    ):
        self._options = options
        self._default_n_jobs = default_n_jobs

    def _render_menu(self) -> None:
        n_info = (
            f" (n_jobs={self._default_n_jobs})"
            if self._default_n_jobs and self._default_n_jobs > 1
            else ""
        )
        print("\n=== Pipeline Tienda Inglesa ===")
        for opt in self._options:
            extra = n_info if opt.supports_parallel else ""
            print(f"{opt.key}. {opt.label}{extra}")
        print(f"{self._EXIT_KEY}. salir")

    def _prompt_choice(self) -> str:
        return input("\nSeleccione una opción: ").strip()

    def _prompt_n_jobs(self) -> int | None:
        """Pregunta workers; Enter conserva el default."""
        default = self._default_n_jobs
        hint = f" [{default}]" if default else " [secuencial]"
        raw = input(f"Nº de procesos paralelos para RLS (--n-jobs){hint}: ").strip()
        if not raw:
            return default
        try:
            n = int(raw)
            return n if n > 0 else None
        except ValueError:
            print("Valor inválido; se usa el default.")
            return default

    def _find_option(self, key: str) -> MenuOption | None:
        return next((opt for opt in self._options if opt.key == key), None)

    def _build_command(self, option: MenuOption, n_jobs: int | None) -> list[str]:
        cmd = list(option.command)
        if option.supports_parallel and n_jobs and n_jobs > 1:
            cmd.extend(["--n-jobs", str(n_jobs)])
        return cmd

    def _execute(self, option: MenuOption, n_jobs: int | None = None) -> None:
        if not option.script_exists():
            logger.error(
                "No se encontró el script: %s",
                option.script_path(),
            )
            return

        if option.supports_parallel:
            n_jobs = n_jobs if n_jobs is not None else self._prompt_n_jobs()

        cmd = self._build_command(option, n_jobs)
        logger.info("Ejecutando: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError as exc:
            logger.error("No se pudo ejecutar el comando (%s): %s", cmd[0], exc)
            return

        if result.returncode == 0:
            logger.info("✓ '%s' finalizó correctamente", option.label)
        else:
            logger.error(
                "'%s' terminó con código de salida %d",
                option.label,
                result.returncode,
            )

    def run(self) -> None:
        while True:
            self._render_menu()
            try:
                choice = self._prompt_choice()
            except EOFError, KeyboardInterrupt:
                print("\nSaliendo...")
                return

            if choice == self._EXIT_KEY:
                print("Saliendo...")
                return

            option = self._find_option(choice)
            if option is None:
                print("Opción inválida, intente de nuevo.")
                continue

            self._execute(option)


def resolve_project_root() -> Path:
    """
    main.py puede vivir en:
      - <root>/app/main.py  → root = parent.parent
      - <root>/main.py      → root = parent
    """
    here = Path(__file__).resolve().parent
    if (here / "forecasts.py").exists() or (here / "ingestor.py").exists():
        # estamos dentro de app/
        return here.parent
    if (here / "app" / "forecasts.py").exists():
        return here
    return here.parent


def build_default_options(project_root: Path) -> list[MenuOption]:
    python = sys.executable
    app = project_root / "app"
    return [
        MenuOption(
            "1",
            "data ingestion",
            [python, str(app / "ingestor.py")],
        ),
        MenuOption(
            "2",
            "seleccionar categorías / secciones",
            [python, str(app / "categories_selector.py")],
        ),
        MenuOption(
            "3",
            "crear pronósticos (RLS, paralelizable)",
            [python, str(app / "forecasts.py")],
            supports_parallel=True,
        ),
        MenuOption(
            "4",
            "construir artefactos dashboard (rápido)",
            [python, str(app / "dashboard_artifacts")],
        ),
        MenuOption(
            "5",
            "cargar dashboard",
            [
                python,
                "-m",
                "streamlit",
                "run",
                str(app / "dashboard.py"),
                "--server.headless",
                "true",
            ],
        ),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI Pipeline Tienda Inglesa – secciones 1 y 23"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Workers para forecasts RLS (ProcessPool). "
            "También: env FORECAST_N_JOBS. Default: secuencial."
        ),
    )
    parser.add_argument(
        "--run",
        type=str,
        choices=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "ingest",
            "select",
            "forecast",
            "artifacts",
            "dashboard",
        ],
        default=None,
        help="Ejecuta una etapa y sale (sin menú interactivo).",
    )
    return parser.parse_args(argv)


def _resolve_n_jobs(cli_n_jobs: int | None) -> int | None:
    if cli_n_jobs is not None and cli_n_jobs > 0:
        return cli_n_jobs
    env = os.environ.get("FORECAST_N_JOBS")
    if env:
        try:
            n = int(env)
            return n if n > 0 else None
        except ValueError:
            pass
    return None


_RUN_ALIASES = {
    "ingest": "1",
    "select": "2",
    "forecast": "3",
    "artifacts": "4",
    "dashboard": "5",
}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    n_jobs = _resolve_n_jobs(args.n_jobs)
    project_root = resolve_project_root()
    # Asegurar que el root esté en PYTHONPATH del subproceso vía cwd
    options = build_default_options(project_root)
    cli = PipelineCLI(options, default_n_jobs=n_jobs)

    if args.run is not None:
        key = _RUN_ALIASES.get(args.run, args.run)
        option = cli._find_option(key)
        if option is None:
            logger.error("Opción --run inválida: %s", args.run)
            sys.exit(2)
        # En modo no interactivo, no preguntar n_jobs
        cli._execute(option, n_jobs=n_jobs)
        return

    cli.run()


if __name__ == "__main__":
    main()
