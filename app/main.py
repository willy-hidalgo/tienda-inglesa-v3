"""
Interfaz operativa del pipeline Tienda Inglesa
==============================================

Todos los procesos operativos y de validación de la versión actual pueden
lanzarse desde este menú. Cada etapa corre como subproceso con el mismo
intérprete Python/uv y con el directorio raíz del proyecto como ``cwd``.

Uso habitual:

    uv run python main.py

También admite ejecución no interactiva, por ejemplo:

    uv run python main.py --run forecast --update-block-days 28 --n-jobs 8
    uv run python main.py --run forecast-all --n-jobs 8
    uv run python main.py --run validate-all
    uv run python main.py --run dashboard

La optimización expuesta aquí mantiene la misma familia productiva SES + RLS;
``optimization-phase2`` genera diagnósticos/refits del mismo RLS, análisis
causal de residuos extremos, calibración diagnóstica del efecto RLS, recurrencia calendario y selección diagnóstica Section/Store por hoja; no introduce modelos alternativos.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MenuOption:
    key: str
    label: str
    command: tuple[str, ...]
    supports_parallel: bool = False
    supports_update_block: bool = False

    def script_path(self) -> Path | None:
        """Devuelve el script .py explícito, si el comando usa uno."""
        for arg in reversed(self.command):
            if arg.endswith(".py"):
                return Path(arg)
        return None

    def command_is_available(self) -> bool:
        """Los comandos ``python -m ...`` no necesitan un path .py explícito."""
        script = self.script_path()
        return script is None or script.exists()


class PipelineCLI:
    _EXIT_KEY = "0"

    def __init__(
        self,
        options: list[MenuOption],
        *,
        project_root: Path,
        default_n_jobs: int | None = None,
        default_update_block_days: int | None = None,
    ) -> None:
        self._options = options
        self._project_root = project_root
        self._default_n_jobs = default_n_jobs
        self._default_update_block_days = self._normalize_update_block(
            default_update_block_days
        )

    @staticmethod
    def _allowed_update_blocks() -> tuple[int, ...]:
        return tuple(int(x) for x in getattr(settings, "UPDATE_BLOCK_OPTIONS", (1, 7, 14, 28)))

    def _normalize_update_block(self, value: int | None) -> int:
        allowed = self._allowed_update_blocks()
        if value in allowed:
            return int(value)
        configured = int(getattr(settings, "UPDATE_BLOCK_DAYS", 28))
        return configured if configured in allowed else allowed[-1]

    def _render_menu(self) -> None:
        print(f"\n=== Pipeline Tienda Inglesa v{settings.APP_VERSION} ===")
        print("Todos los procesos se ejecutan desde la raíz del proyecto.\n")
        for option in self._options:
            suffix: list[str] = []
            if option.supports_update_block:
                suffix.append("bloque 1/7/14/28d")
            if option.supports_parallel:
                suffix.append(
                    f"n_jobs={self._default_n_jobs}"
                    if self._default_n_jobs and self._default_n_jobs > 1
                    else "n_jobs configurable"
                )
            extra = f" [{' · '.join(suffix)}]" if suffix else ""
            print(f"{option.key:>2}. {option.label}{extra}")
        print(f"{self._EXIT_KEY:>2}. salir")

    @staticmethod
    def _prompt_choice() -> str:
        return input("\nSeleccione una opción: ").strip()

    def _prompt_n_jobs(self) -> int | None:
        default = self._default_n_jobs
        hint = f" [{default}]" if default else " [secuencial]"
        raw = input(f"Nº de threads RLS (--n-jobs){hint}: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Valor inválido; se usa el default.")
            return default
        return value if value > 0 else None

    def _prompt_update_block(self) -> int:
        allowed = self._allowed_update_blocks()
        default = self._default_update_block_days
        raw = input(
            f"Bloque de actualización {list(allowed)} [{default}]: "
        ).strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"Valor inválido; se usa {default}d.")
            return default
        if value not in allowed:
            print(f"Bloque no permitido; se usa {default}d.")
            return default
        return value

    def _find_option(self, key: str) -> MenuOption | None:
        return next((option for option in self._options if option.key == key), None)

    def _build_command(
        self,
        option: MenuOption,
        *,
        n_jobs: int | None,
        update_block_days: int | None,
    ) -> list[str]:
        command = list(option.command)
        if option.supports_update_block:
            block = self._normalize_update_block(update_block_days)
            command.extend(["--update-block-days", str(block)])
        if option.supports_parallel and n_jobs and n_jobs > 1:
            command.extend(["--n-jobs", str(n_jobs)])
        return command

    def _execute(
        self,
        option: MenuOption,
        *,
        n_jobs: int | None = None,
        update_block_days: int | None = None,
        interactive: bool = True,
    ) -> int:
        if not option.command_is_available():
            logger.error("No se encontró el script: %s", option.script_path())
            return 2

        if option.supports_update_block:
            if interactive:
                update_block_days = self._prompt_update_block()
            elif update_block_days is None:
                update_block_days = self._default_update_block_days

        if option.supports_parallel:
            if interactive:
                n_jobs = self._prompt_n_jobs()
            elif n_jobs is None:
                n_jobs = self._default_n_jobs

        command = self._build_command(
            option,
            n_jobs=n_jobs,
            update_block_days=update_block_days,
        )
        logger.info("Ejecutando: %s", " ".join(command))
        try:
            result = subprocess.run(
                command,
                check=False,
                cwd=self._project_root,
            )
        except FileNotFoundError as exc:
            logger.error("No se pudo ejecutar el comando (%s): %s", command[0], exc)
            return 127

        if result.returncode == 0:
            logger.info("✓ '%s' finalizó correctamente", option.label)
        else:
            logger.error(
                "'%s' terminó con código de salida %d",
                option.label,
                result.returncode,
            )
        return int(result.returncode)

    def run(self) -> None:
        while True:
            self._render_menu()
            try:
                choice = self._prompt_choice()
            except (EOFError, KeyboardInterrupt):
                print("\nSaliendo...")
                return

            if choice == self._EXIT_KEY:
                print("Saliendo...")
                return

            option = self._find_option(choice)
            if option is None:
                print("Opción inválida, intente de nuevo.")
                continue

            self._execute(option, interactive=True)


def resolve_project_root() -> Path:
    """Resuelve la raíz tanto desde ``app/main.py`` como desde ``main.py``."""
    here = Path(__file__).resolve().parent
    if (here / "forecasts.py").exists() and (here.parent / "settings.py").exists():
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
            "ingestar / actualizar datos",
            (python, str(app / "ingestor.py")),
        ),
        MenuOption(
            "2",
            "seleccionar categorías / secciones",
            (python, str(app / "categories_selector.py")),
        ),
        MenuOption(
            "3",
            "crear pronósticos · un bloque",
            (python, str(app / "forecasts.py")),
            supports_parallel=True,
            supports_update_block=True,
        ),
        MenuOption(
            "4",
            "crear pronósticos · todos los bloques 1/7/14/28d",
            (python, str(app / "forecasts.py"), "--all-update-blocks"),
            supports_parallel=True,
        ),
        MenuOption(
            "5",
            "optimización estadística · SES+RLS + refit drivers + residuos · un bloque · parent leaf",
            (python, str(app / "forecasts.py"), "--optimization-phase2"),
            supports_parallel=True,
            supports_update_block=True,
        ),
        MenuOption(
            "6",
            "construir / reparar artefactos dashboard · un bloque",
            (python, "-m", "app.dashboard_artifacts"),
            supports_update_block=True,
        ),
        MenuOption(
            "7",
            "construir artefactos dashboard para todos los bloques existentes",
            (python, "-m", "app.dashboard_artifacts", "--all-update-blocks"),
        ),
        MenuOption(
            "8",
            "validar modelo v13 · un bloque",
            (python, "-m", "app.forecasting.validate_v13"),
            supports_update_block=True,
        ),
        MenuOption(
            "9",
            "validar modelo v13 · todos los bloques",
            (python, "-m", "app.forecasting.validate_v13", "--all-update-blocks"),
        ),
        MenuOption(
            "10",
            "validar consistencia dashboard · un bloque",
            (python, "-m", "app.dashboard_consistency"),
            supports_update_block=True,
        ),
        MenuOption(
            "11",
            "validar consistencia dashboard · todos los bloques",
            (python, "-m", "app.dashboard_consistency", "--all-update-blocks"),
        ),
        MenuOption(
            "12",
            "reporte de aceptación estadística",
            (python, "-m", "app.forecasting_acceptance_report"),
        ),
        MenuOption(
            "13",
            "ejecutar suite de pruebas pytest",
            (python, "-m", "pytest", "-q"),
        ),
        MenuOption(
            "14",
            "cargar dashboard Streamlit",
            (
                python,
                "-m",
                "streamlit",
                "run",
                str(app / "dashboard.py"),
                "--server.headless",
                "true",
            ),
        ),
        MenuOption(
            "15",
            "PREPARAR DASHBOARD COMPLETO · 1d/7d/14d/28d + auditorías + gate",
            (python, "-m", "app.dashboard_ready"),
            supports_parallel=True,
        ),
        MenuOption(
            "16",
            "gate end-to-end de no-regresión · 1d/7d/14d/28d",
            (python, "-m", "app.forecasting.regression_gate", "--all-update-blocks"),
        ),
    ]


_RUN_ALIASES = {
    "ingest": "1",
    "select": "2",
    "forecast": "3",
    "forecast-all": "4",
    "optimize": "5",
    "optimization-phase2": "5",
    "artifacts": "6",
    "artifacts-all": "7",
    "validate": "8",
    "validate-all": "9",
    "dashboard-check": "10",
    "dashboard-check-all": "11",
    "acceptance": "12",
    "tests": "13",
    "dashboard": "14",
    "dashboard-ready-all": "15",
    "prepare-dashboard-all": "15",
    "regression-gate-all": "16",
    "release-gate": "16",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interfaz operativa Tienda Inglesa v13 · SES + RLS"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Threads para forecasts/RLS. También puede usarse FORECAST_N_JOBS.",
    )
    parser.add_argument(
        "--update-block-days",
        type=int,
        choices=list(getattr(settings, "UPDATE_BLOCK_OPTIONS", (1, 7, 14, 28))),
        default=None,
        help="Bloque para procesos --run que operan sobre un único escenario.",
    )
    run_choices = sorted(set(_RUN_ALIASES) | {str(i) for i in range(1, 17)})
    parser.add_argument(
        "--run",
        type=str,
        choices=run_choices,
        default=None,
        help="Ejecuta una etapa y sale, sin mostrar el menú interactivo.",
    )
    return parser.parse_args(argv)


def _resolve_n_jobs(cli_n_jobs: int | None) -> int | None:
    if cli_n_jobs is not None and cli_n_jobs > 0:
        return cli_n_jobs
    raw = os.environ.get("FORECAST_N_JOBS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    project_root = resolve_project_root()
    cli = PipelineCLI(
        build_default_options(project_root),
        project_root=project_root,
        default_n_jobs=_resolve_n_jobs(args.n_jobs),
        default_update_block_days=args.update_block_days,
    )

    if args.run is None:
        cli.run()
        return

    key = _RUN_ALIASES.get(args.run, args.run)
    option = cli._find_option(key)
    if option is None:
        logger.error("Opción --run inválida: %s", args.run)
        raise SystemExit(2)
    code = cli._execute(
        option,
        n_jobs=_resolve_n_jobs(args.n_jobs),
        update_block_days=args.update_block_days,
        interactive=False,
    )
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
