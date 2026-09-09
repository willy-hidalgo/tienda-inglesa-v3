"""Input preflight for the forecasting CLI.

`forecasts.py` is a production entry point.  A clean checkout/package does not
ship generated files under ``data/output``; therefore this module guarantees
that ``selected.parquet`` exists before the RLS pipeline starts.

The preflight is intentionally idempotent:
- existing selected.parquet -> no work;
- existing master.parquet + sales.parquet -> run selection only;
- raw source files available -> run ingestion, then selection;
- otherwise -> fail immediately with an actionable message.
"""
from __future__ import annotations

import logging
from pathlib import Path

import settings

logger = logging.getLogger(__name__)


def _usable_file(path: Path) -> bool:
    """True when *path* is a non-empty regular file."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def ensure_selected_input(selected_path: Path | str | None = None) -> Path:
    """Ensure the forecasting input parquet exists, building prerequisites only when needed.

    This function never regenerates an already-existing ``selected.parquet``;
    normal forecast runs therefore pay only a couple of filesystem checks.
    """
    selected = Path(selected_path or settings.SELECTED_PATH)
    if _usable_file(selected):
        logger.info("Preflight datos: selected listo | %s", selected)
        return selected

    master = Path(settings.MASTER_PATH)
    sales = Path(settings.SALES_PATH)

    # Fast recovery: generated ingestion artifacts survived, only selection is missing.
    if _usable_file(master) and _usable_file(sales):
        logger.warning(
            "selected.parquet no existe; reconstruyendo selección desde master/sales existentes"
        )
        from app.categories_selector import DemandAnalysisPipeline

        DemandAnalysisPipeline().run()
        if not _usable_file(selected):
            raise RuntimeError(
                f"La selección terminó sin generar el archivo esperado: {selected}"
            )
        return selected

    # Clean/re-extracted project: try to rebuild output artifacts from raw input.
    input_dir = Path(settings.INPUT_DIR)
    master_xlsx = input_dir / settings.MASTER_XLSX_FILENAME
    sales_files = sorted(
        p for p in input_dir.glob(settings.SALES_FILES_GLOB) if p.is_file()
    ) if input_dir.exists() else []

    if not _usable_file(master_xlsx) or not sales_files:
        missing: list[str] = []
        if not _usable_file(master_xlsx):
            missing.append(str(master_xlsx))
        if not sales_files:
            missing.append(str(input_dir / settings.SALES_FILES_GLOB))
        missing_txt = "\n  - ".join(missing)
        raise FileNotFoundError(
            "No se puede preparar selected.parquet porque faltan datos fuente.\n"
            f"Archivos/patrones requeridos:\n  - {missing_txt}\n"
            "Copia los datos a data/input/ y vuelve a ejecutar el mismo comando de forecast."
        )

    logger.warning(
        "Artefactos de salida ausentes; ejecutando preflight automático: ingesta → selección"
    )
    from app.ingestor import DataIngestionPipeline
    from app.categories_selector import DemandAnalysisPipeline

    DataIngestionPipeline().run()
    DemandAnalysisPipeline().run()

    if not _usable_file(selected):
        raise RuntimeError(
            f"El preflight terminó sin generar el archivo esperado: {selected}"
        )
    return selected
