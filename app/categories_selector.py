"""
Selección de datos de demanda por sección (1 y 23)
==================================================
Pipeline lazy de Polars: predicados push-down, columnas mínimas,
filtro de locales por sección en un solo join.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DemandAnalysisConfig:
    master_path: Path
    sales_path: Path
    selected_path: Path
    out_dir: Path
    focus_sections: list[str]

    @classmethod
    def from_settings(cls) -> DemandAnalysisConfig:
        # Keep every available actual. OOS/forecast-only horizons are resolved
        # later from the real first/last actual date of each section.
        return cls(
            master_path=Path(settings.MASTER_PATH),
            sales_path=Path(settings.SALES_PATH),
            selected_path=Path(settings.SELECTED_PATH),
            out_dir=Path(settings.OUT_DIR),
            focus_sections=list(settings.FOCUS_SECTIONS),
        )


def _allowed_locales_df() -> pl.DataFrame:
    rows = [
        {"SECCION": sec, "STORE_ID": loc}
        for sec, cfg in settings.SECCIONES.items()
        for loc in cfg["locales"]
    ]
    return pl.DataFrame(rows)


class SalesCatalogLoader:
    """Carga master + ventas en modo lazy y materializa una sola vez."""

    _SALES_COLUMNS = ["STORE_ID", "SKU_ID", "SALES_DATE", "TRAN_TYPE", "SLS_VAL", "SLS_QTY"]
    _MASTER_KEEP = [
        "SKU",
        "SECCION",
        "DESCRIPCION",
        "DESC_CATEGORIA",
        "DESC_SUBCATEGORIA",
        "DESC_FAMILIA",
        "DESC_SUBFAMILIA",
    ]

    def __init__(self, config: DemandAnalysisConfig):
        self._cfg = config

    def load(self) -> pl.DataFrame:
        logger.info("Scan maestro: %s", self._cfg.master_path)
        # Schema sin materializar el dataset completo
        try:
            master_schema = pl.scan_parquet(self._cfg.master_path).collect_schema()
            master_names = set(master_schema.names())
        except Exception:
            master_names = set(pl.read_parquet(self._cfg.master_path, n_rows=0).columns)
        master_cols = [c for c in self._MASTER_KEEP if c in master_names]
        if "SKU" not in master_cols or "SECCION" not in master_cols:
            raise ValueError(f"Master sin SKU/SECCION; columnas: {master_cols}")
        master_lf = (
            pl.scan_parquet(self._cfg.master_path)
            .select(master_cols)
            .with_columns(
                pl.col("SKU").cast(pl.Utf8).str.strip_chars(),
                pl.col("SECCION").cast(pl.Utf8).str.strip_chars(),
            )
            .filter(pl.col("SECCION").is_in(self._cfg.focus_sections))
        )

        logger.info("Scan ventas: %s", self._cfg.sales_path)
        sales_scan = pl.scan_parquet(self._cfg.sales_path)
        sales_names = set(sales_scan.collect_schema().names())
        required_sales = {"STORE_ID", "SKU_ID", "SALES_DATE", "SLS_VAL", "SLS_QTY"}
        missing_sales = sorted(required_sales - sales_names)
        if missing_sales:
            raise ValueError(
                f"Ventas sin columnas requeridas: {missing_sales}"
            )
        sales_cols = [c for c in self._SALES_COLUMNS if c in sales_names]
        sales_lf = (
            sales_scan
            .select(sales_cols)
            .filter((pl.col("SLS_VAL") > 0) | (pl.col("SLS_QTY") > 0))
            .with_columns(
                pl.col("SKU_ID").cast(pl.Utf8).str.strip_chars(),
                pl.col("STORE_ID").cast(pl.Utf8).str.strip_chars(),
                pl.col("SALES_DATE").dt.date().alias("SALES_DAY"),
            )
        )

        allowed = _allowed_locales_df().lazy()

        df = (
            sales_lf.join(master_lf, left_on="SKU_ID", right_on="SKU", how="inner")
            .join(allowed, on=["SECCION", "STORE_ID"], how="inner")
            .collect()
        )
        logger.info("Filas tras join+filtros (lazy→collect): %d", df.height)
        return df


def _log_full_selection_coverage(df: pl.DataFrame) -> None:
    """Audita que la selección materializada contiene todos los SKU+Tienda cargados.

    ``SalesCatalogLoader`` ya aplica únicamente los filtros de dominio del proyecto:
    secciones foco, locales configurados y transacciones de venta positivas. A partir
    de ese punto no existe muestreo, ranking, top-N ni whitelist de SKU.
    """
    if df.height == 0:
        logger.warning("Selección vacía para las secciones foco: %s", settings.FOCUS_SECTIONS)
        return

    pairs = df.select(["SECCION", "STORE_ID", "SKU_ID"]).unique()
    by_sec = (
        pairs.group_by("SECCION")
        .agg(
            pl.len().alias("sku_tienda"),
            pl.col("SKU_ID").n_unique().alias("skus"),
            pl.col("STORE_ID").n_unique().alias("tiendas"),
        )
        .sort("SECCION")
    )
    dates = (
        df.group_by("SECCION")
        .agg(
            pl.col("SALES_DAY").min().alias("min_date"),
            pl.col("SALES_DAY").max().alias("max_date"),
        )
        .sort("SECCION")
    )
    logger.info("Cobertura SKU+Tienda completa (sin muestreo/whitelist):\n%s", by_sec)
    logger.info("Rango de fechas por sección:\n%s", dates)
    logger.info("Filas seleccionadas (historia completa): %d", df.height)


class DemandAnalysisPipeline:
    def __init__(self, config: DemandAnalysisConfig | None = None):
        self._cfg = config or DemandAnalysisConfig.from_settings()

    def run(self) -> pl.DataFrame:
        # El loader ya devuelve el universo completo elegible de las secciones foco.
        # No existe una segunda etapa de selección de SKU: se persiste exactamente
        # todo SKU+Tienda que sobrevivió a los filtros de dominio del loader.
        df_selected = SalesCatalogLoader(self._cfg).load()
        _log_full_selection_coverage(df_selected)
        self._cfg.out_dir.mkdir(parents=True, exist_ok=True)
        # Compresión zstd: lectura más rápida y archivo más pequeño
        df_selected.write_parquet(
            self._cfg.selected_path,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        logger.info("✓ Archivo guardado en: %s", self._cfg.selected_path)
        return df_selected


def main() -> None:
    DemandAnalysisPipeline().run()


if __name__ == "__main__":
    main()
