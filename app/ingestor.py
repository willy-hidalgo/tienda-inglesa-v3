"""
Ingesta de datos maestros (mercadológico) y de ventas
======================================================
100% Polars. Secciones 1 y 23.
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

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    class _NoOp:
        def __init__(self, iterable=None, *a, **k):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable if self._iterable is not None else [])

        def update(self, n: int = 1) -> None:
            pass

        def set_postfix_str(self, *a, **k) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(iterable=None, *a, **k):
        return _NoOp(iterable, *a, **k)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionConfig:
    input_dir: Path
    output_dir: Path
    master_xlsx_filename: str
    master_dat_filename: str
    sales_glob: str
    encoding: str
    sales_date_format: str

    @classmethod
    def from_settings(cls) -> IngestionConfig:
        return cls(
            input_dir=Path(settings.INPUT_DIR),
            output_dir=Path(settings.OUT_DIR),
            master_xlsx_filename=settings.MASTER_XLSX_FILENAME,
            master_dat_filename=settings.MASTER_DAT_FILENAME,
            sales_glob=settings.SALES_FILES_GLOB,
            encoding=getattr(settings, "SOURCE_ENCODING", "cp1252"),
            sales_date_format=getattr(
                settings, "SALES_DATE_FORMAT", "%d-%m-%Y %H:%M:%S"
            ),
        )


class MasterCatalogIngestor:
    _RAW_COLUMN_NAMES = [
        "SKU",
        "DESCRIPCION",
        "SECCION",
        "CLASIFICACION",
        "CATEGORIA",
        "DESC CATEGORIA",
        "SUBCATEGORIA",
        "DESCSUBCATEGORIA",
        "FAMILIA",
        "DESC FAMILIA",
        "SUBFAMILIA",
        "DESC SUBFAMILIA",
    ]

    def __init__(self, config: IngestionConfig):
        self._cfg = config
        self._column_names = self._normalize_column_names(self._RAW_COLUMN_NAMES)

    @staticmethod
    def _normalize_column_names(raw_names: list[str]) -> list[str]:
        return [
            name.replace(" ", "_")
            if name != "DESCSUBCATEGORIA"
            else "DESC_SUBCATEGORIA"
            for name in raw_names
        ]

    def _read_xlsx(self) -> pl.DataFrame:
        xls_path = self._cfg.input_dir / self._cfg.master_xlsx_filename
        logger.info("Leyendo catálogo maestro (xlsx): %s", xls_path)
        df = pl.read_excel(
            xls_path,
            schema_overrides={name: pl.Utf8 for name in self._RAW_COLUMN_NAMES},
        )
        df.columns = self._column_names
        # Normalizar SECCION a string limpio
        if "SECCION" in df.columns:
            df = df.with_columns(
                pl.col("SECCION").cast(pl.Utf8).str.strip_chars().alias("SECCION")
            )
        return df

    def _read_dat(self) -> pl.DataFrame | None:
        dat_path = self._cfg.input_dir / self._cfg.master_dat_filename
        if not dat_path.exists():
            logger.warning("Archivo .dat no encontrado, se omite: %s", dat_path)
            return None
        try:
            df_dat = pl.read_csv(
                dat_path,
                separator="|",
                has_header=False,
                encoding=self._cfg.encoding,
                ignore_errors=True,
                truncate_ragged_lines=True,
                quote_char=None,
            )
        except Exception:
            logger.exception("Error al leer el archivo .dat: %s", dat_path)
            return None
        if df_dat.width > len(self._column_names):
            df_dat = df_dat.select(df_dat.columns[: len(self._column_names)])
        df_dat.columns = self._column_names
        df_dat = df_dat.select([pl.col(c).cast(pl.Utf8) for c in df_dat.columns])
        if "SECCION" in df_dat.columns:
            df_dat = df_dat.with_columns(
                pl.col("SECCION").str.strip_chars().alias("SECCION")
            )
        if df_dat.height == 0:
            return None
        logger.info("Se agregarán %d filas del archivo .dat", df_dat.height)
        return df_dat

    def run(self) -> pl.DataFrame:
        df = self._read_xlsx()
        df_dat = self._read_dat()
        if df_dat is not None:
            df = pl.concat([df, df_dat], how="vertical").unique(keep="first")
        # Filtrar solo secciones de interés
        if "SECCION" in df.columns:
            before = df.height
            df = df.filter(pl.col("SECCION").is_in(settings.FOCUS_SECTIONS))
            logger.info(
                "Filtro secciones %s: %d → %d filas",
                settings.FOCUS_SECTIONS,
                before,
                df.height,
            )
        logger.info(
            "Catálogo maestro consolidado: %d filas, %d columnas", df.height, df.width
        )
        return df

    def save(self, df: pl.DataFrame) -> Path:
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._cfg.output_dir / "master.parquet"
        df.write_parquet(
            out_path, compression="zstd", compression_level=3, statistics=True
        )
        logger.info("✓ Archivo guardado en: %s", out_path)
        return out_path


class SalesIngestor:
    _COLUMN_NAMES = [
        "SALES_DATE",
        "SKU_ID",
        "STORE_ID",
        "TRAN_TYPE",
        "SLS_VAL",
        "SLS_QTY",
        "RTRN_QTY",
        "RTRN_VAL",
    ]
    _NUMERIC_COLUMNS = ("SLS_VAL", "SLS_QTY", "RTRN_QTY", "RTRN_VAL")

    def __init__(self, config: IngestionConfig):
        self._cfg = config

    def _discover_files(self) -> list[Path]:
        files = sorted(self._cfg.input_dir.glob(self._cfg.sales_glob))
        logger.info("Archivos de ventas encontrados: %d", len(files))
        return files

    def _scan_one(self, path: Path) -> pl.LazyFrame:
        return (
            pl.read_csv(
                path,
                separator="|",
                has_header=False,
                encoding=self._cfg.encoding,
                new_columns=self._COLUMN_NAMES,
                schema_overrides={name: pl.Utf8 for name in self._COLUMN_NAMES},
            )
            .lazy()
            .with_columns(
                pl.col("SALES_DATE").str.strptime(
                    pl.Datetime, self._cfg.sales_date_format
                ),
                *[pl.col(c).cast(pl.Float64) for c in self._NUMERIC_COLUMNS],
            )
        )

    @staticmethod
    def _empty_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "SALES_DATE": pl.Datetime,
                "SKU_ID": pl.Utf8,
                "STORE_ID": pl.Utf8,
                "TRAN_TYPE": pl.Utf8,
                "SLS_VAL": pl.Float64,
                "SLS_QTY": pl.Float64,
                "RTRN_QTY": pl.Float64,
                "RTRN_VAL": pl.Float64,
            }
        )

    def run(self) -> pl.DataFrame:
        files = self._discover_files()
        if not files:
            logger.warning(
                "No se encontraron archivos con patrón '%s'", self._cfg.sales_glob
            )
            return self._empty_frame()
        lazy_frames = [
            self._scan_one(path)
            for path in tqdm(
                files,
                desc="Escaneando ventas",
                unit="archivo",
                ncols=50,  # Controla el ancho total
                ascii="░█",  # Define los caracteres de llenado (vacío/lleno)
                bar_format="{bar} [{n_fmt}/{total_fmt}] {desc}...",  # Estructura del texto
            )
        ]
        sales_df = (
            pl.concat(lazy_frames, how="vertical_relaxed")
            .unique(keep="first")
            .collect()
        )
        logger.info("✓ Éxito: %d filas de ventas consolidadas", sales_df.height)
        return sales_df

    def save(self, df: pl.DataFrame) -> Path:
        self._cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._cfg.output_dir / "sales.parquet"
        df.write_parquet(
            out_path, compression="zstd", compression_level=3, statistics=True
        )
        logger.info("✓ Archivo guardado en: %s", out_path)
        return out_path


class DataIngestionPipeline:
    def __init__(self, config: IngestionConfig | None = None):
        self._cfg = config or IngestionConfig.from_settings()

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        master_ingestor = MasterCatalogIngestor(self._cfg)
        master_df = master_ingestor.run()
        master_ingestor.save(master_df)

        sales_ingestor = SalesIngestor(self._cfg)
        sales_df = sales_ingestor.run()
        sales_ingestor.save(sales_df)
        return master_df, sales_df


def main() -> None:
    DataIngestionPipeline().run()


if __name__ == "__main__":
    main()
