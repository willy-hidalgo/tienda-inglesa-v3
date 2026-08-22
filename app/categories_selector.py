"""
Selección de datos de demanda por sección (1 y 23)
==================================================
Pipeline lazy de Polars: predicados push-down, columnas mínimas,
filtro de locales por sección en un solo join.
"""

from __future__ import annotations

import datetime as dt
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
    analysis_end_date: dt.date
    demo_mode: bool

    @classmethod
    def from_settings(cls) -> DemandAnalysisConfig:
        max_end = max(
            cfg["test_end"] for cfg in settings.SECCIONES.values()
        ) + dt.timedelta(days=settings.METRIC_HORIZON_DAYS)
        return cls(
            master_path=Path(settings.MASTER_PATH),
            sales_path=Path(settings.SALES_PATH),
            selected_path=Path(settings.SELECTED_PATH),
            out_dir=Path(settings.OUT_DIR),
            focus_sections=list(settings.FOCUS_SECTIONS),
            analysis_end_date=max_end,
            demo_mode=settings.DEMO_MODE,
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

    _SALES_COLUMNS = ["STORE_ID", "SKU_ID", "SALES_DATE", "SLS_VAL", "SLS_QTY"]
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
        sales_lf = (
            pl.scan_parquet(self._cfg.sales_path)
            .select(self._SALES_COLUMNS)
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


class SectionDemandSelector:
    def __init__(self, focus_sections: list[str], analysis_end_date: dt.date):
        self._focus_sections = focus_sections
        self._analysis_end_date = analysis_end_date

    def select(self, df: pl.DataFrame, demo_mode) -> pl.DataFrame:

        if demo_mode:
            df = self.select_best_skus(df)

        by_sec = (
            df.group_by("SECCION")
            .agg(
                pl.col("SALES_DAY").min().alias("min_date"),
                pl.col("SALES_DAY").max().alias("max_date"),
            )
            .sort("SECCION")
        )
        logger.info("Rango de fechas por sección:\n%s", by_sec)

        min_date = by_sec["min_date"].min()
        out = df.filter(
            (pl.col("SALES_DAY") >= min_date)
            & (pl.col("SALES_DAY") <= self._analysis_end_date)
        )
        logger.info("Filas seleccionadas: %d", out.height)
        return out

    @staticmethod
    def select_best_skus(df: pl.DataFrame) -> pl.DataFrame:
        best_skus = [
            "412091",
            "278120",
            "39472",
            "581783",
            "580661",
            "459977",
            "541427",
            "106356",
            "495933",
            "516028",
            "568160",
            "567332",
            "41796",
            "46499",
            "46853",
            "75048",
            "58905",
            "143871",
            "85860",
            "208",
            "483046",
            "64048",
            "606556",
            "39472",
            "568161",
            "58905",
            "2322",
            "568160",
            "489263",
            "218557",
            "43549",
            "190454",
            "168765",
            "478352",
            "565871",
            "447684",
            "42231",
            "473932",
            "456855",
            "43422",
            "568160",
            "127359",
            "565118",
            "6656",
            "61722",
            "116741",
            "100622",
            "46713",
            "299478",
            "495933",
            "603595",
            "195295",
            "219991",
            "587387",
            "113618",
            "514081",
            "582947",
            "333113",
            "113618",
            "501237",
            "113618",
            "218058",
            "103986",
            "71765",
            "113618",
            "448634",
            "32757",
            "71765",
            "45642",
            "197305",
            "474665",
            "28303",
            "333110",
            "197305",
            "306915",
            "593461",
            "306915",
            "5898",
            "598287",
            "218058",
            "392615",
            "600005",
            "582946",
            "45642",
            "385301",
            "519600",
            "448849",
            "17245",
            "25879",
            "126196",
            "523647",
            "288718",
            "466270",
            "419464",
            "90052",
            "306918",
            "566325",
            "197305",
            "590253",
            "448848",
        ]
        return df.filter(pl.col("SKU_ID").is_in(best_skus))


class DemandAnalysisPipeline:
    def __init__(self, config: DemandAnalysisConfig | None = None):
        self._cfg = config or DemandAnalysisConfig.from_settings()
        self._selector = SectionDemandSelector(
            self._cfg.focus_sections, self._cfg.analysis_end_date
        )

    def run(self) -> pl.DataFrame:
        df = SalesCatalogLoader(self._cfg).load()
        df_common = self._selector.select(df, self._cfg.demo_mode)
        self._cfg.out_dir.mkdir(parents=True, exist_ok=True)
        # Compresión zstd: lectura más rápida y archivo más pequeño
        df_common.write_parquet(
            self._cfg.selected_path,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        logger.info("✓ Archivo guardado en: %s", self._cfg.selected_path)
        return df_common


def main() -> None:
    DemandAnalysisPipeline().run()


if __name__ == "__main__":
    main()
