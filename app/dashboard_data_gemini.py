"""
Orquestador de datos del Dashboard.
Soporta consultas de series temporales y rankings filtrados por
Valor (value, valuehat) o por Cantidad (y, yhat).
"""

import os
from pathlib import Path

import polars as pl

ROOT = Path.cwd()
OUT_DIR = ROOT / "data" / "output"
if not OUT_DIR.exists():
    OUT_DIR = ROOT.parent / "data" / "output"


class DashboardDataOrchestrator:
    def __init__(
        self,
        forecast_path: str = "forecast.parquet",
        rankings_path: str = "rankings.parquet",
    ):

        self.forecast_path = OUT_DIR / forecast_path
        self.rankings_path = OUT_DIR / rankings_path

        self.df_forecast: pl.DataFrame | None = None
        self.df_rankings: pl.DataFrame | None = None

        self._load_data()

    def _load_data(self):
        if os.path.exists(self.forecast_path):
            self.df_forecast = pl.read_parquet(self.forecast_path)

        if os.path.exists(self.rankings_path):
            self.df_rankings = pl.read_parquet(self.rankings_path)
        else:
            from forecasting_pipeline import build_and_save_rankings

            if self.df_forecast is not None:
                self.df_rankings = build_and_save_rankings(
                    self.df_forecast, self.rankings_path
                )

    def get_sections(self) -> list[str]:
        if self.df_rankings is None:
            return []
        return (
            self.df_rankings.select("seccion")
            .filter(pl.col("seccion").is_not_null())
            .unique()
            .sort("seccion")
            .get_column("seccion")
            .to_list()
        )

    def get_stores_by_section(self, seccion: str) -> list[str]:
        if self.df_rankings is None:
            return []
        return (
            self.df_rankings.filter(
                (pl.col("seccion") == seccion)
                & (pl.col("store").is_not_null())
                & (pl.col("sku").is_not_null())
            )
            .select("store")
            .unique()
            .sort("store")
            .get_column("store")
            .to_list()
        )

    def get_skus_by_section(self, seccion: str, store: str | None = None) -> list[str]:
        if self.df_rankings is None:
            return []
        query = (pl.col("seccion") == seccion) & (pl.col("sku").is_not_null())
        if store:
            query = query & (pl.col("store") == store)

        return (
            self.df_rankings.filter(query)
            .select("sku")
            .unique()
            .sort("sku")
            .get_column("sku")
            .to_list()
        )

    def get_ranking_table(
        self,
        seccion: str,
        axis: str = "store",
        fixed_peer: str | None = None,
        metric_mode: str = "value",  # "value" o "quantity"
        top_n: int = 10,
    ) -> pl.DataFrame:
        """Retorna la tabla de ranking ordenable por WMAPE de Valor o de Cantidad."""
        if self.df_rankings is None:
            return pl.DataFrame()

        if axis == "store":
            if fixed_peer:
                res = self.df_rankings.filter(
                    (pl.col("seccion") == seccion)
                    & (pl.col("sku") == fixed_peer)
                    & (pl.col("store").is_not_null())
                )
            else:
                res = self.df_rankings.filter(
                    (pl.col("seccion") == seccion)
                    & (pl.col("store").is_not_null())
                    & (pl.col("sku").is_null())
                )
        else:
            if fixed_peer:
                res = self.df_rankings.filter(
                    (pl.col("seccion") == seccion)
                    & (pl.col("store") == fixed_peer)
                    & (pl.col("sku").is_not_null())
                )
            else:
                res = self.df_rankings.filter(
                    (pl.col("seccion") == seccion)
                    & (pl.col("sku").is_not_null())
                    & (pl.col("store").is_null())
                )

        sort_col = "wmape_val" if metric_mode == "value" else "wmape_qty"
        return res.sort(sort_col, descending=True).head(top_n)

    def get_timeseries(
        self,
        seccion: str,
        store: str | None = None,
        sku: str | None = None,
        metric_mode: str = "value",  # "value" o "quantity"
    ) -> pl.DataFrame:
        """Recupera la serie temporal adecuada (y/yhat o value/valuehat)."""
        if self.df_forecast is None:
            return pl.DataFrame()

        real_col = "value" if metric_mode == "value" else "y"
        pred_col = "valuehat" if metric_mode == "value" else "yhat"

        if store and sku:
            target_id = f"{seccion}||T:{store}||S:{sku}"
            filtered = self.df_forecast.filter(pl.col("unique_id") == target_id)
        elif store and not sku:
            target_id_prefix = f"{seccion}||T:{store}||"
            filtered = self.df_forecast.filter(
                pl.col("unique_id").str.starts_with(target_id_prefix)
            )
        elif sku and not store:
            target_id_suffix = f"||S:{sku}"
            filtered = self.df_forecast.filter(
                (pl.col("unique_id").str.starts_with(f"{seccion}||"))
                & (pl.col("unique_id").str.ends_with(target_id_suffix))
            )
        else:
            section_prefix = f"{seccion}||"
            filtered = self.df_forecast.filter(
                pl.col("unique_id").str.starts_with(section_prefix)
            )

        return (
            filtered.group_by("ds")
            .agg(
                [
                    pl.col(real_col).sum().alias("real"),
                    pl.col(pred_col).sum().alias("pred"),
                ]
            )
            .sort("ds")
        )
