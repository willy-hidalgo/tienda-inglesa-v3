"""Hierarchical Polars aggregation."""
from __future__ import annotations
import polars as pl
import settings
from app.forecasting.utils import _collect_streaming as collect_streaming, _lf_columns as lf_columns

class DataAggregator:
    """
    unique_id — filtros independientes tienda/sku (ver settings.make_unique_id):
      - "1"                      (sección)
      - "1||T:00122"             (tienda, todos los SKU)
      - "1||S:SKU123"            (sku, todas las tiendas)
      - "1||T:00122||S:SKU123"   (tienda + sku)
    Columnas extra: sku_desc, store_name, seccion
    """

    def __init__(
        self,
        date_column: str,
        quantity_column: str,
        price_column: str,
        aggregation_levels: dict,
    ):
        self._date_col = date_column
        self._qty_col = quantity_column
        self._prc_col = price_column
        self._levels = aggregation_levels

    @staticmethod
    def _base_aggs() -> list[pl.Expr]:
        return [
            pl.col("y").sum(),
            pl.col("value").sum(),
            pl.count("y").alias("conteo_sku"),
        ]

    def aggregate(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """
        Agrega solo los niveles de pronóstico: sección, tienda, sku-tienda.
        Acepta DataFrame o LazyFrame; el plan se ejecuta con collect streaming.
        """
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        cols_in = set(lf_columns(lf))

        rename_map = {
            self._date_col: "ds",
            self._qty_col: "y",
            self._prc_col: "value",
        }
        rename_map = {k: v for k, v in rename_map.items() if k in cols_in}
        if rename_map:
            lf = lf.rename(rename_map)
            cols_in = (cols_in - set(rename_map)) | set(rename_map.values())

        lf = lf.filter(pl.col("y") >= 0)

        cast_cols = [
            pl.col(c).cast(pl.Utf8).str.strip_chars()
            for c in ("SECCION", "SKU_ID", "STORE_ID")
            if c in cols_in
        ]
        if cast_cols:
            lf = lf.with_columns(cast_cols)

        # Dimensiones pequeñas en eager (lookup tables)
        store_rows = [
            {"SECCION": sec, "STORE_ID": loc, "store_name": name}
            for sec, cfg in settings.SECCIONES.items()
            for loc, name in cfg["local_names"].items()
        ]
        store_name_df = (
            pl.DataFrame(store_rows)
            if store_rows
            else pl.DataFrame(
                schema={"SECCION": pl.Utf8, "STORE_ID": pl.Utf8, "store_name": pl.Utf8}
            )
        )

        if "DESCRIPCION" in cols_in:
            sku_desc_lf = (
                lf.select(["SKU_ID", "DESCRIPCION"])
                .drop_nulls()
                .unique(subset=["SKU_ID"], maintain_order=False)
                .rename({"DESCRIPCION": "sku_desc"})
            )
        else:
            sku_desc_lf = pl.DataFrame(
                schema={"SKU_ID": pl.Utf8, "sku_desc": pl.Utf8}
            ).lazy()

        base_aggs = self._base_aggs()

        # 1) sección
        lvl_sec = (
            lf.group_by(["ds", "SECCION"])
            .agg(base_aggs)
            .with_columns(
                pl.col("SECCION").alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("sku_desc"),
                pl.lit("").alias("store_name"),
            )
        )

        # 2) sección + tienda
        lvl_store = (
            lf.group_by(["ds", "SECCION", "STORE_ID"])
            .agg(base_aggs)
            .with_columns(
                pl.concat_str(
                    [pl.col("SECCION"), pl.lit("||T:"), pl.col("STORE_ID")]
                ).alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
                pl.lit("").alias("sku_desc"),
            )
            .join(store_name_df.lazy(), on=["SECCION", "STORE_ID"], how="left")
            .with_columns(pl.col("store_name").fill_null(""))
        )

        # 3) sección + tienda + sku
        lvl_store_sku = (
            lf.group_by(["ds", "SECCION", "STORE_ID", "SKU_ID"])
            .agg(base_aggs)
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("SECCION"),
                        pl.lit("||T:"),
                        pl.col("STORE_ID"),
                        pl.lit("||S:"),
                        pl.col("SKU_ID"),
                    ]
                ).alias("unique_id"),
                pl.col("SECCION").alias("seccion"),
            )
            .join(store_name_df.lazy(), on=["SECCION", "STORE_ID"], how="left")
            .join(sku_desc_lf, on="SKU_ID", how="left")
            .with_columns(
                pl.col("store_name").fill_null(""),
                pl.col("sku_desc").fill_null(""),
            )
        )

        out_cols = [
            "unique_id",
            "ds",
            "y",
            "value",
            "conteo_sku",
            "seccion",
            "sku_desc",
            "store_name",
        ]
        out_lf = (
            pl.concat(
                [
                    lvl_sec.select(out_cols),
                    lvl_store.select(out_cols),
                    lvl_store_sku.select(out_cols),
                ],
                how="vertical",
            )
            .with_columns(pl.lit(1).cast(pl.Int8).alias("intercept"))
            .sort(["unique_id", "ds"])
        )
        return collect_streaming(out_lf)
