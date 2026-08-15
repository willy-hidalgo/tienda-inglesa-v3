"""
Pipeline principal para el cálculo de pronósticos y materialización de artefactos para Dashboard.
Genera tanto las series de tiempo como la tabla de rankings precalculada ('rankings.parquet')
soportando evaluación por Cantidad (y, yhat) y por Valor (value, valuehat).
"""

import os

import polars as pl


def build_and_save_rankings(
    df_forecast: pl.DataFrame, output_path: str = "rankings.parquet"
) -> pl.DataFrame:
    """
    Extrae, agrega y precalcula los métricos de ranking (WMAPE, totales, conteos)
    por Sección, Tienda y SKU, para Cantidad y Valor.
    """
    # Parsear componentes del unique_id
    df_parsed = df_forecast.with_columns(
        [
            pl.col("unique_id").str.extract(r"^(\d+)\|\|", 1).alias("seccion"),
            pl.col("unique_id").str.extract(r"\|\|T:([^|]+)", 1).alias("store"),
            pl.col("unique_id").str.extract(r"\|\|S:([^|]+)", 1).alias("sku"),
        ]
    )

    def calc_aggregations(group_cols: list) -> pl.DataFrame:
        """Agrega métricas para Cantidad (y, yhat) y Valor (value, valuehat)."""
        return df_parsed.group_by(group_cols).agg(
            [
                # Cantidad
                pl.col("y").sum().alias("sum_y"),
                pl.col("yhat").sum().alias("sum_yhat"),
                (pl.col("y") - pl.col("yhat")).abs().sum().alias("sum_abs_err_qty"),
                # Valor
                pl.col("value").sum().alias("sum_val"),
                pl.col("valuehat").sum().alias("sum_valuehat"),
                (pl.col("value") - pl.col("valuehat"))
                .abs()
                .sum()
                .alias("sum_abs_err_val"),
                # Conteos
                (pl.col("y") > 0).sum().alias("n_with_sales"),
                pl.len().alias("n_points"),
            ]
        )

    def add_wmape_cols(df: pl.DataFrame) -> pl.DataFrame:
        """Calcula los WMAPE ponderados para cantidad y valor."""
        return df.with_columns(
            [
                pl.when(pl.col("sum_y") > 0)
                .then((pl.col("sum_abs_err_qty") / pl.col("sum_y")) * 100.0)
                .otherwise(0.0)
                .alias("wmape_qty"),
                pl.when(pl.col("sum_val") > 0)
                .then((pl.col("sum_abs_err_val") / pl.col("sum_val")) * 100.0)
                .otherwise(0.0)
                .alias("wmape_val"),
            ]
        )

    # 1. Nivel Hoja (Sección + Tienda + SKU)
    grouped_leaf = calc_aggregations(["seccion", "store", "sku", "unique_id"]).filter(
        pl.col("store").is_not_null()
    )
    rankings_leaf = add_wmape_cols(grouped_leaf)

    # 2. Nivel SKU Puro (Sección + SKU global)
    grouped_sku = calc_aggregations(["seccion", "sku"]).with_columns(
        [
            pl.lit(None).alias("store"),
            (pl.col("seccion") + pl.lit("||S:") + pl.col("sku")).alias("unique_id"),
        ]
    )
    rankings_sku = add_wmape_cols(grouped_sku)

    # 3. Nivel Tienda Pura (Sección + Tienda global)
    grouped_store = (
        calc_aggregations(["seccion", "store"])
        .filter(pl.col("store").is_not_null())
        .with_columns(
            [
                pl.lit(None).alias("sku"),
                (pl.col("seccion") + pl.lit("||T:") + pl.col("store")).alias(
                    "unique_id"
                ),
            ]
        )
    )
    rankings_store = add_wmape_cols(grouped_store)

    cols_select = [
        "seccion",
        "store",
        "sku",
        "unique_id",
        "wmape_qty",
        "wmape_val",
        "sum_y",
        "sum_yhat",
        "sum_val",
        "sum_valuehat",
        "n_with_sales",
        "n_points",
    ]

    rankings_df = pl.concat(
        [
            rankings_leaf.select(cols_select),
            rankings_sku.select(cols_select),
            rankings_store.select(cols_select),
        ],
        how="vertical",
    )

    rankings_df.write_parquet(output_path)
    return rankings_df


def run_pipeline(data_input_path: str, output_dir: str = "."):
    os.makedirs(output_dir, exist_ok=True)
    forecast_file = os.path.join(output_dir, "forecast.parquet")
    rankings_file = os.path.join(output_dir, "rankings.parquet")

    df_forecast = pl.read_parquet(data_input_path)
    df_forecast = df_forecast.sort(["unique_id", "ds"])
    df_forecast.write_parquet(forecast_file)

    build_and_save_rankings(df_forecast, output_path=rankings_file)
    print(f"Pipeline completado. Artefactos guardados en '{output_dir}'.")


if __name__ == "__main__":
    from pathlib import Path

    import polars as pl

    ROOT = Path.cwd()
    OUT_DIR = ROOT / "data" / "output"
    if not OUT_DIR.exists():
        OUT_DIR = ROOT.parent / "data" / "output"

    forecast_path = OUT_DIR / "forecast.parquet"
    rankings_path = str(OUT_DIR / "rankings.parquet")

    if os.path.exists(forecast_path):
        df = pl.read_parquet(forecast_path)
        build_and_save_rankings(df, rankings_path)
